import numpy as np
from mpi4py import MPI
from . import partition

def _prepare_alltoallv_buffers(items_to_send, p, dtype=np.int64):
    """Prepare buffers for Alltoallv communication."""
    send_data = [items_to_send.get(i, []) for i in range(p)]
    send_counts = np.array([len(x) for x in send_data], dtype=int)
    send_buffer = np.array([item for sublist in send_data for item in sublist], dtype=dtype)
    send_displacements = np.insert(np.cumsum(send_counts)[:-1], 0, 0)
    return send_buffer, send_counts, send_displacements

def _is_vertex_matched(vertex, rank, p, owned_matched, foreign_matched):
    """Check if a vertex is matched using owned and foreign sets."""
    owner = partition.get_owner(vertex, p)
    if owner == rank:
        return vertex in owned_matched
    else:
        return vertex in foreign_matched

def peeling_round(comm, local_edges, owned_matched, foreign_matched, tau, seed, metrics_tracker):
    """
    Performs one round of degree peeling with mutual agreement protocol and targeted matched notifications.

    Memory compliance changes:
    - No global gather/broadcast of matched edges/vertices.
    - Owners send targeted 'matched vertex' notices to only ranks that contributed edges incident to that vertex.
    - Each rank maintains:
        * owned_matched: vertices owned by this rank that are matched
        * foreign_matched: non-owned vertices this rank has learned are matched (only those that appear in local edges)

    Algorithm:
    1. Degree estimation with local aggregation; track contributor ranks per owned vertex.
    2. Identify high-degree owned vertices.
    3. Sample bounded proposals per high-degree vertex (k=2).
    4. Three-way Alltoallv for mutual agreement:
       - Proposals to owner(u)
       - Forward to owner(v)
       - Accept back to owner(u)
    5. Confirm and match (owner(u)); also record matches (owner(v)).
    6. Targeted Alltoallv: owners send 'matched vertex' notices to contributor ranks.
    7. Update foreign_matched with received notices.

    Returns:
        local_matched_count (int): Number of edges matched this round where this rank participated (confirmation at owner(u)).
    """
    rank = comm.Get_rank()
    p = comm.Get_size()

    # Step 1: Degree Estimation (with local aggregation and contributor tracking)
    # Build contributions only for alive edges (both endpoints unmatched per local knowledge)
    degree_contribs = {}  # dest_rank -> [vertex, count, ...] with dtype int32
    for u, v in local_edges:
        if not _is_vertex_matched(u, rank, p, owned_matched, foreign_matched) and \
           not _is_vertex_matched(v, rank, p, owned_matched, foreign_matched):
            owner_u = partition.get_owner(u, p)
            owner_v = partition.get_owner(v, p)
            degree_contribs.setdefault(owner_u, []).extend([u, 1])
            degree_contribs.setdefault(owner_v, []).extend([v, 1])

    send_buffer, send_counts, send_displacements = _prepare_alltoallv_buffers(
        degree_contribs, p, dtype=np.int32
    )
    recv_counts = np.empty_like(send_counts)
    comm.Alltoall(send_counts, recv_counts)
    recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)
    recv_buffer = np.empty(sum(recv_counts), dtype=np.int32)

    comm.Alltoallv(
        [send_buffer, (send_counts, send_displacements), MPI.INT],
        [recv_buffer, (recv_counts, recv_displacements), MPI.INT],
    )
    metrics_tracker.record_communication(send_counts, np.dtype(np.int32).itemsize, "degree_estimation")

    # Reconstruct degrees and contributor ranks from [vertex, count] pairs
    local_degrees = {}            # v -> deg
    contributor_ranks = {}        # v_owned_by_rank -> set(source_ranks)
    offset = 0
    for src in range(p):
        count_elems = recv_counts[src]
        i = offset
        while i < offset + count_elems:
            v = recv_buffer[i]
            c = recv_buffer[i + 1]
            local_degrees[v] = local_degrees.get(v, 0) + c
            # Only track contributors for vertices we own
            if partition.get_owner(v, p) == rank:
                s = contributor_ranks.get(v)
                if s is None:
                    s = set()
                    contributor_ranks[v] = s
                s.add(src)
            i += 2
        offset += count_elems

    # High-degree vertices owned by this rank
    high_degree_vertices = {v for v, deg in local_degrees.items() if deg > tau and partition.get_owner(v, p) == rank}

    # Step 2: Proposal Generation with Sampling
    edges_by_vertex = {}  # owned high-degree vertex -> [(u, v)]
    for u, v in local_edges:
        if not _is_vertex_matched(u, rank, p, owned_matched, foreign_matched) and \
           not _is_vertex_matched(v, rank, p, owned_matched, foreign_matched):
            # If u is owned and high-degree, include (u, v)
            if partition.get_owner(u, p) == rank and u in high_degree_vertices:
                edges_by_vertex.setdefault(u, []).append((u, v))
            # If v is owned and high-degree, include (v, u)
            if partition.get_owner(v, p) == rank and v in high_degree_vertices:
                edges_by_vertex.setdefault(v, []).append((v, u))

    # Sample at most k=2 proposals per high-degree vertex
    proposals_to_send = {}  # dest_rank -> [u, v, weight_scaled, ...] dtype int64
    rng = np.random.default_rng(seed=rank + seed)
    k = 2

    for vertex, incident_edges in edges_by_vertex.items():
        num_samples = min(k, len(incident_edges))
        if num_samples > 0:
            sampled_indices = rng.choice(len(incident_edges), size=num_samples, replace=False)
            for idx in sampled_indices:
                u, v = incident_edges[idx]
                w = rng.random()
                owner_u = partition.get_owner(u, p)
                proposals_to_send.setdefault(owner_u, []).extend([u, v, int(w * 1e9)])

    # Step 3: First Alltoallv - Send proposals to owner(u)
    send_buffer, send_counts, send_displacements = _prepare_alltoallv_buffers(
        proposals_to_send, p, dtype=np.int64
    )
    recv_counts = np.empty_like(send_counts)
    comm.Alltoall(send_counts, recv_counts)
    recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)
    recv_buffer = np.empty(sum(recv_counts), dtype=np.int64)

    comm.Alltoallv(
        [send_buffer, (send_counts, send_displacements), MPI.LONG_LONG],
        [recv_buffer, (recv_counts, recv_displacements), MPI.LONG_LONG],
    )
    metrics_tracker.record_communication(send_counts, np.dtype(np.int64).itemsize, "proposals")

    # Step 4: Owner(u) selects best proposal per u
    received_proposals = {}  # u -> [(u, v, w)]
    for i in range(0, len(recv_buffer), 3):
        u, v, w = recv_buffer[i], recv_buffer[i + 1], recv_buffer[i + 2]
        received_proposals.setdefault(u, []).append((u, v, w / 1e9))

    best_proposals_u = {}  # u_owned -> (u, v, w)
    for u in high_degree_vertices:
        if u in received_proposals and not _is_vertex_matched(u, rank, p, owned_matched, foreign_matched):
            best_prop = max(received_proposals[u], key=lambda x: x[2])
            best_proposals_u[u] = best_prop

    # Step 5: Second Alltoallv - Forward best proposals to owner(v)
    forward_to_v = {}
    for u, (u_val, v, w) in best_proposals_u.items():
        owner_v = partition.get_owner(v, p)
        forward_to_v.setdefault(owner_v, []).extend([u, v, int(w * 1e9)])

    send_buffer, send_counts, send_displacements = _prepare_alltoallv_buffers(
        forward_to_v, p, dtype=np.int64
    )
    recv_counts = np.empty_like(send_counts)
    comm.Alltoall(send_counts, recv_counts)
    recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)
    recv_buffer = np.empty(sum(recv_counts), dtype=np.int64)

    comm.Alltoallv(
        [send_buffer, (send_counts, send_displacements), MPI.LONG_LONG],
        [recv_buffer, (recv_counts, recv_displacements), MPI.LONG_LONG],
    )
    metrics_tracker.record_communication(send_counts, np.dtype(np.int64).itemsize, "forward_to_v")

    # Step 6: Owner(v) selects winner per v (only for owned v)
    proposals_at_v = {}  # v_owned -> [(u, v, w)]
    for i in range(0, len(recv_buffer), 3):
        u, v, w = recv_buffer[i], recv_buffer[i + 1], recv_buffer[i + 2]
        proposals_at_v.setdefault(v, []).append((u, v, w / 1e9))

    winners_at_v = {}  # v_owned -> (u, v, w)
    matched_owned_vertices_v = set()
    for v, props in proposals_at_v.items():
        if partition.get_owner(v, p) == rank and not _is_vertex_matched(v, rank, p, owned_matched, foreign_matched):
            best_prop = max(props, key=lambda x: x[2])
            winners_at_v[v] = best_prop
            matched_owned_vertices_v.add(v)  # v will be matched

    # Step 7: Third Alltoallv - Send ACCEPT back to owner(u)
    accepts_to_u = {}
    for v, (u, v_val, w) in winners_at_v.items():
        owner_u = partition.get_owner(u, p)
        accepts_to_u.setdefault(owner_u, []).extend([u, v, int(w * 1e9)])

    send_buffer, send_counts, send_displacements = _prepare_alltoallv_buffers(
        accepts_to_u, p, dtype=np.int64
    )
    recv_counts = np.empty_like(send_counts)
    comm.Alltoall(send_counts, recv_counts)
    recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)
    recv_buffer = np.empty(sum(recv_counts), dtype=np.int64)

    comm.Alltoallv(
        [send_buffer, (send_counts, send_displacements), MPI.LONG_LONG],
        [recv_buffer, (recv_counts, recv_displacements), MPI.LONG_LONG],
    )
    metrics_tracker.record_communication(send_counts, np.dtype(np.int64).itemsize, "accepts")

    # Step 8: Owner(u) confirms match (mutual agreement) and record local matches
    local_matched_edges = []
    matched_owned_vertices_u = set()
    matched_foreign_vertices_local = set()
    for i in range(0, len(recv_buffer), 3):
        u, v, w_scaled = recv_buffer[i], recv_buffer[i + 1], recv_buffer[i + 2]
        w = w_scaled / 1e9
        if u in best_proposals_u:
            u_best_u, u_best_v, u_best_w = best_proposals_u[u]
            if u_best_v == v and abs(u_best_w - w) < 1e-6:
                # Ensure endpoints are currently unmatched
                if not _is_vertex_matched(u, rank, p, owned_matched, foreign_matched) and \
                   not _is_vertex_matched(v, rank, p, owned_matched, foreign_matched):
                    local_matched_edges.append((u, v))
                    matched_owned_vertices_u.add(u)
                    # we learned v is matched; mark foreign locally for faster filtering
                    matched_foreign_vertices_local.add(v)

    # Update owned/foreign matched for this rank immediately
    for u in matched_owned_vertices_u:
        owned_matched.add(u)
    for v in matched_foreign_vertices_local:
        foreign_matched.add(v)
    for v in matched_owned_vertices_v:
        owned_matched.add(v)

    # Step 9: Targeted matched vertex notifications (owners notify contributor ranks)
    # Build notices for vertices owned by this rank that became matched (union of u-owned and v-owned)
    owned_matched_vertices_this_round = set()
    owned_matched_vertices_this_round.update(matched_owned_vertices_u)
    owned_matched_vertices_this_round.update(matched_owned_vertices_v)

    notices_to_send = {}  # dest_rank -> [vertex_id, ...] dtype int64
    for v in owned_matched_vertices_this_round:
        dests = contributor_ranks.get(v, set())
        for d in dests:
            if d == rank:
                # Local contributor: nothing to send; already updated locally
                continue
            notices_to_send.setdefault(d, []).append(v)

    send_buffer, send_counts, send_displacements = _prepare_alltoallv_buffers(
        notices_to_send, p, dtype=np.int64
    )
    recv_counts = np.empty_like(send_counts)
    comm.Alltoall(send_counts, recv_counts)
    recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)
    recv_buffer = np.empty(sum(recv_counts), dtype=np.int64)

    comm.Alltoallv(
        [send_buffer, (send_counts, send_displacements), MPI.LONG_LONG],
        [recv_buffer, (recv_counts, recv_displacements), MPI.LONG_LONG],
    )
    metrics_tracker.record_communication(send_counts, np.dtype(np.int64).itemsize, "matched_notices")

    # Step 10: Process received matched vertex notices (mark foreign_matched)
    for i in range(0, len(recv_buffer)):
        v = recv_buffer[i]
        # If we do not own v, record it as foreign matched (only if relevant for local edges)
        if partition.get_owner(v, p) != rank:
            foreign_matched.add(v)

    # Return the number of locally confirmed matches (for metrics/progress)
    return len(local_matched_edges)
