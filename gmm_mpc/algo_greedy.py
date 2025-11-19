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

def final_greedy(comm, local_edges, owned_matched, foreign_matched, seed, metrics_tracker):
    """
    Randomized greedy maximal matching with mutual agreement and targeted matched notifications.

    Memory compliance changes:
    - No global gather/broadcast of matched edges/vertices.
    - Owners send targeted 'matched vertex' notices to only ranks that contributed edges incident to that vertex.
    - Each rank maintains:
        * owned_matched: vertices owned by this rank that are matched
        * foreign_matched: non-owned vertices learned as matched (limited to those appearing in local edges)

    Returns:
        local_matched_total (int): Number of edges matched in greedy phase by this rank (confirmation at owner(u)).
    """
    rank = comm.Get_rank()
    p = comm.Get_size()

    # Assign priorities to all local alive edges using a deterministic, symmetric hash
    # This ensures that the priority for an edge (u, v) is the same regardless of which rank computes it or the total number of ranks.
    edge_priorities = {}  # (min(u,v), max(u,v)) -> priority
    for u, v in local_edges:
        if not _is_vertex_matched(u, rank, p, owned_matched, foreign_matched) and \
           not _is_vertex_matched(v, rank, p, owned_matched, foreign_matched):
            k = (u, v) if u < v else (v, u)
            if k not in edge_priorities:
                # Use a simple hashing function based on vertex IDs and the global seed
                # to generate a deterministic and consistent priority.
                hash_val = (k[0] * 31 + k[1] + seed) * (k[0] * 31 + k[1] + seed)
                edge_priorities[k] = (hash_val % 1_000_000_007) / 1_000_000_007

    # Initial contributor tracking via Alltoallv of [vertex, 1] for alive edges
    contribs = {}
    for u, v in local_edges:
        if not _is_vertex_matched(u, rank, p, owned_matched, foreign_matched) and \
           not _is_vertex_matched(v, rank, p, owned_matched, foreign_matched):
            owner_u = partition.get_owner(u, p)
            owner_v = partition.get_owner(v, p)
            contribs.setdefault(owner_u, []).extend([u, 1])
            contribs.setdefault(owner_v, []).extend([v, 1])

    send_buffer, send_counts, send_displacements = _prepare_alltoallv_buffers(contribs, p, dtype=np.int32)
    recv_counts = np.empty_like(send_counts)
    comm.Alltoall(send_counts, recv_counts)
    recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)
    recv_buffer = np.empty(sum(recv_counts), dtype=np.int32)

    comm.Alltoallv(
        [send_buffer, (send_counts, send_displacements), MPI.INT],
        [recv_buffer, (recv_counts, recv_displacements), MPI.INT],
    )
    metrics_tracker.record_communication(send_counts, np.dtype(np.int32).itemsize, "greedy_contributors")

    contributor_ranks = {}  # v_owned_by_rank -> set(source_ranks)
    offset = 0
    for src in range(p):
        count_elems = recv_counts[src]
        i = offset
        while i < offset + count_elems:
            v = recv_buffer[i]
            # c = recv_buffer[i + 1]  # count unused here
            if partition.get_owner(v, p) == rank:
                s = contributor_ranks.get(v)
                if s is None:
                    s = set()
                    contributor_ranks[v] = s
                s.add(src)
            i += 2
        offset += count_elems

    local_matched_total = 0
    micro_round = 0
    max_micro_rounds = 100  # Safety bound

    while micro_round < max_micro_rounds:
        micro_round += 1

        # Step 1: Each owned vertex proposes its best incident edge (by priority)
        edges_by_owned_vertex = {}  # vertex_owned -> (u, v, priority)
        for u, v in local_edges:
            if not _is_vertex_matched(u, rank, p, owned_matched, foreign_matched) and \
               not _is_vertex_matched(v, rank, p, owned_matched, foreign_matched):
                key = (u, v) if u < v else (v, u)
                pr = edge_priorities.get(key)
                if pr is None:
                    continue
                owner_u = partition.get_owner(u, p)
                owner_v = partition.get_owner(v, p)
                # If u is owned by this rank, consider u->v
                if owner_u == rank:
                    prev = edges_by_owned_vertex.get(u)
                    if prev is None or pr > prev[2]:
                        edges_by_owned_vertex[u] = (u, v, pr)
                # If v is owned by this rank, consider v->u
                if owner_v == rank:
                    prev = edges_by_owned_vertex.get(v)
                    if prev is None or pr > prev[2]:
                        edges_by_owned_vertex[v] = (v, u, pr)

        proposals_to_send = {}  # dest_rank -> [u, v, priority_scaled, ...]
        for vertex, (u, v, pr) in edges_by_owned_vertex.items():
            owner_u = partition.get_owner(u, p)
            proposals_to_send.setdefault(owner_u, []).extend([u, v, int(pr * 1e9)])

        # Step 2: First Alltoallv - proposals to owner(u)
        send_buffer, send_counts, send_displacements = _prepare_alltoallv_buffers(proposals_to_send, p, dtype=np.int64)
        recv_counts = np.empty_like(send_counts)
        comm.Alltoall(send_counts, recv_counts)
        recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)
        recv_buffer = np.empty(sum(recv_counts), dtype=np.int64)

        comm.Alltoallv(
            [send_buffer, (send_counts, send_displacements), MPI.LONG_LONG],
            [recv_buffer, (recv_counts, recv_displacements), MPI.LONG_LONG],
        )
        metrics_tracker.record_communication(send_counts, np.dtype(np.int64).itemsize, "greedy_proposals")

        proposals_at_u = {}  # u -> (u, v, pr)
        for i in range(0, len(recv_buffer), 3):
            u, v, pr_scaled = recv_buffer[i], recv_buffer[i + 1], recv_buffer[i + 2]
            pr = pr_scaled / 1e9
            if not _is_vertex_matched(u, rank, p, owned_matched, foreign_matched):
                prev = proposals_at_u.get(u)
                if prev is None or pr > prev[2]:
                    proposals_at_u[u] = (u, v, pr)

        # Step 3: Second Alltoallv - forward winners at u to owner(v)
        forward_to_v = {}
        for u, (uu, v, pr) in proposals_at_u.items():
            owner_v = partition.get_owner(v, p)
            forward_to_v.setdefault(owner_v, []).extend([u, v, int(pr * 1e9)])

        send_buffer, send_counts, send_displacements = _prepare_alltoallv_buffers(forward_to_v, p, dtype=np.int64)
        recv_counts = np.empty_like(send_counts)
        comm.Alltoall(send_counts, recv_counts)
        recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)
        recv_buffer = np.empty(sum(recv_counts), dtype=np.int64)

        comm.Alltoallv(
            [send_buffer, (send_counts, send_displacements), MPI.LONG_LONG],
            [recv_buffer, (recv_counts, recv_displacements), MPI.LONG_LONG],
        )
        metrics_tracker.record_communication(send_counts, np.dtype(np.int64).itemsize, "greedy_forward")

        winners_at_v = {}  # v_owned -> (u, v, pr)
        matched_owned_vertices_v = set()
        for i in range(0, len(recv_buffer), 3):
            u, v, pr_scaled = recv_buffer[i], recv_buffer[i + 1], recv_buffer[i + 2]
            pr = pr_scaled / 1e9
            if partition.get_owner(v, p) == rank and not _is_vertex_matched(v, rank, p, owned_matched, foreign_matched):
                prev = winners_at_v.get(v)
                if prev is None or pr > prev[2]:
                    winners_at_v[v] = (u, v, pr)

        # Owner(v) marks v as matched immediately for accepted winners
        for v, (u, vv, pr) in winners_at_v.items():
            if partition.get_owner(v, p) == rank:
                owned_matched.add(v)
                matched_owned_vertices_v.add(v)
                # We learned u will be matched; mark foreign locally for faster filtering
                if partition.get_owner(u, p) != rank:
                    foreign_matched.add(u)

        # Step 4: Third Alltoallv - send ACCEPT back to owner(u)
        accepts_to_u = {}
        for v, (u, vv, pr) in winners_at_v.items():
            owner_u = partition.get_owner(u, p)
            accepts_to_u.setdefault(owner_u, []).extend([u, v, int(pr * 1e9)])

        send_buffer, send_counts, send_displacements = _prepare_alltoallv_buffers(accepts_to_u, p, dtype=np.int64)
        recv_counts = np.empty_like(send_counts)
        comm.Alltoall(send_counts, recv_counts)
        recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)
        recv_buffer = np.empty(sum(recv_counts), dtype=np.int64)

        comm.Alltoallv(
            [send_buffer, (send_counts, send_displacements), MPI.LONG_LONG],
            [recv_buffer, (recv_counts, recv_displacements), MPI.LONG_LONG],
        )
        metrics_tracker.record_communication(send_counts, np.dtype(np.int64).itemsize, "greedy_accepts")

        # Step 5: Owner(u) confirms, matches and updates local sets
        local_matched_this_round = 0
        matched_owned_vertices_u = set()
        matched_foreign_vertices_local = set()
        for i in range(0, len(recv_buffer), 3):
            u, v, pr_scaled = recv_buffer[i], recv_buffer[i + 1], recv_buffer[i + 2]
            pr = pr_scaled / 1e9
            if u in proposals_at_u:
                uu, vv, upr = proposals_at_u[u]
                if vv == v and abs(upr - pr) < 1e-6:
                    if not _is_vertex_matched(u, rank, p, owned_matched, foreign_matched) and \
                       not _is_vertex_matched(v, rank, p, owned_matched, foreign_matched):
                        owned_matched.add(u)
                        matched_owned_vertices_u.add(u)
                        local_matched_this_round += 1
                        # mark v foreign locally for filtering
                        if partition.get_owner(v, p) != rank:
                            foreign_matched.add(v)
                            matched_foreign_vertices_local.add(v)

        # Step 6: Targeted matched vertex notifications for owned vertices matched this round
        owned_matched_vertices_this_round = set()
        owned_matched_vertices_this_round.update(matched_owned_vertices_u)
        owned_matched_vertices_this_round.update(matched_owned_vertices_v)

        notices_to_send = {}  # dest_rank -> [vertex_id, ...] dtype int64
        for v in owned_matched_vertices_this_round:
            dests = contributor_ranks.get(v, set())
            for d in dests:
                if d == rank:
                    continue
                notices_to_send.setdefault(d, []).append(v)

        send_buffer, send_counts, send_displacements = _prepare_alltoallv_buffers(notices_to_send, p, dtype=np.int64)
        recv_counts = np.empty_like(send_counts)
        comm.Alltoall(send_counts, recv_counts)
        recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)
        recv_buffer = np.empty(sum(recv_counts), dtype=np.int64)

        comm.Alltoallv(
            [send_buffer, (send_counts, send_displacements), MPI.LONG_LONG],
            [recv_buffer, (recv_counts, recv_displacements), MPI.LONG_LONG],
        )
        metrics_tracker.record_communication(send_counts, np.dtype(np.int64).itemsize, "greedy_matched_notices")

        # Process received matched notices (foreign markers)
        for i in range(0, len(recv_buffer)):
            v = recv_buffer[i]
            if partition.get_owner(v, p) != rank:
                foreign_matched.add(v)

        # Convergence check
        any_changes = comm.allreduce(local_matched_this_round > 0, op=MPI.LOR)
        # Record per-micro-round matched count (optional granularity)
        metrics_tracker.record_matched(comm.allreduce(local_matched_this_round, op=MPI.SUM))
        local_matched_total += local_matched_this_round

        if not any_changes:
            if rank == 0:
                print(f"  Greedy converged after {micro_round} micro-rounds")
            break

    if micro_round >= max_micro_rounds and rank == 0:
        print(f"  Warning: Greedy reached max micro-rounds ({max_micro_rounds})")

    return local_matched_total
