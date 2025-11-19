def get_owner(vertex_id, p):
    """
    Determines the owner rank for a vertex using modulo partitioning (Fix B.8).

    Args:
        vertex_id: Integer vertex identifier
        p: Number of MPI ranks

    Returns:
        Owner rank in [0, p-1]

    Note:
        This function must be pure and consistent across all ranks.
        Assumes vertex_id is an integer.
    """
    return vertex_id % p

def partition_edges(edges, p, rank):
    """
    Partitions edges among p ranks.
    Each rank gets the edges where the owner of the min vertex id is the rank.

    Args:
        edges: List of (u, v) tuples
        p: Number of MPI ranks
        rank: Current rank

    Returns:
        local_edges: List of edges owned by this rank
    """
    local_edges = []
    for u, v in edges:
        min_v = min(u, v)
        if get_owner(min_v, p) == rank:
            local_edges.append((u, v))
    return local_edges
