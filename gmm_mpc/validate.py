def preprocess_edges(edges):
    """
    Remove self-loops and deduplicate edges (Fix B.9).

    Args:
        edges: List of (u, v) tuples

    Returns:
        clean_edges: List of edges without self-loops or duplicates
    """
    clean_edges = []
    seen = set()
    for u, v in edges:
        if u == v:  # Self-loop
            continue
        # Normalize edge representation
        edge = tuple(sorted([u, v]))
        if edge not in seen:
            seen.add(edge)
            clean_edges.append((u, v))
    return clean_edges

def is_matching(matching):
    """
    Checks if a set of edges is a valid matching.

    A valid matching has no vertex appearing in more than one edge.

    Args:
        matching: List of (u, v) tuples

    Returns:
        True if valid matching, False otherwise
    """
    # First preprocess to handle any edge cases
    matching = preprocess_edges(matching)

    seen_vertices = set()
    for u, v in matching:
        if u in seen_vertices or v in seen_vertices:
            return False
        seen_vertices.add(u)
        seen_vertices.add(v)
    return True

def is_maximal(matching, all_edges):
    """
    Checks if a matching is maximal.

    A matching is maximal if no additional edge can be added without
    violating the matching property (i.e., no edge exists with both
    endpoints unmatched).

    Args:
        matching: List of (u, v) tuples representing the matching
        all_edges: List of all edges in the original graph

    Returns:
        True if maximal, False otherwise
    """
    # Preprocess both sets
    matching = preprocess_edges(matching)
    all_edges = preprocess_edges(all_edges)

    matched_vertices = set()
    for u, v in matching:
        matched_vertices.add(u)
        matched_vertices.add(v)

    for u, v in all_edges:
        if u not in matched_vertices and v not in matched_vertices:
            return False  # Found an edge that could be added

    return True
