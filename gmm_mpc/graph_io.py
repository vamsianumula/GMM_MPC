import numpy as np

def generate_erdos_renyi(n, p, seed=None):
    """Generates an Erdos-Renyi graph G(n, p)."""
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    return edges

def generate_barabasi_albert(n, m, seed=None):
    """Generates a Barabasi-Albert graph with n vertices and m edges to attach from a new vertex."""
    if m < 1 or m >= n:
        raise ValueError("Barabasi-Albert model requires 1 <= m < n")

    rng = np.random.default_rng(seed)
    edges = []
    # Start with a small connected graph
    for i in range(m):
        edges.append((i, (i + 1) % m))

    repeated_nodes = []
    for i in range(m):
        repeated_nodes.extend([i, (i+1)%m])

    source = m
    while source < n:
        # Add m edges from the new node to existing nodes
        targets = rng.choice(repeated_nodes, size=m, replace=False)
        for target in targets:
            edges.append((source, target))
            repeated_nodes.extend([source, target])
        source += 1
    return edges
