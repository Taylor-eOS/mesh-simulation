from utils import euclidean, normalize_signal, segment_intersection

PROPAGATION_ITERATIONS = 25
COVERAGE_WEIGHT = 1.0
REDUNDANCY_PENALTY = 0.4
CONGESTION_PENALTY = 0.15
MAX_RANGE = 500.0

def nodes_connected(i, j, nodes, walls):
    for w in walls:
        if segment_intersection(nodes[i], nodes[j], w[0], w[1]):
            return False
    return True

def signal_strength(i, j, nodes):
    d = euclidean(nodes[i], nodes[j])
    if d >= MAX_RANGE:
        return None
    return -130.0 + (1.0 - d / MAX_RANGE) * 100.0

def reception_prob(sig):
    norm = normalize_signal(sig)
    return norm ** 2

def build_adjacency(nodes, walls):
    n = len(nodes)
    adj = {i: [] for i in range(n)}
    radj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if not nodes_connected(i, j, nodes, walls):
                continue
            sig = signal_strength(i, j, nodes)
            if sig is None:
                continue
            adj[i].append((j, sig))
            radj[j].append((i, sig))
    return adj, radj

def simulate_propagation(source, relay_set, adj, radj, n):
    relay_set_with_source = relay_set | {source}
    p_reach = [0.0] * n
    p_reach[source] = 1.0
    for _ in range(PROPAGATION_ITERATIONS):
        new_p = list(p_reach)
        for i in range(n):
            if i == source:
                continue
            p_not_rx = 1.0
            for j, sig in radj[i]:
                if j not in relay_set_with_source:
                    continue
                p_not_rx *= (1.0 - reception_prob(sig) * p_reach[j])
            new_p[i] = 1.0 - p_not_rx
        delta = max(abs(new_p[i] - p_reach[i]) for i in range(n))
        p_reach = new_p
        if delta < 1e-6:
            break
    return p_reach

def propagation_quality(source, relay_set, adj, radj, n, p_reach=None):
    if p_reach is None:
        p_reach = simulate_propagation(source, relay_set, adj, radj, n)
    relay_set_with_source = relay_set | {source}
    coverage = sum(p_reach[i] for i in range(n) if i != source) / max(n - 1, 1)
    total_redundancy = 0.0
    for i in range(n):
        if i == source:
            continue
        expected_rx = sum(
            reception_prob(sig) * p_reach[j]
            for j, sig in radj[i]
            if j in relay_set_with_source
        )
        total_redundancy += max(0.0, expected_rx - p_reach[i])
    redundancy = total_redundancy / max(n - 1, 1)
    congestion = sum(p_reach[j] for j in relay_set) / max(n - 1, 1)
    return (
        COVERAGE_WEIGHT * coverage
        - REDUNDANCY_PENALTY * redundancy
        - CONGESTION_PENALTY * congestion
    )

def bfs_hops(source, adj, n):
    hops = [-1] * n
    hops[source] = 0
    queue = [source]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v, _sig in adj[u]:
            if hops[v] == -1:
                hops[v] = hops[u] + 1
                queue.append(v)
    return hops

def oracle_marginal_utility(node, source, relay_set, adj, radj, n, p_full=None):
    q_full = propagation_quality(source, relay_set, adj, radj, n, p_full)
    p_reduced = simulate_propagation(source, relay_set - {node}, adj, radj, n)
    q_reduced = propagation_quality(source, relay_set - {node}, adj, radj, n, p_reduced)
    return q_full - q_reduced

def extract_local_features(node, source, relay_set, adj, radj, n, p_reach=None, hops=None):
    if p_reach is None:
        p_reach = simulate_propagation(source, relay_set, adj, radj, n)
    if hops is None:
        hops = bfs_hops(source, adj, n)
    relay_set_with_source = relay_set | {source}
    incoming = [
        (sig, p_reach[j])
        for j, sig in radj[node]
        if j in relay_set_with_source
    ]
    expected_rx_count = sum(reception_prob(sig) * pj for sig, pj in incoming)
    best_signal = max((sig for sig, _pj in incoming), default=-130.0)
    mean_signal = (
        sum(sig for sig, _pj in incoming) / len(incoming)
        if incoming else -130.0
    )
    return {
        "expected_rx_count": expected_rx_count,
        "best_signal": best_signal,
        "mean_signal": mean_signal,
        "neighbor_count": len(adj[node]),
        "hop_count": hops[node] if hops[node] >= 0 else -1,
        "node_p_reach": p_reach[node],
    }

def generate_training_sample(node, source, nodes, adj, radj):
    n = len(nodes)
    relay_set = set(range(n)) - {source}
    p_full = simulate_propagation(source, relay_set, adj, radj, n)
    hops = bfs_hops(source, adj, n)
    features = extract_local_features(node, source, relay_set, adj, radj, n, p_full, hops)
    utility = oracle_marginal_utility(node, source, relay_set, adj, radj, n, p_full)
    return features, utility

def oracle_label(node, source, nodes, adj, radj):
    _features, utility = generate_training_sample(node, source, nodes, adj, radj)
    return utility

def load_nodes(path):
    nodes = []
    walls = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.rstrip(",").split("),")]
            if len(parts) == 1:
                coords = parts[0].strip("()")
                x, y = coords.split(",")
                nodes.append((int(x.strip()), int(y.strip())))
            elif len(parts) == 2:
                c1 = parts[0].strip("()")
                c2 = parts[1].strip("()")
                x1, y1 = c1.split(",")
                x2, y2 = c2.split(",")
                walls.append((
                    (int(x1.strip()), int(y1.strip())),
                    (int(x2.strip()), int(y2.strip()))
                ))
    return nodes, walls
