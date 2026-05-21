from utils import euclidean, segment_intersection

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

def link_quality(i, j, nodes):
    d = euclidean(nodes[i], nodes[j])
    if d >= MAX_RANGE:
        return None
    return (1.0 - d / MAX_RANGE) ** 2

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
            q = link_quality(i, j, nodes)
            if q is None:
                continue
            adj[i].append((j, q))
            radj[j].append((i, q))
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
            for j, q in radj[i]:
                if j not in relay_set_with_source:
                    continue
                p_not_rx *= (1.0 - q * p_reach[j])
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
            q * p_reach[j]
            for j, q in radj[i]
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

def compute_node_fingerprints(nodes, adj, radj):
    n = len(nodes)
    utility_sum = [0.0] * n
    utility_sq_sum = [0.0] * n
    coverage_sum = [0.0] * n
    redundancy_sum = [0.0] * n
    participation_count = [0] * n
    for source in range(n):
        relay_set = set(range(n)) - {source}
        p_full = simulate_propagation(source, relay_set, adj, radj, n)
        q_full = propagation_quality(source, relay_set, adj, radj, n, p_full)
        for node in relay_set:
            reduced_relay_set = relay_set - {node}
            p_reduced = simulate_propagation(source, reduced_relay_set, adj, radj, n)
            q_reduced = propagation_quality(source, reduced_relay_set, adj, radj, n, p_reduced)
            utility = q_full - q_reduced
            utility_sum[node] += utility
            utility_sq_sum[node] += utility * utility
            coverage_sum[node] += p_full[node]
            expected_rx = sum(
                q * p_full[j]
                for j, q in radj[node]
                if j in relay_set | {source}
            )
            redundancy_sum[node] += max(0.0, expected_rx - p_full[node])
            participation_count[node] += 1
    fingerprints = {}
    for node in range(n):
        count = max(participation_count[node], 1)
        mean_utility = utility_sum[node] / count
        mean_utility_sq = utility_sq_sum[node] / count
        utility_variance = max(0.0, mean_utility_sq - mean_utility * mean_utility)
        fingerprints[node] = {
            "mean_utility": mean_utility,
            "utility_variance": utility_variance,
            "mean_coverage": coverage_sum[node] / count,
            "mean_redundancy": redundancy_sum[node] / count,
            "degree": len(adj[node]),
            "weighted_in_degree": sum(q for _j, q in radj[node]),
            "mean_out_signal": (
                sum(q for _j, q in adj[node]) / len(adj[node])
                if adj[node] else 0.0
            ),
            "mean_in_signal": (
                sum(q for _j, q in radj[node]) / len(radj[node])
                if radj[node] else 0.0
            ),
        }
    return fingerprints

def extract_structural_features(node, adj, radj):
    out_q = [q for _j, q in adj[node]]
    in_q = [q for _j, q in radj[node]]
    neighbor_degrees = [len(adj[j]) for j, _q in adj[node]]
    return {
        "degree": len(out_q),
        "mean_out_signal": sum(out_q) / len(out_q) if out_q else 0.0,
        "mean_in_signal": sum(in_q) / len(in_q) if in_q else 0.0,
        "neighbor_mean_degree": (
            sum(neighbor_degrees) / len(neighbor_degrees)
            if neighbor_degrees else 0.0
        ),
        "weighted_in_degree": sum(in_q),
    }

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
