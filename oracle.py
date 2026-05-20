import math
from utils import euclidean, normalize_signal, segment_intersection

PRESSURE_ITERATIONS = 30
PRESSURE_DECAY = 0.92
PRESSURE_GAIN = 0.18
PRESSURE_COST = 4.0
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
    return -130 + (1.0 - d / MAX_RANGE) * 100.0

def build_adjacency(nodes, walls):
    n = len(nodes)
    adj = {i: [] for i in range(n)}
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
    return adj

def link_cost(sig):
    norm = normalize_signal(sig)
    return 1.0 / (norm ** 3)

def pressure_weight(pressure):
    return 1.0 + PRESSURE_COST * pressure

def shortest_path(source, dest, adj, pressure):
    n = len(adj)
    dist = [float("inf")] * n
    pred = [-1] * n
    used = [False] * n
    dist[source] = 0.0
    for _ in range(n):
        u = -1
        best = float("inf")
        for i in range(n):
            if not used[i] and dist[i] < best:
                best = dist[i]
                u = i
        if u == -1:
            break
        if u == dest:
            break
        used[u] = True
        for v, sig in adj[u]:
            base = link_cost(sig)
            congestion = pressure_weight(pressure[v])
            nd = dist[u] + base * congestion
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = u
    if dist[dest] == float("inf"):
        return None
    path = []
    cur = dest
    while cur != -1:
        path.append(cur)
        cur = pred[cur]
    path.reverse()
    return path

def dijkstra(origin, adj, pressure):
    n = len(adj)
    paths = {}
    for dest in range(n):
        if dest == origin:
            continue
        path = shortest_path(origin, dest, adj, pressure)
        if path is not None:
            paths[dest] = path
    return paths

def compute_pressure(adj):
    n = len(adj)
    pressure = [0.0 for _ in range(n)]
    for _ in range(PRESSURE_ITERATIONS):
        traffic = [0.0 for _ in range(n)]
        for source in range(n):
            for dest in range(n):
                if source == dest:
                    continue
                path = shortest_path(source, dest, adj, pressure)
                if path is None:
                    continue
                if len(path) <= 2:
                    continue
                for hop_index, node in enumerate(path[1:-1], start=1):
                    remaining = len(path) - hop_index
                    load = 1.0 + remaining * 0.25
                    traffic[node] += load
        peak = max(max(traffic), 1e-9)
        next_pressure = [0.0 for _ in range(n)]
        for i in range(n):
            normalized = traffic[i] / peak
            next_pressure[i] = pressure[i] * PRESSURE_DECAY + normalized * PRESSURE_GAIN
        pressure = next_pressure
    peak = max(max(pressure), 1e-9)
    for i in range(n):
        pressure[i] /= peak
    return pressure

def estimate_relay_utility(current, neighbor, dest, nodes, pressure, sig):
    current_distance = euclidean(nodes[current], nodes[dest])
    neighbor_distance = euclidean(nodes[neighbor], nodes[dest])
    progress_gain = (current_distance - neighbor_distance) / MAX_RANGE
    signal_quality = normalize_signal(sig)
    congestion_penalty = pressure[neighbor]
    utility = (progress_gain * 0.50 + signal_quality * 0.35 - congestion_penalty * 0.40)
    return utility

def oracle_label(current, neighbor, dest, nodes, pressure, sig):
    utility = estimate_relay_utility(current, neighbor, dest, nodes, pressure, sig)
    return max(0.0, min(1.0, utility + 0.5))

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

