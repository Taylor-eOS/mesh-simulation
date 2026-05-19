import math
import random
from utils import ccw, segment_intersection, dot, sigmoid

NODES = [
    (100, 80),
    (80, 420),
    (210, 260),
    (490, 270),
    (580, 110),
    (620, 390),
    (150, 320),
    (300, 150),
    (400, 400),
    (650, 200),
    (210, 200),
]
WALLS = [
    ((320, 0), (300, 210)),
    ((360, 500), (340, 290)),
    ((560, 270), (700, 270)),
]
LATENT_DIM = 3
LEARN_RATE = 0.01
EPOCHS = 10000
EPOCH_INTERVAL = 100
PRESSURE_ITERATIONS = 30
PRESSURE_DECAY = 0.92
PRESSURE_GAIN = 0.18
PRESSURE_COST = 4.0

def nodes_connected(i, j):
    for w in WALLS:
        if segment_intersection(NODES[i], NODES[j], w[0], w[1]):
            return False
    return True

def signal_strength(i, j):
    d = math.hypot(
        NODES[j][0] - NODES[i][0],
        NODES[j][1] - NODES[i][1]
    )
    if d >= 500:
        return None
    return -130 + (1 - d / 500) * 100

def build_adjacency():
    n = len(NODES)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if not nodes_connected(i, j):
                continue
            s = signal_strength(i, j)
            if s is not None:
                adj[i].append((j, s))
    return adj

def link_cost(sig):
    norm = (sig + 130) / 100
    return 1.0 / (norm ** 3)

def pressure_weight(pressure):
    return 1.0 + PRESSURE_COST * pressure

def dijkstra(origin, adj, n, pressure):
    dist = [float("inf")] * n
    pred = [-1] * n
    used = [False] * n
    dist[origin] = 0
    for _ in range(n):
        u = -1
        best = float("inf")
        for i in range(n):
            if not used[i] and dist[i] < best:
                best = dist[i]
                u = i
        if u == -1:
            break
        used[u] = True
        for v, sig in adj[u]:
            base = link_cost(sig)
            congestion = pressure_weight(pressure[v])
            cost = base * congestion
            nd = dist[u] + cost
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = u
    paths = {}
    for t in range(n):
        if t == origin:
            continue
        if dist[t] == float("inf"):
            continue
        cur = t
        path = []
        while cur != -1:
            path.append(cur)
            cur = pred[cur]
        path.reverse()
        paths[t] = path
    return paths

def compute_pressure(adj, n):
    pressure = [0.0 for _ in range(n)]
    for _ in range(PRESSURE_ITERATIONS):
        traffic = [0.0 for _ in range(n)]
        for source in range(n):
            paths = dijkstra(source, adj, n, pressure)
            for dest, path in paths.items():
                if len(path) <= 2:
                    continue
                for hop_index, node in enumerate(path[1:-1], start=1):
                    remaining = len(path) - hop_index
                    traffic[node] += 1.0 + remaining * 0.15
        peak = max(max(traffic), 1e-9)
        for i in range(n):
            norm = traffic[i] / peak
            pressure[i] = (
                pressure[i] * PRESSURE_DECAY +
                norm * PRESSURE_GAIN
            )
    peak = max(max(pressure), 1e-9)
    for i in range(n):
        pressure[i] /= peak
    return pressure

def relay_probability(node_pressure, path_len):
    congestion = 1.0 - node_pressure
    stretch = 1.0 / max(path_len, 1)
    x = congestion * 0.75 + stretch * 0.25
    return max(0.02, min(0.98, x))

def build_q_table(adj, n, pressure):
    q = {}
    for source in range(n):
        paths = dijkstra(source, adj, n, pressure)
        for dest, path in paths.items():
            relay_nodes = set(path[1:-1])
            for prev_hop in range(n):
                for node in range(n):
                    if node == source:
                        continue
                    if node == dest:
                        continue
                    state = (source, dest, prev_hop, node)
                    if node in relay_nodes:
                        target = relay_probability(
                            pressure[node],
                            len(path)
                        )
                    else:
                        target = 0.0
                    q[state] = target
    return q

class PolicyModel:
    def __init__(self, n, d):
        self.n = n
        self.d = d
        self.S = [
            [random.gauss(0, 0.2) for _ in range(d)]
            for _ in range(n)
        ]
        self.D = [
            [random.gauss(0, 0.2) for _ in range(d)]
            for _ in range(n)
        ]
        self.P = [
            [random.gauss(0, 0.2) for _ in range(d)]
            for _ in range(n)
        ]
        self.N = [
            [random.gauss(0, 0.2) for _ in range(d)]
            for _ in range(n)
        ]
        self.pressure_bias = [
            random.gauss(0, 0.1)
            for _ in range(n)
        ]
        self.bias = random.gauss(0, 0.1)

    def score(self, source, dest, prev_hop, node, pressure):
        s = 0.0
        s += dot(self.S[source], self.N[node])
        s += dot(self.D[dest], self.N[node])
        s += dot(self.P[prev_hop], self.N[node])
        s += dot(self.S[source], self.D[dest])
        s += pressure[node] * self.pressure_bias[node]
        s += self.bias
        return s

    def predict(self, source, dest, prev_hop, node, pressure):
        return sigmoid(
            self.score(
                source,
                dest,
                prev_hop,
                node,
                pressure
            )
        )

def train(model, q_table, pressure):
    states = list(q_table.keys())
    for epoch in range(EPOCHS):
        random.shuffle(states)
        total_loss = 0.0
        correct = 0
        for state in states:
            source, dest, prev_hop, node = state
            target = q_table[state]
            pred = model.predict(
                source,
                dest,
                prev_hop,
                node,
                pressure
            )
            err = pred - target
            if (pred >= 0.5) == (target >= 0.5):
                correct += 1
            total_loss += -(
                target * math.log(pred + 1e-9) +
                (1 - target) * math.log(1 - pred + 1e-9)
            )
            grad = err
            s_vec = model.S[source]
            d_vec = model.D[dest]
            p_vec = model.P[prev_hop]
            n_vec = model.N[node]
            s_old = s_vec[:]
            d_old = d_vec[:]
            p_old = p_vec[:]
            n_old = n_vec[:]
            for k in range(model.d):
                s_vec[k] -= LEARN_RATE * grad * (
                    n_old[k] + d_old[k]
                )
                d_vec[k] -= LEARN_RATE * grad * (
                    n_old[k] + s_old[k]
                )
                p_vec[k] -= LEARN_RATE * grad * (
                    n_old[k]
                )
                n_vec[k] -= LEARN_RATE * grad * (
                    s_old[k] +
                    d_old[k] +
                    p_old[k]
                )
            model.pressure_bias[node] -= (
                LEARN_RATE *
                grad *
                pressure[node]
            )
            model.bias -= LEARN_RATE * grad
        acc = correct / len(states)
        if epoch % EPOCH_INTERVAL == 0:
            print(
                f"epoch={epoch:5d}  "
                f"loss={total_loss/len(states):.5f}  "
                f"acc={acc:.1%}"
            )
        if acc > 0.995:
            print(f"[✓] converged at epoch {epoch}")
            break

def print_pressure(pressure):
    print("PRESSURE FIELD")
    for i, p in enumerate(pressure):
        print(f"Node {i}  pressure={p:.3f}")

def print_embeddings(model):
    print("LATENT POLICY FINGERPRINTS")
    for i in range(model.n):
        s = " ".join(f"{v:+.3f}" for v in model.S[i])
        d = " ".join(f"{v:+.3f}" for v in model.D[i])
        p = " ".join(f"{v:+.3f}" for v in model.P[i])
        n = " ".join(f"{v:+.3f}" for v in model.N[i])
        print(f"\nNode {i}")
        print(f"  source-role : {s}")
        print(f"  dest-role   : {d}")
        print(f"  prev-role   : {p}")
        print(f"  node-role   : {n}")
        print(f"  pressure-b  : {model.pressure_bias[i]:+.3f}")

def print_policy(model, q_table, pressure, sample_size=40):
    print("\nPOLICY RECONSTRUCTION")
    count = 0
    for state, target in sorted(q_table.items()):
        if count >= sample_size:
            break
        source, dest, prev_hop, node = state
        pred = model.predict(
            source,
            dest,
            prev_hop,
            node,
            pressure
        )
        if pred < 0.01 and target < 0.01:
            continue
        label = 1 if pred >= 0.5 else 0
        actual = 1 if target >= 0.5 else 0
        ok = "✓" if label == actual else "✗"
        print(
            f"S={source} "
            f"D={dest} "
            f"P={prev_hop} "
            f"N={node} "
            f"pressure={pressure[node]:.2f} "
            f"target={target:.3f} "
            f"pred={pred:.3f} "
            f"{ok}"
        )
        count += 1

def main():
    random.seed(42)
    n = len(NODES)
    adj = build_adjacency()
    pressure = compute_pressure(adj, n)
    q_table = build_q_table(
        adj,
        n,
        pressure
    )
    print(f"nodes: {n}")
    print(f"states: {len(q_table)}")
    print_pressure(pressure)
    model = PolicyModel(
        n=n,
        d=LATENT_DIM
    )
    train(
        model,
        q_table,
        pressure
    )
    print_embeddings(model)
    print_policy(
        model,
        q_table,
        pressure
    )

if __name__ == "__main__":
    main()

