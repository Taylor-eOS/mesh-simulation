import os
import math
import random
import pickle
from concurrent.futures import ProcessPoolExecutor
from utils import ccw, segment_intersection, dot, sigmoid

LATENT_DIM = 16
MAX_EPOCHS = 10000
EPOCH_INTERVAL = 10
PRESSURE_ITERATIONS = 30
PRESSURE_DECAY = 0.92
PRESSURE_GAIN = 0.18
PRESSURE_COST = 4.0

def nodes_connected(i, j, nodes, walls):
    for w in walls:
        if segment_intersection(nodes[i], nodes[j], w[0], w[1]):
            return False
    return True

def signal_strength(i, j, nodes):
    d = math.hypot(
        nodes[j][0] - nodes[i][0],
        nodes[j][1] - nodes[i][1]
    )
    if d >= 500:
        return None
    return -130 + (1 - d / 500) * 100

def build_adjacency(nodes, walls):
    n = len(nodes)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if not nodes_connected(i, j, nodes, walls):
                continue
            s = signal_strength(i, j, nodes)
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

def _dijkstra_worker(args):
    source, adj, n, pressure = args
    return source, dijkstra(source, adj, n, pressure)

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
            congestion = 1.0 + pressure[v] * PRESSURE_COST
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

def compute_pressure(adj, n):
    pressure = [0.0 for _ in range(n)]
    for iteration in range(PRESSURE_ITERATIONS):
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
            next_pressure[i] = (
                pressure[i] * PRESSURE_DECAY + normalized * PRESSURE_GAIN
            )
        pressure = next_pressure
    peak = max(max(pressure), 1e-9)
    for i in range(n):
        pressure[i] /= peak
    return pressure

def relay_probability(node_pressure, path_len):
    congestion = 1.0 - node_pressure
    stretch = 1.0 / max(path_len, 1)
    x = congestion * 0.75 + stretch * 0.25
    return max(0.02, min(0.98, x))

def build_training_data(adj, nodes, pressure):
    samples = []
    n = len(nodes)
    for source in range(n):
        paths = dijkstra(source, adj, n, pressure)
        for dest, path in paths.items():
            if len(path) < 2:
                continue
            for hop_index in range(len(path) - 1):
                current = path[hop_index]
                if hop_index == 0:
                    prev_hop = current
                else:
                    prev_hop = path[hop_index - 1]
                correct_next = path[hop_index + 1]
                current_distance = math.hypot(
                    nodes[dest][0] - nodes[current][0],
                    nodes[dest][1] - nodes[current][1],
                )
                for neighbor, sig in adj[current]:
                    if neighbor == prev_hop:
                        continue
                    neighbor_distance = math.hypot(
                        nodes[dest][0] - nodes[neighbor][0],
                        nodes[dest][1] - nodes[neighbor][1],
                    )
                    progress = (current_distance - neighbor_distance) / 500.0
                    signal = (sig + 130.0) / 100.0
                    congestion = pressure[neighbor]
                    features = [progress, signal, congestion]
                    target = 1.0 if neighbor == correct_next else 0.0
                    samples.append((features, target))
    return samples

class PolicyModel:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        self.weights = [
            random.uniform(-0.1, 0.1)
            for _ in range(feature_dim)
        ]
        self.bias = 0.0

    def predict(self, features):
        s = self.bias
        for w, x in zip(self.weights, features):
            s += w * x
        if s >= 0.0:
            z = math.exp(-s)
            return 1.0 / (1.0 + z)
        z = math.exp(s)
        return z / (1.0 + z)

class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def update(self, key, param, grad):
        if key not in self.m:
            self.m[key] = 0.0
            self.v[key] = 0.0
        self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grad
        self.v[key] = (
            self.beta2 * self.v[key] + (1.0 - self.beta2) * grad * grad
        )
        m_hat = self.m[key] / (1.0 - self.beta1**self.t)
        v_hat = self.v[key] / (1.0 - self.beta2**self.t)
        param -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
        return param

    def step(self):
        self.t += 1

def train(model, samples):
    opt = Adam(lr=0.003)
    best_f1 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    patience = 120
    min_delta = 0.002
    for epoch in range(MAX_EPOCHS):
        random.shuffle(samples)
        total_loss = 0.0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for features, target in samples:
            opt.step()
            pred = model.predict(features)
            err = pred - target
            pred_label = 1 if pred >= 0.5 else 0
            target_label = 1 if target >= 0.5 else 0
            if pred_label == 1 and target_label == 1:
                tp += 1
            elif pred_label == 1 and target_label == 0:
                fp += 1
            elif pred_label == 0 and target_label == 0:
                tn += 1
            else:
                fn += 1
            total_loss += -(
                target * math.log(pred + 1e-9)
                + (1.0 - target) * math.log(1.0 - pred + 1e-9)
            )
            for i in range(model.feature_dim):
                grad = err * features[i]
                model.weights[i] = opt.update(("W", i), model.weights[i], grad)
            model.bias = opt.update(("B",), model.bias, err)
        total = tp + fp + tn + fn
        acc = (tp + tn) / max(total, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall > 0.0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        improvement = f1 - best_f1
        if improvement > min_delta:
            best_f1 = f1
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch % EPOCH_INTERVAL == 0:
            print(
                f"epoch={epoch:5d}  "
                f"loss={total_loss/len(samples):.5f}  "
                f"acc={acc:.1%}  "
                f"precision={precision:.4f}  "
                f"recall={recall:.4f}  "
                f"f1={f1:.4f}"
            )
        if f1 >= 0.995:
            print(f"Perfect convergence at epoch {epoch}. " f"f1={f1:.4f}")
            break
        if epochs_without_improvement >= patience:
            print(
                f"Stabilized at epoch {epoch}. "
                f"best_f1={best_f1:.4f} "
                f"(best epoch={best_epoch})"
            )
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
        print(f"Node {i}")
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
        pred = model.predict(source, dest, prev_hop, node, pressure)
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
                walls.append(((int(x1.strip()), int(y1.strip())), (int(x2.strip()), int(y2.strip()))))
    return nodes, walls

def main():
    random.seed(42)
    nodes, walls = load_nodes("points.txt")
    n = len(nodes)
    adj = build_adjacency(nodes, walls)
    pressure = compute_pressure(adj, n)
    samples = build_training_data(adj, nodes, pressure)
    print(f"nodes: {n}")
    print(f"samples: {len(samples)}")
    print_pressure(pressure)
    feature_dim = len(samples[0][0])
    model = PolicyModel(
        feature_dim=feature_dim
    )
    train(model, samples)
    with open("policy_model.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "pressure": pressure
        }, f)

if __name__ == "__main__":
    main()
