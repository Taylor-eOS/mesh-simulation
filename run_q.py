import os
import math
import random
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
            temperature = 0.15
            if nd < dist[v]:
                if dist[v] < float("inf"):
                    delta = dist[v] - nd
                    accept_prob = sigmoid(
                        delta / temperature
                    )
                    if random.random() > accept_prob:
                        continue
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

def compute_pressure(adj, n):
    pressure = [0.0 for _ in range(n)]
    workers = os.cpu_count() or 1
    for _ in range(PRESSURE_ITERATIONS):
        traffic = [0.0 for _ in range(n)]
        args = [
            (source, adj, n, pressure)
            for source in range(n)
        ]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for source, paths in ex.map(_dijkstra_worker, args):
                for dest, path in paths.items():
                    if len(path) <= 2:
                        continue
                    path_pressure = 0.0
                    for node in path[1:-1]:
                        path_pressure += pressure[node]
                    reroute_factor = 1.0 / (
                        1.0 + path_pressure * 2.5
                    )
                    for hop_index, node in enumerate(path[1:-1], start=1):
                        remaining = len(path) - hop_index
                        load = (
                            1.0 +
                            remaining * 0.15
                        )
                        traffic[node] += (
                            load * reroute_factor
                        )
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
    workers = os.cpu_count() or 1
    args = [(source, adj, n, pressure) for source in range(n)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for source, paths in ex.map(_dijkstra_worker, args):
            for dest, path in paths.items():
                relay_nodes = set(path[1:-1])
                legal_prev = set()
                for path_index in range(1, len(path)):
                    current = path[path_index]
                    previous = path[path_index - 1]
                    legal_prev.add((current, previous))
                for node in range(n):
                    if node == source:
                        continue
                    if node == dest:
                        continue
                    incoming_prev = [
                        prev for current, prev in legal_prev
                        if current == node
                    ]
                    if not incoming_prev:
                        incoming_prev = [source]
                    for prev_hop in incoming_prev:
                        state = (source, dest, prev_hop, node)
                        if node in relay_nodes:
                            target = relay_probability(
                                pressure[node],
                                len(path)
                            )
                        else:
                            target = 0.0
                            if random.random() < 0.8:
                                continue
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

class Adam:
    def __init__(
        self,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8
    ):
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
        self.m[key] = (
            self.beta1 * self.m[key] +
            (1.0 - self.beta1) * grad
        )
        self.v[key] = (
            self.beta2 * self.v[key] +
            (1.0 - self.beta2) * grad * grad
        )
        m_hat = self.m[key] / (
            1.0 - self.beta1 ** self.t
        )
        v_hat = self.v[key] / (
            1.0 - self.beta2 ** self.t
        )
        param -= (
            self.lr *
            m_hat /
            (math.sqrt(v_hat) + self.eps)
        )
        return param

    def step(self):
        self.t += 1

def train(model, q_table, pressure):
    states = list(q_table.keys())
    opt = Adam(lr=0.01)
    best_f1 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    patience = 120
    min_delta = 0.002
    for epoch in range(MAX_EPOCHS):
        random.shuffle(states)
        total_loss = 0.0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for state in states:
            opt.step()
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
            pred_label = 1 if pred >= 0.5 else 0
            target_label = 1 if target >= 0.5 else 0
            if pred_label == 1 and target_label == 1:
                tp += 1
            elif pred_label == 1 and target_label == 0:
                fp += 1
            elif pred_label == 0 and target_label == 0:
                tn += 1
            elif pred_label == 0 and target_label == 1:
                fn += 1
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
                s_grad = grad * (
                    n_old[k] + d_old[k]
                )
                d_grad = grad * (
                    n_old[k] + s_old[k]
                )
                p_grad = grad * (
                    n_old[k]
                )
                n_grad = grad * (
                    s_old[k] +
                    d_old[k] +
                    p_old[k]
                )
                s_vec[k] = opt.update(
                    ("S", source, k),
                    s_vec[k],
                    s_grad
                )
                d_vec[k] = opt.update(
                    ("D", dest, k),
                    d_vec[k],
                    d_grad
                )
                p_vec[k] = opt.update(
                    ("P", prev_hop, k),
                    p_vec[k],
                    p_grad
                )
                n_vec[k] = opt.update(
                    ("N", node, k),
                    n_vec[k],
                    n_grad
                )
            pb_grad = grad * pressure[node]
            model.pressure_bias[node] = opt.update(
                ("PB", node),
                model.pressure_bias[node],
                pb_grad
            )
            model.bias = opt.update(
                ("B",),
                model.bias,
                grad
            )
        total = tp + fp + tn + fn
        acc = (tp + tn) / max(total, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall > 0:
            f1 = (
                2.0 *
                precision *
                recall /
                (precision + recall)
            )
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
                f"loss={total_loss/len(states):.5f}  "
                f"acc={acc:.1%}"
            )
            print(
                f"tp={tp}  "
                f"fp={fp}  "
                f"tn={tn}  "
                f"fn={fn}"
            )
            print(
                f"precision={precision:.4f}  "
                f"recall={recall:.4f}  "
                f"f1={f1:.4f}  "
                f"best_f1={best_f1:.4f}"
            )
        if f1 >= 0.995:
            print(
                f"Perfect convergence at epoch {epoch}. "
                f"f1={f1:.4f}"
            )
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
    q_table = build_q_table(adj, n, pressure)
    print(f"nodes: {n}")
    print(f"states: {len(q_table)}")
    print_pressure(pressure)
    model = PolicyModel(n=n, d=LATENT_DIM)
    train(model, q_table, pressure)
    print_embeddings(model)
    print_policy(model, q_table, pressure)

if __name__ == "__main__":
    main()
