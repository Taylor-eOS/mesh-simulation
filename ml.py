import math
import random
from utils import euclidean
from utils import normalize_signal

MAX_EPOCHS = 10000
EPOCH_INTERVAL = 10

def extract_observation(current, neighbor, dest, nodes, pressure, sig, adj):
    current_distance = euclidean(nodes[current], nodes[dest])
    neighbor_distance = euclidean(nodes[neighbor], nodes[dest])
    local_density = len(adj[neighbor]) / max(len(nodes), 1)
    observation = {
        "progress": (current_distance - neighbor_distance) / 500.0,
        "signal": normalize_signal(sig),
        "congestion": pressure[neighbor],
        "neighbor_density": local_density,
    }
    return observation

def observation_to_vector(obs):
    return [
        obs["progress"],
        obs["signal"],
        obs["congestion"],
        obs["neighbor_density"],
    ]

def build_training_data(adj, nodes, pressure, oracle_label_fn):
    samples = []
    n = len(nodes)
    for source in range(n):
        for dest in range(n):
            if source == dest:
                continue
            current = source
            visited = set()
            while current not in visited:
                visited.add(current)
                candidates = adj[current]
                for neighbor, sig in candidates:
                    observation = extract_observation(
                        current=current,
                        neighbor=neighbor,
                        dest=dest,
                        nodes=nodes,
                        pressure=pressure,
                        sig=sig,
                        adj=adj,
                    )
                    target = oracle_label_fn(
                        current=current,
                        neighbor=neighbor,
                        dest=dest,
                        nodes=nodes,
                        pressure=pressure,
                        sig=sig,
                    )
                    samples.append((
                        observation_to_vector(observation),
                        target,
                    ))
                best_neighbor = None
                best_score = -1e9
                for neighbor, sig in candidates:
                    score = oracle_label_fn(
                        current=current,
                        neighbor=neighbor,
                        dest=dest,
                        nodes=nodes,
                        pressure=pressure,
                        sig=sig,
                    )
                    if score > best_score:
                        best_score = score
                        best_neighbor = neighbor
                if best_neighbor is None:
                    break
                current = best_neighbor
    return samples

class RelayUtilityModel:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(feature_dim)]
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
    def __init__(self, lr=0.003, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1

    def update(self, key, param, grad):
        if key not in self.m:
            self.m[key] = 0.0
            self.v[key] = 0.0
        self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * grad * grad
        m_hat = self.m[key] / (1.0 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1.0 - self.beta2 ** self.t)
        param -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
        return param

def train(model, samples):
    opt = Adam()
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
                f"loss={total_loss / len(samples):.5f}  "
                f"acc={acc:.1%}  "
                f"precision={precision:.4f}  "
                f"recall={recall:.4f}  "
                f"f1={f1:.4f}"
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

