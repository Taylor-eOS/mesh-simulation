import math
import random
from oracle import compute_node_fingerprints, extract_structural_features
from utils import normalize_signal

MAX_EPOCHS = 10000
EPOCH_INTERVAL = 10
MAX_DEGREE_NORM = 64.0
FEATURE_DIM = 5
TARGET_KEY = "mean_utility"

def features_to_vector(features):
    return [
        min(features["degree"], MAX_DEGREE_NORM) / MAX_DEGREE_NORM,
        normalize_signal(features["mean_out_signal"]),
        normalize_signal(features["mean_in_signal"]),
        min(features["neighbor_mean_degree"], MAX_DEGREE_NORM) / MAX_DEGREE_NORM,
        min(features["weighted_in_degree"], MAX_DEGREE_NORM) / MAX_DEGREE_NORM,
    ]

def build_training_data(nodes, adj, radj):
    fingerprints = compute_node_fingerprints(nodes, adj, radj)
    samples = []
    for node, fp in fingerprints.items():
        features = extract_structural_features(node, adj, radj)
        feature_vector = features_to_vector(features)
        target = fp[TARGET_KEY]
        samples.append((feature_vector, target))
    return samples, fingerprints

class RelayFingerprintModel:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(feature_dim)]
        self.bias = 0.0

    def predict(self, features):
        s = self.bias
        for w, x in zip(self.weights, features):
            s += w * x
        return s

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

def evaluate(model, samples):
    total_sq_err = 0.0
    total_abs_err = 0.0
    targets = [target for _features, target in samples]
    mean_target = sum(targets) / len(targets)
    ss_tot = sum((t - mean_target) ** 2 for t in targets)
    for features, target in samples:
        pred = model.predict(features)
        err = pred - target
        total_sq_err += err * err
        total_abs_err += abs(err)
    mse = total_sq_err / len(samples)
    mae = total_abs_err / len(samples)
    r2 = 1.0 - total_sq_err / max(ss_tot, 1e-12)
    return mse, mae, r2

def train(model, samples):
    opt = Adam()
    best_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    patience = 120
    min_delta = 1e-5
    for epoch in range(MAX_EPOCHS):
        random.shuffle(samples)
        for features, target in samples:
            opt.step()
            pred = model.predict(features)
            err = pred - target
            for i in range(model.feature_dim):
                grad = 2.0 * err * features[i]
                model.weights[i] = opt.update(("W", i), model.weights[i], grad)
            model.bias = opt.update(("B",), model.bias, 2.0 * err)
        mse, mae, r2 = evaluate(model, samples)
        improvement = best_mae - mae
        if improvement > min_delta:
            best_mae = mae
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch % EPOCH_INTERVAL == 0:
            print(
                f"epoch={epoch:5d}  "
                f"mse={mse:.6f}  "
                f"mae={mae:.6f}  "
                f"r2={r2:.4f}"
            )
            print("weights:", [round(w, 6) for w in model.weights])
            print("bias:", round(model.bias, 6))
        if epochs_without_improvement >= patience:
            print(
                f"stabilized epoch={epoch}  "
                f"best_mae={best_mae:.6f}  "
                f"best_epoch={best_epoch}"
            )
            break

def print_node_analysis(model, samples, fingerprints):
    print()
    print("node analysis")
    print()
    for node, (_features, target) in enumerate(samples):
        pred = model.predict(samples[node][0])
        fp = fingerprints[node]
        print(
            f"node={node:3d}  "
            f"target={target:.6f}  "
            f"pred={pred:.6f}  "
            f"err={abs(pred - target):.6f}  "
            f"utility_var={fp['utility_variance']:.6f}  "
            f"coverage={fp['mean_coverage']:.6f}  "
            f"redundancy={fp['mean_redundancy']:.6f}"
        )

