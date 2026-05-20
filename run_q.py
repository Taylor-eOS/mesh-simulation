import random
from oracle import build_adjacency, load_nodes
from ml import FEATURE_DIM, RelayFingerprintModel, build_training_data, print_node_analysis, train
from utils import save_pickle

def print_fingerprint_stats(fingerprints):
    utilities = [fp["mean_utility"] for fp in fingerprints.values()]
    coverages = [fp["mean_coverage"] for fp in fingerprints.values()]
    redundancies = [fp["mean_redundancy"] for fp in fingerprints.values()]
    mean_u = sum(utilities) / len(utilities)
    min_u = min(utilities)
    max_u = max(utilities)
    mean_cov = sum(coverages) / len(coverages)
    mean_red = sum(redundancies) / len(redundancies)
    n_negative = sum(1 for u in utilities if u < 0.0)
    print("fingerprint statistics")
    print(
        f"nodes={len(fingerprints)}  "
        f"utility_min={min_u:.6f}  "
        f"utility_max={max_u:.6f}  "
        f"utility_mean={mean_u:.6f}"
    )
    print(
        f"coverage_mean={mean_cov:.6f}  redundancy_mean={mean_red:.6f}")
    print(f"negative_utility_nodes={n_negative}/{len(fingerprints)}")

def main():
    random.seed(42)
    nodes, walls = load_nodes("points.txt")
    adj, radj = build_adjacency(nodes, walls)
    samples, fingerprints = build_training_data(nodes, adj, radj,)
    print_fingerprint_stats(fingerprints)
    model = RelayFingerprintModel(FEATURE_DIM)
    train(model, samples)
    print_node_analysis(model,
        samples,
        fingerprints,)
    save_pickle("relay_model.pkl", { "weights": model.weights, "bias": model.bias, "fingerprints": fingerprints,})

if __name__ == "__main__":
    main()

