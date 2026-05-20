import random
from oracle import load_nodes, build_adjacency
from ml import RelayUtilityModel, build_training_data, train, FEATURE_DIM
from utils import save_pickle

def print_sample_stats(samples):
    utilities = [t for _f, t in samples]
    mean_u = sum(utilities) / len(utilities)
    min_u = min(utilities)
    max_u = max(utilities)
    n_neg = sum(1 for u in utilities if u < 0)
    print(
        f"samples={len(samples)}  "
        f"utility  min={min_u:.4f}  max={max_u:.4f}  mean={mean_u:.4f}  "
        f"negative={n_neg}/{len(utilities)}"
    )

def main():
    random.seed(42)
    nodes, walls = load_nodes("points.txt")
    adj, radj = build_adjacency(nodes, walls)
    samples = build_training_data(nodes, adj, radj)
    print_sample_stats(samples)
    model = RelayUtilityModel(FEATURE_DIM)
    train(model, samples)
    save_pickle("relay_model.pkl", {"model": model})

if __name__ == "__main__":
    main()
