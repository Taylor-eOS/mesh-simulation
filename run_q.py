import random
from oracle import load_nodes, build_adjacency, compute_pressure, oracle_label
from ml import RelayUtilityModel, build_training_data, train
from utils import save_pickle

def print_pressure(pressure):
    for i, p in enumerate(pressure):
        print(f"Node {i} pressure={p:.3f}")

def main():
    random.seed(42)
    nodes, walls = load_nodes("points.txt")
    adj = build_adjacency(nodes, walls)
    pressure = compute_pressure(adj)
    print_pressure(pressure)
    samples = build_training_data(
        adj=adj,
        nodes=nodes,
        pressure=pressure,
        oracle_label_fn=oracle_label,
    )
    print(f"samples={len(samples)}")
    model = RelayUtilityModel(
        feature_dim=len(samples[0][0])
    )
    train(model, samples)
    save_pickle("relay_model.pkl", {
        "model": model,
        "pressure": pressure,
    })

if __name__ == "__main__":
    main()

