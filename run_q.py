import pickle
import torch
from graph import load_graph, structural_features
from model import train, evaluate

def main():
    link, n = load_graph("points.txt")
    features = structural_features(link)
    print(f"nodes={n}  link_density={(link > 0).float().mean().item():.3f}")
    model = train(features, link, n)
    relay_probs = evaluate(model, features, link, n)
    with open("relay_model.pkl", "wb") as f:
        pickle.dump({
            "state_dict": model.state_dict(),
            "relay_probs": relay_probs.tolist(),
        }, f)

if __name__ == "__main__":
    main()
