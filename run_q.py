import pickle
from graph import load_precomputed
from model import train, evaluate

def main():
    link, n, features = load_precomputed()
    print(f"nodes={n}  link_density={(link > 0).float().mean().item():.3f}")
    model, redundancy_penalty = train(features, link, n)
    scores = evaluate(model, features, link, n, redundancy_penalty)
    with open("relay_model.pkl", "wb") as f:
        pickle.dump({
            "state_dict": model.state_dict(),
            "utility_scores": scores.tolist(),
            "redundancy_penalty": redundancy_penalty,
        }, f)

if __name__ == "__main__":
    main()
