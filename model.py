import torch
import torch.nn as nn
from propagation import propagation_loss

FEATURE_DIM = 5
HIDDEN_DIM = 16
MAX_EPOCHS = 10000
LR = 0.01
LOG_INTERVAL = 50
PATIENCE = 150
MIN_DELTA = 1e-5
N_SOURCES_PER_STEP = None
RELAY_THRESHOLD = 0.5


class RelayPolicy(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


def train(features, link, n):
    model = RelayPolicy()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(MAX_EPOCHS):
        opt.zero_grad()
        relay_probs = model(features)
        loss, coverage, redundancy = propagation_loss(relay_probs, link, n, N_SOURCES_PER_STEP)
        loss.backward()
        opt.step()
        if best_loss - loss.item() > MIN_DELTA:
            best_loss = loss.item()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch % LOG_INTERVAL == 0:
            airtime = relay_probs.mean().item()
            print(
                f"ep={epoch:4d}  loss={loss.item():.4f}  "
                f"coverage={coverage:.4f}  redundancy={redundancy:.4f}  airtime={airtime:.4f}"
            )
        if epochs_without_improvement >= PATIENCE:
            print(f"stabilized ep={epoch}  best_loss={best_loss:.4f}")
            break
    return model


def evaluate(model, features, link, n):
    with torch.no_grad():
        relay_probs = model(features)
        loss, coverage, redundancy = propagation_loss(relay_probs, link, n)
    airtime = relay_probs.mean().item()
    print(f"\nfinal  coverage={coverage:.4f}  redundancy={redundancy:.4f}  airtime={airtime:.4f}  loss={loss:.4f}")
    relay_nodes = [i for i, p in enumerate(relay_probs.tolist()) if p >= RELAY_THRESHOLD]
    suppress_nodes = [i for i, p in enumerate(relay_probs.tolist()) if p < RELAY_THRESHOLD]
    print(f"\nrelay backbone (threshold={RELAY_THRESHOLD}):")
    print(f"  relay    ({len(relay_nodes):2d} nodes): {relay_nodes}")
    print(f"  suppress ({len(suppress_nodes):2d} nodes): {suppress_nodes}")
    print("\nper-node relay probabilities:")
    for i, p in enumerate(relay_probs.tolist()):
        marker = "  <-- relay" if p >= RELAY_THRESHOLD else ""
        print(f"  node={i:2d}  relay_prob={p:.4f}{marker}")
    return relay_probs
