import torch
import torch.nn as nn
from propagation import propagation_loss, node_redundancy, auto_redundancy_penalty

FEATURE_DIM = 5
HIDDEN_DIM = 16
MAX_EPOCHS = 10000
LR = 0.01
LOG_INTERVAL = 50
PATIENCE = 100
MIN_DELTA = 1e-5

class RelayPolicy(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return torch.sigmoid(self.net(features).squeeze(-1))

def train(features, link, n):
    redundancy_penalty = auto_redundancy_penalty(link)
    print(f"auto redundancy_penalty={redundancy_penalty:.4f}")
    model = RelayPolicy()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(MAX_EPOCHS):
        opt.zero_grad()
        relay_probs = model(features)
        loss, coverage, redundancy = propagation_loss(relay_probs, link, n, redundancy_penalty)
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
    return model, redundancy_penalty

def relay_utility_scores(model, features):
    with torch.no_grad():
        return model.net(features).squeeze(-1)

def evaluate(model, features, link, n, redundancy_penalty):
    with torch.no_grad():
        relay_probs = model(features)
        loss, coverage, redundancy = propagation_loss(relay_probs, link, n, redundancy_penalty)
        per_node_redundancy, per_node_coverage = node_redundancy(relay_probs, link, n)
        scores = relay_utility_scores(model, features)
    airtime = relay_probs.mean().item()
    print(f"\nfinal  coverage={coverage:.4f}  redundancy={redundancy:.4f}  airtime={airtime:.4f}  loss={loss:.4f}")
    print("\nper-node analysis:")
    print(f"  {'node':>4}  {'utility':>9}  {'redundancy':>10}  {'coverage':>10}  note")
    for i, score in enumerate(scores.tolist()):
        p = relay_probs[i].item()
        red = per_node_redundancy[i].item()
        cov = per_node_coverage[i].item()
        if score > 2.0:
            note = "clear relay"
        elif score > 0.0:
            note = "marginal relay"
        elif score > -2.0:
            note = "marginal suppress"
        else:
            note = "clear suppress"
        print(f"  {i:>4}  {score:>9.4f}  {red:>10.4f}  {cov:>10.4f}  {note}")
    return scores
