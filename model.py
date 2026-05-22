import torch
import torch.nn as nn
from propagation import propagation_loss, node_redundancy, auto_redundancy_penalty, transit_utility
import settings

class RelayPolicy(nn.Module):
    def __init__(self, feature_dim=settings.FEATURE_DIM, hidden_dim=settings.HIDDEN_DIM):
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
    opt = torch.optim.Adam(model.parameters(), lr=settings.LR)
    best_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(settings.MAX_EPOCHS):
        opt.zero_grad()
        relay_probs = model(features)
        loss, coverage, redundancy = propagation_loss(relay_probs, link, n, redundancy_penalty)
        loss.backward()
        opt.step()
        if best_loss - loss.item() > settings.MIN_DELTA:
            best_loss = loss.item()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch % settings.LOG_INTERVAL == 0:
            airtime = relay_probs.mean().item()
            print(f"ep={epoch:4d}, loss={loss.item():.3f}, coverage={coverage:.3f}, redundancy={redundancy:.3f}, airtime={airtime:.3f}")
        if epochs_without_improvement >= settings.PATIENCE:
            print(f"stabilized ep={epoch}, best_loss={best_loss:.4f}")
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
        transit_utilities = transit_utility(relay_probs, link, n)
    airtime = relay_probs.mean().item()
    print(f"final coverage={coverage:.4f}, redundancy={redundancy:.4f}, airtime={airtime:.4f}, loss={loss:.4f}")
    print(f"  {'node':>4}  {'utility':>9}  {'transit':>9}  {'redundancy':>10}  {'coverage':>10}")
    for i, score in enumerate(scores.tolist()):
        transit = transit_utilities[i].item()
        red = per_node_redundancy[i].item()
        cov = per_node_coverage[i].item()
        print(f"  {i:>4}  {score:>9.1f}  {transit:>9.3f}  {red:>10.3f}  {cov:>10.3f}")
    return scores
