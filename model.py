import torch
import torch.nn as nn
from propagation import propagation_loss

FEATURE_DIM = 5
HIDDEN_DIM = 16
MAX_EPOCHS = 2000
LR = 0.01
LOG_INTERVAL = 50
N_SOURCES_PER_STEP = None  #None = all sources; set to int for stochastic

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
    for epoch in range(MAX_EPOCHS):
        opt.zero_grad()
        relay_probs = model(features)
        loss, coverage, airtime = propagation_loss(relay_probs, link, n, N_SOURCES_PER_STEP)
        loss.backward()
        opt.step()
        if epoch % LOG_INTERVAL == 0:
            print(
                f"ep={epoch:4d}  loss={loss.item():.4f}  "
                f"coverage={coverage:.4f}  airtime={airtime:.4f}"
            )
    return model

def evaluate(model, features, link, n):
    with torch.no_grad():
        relay_probs = model(features)
        loss, coverage, airtime = propagation_loss(relay_probs, link, n)
    print(f"\nfinal  coverage={coverage:.4f}  airtime={airtime:.4f}  loss={loss:.4f}")
    print("\nper-node relay probabilities:")
    for i, p in enumerate(relay_probs.tolist()):
        print(f"  node={i:2d}  relay_prob={p:.4f}")
    return relay_probs
