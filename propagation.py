import torch

PROPAGATION_ITERATIONS = 25
REDUNDANCY_PENALTY = 0.4

def soft_propagate(source_idx, relay_probs, link, n):
    source_relay = torch.cat([
        relay_probs[:source_idx],
        torch.ones(1, dtype=relay_probs.dtype),
        relay_probs[source_idx + 1:],
    ])
    p_init = torch.zeros(n, dtype=relay_probs.dtype)
    p_reach = torch.cat([
        p_init[:source_idx],
        torch.ones(1, dtype=relay_probs.dtype),
        p_init[source_idx + 1:],
    ])
    source_onehot = torch.zeros(n, dtype=relay_probs.dtype)
    source_onehot[source_idx] = 1.0
    for _ in range(PROPAGATION_ITERATIONS):
        relay_signal = source_relay * p_reach
        log_survival = torch.log1p(-(link * relay_signal.unsqueeze(0)).clamp(max=1.0 - 1e-7))
        p_rx = 1.0 - torch.exp(log_survival.sum(dim=1))
        p_reach = p_rx * (1.0 - source_onehot) + source_onehot
    return p_reach, source_relay

def propagation_loss(relay_probs, link, n, n_sources=None):
    sources = list(range(n)) if n_sources is None else torch.randperm(n)[:n_sources].tolist()
    coverage_terms = []
    redundancy_terms = []
    for src in sources:
        p_reach, source_relay = soft_propagate(src, relay_probs, link, n)
        non_src = torch.ones(n, dtype=torch.bool)
        non_src[src] = False
        coverage_terms.append(p_reach[non_src].mean())
        expected_rx = (link * (source_relay * p_reach).unsqueeze(0)).sum(dim=1)
        redundancy = (expected_rx - p_reach).clamp(min=0.0)
        redundancy_terms.append(redundancy[non_src].mean())
    coverage = torch.stack(coverage_terms).mean()
    redundancy = torch.stack(redundancy_terms).mean()
    loss = -coverage + REDUNDANCY_PENALTY * redundancy
    return loss, coverage.item(), redundancy.item()
