import torch

PROPAGATION_ITERATIONS = 25
AIRTIME_PENALTY = 0.25

def soft_propagate(source_idx, relay_probs, link, n):
    source_mask = torch.ones(n, dtype=relay_probs.dtype)
    source_mask[source_idx] = 0.0
    effective_relay = relay_probs * source_mask
    effective_relay = torch.cat([
        effective_relay[:source_idx],
        torch.ones(1, dtype=relay_probs.dtype),
        effective_relay[source_idx + 1:],
    ])
    p_reach = torch.zeros(n)
    p_reach[source_idx] = 1.0
    for _ in range(PROPAGATION_ITERATIONS):
        relay_signal = effective_relay * p_reach
        log_p_not_rx = torch.log1p(-link * relay_signal.unsqueeze(0) + 1e-12)
        log_p_not_rx = log_p_not_rx.sum(dim=1)
        new_p = 1.0 - torch.exp(log_p_not_rx)
        new_p[source_idx] = 1.0
        if torch.max(torch.abs(new_p - p_reach)) < 1e-6:
            p_reach = new_p
            break
        p_reach = new_p
    return p_reach

def propagation_loss(relay_probs, link, n, n_sources=None):
    sources = list(range(n)) if n_sources is None else torch.randperm(n)[:n_sources].tolist()
    total_coverage = torch.tensor(0.0)
    for src in sources:
        p_reach = soft_propagate(src, relay_probs, link, n)
        mask = torch.ones(n, dtype=torch.bool)
        mask[src] = False
        total_coverage = total_coverage + p_reach[mask].mean()
    coverage = total_coverage / len(sources)
    airtime = relay_probs.mean()
    return -coverage + AIRTIME_PENALTY * airtime, coverage.item(), airtime.item()
