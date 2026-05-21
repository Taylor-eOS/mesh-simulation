import torch
from settings import PROPAGATION_ITERATIONS

def soft_propagate_all(relay_probs, link, n):
    source_relay = relay_probs.unsqueeze(0).expand(n, n).clone()
    source_relay[torch.arange(n), torch.arange(n)] = 1.0
    p_reach = torch.zeros(n, n)
    p_reach[torch.arange(n), torch.arange(n)] = 1.0
    for _ in range(PROPAGATION_ITERATIONS):
        relay_signal = source_relay * p_reach
        log_survival = torch.log1p(
            -(link.unsqueeze(0) * relay_signal.unsqueeze(1)).clamp(max=1.0 - 1e-7)
        ).sum(dim=2)
        p_rx = 1.0 - torch.exp(log_survival)
        src_idx = torch.arange(n)
        p_rx[src_idx, src_idx] = 1.0
        if (p_rx - p_reach).abs().max() < 1e-6:
            p_reach = p_rx
            break
        p_reach = p_rx
    return p_reach, source_relay

def auto_redundancy_penalty(link):
    weighted_in = link.sum(dim=1)
    return 0.3 + 0.5 * weighted_in.mean().item()

def propagation_loss(relay_probs, link, n, redundancy_penalty):
    p_reach, source_relay = soft_propagate_all(relay_probs, link, n)
    src_idx = torch.arange(n)
    non_src_mask = torch.ones(n, n, dtype=torch.bool)
    non_src_mask[src_idx, src_idx] = False
    coverage = p_reach[non_src_mask].mean()
    expected_rx = (link.unsqueeze(0) * (source_relay * p_reach).unsqueeze(1)).sum(dim=2)
    redundancy = (expected_rx - p_reach).clamp(min=0.0)[non_src_mask].mean()
    loss = -coverage + redundancy_penalty * redundancy
    return loss, coverage.item(), redundancy.item()

def node_redundancy(relay_probs, link, n):
    p_reach, source_relay = soft_propagate_all(relay_probs, link, n)
    expected_rx = (link.unsqueeze(0) * (source_relay * p_reach).unsqueeze(1)).sum(dim=2)
    redundancy_per_node = (expected_rx - p_reach).clamp(min=0.0).mean(dim=0)
    coverage_per_node = p_reach.mean(dim=0)
    return redundancy_per_node, coverage_per_node
