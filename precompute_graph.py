import math
import torch
from utils import euclidean, segment_intersection
from settings import MAX_RANGE, OUTPUT_FILE, PRECOMPUTED_PATH

def _nodes_connected(i, j, nodes, walls):
    for w in walls:
        if segment_intersection(nodes[i], nodes[j], w[0], w[1]):
            return False
    return True

def _load_graph(path):
    nodes = []
    walls = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.rstrip(",").split("),")]
            if len(parts) == 1:
                coords = parts[0].strip("()")
                x, y = coords.split(",")
                nodes.append((int(x.strip()), int(y.strip())))
            elif len(parts) == 2:
                c1 = parts[0].strip("()")
                c2 = parts[1].strip("()")
                x1, y1 = c1.split(",")
                x2, y2 = c2.split(",")
                walls.append((
                    (int(x1.strip()), int(y1.strip())),
                    (int(x2.strip()), int(y2.strip()))
                ))
    n = len(nodes)
    link = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if not _nodes_connected(i, j, nodes, walls):
                continue
            d = euclidean(nodes[i], nodes[j])
            if d < MAX_RANGE:
                link[i, j] = (1.0 - d / MAX_RANGE) ** 2
    return link, n

def _structural_features(link, max_degree_norm=64.0):
    n = link.shape[0]
    degree = (link > 0).float().sum(dim=1)
    mean_out = link.sum(dim=1) / degree.clamp(min=1)
    in_link = link.t()
    in_degree = (in_link > 0).float().sum(dim=1)
    mean_in = in_link.sum(dim=1) / in_degree.clamp(min=1)
    weighted_in = in_link.sum(dim=1)
    neighbor_degree = torch.zeros(n)
    for i in range(n):
        neighbors = (link[i] > 0).nonzero(as_tuple=True)[0]
        if len(neighbors) > 0:
            neighbor_degree[i] = degree[neighbors].mean()
    feats = torch.stack([
        (degree / max_degree_norm).clamp(max=1.0),
        mean_out,
        mean_in,
        (neighbor_degree / max_degree_norm).clamp(max=1.0),
        (weighted_in / max_degree_norm).clamp(max=1.0),
    ], dim=1)
    return feats

def precompute(source_path=OUTPUT_FILE, out_path=PRECOMPUTED_PATH):
    link, n = _load_graph(source_path)
    features = _structural_features(link)
    torch.save({"link": link, "n": n, "features": features}, out_path)
    print(f"saved {out_path}  nodes={n}")
    for i in range(n):
        neighbors = [(j, round(link[i, j].item(), 2)) for j in range(n) if link[i, j] > 0]
        print(f"  node {i:>2}: {neighbors}")

if __name__ == "__main__":
    precompute()
