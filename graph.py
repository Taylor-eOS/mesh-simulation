import math
import torch

MAX_RANGE = 500.0

def _euclidean(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])

def _orient(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

def _on_segment(p, q, r):
    return (
        min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
        and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
    )

def _segment_intersection(a, b, c, d):
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if o1 == 0 and _on_segment(a, c, b):
        return True
    if o2 == 0 and _on_segment(a, d, b):
        return True
    if o3 == 0 and _on_segment(c, a, d):
        return True
    if o4 == 0 and _on_segment(c, b, d):
        return True
    return False

def _nodes_connected(i, j, nodes, walls):
    for w in walls:
        if _segment_intersection(nodes[i], nodes[j], w[0], w[1]):
            return False
    return True

def load_graph(path):
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
            d = _euclidean(nodes[i], nodes[j])
            if d < MAX_RANGE:
                link[i, j] = (1.0 - d / MAX_RANGE) ** 2
    return link, n

def structural_features(link, max_degree_norm=64.0):
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
