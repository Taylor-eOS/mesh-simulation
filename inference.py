import math
import pickle
import random
import tkinter as tk
from run_calc import COLORS

WIDTH = 700
HEIGHT = 700
NODE_RADIUS = 16
MAX_DISTANCE = 500
MODEL_PATH = "policy_model.pkl"
POINTS_PATH = "points.txt"
selected_origin = None
selected_destination = None
hover_node = None

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s

def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

def segment_intersection(a, b, c, d):
    if a == c or a == d or b == c or b == d:
        return False
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

def nodes_connected(i, j, nodes, walls):
    for w in walls:
        if segment_intersection(nodes[i], nodes[j], w[0], w[1]):
            return False
    return True

def signal_strength(i, j, nodes):
    d = math.hypot(nodes[j][0] - nodes[i][0], nodes[j][1] - nodes[i][1])
    if d >= MAX_DISTANCE:
        return None
    return -130 + (1 - d / MAX_DISTANCE) * 100

def build_adjacency(nodes, walls):
    n = len(nodes)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if not nodes_connected(i, j, nodes, walls):
                continue
            sig = signal_strength(i, j, nodes)
            if sig is not None:
                adj[i].append((j, sig))
    return adj

def load_nodes(path):
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
                walls.append(((int(x1.strip()), int(y1.strip())), (int(x2.strip()), int(y2.strip()))))
    return nodes, walls

class PolicyModel:
    def __init__(self):
        self.n = 0
        self.d = 0
        self.S = []
        self.D = []
        self.P = []
        self.N = []
        self.pressure_bias = []
        self.bias = 0.0

    def score(self, source, dest, prev_hop, node, pressure):
        s = 0.0
        s += dot(self.S[source], self.N[node])
        s += dot(self.D[dest], self.N[node])
        s += dot(self.P[prev_hop], self.N[node])
        s += dot(self.S[source], self.D[dest])
        s += pressure[node] * self.pressure_bias[node]
        s += self.bias
        return s

    def predict(self, source, dest, prev_hop, node, pressure):
        return sigmoid(self.score(source, dest, prev_hop, node, pressure))

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def reconstruct_route(model, adj, pressure, source, dest, nodes, max_steps=256, beam_width=16):
    def distance(a, b):
        return math.hypot(nodes[b][0] - nodes[a][0], nodes[b][1] - nodes[a][1])
    beams = [([source], source, source, set([source]), 0.0)]
    best_partial = None
    best_distance = float("inf")
    for _ in range(max_steps):
        expanded = []
        for route, current, previous, visited, total_score in beams:
            if current == dest:
                return route
            current_dist = distance(current, dest)
            if current_dist < best_distance:
                best_distance = current_dist
                best_partial = route
            neighbors = adj.get(current, [])
            for neighbor, sig in neighbors:
                if neighbor in visited:
                    continue
                prob = model.predict(source, dest, previous, neighbor, pressure)
                next_dist = distance(neighbor, dest)
                progress = (current_dist - next_dist) / MAX_DISTANCE
                signal_bonus = (sig + 130.0) / 100.0
                score = total_score + prob * 2.5 + progress * 4.0 + signal_bonus * 0.5
                if neighbor == dest:
                    score += 1000.0
                expanded.append((route + [neighbor], neighbor, current, visited | {neighbor}, score))
        if not expanded:
            break
        expanded.sort(key=lambda x: x[4], reverse=True)
        beams = expanded[:beam_width]
    return best_partial

def path_metrics(route, adj):
    if not route:
        return None
    total_signal = 0.0
    weakest = 999.0
    for i in range(len(route) - 1):
        u = route[i]
        v = route[i + 1]
        sig = None
        for neighbor, s in adj[u]:
            if neighbor == v:
                sig = s
                break
        if sig is None:
            continue
        total_signal += sig
        weakest = min(weakest, sig)
    hops = len(route) - 1
    if hops == 0:
        avg_signal = 0.0
    else:
        avg_signal = total_signal / hops
    return {"hops": hops, "avg_signal": avg_signal, "weakest_signal": weakest}

def draw_connection_graph(canvas, nodes, walls, adj):
    for i in range(len(nodes)):
        for j, sig in adj[i]:
            if j <= i:
                continue
            x1, y1 = nodes[i]
            x2, y2 = nodes[j]
            t = (sig + 130) / 100
            width = 1.5 + t * 3.5
            canvas.create_line(x1, y1, x2, y2, fill="#e6e6e6", width=width)

def draw_route(canvas, nodes, route, color):
    if not route:
        return
    for i in range(len(route) - 1):
        u = route[i]
        v = route[i + 1]
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        canvas.create_line(x1, y1, x2, y2, fill=color, width=6)

def draw_nodes(canvas, nodes, route):
    route_set = set(route or [])
    for idx, (x, y) in enumerate(nodes):
        fill = "white"
        text = "black"
        if idx == selected_origin:
            fill = "#3cb44b"
            text = "white"
        elif idx == selected_destination:
            fill = "#e6194b"
            text = "white"
        elif idx in route_set:
            fill = "#4363d8"
            text = "white"
        elif idx == hover_node:
            fill = "#ffe599"
        canvas.create_oval(x - NODE_RADIUS, y - NODE_RADIUS, x + NODE_RADIUS, y + NODE_RADIUS, fill=fill, outline="black", width=2)
        canvas.create_text(x, y, text=str(idx), fill=text, font=("Arial", 10, "bold"))

def draw_walls(canvas, walls):
    for w in walls:
        canvas.create_line(w[0][0], w[0][1], w[1][0], w[1][1], fill="black", width=6)

def draw_info_panel(canvas, route, metrics):
    if selected_origin is None or selected_destination is None or route is None:
        return
    panel_width = 520
    panel_height = 120
    x, y = 20, HEIGHT - panel_height - 20
    canvas.create_rectangle(x, y, x + panel_width, y + panel_height, fill="#ffffff", outline="#cccccc", width=2)
    canvas.create_text(x + 15, y + 20, anchor="w", text=f"{selected_origin} -> {selected_destination}", font=("Arial", 14, "bold"))
    canvas.create_text(x + 15, y + 50, anchor="w", text=f"Route: {' -> '.join(map(str, route))}", font=("Arial", 11))
    canvas.create_text(x + 15, y + 80, anchor="w", text=f"Hops={metrics['hops']}   Avg={metrics['avg_signal']:.1f} dBm   Weakest={metrics['weakest_signal']:.1f} dBm", font=("Arial", 11))

def redraw(canvas, nodes, walls, adj, model, pressure):
    canvas.delete("all")
    route = None
    metrics = None
    if selected_origin is not None and selected_destination is not None and selected_origin != selected_destination:
        route = reconstruct_route(
            model,
            adj,
            pressure,
            selected_origin,
            selected_destination,
            nodes
        )
        if route:
            metrics = path_metrics(route, adj)
    draw_connection_graph(canvas, nodes, walls, adj)
    draw_route(canvas, nodes, route, "#ff8800")
    draw_walls(canvas, walls)
    draw_nodes(canvas, nodes, route)
    draw_info_panel(canvas, route, metrics)

def find_clicked_node(event, nodes):
    for idx, (x, y) in enumerate(nodes):
        if math.hypot(event.x - x, event.y - y) <= NODE_RADIUS + 5:
            return idx
    return None

def on_click(event, canvas, nodes, walls, adj, model, pressure):
    global selected_origin
    global selected_destination
    node = find_clicked_node(event, nodes)
    if node is None:
        return
    if selected_origin is None:
        selected_origin = node
    elif selected_destination is None:
        if node == selected_origin:
            selected_origin = None
        else:
            selected_destination = node
    else:
        selected_origin = node
        selected_destination = None
    redraw(canvas, nodes, walls, adj, model, pressure)

def on_motion(event, canvas, nodes, walls, adj, model, pressure):
    global hover_node
    hover_node = find_clicked_node(event, nodes)
    redraw(canvas, nodes, walls, adj, model, pressure)

def main():
    global selected_origin
    global selected_destination
    model_data = load_model(MODEL_PATH)
    model = model_data["model"]
    pressure = model_data["pressure"]
    nodes, walls = load_nodes(POINTS_PATH)
    adj = build_adjacency(nodes, walls)
    root = tk.Tk()
    root.title("Mesh Policy Inference Viewer")
    root.resizable(False, False)
    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white", highlightthickness=0)
    canvas.pack()
    canvas.bind("<Button-1>", lambda e: on_click(e, canvas, nodes, walls, adj, model, pressure))
    canvas.bind("<Motion>", lambda e: on_motion(e, canvas, nodes, walls, adj, model, pressure))
    root.bind("<Escape>", lambda e: root.destroy())
    redraw(canvas, nodes, walls, adj, model, pressure)
    root.bind("<Control-q>", lambda e: root.destroy())
    root.mainloop()

if __name__ == "__main__":
    main()

