import math
import pickle
import tkinter as tk
WIDTH = 1200
HEIGHT = 950
NODE_RADIUS = 14
MAX_DISTANCE = 500.0
MODEL_PATH = "policy_model.pkl"
POINTS_PATH = "points.txt"
selected_origin = None
selected_destination = None
hovered_node = None

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
    return -130.0 + (1.0 - d / MAX_DISTANCE) * 100.0

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
            if sig is None:
                continue
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
    def predict(self, features):
        s = self.bias
        for w, x in zip(self.weights, features):
            s += w * x
        if s >= 0.0:
            z = math.exp(-s)
            return 1.0 / (1.0 + z)
        z = math.exp(s)
        return z / (1.0 + z)

def load_model(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["pressure"]

def evaluate_neighbors(model, pressure, adj, nodes, current, dest):
    current_distance = math.hypot(
        nodes[dest][0] - nodes[current][0], nodes[dest][1] - nodes[current][1]
    )
    results = []
    for neighbor, sig in adj[current]:
        next_distance = math.hypot(
            nodes[dest][0] - nodes[neighbor][0],
            nodes[dest][1] - nodes[neighbor][1],
        )
        progress = (current_distance - next_distance) / MAX_DISTANCE
        signal = (sig + 130.0) / 100.0
        congestion = pressure[neighbor]
        features = [progress, signal, congestion]
        score = model.predict(features)
        results.append(
            {
                "neighbor": neighbor,
                "score": score,
                "progress": progress,
                "signal": signal,
                "congestion": congestion,
            }
        )
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def infer_route(model, pressure, adj, nodes, source, dest):
    current = source
    route = [source]
    visited = {source}
    decisions = []
    limit = len(nodes)
    for _ in range(limit):
        if current == dest:
            return route, decisions, True
        evaluations = evaluate_neighbors(
            model, pressure, adj, nodes, current, dest
        )
        next_hop = None
        for item in evaluations:
            if item["neighbor"] not in visited:
                next_hop = item["neighbor"]
                break
        decisions.append(
            {
                "current": current,
                "evaluations": evaluations,
                "selected": next_hop,
            }
        )
        if next_hop is None:
            return route, decisions, False
        route.append(next_hop)
        visited.add(next_hop)
        current = next_hop
    return route, decisions, False

def draw_edges(canvas, nodes, adj):
    drawn = set()
    for i in adj:
        for j, sig in adj[i]:
            key = tuple(sorted((i, j)))
            if key in drawn:
                continue
            drawn.add(key)
            x1, y1 = nodes[i]
            x2, y2 = nodes[j]
            canvas.create_line(x1, y1, x2, y2, fill="#dddddd", width=1)

def draw_route(canvas, nodes, route):
    if not route:
        return
    for i in range(len(route) - 1):
        u = route[i]
        v = route[i + 1]
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        canvas.create_line(x1, y1, x2, y2, fill="#ff8800", width=6)

def draw_walls(canvas, walls):
    for w in walls:
        canvas.create_line(
            w[0][0], w[0][1], w[1][0], w[1][1], fill="black", width=5
        )

def draw_nodes(canvas, nodes, pressure, route, decisions):
    route_set = set(route or [])
    decision_map = {}
    if decisions:
        for d in decisions:
            current = d["current"]
            if d["selected"] is not None:
                decision_map[current] = d["selected"]
    for idx, (x, y) in enumerate(nodes):
        p = pressure[idx]
        shade = int(255 - p * 180)
        color = f"#{shade:02x}{shade:02x}ff"
        text = "black"
        outline = "black"
        width = 2
        if idx == selected_origin:
            color = "#00cc44"
            text = "white"
        elif idx == selected_destination:
            color = "#ff2244"
            text = "white"
        elif idx in route_set:
            color = "#ff8800"
            text = "white"
        if idx == hovered_node:
            outline = "#00ffff"
            width = 4
        canvas.create_oval(
            x - NODE_RADIUS,
            y - NODE_RADIUS,
            x + NODE_RADIUS,
            y + NODE_RADIUS,
            fill=color,
            outline=outline,
            width=width,
        )
        canvas.create_text(
            x, y, text=str(idx), fill=text, font=("Arial", 9, "bold")
        )
        if idx in decision_map:
            target = decision_map[idx]
            tx, ty = nodes[target]
            dx = tx - x
            dy = ty - y
            length = math.hypot(dx, dy)
            if length > 0:
                dx /= length
                dy /= length
                ax = x + dx * 24
                ay = y + dy * 24
                bx = tx - dx * 24
                by = ty - dy * 24
                canvas.create_line(
                    ax, ay, bx, by, fill="#ff0000", width=3, arrow=tk.LAST
                )

def draw_route_text(canvas, route, success):
    if not route:
        return
    text = " -> ".join(str(x) for x in route)
    if success:
        text += "    SUCCESS"
    else:
        text += "    FAILED"
    canvas.create_text(
        20, 20, anchor="w", text=text, font=("Arial", 14, "bold")
    )

def draw_node_analysis(canvas, nodes, pressure, adj, model):
    if hovered_node is None:
        return
    if selected_destination is None:
        return
    evaluations = evaluate_neighbors(
        model, pressure, adj, nodes, hovered_node, selected_destination
    )
    x = 20
    y = HEIGHT - 320
    canvas.create_rectangle(
        x, y, x + 1120, y + 280, fill="white", outline="#cccccc", width=2
    )
    title = (
        f"Node {hovered_node} local policy "
        f"(destination={selected_destination})"
    )
    canvas.create_text(
        x + 10, y + 16, anchor="w", text=title, font=("Arial", 13, "bold")
    )
    py = y + 42
    for item in evaluations[:12]:
        line = (
            f"neighbor={item['neighbor']:3d}   "
            f"score={item['score']:.4f}   "
            f"progress={item['progress']:+.3f}   "
            f"signal={item['signal']:.3f}   "
            f"pressure={item['congestion']:.3f}"
        )
        canvas.create_text(
            x + 10, py, anchor="w", text=line, font=("Courier", 10)
        )
        py += 20

def redraw(canvas, nodes, walls, adj, model, pressure):
    canvas.delete("all")
    route = None
    decisions = None
    success = False
    if (
        selected_origin is not None
        and selected_destination is not None
        and selected_origin != selected_destination
    ):
        route, decisions, success = infer_route(
            model, pressure, adj, nodes, selected_origin, selected_destination
        )
    draw_edges(canvas, nodes, adj)
    draw_route(canvas, nodes, route)
    draw_walls(canvas, walls)
    draw_nodes(canvas, nodes, pressure, route, decisions)
    draw_route_text(canvas, route, success)
    draw_node_analysis(canvas, nodes, pressure, adj, model)

def find_clicked_node(event, nodes):
    for idx, (x, y) in enumerate(nodes):
        d = math.hypot(event.x - x, event.y - y)
        if d <= NODE_RADIUS + 4:
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
    global hovered_node
    hovered_node = find_clicked_node(event, nodes)
    redraw(canvas, nodes, walls, adj, model, pressure)

def main():
    model, pressure = load_model(MODEL_PATH)
    nodes, walls = load_nodes(POINTS_PATH)
    adj = build_adjacency(nodes, walls)
    root = tk.Tk()
    root.title("Mesh Node Policy Visualizer")
    root.resizable(False, False)
    canvas = tk.Canvas(
        root, width=WIDTH, height=HEIGHT, bg="white", highlightthickness=0
    )
    canvas.pack()
    canvas.bind(
        "<Button-1>",
        lambda e: on_click(e, canvas, nodes, walls, adj, model, pressure),
    )
    canvas.bind(
        "<Motion>",
        lambda e: on_motion(e, canvas, nodes, walls, adj, model, pressure),
    )
    root.bind("<Escape>", lambda e: root.destroy())
    redraw(canvas, nodes, walls, adj, model, pressure)
    root.mainloop()

if __name__ == "__main__":
    main()

