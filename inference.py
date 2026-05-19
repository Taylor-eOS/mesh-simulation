import math
import pickle
import tkinter as tk

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
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s

def ccw(a, b, c):
    return (
        (c[1] - a[1]) *
        (b[0] - a[0]) >
        (b[1] - a[1]) *
        (c[0] - a[0])
    )

def segment_intersection(a, b, c, d):
    if a == c or a == d or b == c or b == d:
        return False
    return (
        ccw(a, c, d) != ccw(b, c, d) and
        ccw(a, b, c) != ccw(a, b, d)
    )

def nodes_connected(i, j, nodes, walls):
    for w in walls:
        if segment_intersection(
            nodes[i],
            nodes[j],
            w[0],
            w[1]
        ):
            return False
    return True

def signal_strength(i, j, nodes):
    d = math.hypot(
        nodes[j][0] - nodes[i][0],
        nodes[j][1] - nodes[i][1]
    )
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
            parts = [
                p.strip()
                for p in line.rstrip(",").split("),")
            ]
            if len(parts) == 1:
                coords = parts[0].strip("()")
                x, y = coords.split(",")
                nodes.append((
                    int(x.strip()),
                    int(y.strip())
                ))
            elif len(parts) == 2:
                c1 = parts[0].strip("()")
                c2 = parts[1].strip("()")
                x1, y1 = c1.split(",")
                x2, y2 = c2.split(",")
                walls.append((
                    (
                        int(x1.strip()),
                        int(y1.strip())
                    ),
                    (
                        int(x2.strip()),
                        int(y2.strip())
                    )
                ))
    return nodes, walls

class PolicyModel:
    def ensure_compatibility(self):
        if not hasattr(self, "pressure_bias"):
            self.pressure_bias = [
                0.0 for _ in range(self.n)
            ]
        if not hasattr(self, "bias"):
            self.bias = 0.0

    def score(
        self,
        source,
        dest,
        prev_hop,
        node,
        pressure
    ):
        self.ensure_compatibility()
        s = 0.0
        s += dot(
            self.S[source],
            self.N[node]
        )
        s += dot(
            self.D[dest],
            self.N[node]
        )
        s += dot(
            self.P[prev_hop],
            self.N[node]
        )
        s += dot(
            self.S[source],
            self.D[dest]
        )
        s += (
            pressure[node] *
            self.pressure_bias[node]
        )
        s += self.bias
        return s

def load_model(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    if not hasattr(model, "pressure_bias"):
        model.pressure_bias = [
            0.0 for _ in range(model.n)
        ]
    if not hasattr(model, "bias"):
        model.bias = 0.0
    return data

def reconstruct_route(
    model,
    adj,
    pressure,
    source,
    dest,
    nodes
):
    route = [source]
    visited = {source}
    current = source
    previous = source
    max_steps = len(nodes)

    for _ in range(max_steps):
        if current == dest:
            return route

        best_neighbor = None
        best_score = -1e9

        for neighbor, sig in adj[current]:
            if neighbor in visited:
                continue

            model_score = model.score(
                source,
                dest,
                previous,
                neighbor,
                pressure
            )

            distance_to_dest = math.hypot(
                nodes[dest][0] - nodes[neighbor][0],
                nodes[dest][1] - nodes[neighbor][1]
            )

            current_distance = math.hypot(
                nodes[dest][0] - nodes[current][0],
                nodes[dest][1] - nodes[current][1]
            )

            progress = (
                current_distance -
                distance_to_dest
            ) / MAX_DISTANCE

            signal_bonus = (
                (sig + 130.0) / 100.0
            )

            congestion_penalty = (
                pressure[neighbor] * 0.8
            )

            total_score = (
                model_score * 0.35 +
                progress * 6.0 +
                signal_bonus * 1.5 -
                congestion_penalty
            )

            if neighbor == dest:
                total_score += 1000.0

            if total_score > best_score:
                best_score = total_score
                best_neighbor = neighbor

        if best_neighbor is None:
            break

        previous = current
        current = best_neighbor
        visited.add(current)
        route.append(current)

    return route

def route_metrics(route, adj, pressure):
    if len(route) < 2:
        return None

    total_signal = 0.0
    weakest = 999.0
    total_pressure = 0.0

    for i in range(len(route) - 1):
        u = route[i]
        v = route[i + 1]

        for neighbor, sig in adj[u]:
            if neighbor != v:
                continue

            total_signal += sig
            weakest = min(weakest, sig)
            total_pressure += pressure[v]
            break

    hops = len(route) - 1

    return {
        "hops": hops,
        "avg_signal": (
            total_signal / hops
        ),
        "weakest_signal": weakest,
        "avg_pressure": (
            total_pressure / hops
        )
    }

def draw_edges(canvas, nodes, adj):
    for i in adj:
        for j, sig in adj[i]:
            if j <= i:
                continue

            x1, y1 = nodes[i]
            x2, y2 = nodes[j]

            strength = (
                (sig + 130.0) / 100.0
            )

            width = 1.0 + strength * 3.0

            canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill="#dddddd",
                width=width
            )

def draw_route(canvas, nodes, route):
    if not route:
        return

    for i in range(len(route) - 1):
        u = route[i]
        v = route[i + 1]

        x1, y1 = nodes[u]
        x2, y2 = nodes[v]

        canvas.create_line(
            x1,
            y1,
            x2,
            y2,
            fill="#ff8800",
            width=6
        )

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

        canvas.create_oval(
            x - NODE_RADIUS,
            y - NODE_RADIUS,
            x + NODE_RADIUS,
            y + NODE_RADIUS,
            fill=fill,
            outline="black",
            width=2
        )

        canvas.create_text(
            x,
            y,
            text=str(idx),
            fill=text,
            font=("Arial", 10, "bold")
        )

def draw_walls(canvas, walls):
    for w in walls:
        canvas.create_line(
            w[0][0],
            w[0][1],
            w[1][0],
            w[1][1],
            fill="black",
            width=6
        )

def draw_info(canvas, route, metrics):
    if route is None or metrics is None:
        return

    x = 20
    y = HEIGHT - 140

    canvas.create_rectangle(
        x,
        y,
        x + 620,
        y + 110,
        fill="white",
        outline="#cccccc",
        width=2
    )

    canvas.create_text(
        x + 15,
        y + 20,
        anchor="w",
        text=(
            f"{selected_origin} -> "
            f"{selected_destination}"
        ),
        font=("Arial", 14, "bold")
    )

    canvas.create_text(
        x + 15,
        y + 50,
        anchor="w",
        text=(
            "Route: " +
            " -> ".join(map(str, route))
        ),
        font=("Arial", 11)
    )

    canvas.create_text(
        x + 15,
        y + 80,
        anchor="w",
        text=(
            f"Hops={metrics['hops']}    "
            f"Avg RSSI={metrics['avg_signal']:.1f} dBm    "
            f"Weakest={metrics['weakest_signal']:.1f} dBm    "
            f"Avg Pressure={metrics['avg_pressure']:.2f}"
        ),
        font=("Arial", 11)
    )

def redraw(
    canvas,
    nodes,
    walls,
    adj,
    model,
    pressure
):
    canvas.delete("all")

    route = None
    metrics = None

    if (
        selected_origin is not None and
        selected_destination is not None and
        selected_origin != selected_destination
    ):
        route = reconstruct_route(
            model,
            adj,
            pressure,
            selected_origin,
            selected_destination,
            nodes
        )

        if (
            route and
            route[-1] == selected_destination
        ):
            metrics = route_metrics(
                route,
                adj,
                pressure
            )

    draw_edges(canvas, nodes, adj)
    draw_route(canvas, nodes, route)
    draw_walls(canvas, walls)
    draw_nodes(canvas, nodes, route)
    draw_info(canvas, route, metrics)

def find_clicked_node(event, nodes):
    for idx, (x, y) in enumerate(nodes):
        d = math.hypot(
            event.x - x,
            event.y - y
        )

        if d <= NODE_RADIUS + 5:
            return idx

    return None

def on_click(
    event,
    canvas,
    nodes,
    walls,
    adj,
    model,
    pressure
):
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

    redraw(
        canvas,
        nodes,
        walls,
        adj,
        model,
        pressure
    )

def on_motion(
    event,
    canvas,
    nodes,
    walls,
    adj,
    model,
    pressure
):
    global hover_node

    hover_node = find_clicked_node(
        event,
        nodes
    )

    redraw(
        canvas,
        nodes,
        walls,
        adj,
        model,
        pressure
    )

def main():
    model_data = load_model(MODEL_PATH)

    model = model_data["model"]
    pressure = model_data["pressure"]

    nodes, walls = load_nodes(
        POINTS_PATH
    )

    adj = build_adjacency(
        nodes,
        walls
    )

    root = tk.Tk()

    root.title(
        "Simple Mesh Policy Viewer"
    )

    root.resizable(False, False)

    canvas = tk.Canvas(
        root,
        width=WIDTH,
        height=HEIGHT,
        bg="white",
        highlightthickness=0
    )

    canvas.pack()

    canvas.bind(
        "<Button-1>",
        lambda e: on_click(
            e,
            canvas,
            nodes,
            walls,
            adj,
            model,
            pressure
        )
    )

    canvas.bind(
        "<Motion>",
        lambda e: on_motion(
            e,
            canvas,
            nodes,
            walls,
            adj,
            model,
            pressure
        )
    )

    root.bind(
        "<Escape>",
        lambda e: root.destroy()
    )

    root.bind(
        "<Control-q>",
        lambda e: root.destroy()
    )

    redraw(
        canvas,
        nodes,
        walls,
        adj,
        model,
        pressure
    )

    root.mainloop()

if __name__ == "__main__":
    main()
