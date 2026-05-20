import tkinter as tk
import math
from utils import segment_intersection

WIDTH, HEIGHT = 700, 500
NODE_RADIUS = 12
MAX_DISTANCE = 500
NODES = [
    (100, 80),
    (80, 420),
    (210, 260),
    (490, 270),
    (580, 110),
    (620, 390),
]
WALLS = [
    ((320, 0), (300, 210)),
    ((360, 500), (340, 290)),
    ((560, 270), (700, 270)),
]
COLORS = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6"]
selected_origin = None

def nodes_connected(i, j, nodes, walls):
    for w in walls:
        if segment_intersection(nodes[i], nodes[j], w[0], w[1]):
            return False
    return True

def node_distance(i, j, nodes):
    x1, y1 = nodes[i]
    x2, y2 = nodes[j]
    return math.hypot(x2 - x1, y2 - y1)

def compute_signal_strength(distance):
    if distance >= MAX_DISTANCE:
        return -130
    return int(-130 + (1 - (distance / MAX_DISTANCE)) * 100)

def find_label_position(x1, y1, x2, y2):
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return x1, y1
    nx = -dy / length
    ny = dx / length
    offset = 15
    return mx + nx * offset, my + ny * offset

def dijkstra_paths(origin_idx, nodes, walls):
    n = len(nodes)
    distances = [float("inf")] * n
    predecessors = [-1] * n
    distances[origin_idx] = 0
    visited = [False] * n
    for _ in range(n):
        u = -1
        min_dist = float("inf")
        for i in range(n):
            if not visited[i] and distances[i] < min_dist:
                min_dist = distances[i]
                u = i
        if u == -1:
            break
        visited[u] = True
        for v in range(n):
            if u != v and nodes_connected(u, v, nodes, walls):
                d = node_distance(u, v, nodes)
                signal = compute_signal_strength(d)
                if signal > -130:
                    normalized_signal = (signal + 130) / 100
                    if normalized_signal > 0:
                        weight = 1.0 / (normalized_signal ** 3)
                    else:
                        weight = float("inf")
                    if distances[u] + weight < distances[v]:
                        distances[v] = distances[u] + weight
                        predecessors[v] = u
    all_paths = {}
    for target in range(n):
        if target == origin_idx or distances[target] == float("inf"):
            continue
        path = []
        curr = target
        while curr != -1:
            path.append(curr)
            curr = predecessors[curr]
        path.reverse()
        if len(path) > 2:
            all_paths[target] = path
    filtered_paths = {}
    for t1, p1 in all_paths.items():
        is_subpath = False
        for t2, p2 in all_paths.items():
            if t1 != t2 and len(p2) > len(p1):
                for i in range(len(p2) - len(p1) + 1):
                    if p2[i:i+len(p1)] == p1:
                        is_subpath = True
                        break
            if is_subpath:
                break
        if not is_subpath:
            filtered_paths[t1] = p1
    return filtered_paths

def draw(canvas, nodes, walls):
    canvas.delete("all")
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            x1, y1 = nodes[i]
            x2, y2 = nodes[j]
            if nodes_connected(i, j, nodes, walls):
                dist = node_distance(i, j, nodes)
                signal = compute_signal_strength(dist)
                if signal > -130:
                    t_val = (signal + 130) / 100
                    width = 2.5 + t_val * 3.5
                else:
                    width = 1
                canvas.create_line(x1, y1, x2, y2, fill="#e6e6e6", width=width)
                if signal > -130 and selected_origin is None:
                    label_x, label_y = find_label_position(x1, y1, x2, y2)
                    canvas.create_text(
                        label_x, label_y, text=f"{signal}", fill="#aaaaaa", font=("Arial", 10, "bold")
                    )
            else:
                canvas.create_line(x1, y1, x2, y2, fill="#f2f2f2", width=1, dash=(4, 4))
    for w in walls:
        canvas.create_line(w[0][0], w[0][1], w[1][0], w[1][1], fill="black", width=6)
    if selected_origin is not None:
        multi_hop_paths = dijkstra_paths(selected_origin, nodes, walls)
        color_index = 0
        for target, path in multi_hop_paths.items():
            color = COLORS[color_index % len(COLORS)]
            color_index += 1
            for step in range(len(path) - 1):
                u, v = path[step], path[step + 1]
                x1, y1 = nodes[u]
                x2, y2 = nodes[v]
                offset_multiplier = (color_index - 1) * 4
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                if length > 0:
                    nx = -dy / length
                    ny = dx / length
                    ox = nx * offset_multiplier
                    oy = ny * offset_multiplier
                else:
                    ox, oy = 0, 0
                canvas.create_line(
                    x1 + ox, y1 + oy, x2 + ox, y2 + oy,
                    fill=color, width=3
                )
    for idx, (x, y) in enumerate(nodes):
        if idx == selected_origin:
            fill_color = "#3cb44b"
            outline_color = "black"
            text_color = "white"
        elif selected_origin is not None:
            active_paths = dijkstra_paths(selected_origin, nodes, walls)
            is_target = any(idx == path[-1] for path in active_paths.values())
            is_internal = any(idx in path[1:-1] for path in active_paths.values())
            if is_target or is_internal:
                fill_color = "#4363d8"
                outline_color = "black"
                text_color = "white"
            else:
                fill_color = "white"
                outline_color = "black"
                text_color = "black"
        else:
            fill_color = "white"
            outline_color = "black"
            text_color = "black"
        canvas.create_oval(
            x - NODE_RADIUS, y - NODE_RADIUS, x + NODE_RADIUS, y + NODE_RADIUS,
            fill=fill_color, outline=outline_color, width=2
        )
        canvas.create_text(x, y, text=str(idx), fill=text_color, font=("Arial", 10, "bold"))

def on_canvas_click(event, canvas):
    global selected_origin
    clicked_node = None
    for idx, (x, y) in enumerate(NODES):
        if math.hypot(event.x - x, event.y - y) <= NODE_RADIUS + 5:
            clicked_node = idx
            break
    if clicked_node is not None:
        if selected_origin == clicked_node:
            selected_origin = None
        else:
            selected_origin = clicked_node
        draw(canvas, NODES, WALLS)

def main():
    root = tk.Tk()
    root.title("Mesh Routing Simulation")
    root.resizable(False, False)
    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white", highlightthickness=0)
    canvas.pack()
    canvas.bind("<Button-1>", lambda event: on_canvas_click(event, canvas))
    root.bind("<Control-q>", lambda e: root.destroy())
    draw(canvas, NODES, WALLS)
    root.mainloop()

if __name__ == "__main__":
    main()
