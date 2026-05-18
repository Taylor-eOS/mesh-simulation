import tkinter as tk
import math

WIDTH, HEIGHT = 700, 500
NODE_RADIUS = 8

NODES = [
    (100, 80),
    (580, 110),
    (620, 390),
    (80, 420),
    (210, 260),
    (490, 270),
]

WALLS = [
    ((320, 0), (300, 210)),
    ((360, 500), (340, 290)),
]

def segment_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return False
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    return 0 < t < 1 and 0 < u < 1

def nodes_connected(i, j, nodes, walls):
    for w in walls:
        if segment_intersection(nodes[i], nodes[j], w[0], w[1]):
            return False
    return True

def node_distance(i, j, nodes):
    x1, y1 = nodes[i]
    x2, y2 = nodes[j]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def compute_connected_distance_range(nodes, walls):
    n = len(nodes)
    distances = [
        node_distance(i, j, nodes)
        for i in range(n)
        for j in range(i + 1, n)
        if nodes_connected(i, j, nodes, walls)
    ]
    if len(distances) < 2:
        return (distances[0], distances[0]) if distances else (1, 1)
    return min(distances), max(distances)

def draw(canvas, nodes, walls):
    canvas.delete("all")
    min_dist, max_dist = compute_connected_distance_range(nodes, walls)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            x1, y1 = nodes[i]
            x2, y2 = nodes[j]
            connected = nodes_connected(i, j, nodes, walls)
            if connected:
                dist = node_distance(i, j, nodes)
                t = (dist - min_dist) / (max_dist - min_dist) if max_dist != min_dist else 0.5
                width = 2.5 + (1 - t) * 3.5
                color = "#7ab3d4"
                canvas.create_line(x1, y1, x2, y2, fill=color, width=width)
            else:
                width = 1
                color = "#cccccc"
                canvas.create_line(x1, y1, x2, y2, fill=color, width=width, dash=(4, 4))
    for w in walls:
        canvas.create_line(w[0][0], w[0][1], w[1][0], w[1][1], fill="black", width=3)
    for x, y in nodes:
        canvas.create_oval(
            x - NODE_RADIUS, y - NODE_RADIUS,
            x + NODE_RADIUS, y + NODE_RADIUS,
            fill="white", outline="black", width=2
        )

def main():
    root = tk.Tk()
    root.title("Mesh Routing Simulation")
    root.resizable(False, False)
    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white", highlightthickness=0)
    canvas.pack()
    draw(canvas, NODES, WALLS)
    root.mainloop()

main()
