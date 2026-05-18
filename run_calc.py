import tkinter as tk
import math

WIDTH, HEIGHT = 700, 500
NODE_RADIUS = 8
MAX_DISTANCE = 500
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

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segment_intersection(p1, p2, p3, p4):
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

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
                
                canvas.create_line(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill="#7ab3d4",
                    width=width
                )
                
                if signal > -130:
                    label_x, label_y = find_label_position(x1, y1, x2, y2)
                    canvas.create_text(
                        label_x,
                        label_y,
                        text=f"{signal}",
                        fill="#333333",
                        font=("Arial", 10, "bold")
                    )
            else:
                canvas.create_line(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill="#cccccc",
                    width=1,
                    dash=(4, 4)
                )
    for w in walls:
        canvas.create_line(
            w[0][0],
            w[0][1],
            w[1][0],
            w[1][1],
            fill="black",
            width=3
        )
    for x, y in nodes:
        canvas.create_oval(
            x - NODE_RADIUS,
            y - NODE_RADIUS,
            x + NODE_RADIUS,
            y + NODE_RADIUS,
            fill="white",
            outline="black",
            width=2
        )

def main():
    root = tk.Tk()
    root.title("Mesh Routing Simulation")
    root.resizable(False, False)
    canvas = tk.Canvas(
        root,
        width=WIDTH,
        height=HEIGHT,
        bg="white",
        highlightthickness=0
    )
    canvas.pack()
    draw(canvas, NODES, WALLS)
    root.mainloop()

main()
