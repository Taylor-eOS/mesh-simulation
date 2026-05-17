import tkinter as tk
import math
import random

WIDTH, HEIGHT = 700, 500
NODE_RADIUS = 10
NUM_NODES = 5
WAVE_SPEED = 1.2
FRAME_MS = 16

WALL = ((WIDTH // 2, 80), (WIDTH // 2 - 60, HEIGHT - 80))


def generate_nodes(n, margin=80):
    random.seed(42)
    nodes = []
    attempts = 0
    while len(nodes) < n and attempts < 10000:
        x = random.randint(margin, WIDTH - margin)
        y = random.randint(margin, HEIGHT - margin)
        if all(math.hypot(x - nx, y - ny) > 90 for nx, ny in nodes):
            nodes.append((x, y))
        attempts += 1
    return nodes


def segments_intersect(p1, p2, p3, p4):
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


def wall_blocks(origin, target):
    return segments_intersect(origin, target, WALL[0], WALL[1])


class MeshSimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mesh Routing Simulation")
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white", highlightthickness=0)
        self.canvas.pack()
        self.nodes = generate_nodes(NUM_NODES)
        self.sender_idx = 0
        self.reached = set()
        self.rings = {}
        self._draw_frame()
        self.root.after(500, self._animate)

    def _draw_frame(self):
        self.canvas.delete("all")
        self.canvas.create_line(
            WALL[0][0], WALL[0][1], WALL[1][0], WALL[1][1],
            fill="black", width=3
        )
        for tag, radius in self.rings.items():
            idx = int(tag.split("_")[1])
            sx, sy = self.nodes[idx]
            self.canvas.create_oval(
                sx - radius, sy - radius, sx + radius, sy + radius,
                outline="black", width=1, fill=""
            )
        for i, (x, y) in enumerate(self.nodes):
            fill = "black" if (i in self.reached or i == self.sender_idx) else "white"
            self.canvas.create_oval(
                x - NODE_RADIUS, y - NODE_RADIUS,
                x + NODE_RADIUS, y + NODE_RADIUS,
                fill=fill, outline="black", width=2
            )

    def _animate(self):
        sender_tag = f"ring_{self.sender_idx}"
        if sender_tag not in self.rings:
            self.rings[sender_tag] = NODE_RADIUS
        for tag in list(self.rings):
            self.rings[tag] += WAVE_SPEED
        for i, (nx, ny) in enumerate(self.nodes):
            if i in self.reached or i == self.sender_idx:
                continue
            for tag, radius in self.rings.items():
                origin_idx = int(tag.split("_")[1])
                ox, oy = self.nodes[origin_idx]
                dist = math.hypot(nx - ox, ny - oy)
                if radius >= dist - NODE_RADIUS and not wall_blocks((ox, oy), (nx, ny)):
                    self.reached.add(i)
                    self.rings[f"ring_{i}"] = NODE_RADIUS
                    break
        self._draw_frame()
        self.root.after(FRAME_MS, self._animate)


def main():
    root = tk.Tk()
    MeshSimApp(root)
    root.mainloop()


main()
