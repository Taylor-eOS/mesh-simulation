import tkinter as tk
import math
import random

WIDTH, HEIGHT = 700, 500
NODE_RADIUS = 10
NUM_NODES = 5
WAVE_SPEED = 1.2
FRAME_MS = 16


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
        for tag, radius in self.rings.items():
            idx = int(tag.split("_")[1])
            sx, sy = self.nodes[idx]
            self.canvas.create_oval(
                sx - radius, sy - radius, sx + radius, sy + radius,
                outline="black", width=1, fill=""
            )
        for i, (x, y) in enumerate(self.nodes):
            if i in self.reached:
                fill = "black"
            elif i == self.sender_idx:
                fill = "black"
            else:
                fill = "white"
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
        sx, sy = self.nodes[self.sender_idx]
        for i, (nx, ny) in enumerate(self.nodes):
            if i == self.sender_idx or i in self.reached:
                continue
            dist = math.hypot(nx - sx, ny - sy)
            if self.rings.get(sender_tag, 0) >= dist - NODE_RADIUS:
                self.reached.add(i)
                self.rings[f"ring_{i}"] = NODE_RADIUS
        self._draw_frame()
        self.root.after(FRAME_MS, self._animate)


def main():
    root = tk.Tk()
    MeshSimApp(root)
    root.mainloop()


main()
