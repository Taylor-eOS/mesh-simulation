import tkinter as tk
import math
import random

WIDTH, HEIGHT = 700, 500
NODE_RADIUS = 8
NUM_NODES = 6
PULSE_SPEED = 2.1
FRAME_MS = 40
RING_POINTS = 45

def segment_intersection_t(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if 0 < t < 1 and 0 < u < 1:
        return t
    return None

side = random.choice([True, False])
if side:
    w1_min_x, w1_max_x = 200, 320
    w2_min_x, w2_max_x = 380, 500
else:
    w1_min_x, w1_max_x = 380, 500
    w2_min_x, w2_max_x = 200, 320

wall_x2 = random.randint(w1_min_x, w1_max_x)
wall_height = random.randint(220, 320)
wall_x1 = wall_x2 + random.randint(-40, 40)
wall1 = ((wall_x1, HEIGHT - wall_height), (wall_x2, HEIGHT))
wall2_x1 = random.randint(w2_min_x, w2_max_x)
wall2_height = random.randint(200, 280)
wall2_x2 = wall2_x1 + random.randint(-40, 40)
wall2 = ((wall2_x1, 0), (wall2_x2, wall2_height))
WALL = [wall1, wall2]

def generate_nodes(n, margin=80):
    nodes = []
    attempts = 0
    center_x = WIDTH // 2
    center_y = HEIGHT // 2
    exclude_w = 200
    exclude_h = 150
    while len(nodes) < n and attempts < 10000:
        x = random.randint(margin, WIDTH - margin)
        y = random.randint(margin, HEIGHT - margin)
        in_center = (center_x - exclude_w // 2 <= x <= center_x + exclude_w // 2) and \
                    (center_y - exclude_h // 2 <= y <= center_y + exclude_h // 2)
        if in_center:
            attempts += 1
            continue
        if all(math.hypot(x - nx, y - ny) > 90 for nx, ny in nodes):
            nodes.append((x, y))
        attempts += 1
    return nodes

def precompute_ring_stops(nodes, wall, n_points):
    far = math.hypot(WIDTH, HEIGHT) * 2
    stops = {}
    for i, (ox, oy) in enumerate(nodes):
        ray_stops = []
        for k in range(n_points):
            angle = 2 * math.pi * k / n_points
            dx, dy = math.cos(angle), math.sin(angle)
            tip = (ox + dx * far, oy + dy * far)
            min_t = float("inf")
            for w in wall:
                t = segment_intersection_t((ox, oy), tip, w[0], w[1])
                if t is not None and t < min_t:
                    min_t = t
            ray_stops.append(min_t * far if min_t != float("inf") else float("inf"))
        stops[i] = ray_stops
    return stops

def precompute_can_reach(nodes, wall):
    can_reach = {}
    for i, origin in enumerate(nodes):
        for j, target in enumerate(nodes):
            if i == j:
                continue
            min_t = None
            for w in wall:
                t = segment_intersection_t(origin, target, w[0], w[1])
                if t is not None:
                    if min_t is None or t < min_t:
                        min_t = t
            if min_t is None:
                can_reach[(i, j)] = None
            else:
                path_len = math.hypot(target[0] - origin[0], target[1] - origin[1])
                can_reach[(i, j)] = min_t * path_len
    return can_reach

class Pulse:
    def __init__(self, origin_idx, target_idx, nodes, stop_dist):
        self.origin_idx = origin_idx
        self.target_idx = target_idx
        ox, oy = nodes[origin_idx]
        tx, ty = nodes[target_idx]
        self.path_len = math.hypot(tx - ox, ty - oy)
        self.stop_dist = stop_dist
        self.progress = 0.0
        self.delivered = False
        self.done = False

    def step(self):
        self.progress += PULSE_SPEED
        if self.stop_dist is not None and self.progress >= self.stop_dist:
            self.done = True
        elif self.stop_dist is None and self.progress >= self.path_len - NODE_RADIUS:
            self.delivered = True
            self.done = True

class Ring:
    def __init__(self, origin_idx, ray_stops):
        self.origin_idx = origin_idx
        self.ray_stops = ray_stops
        self.radius = float(NODE_RADIUS)
        self.done = False

    def step(self):
        self.radius += PULSE_SPEED
        if all(r != float("inf") and self.radius >= r for r in self.ray_stops):
            self.done = True
        elif self.radius > math.hypot(WIDTH, HEIGHT):
            self.done = True

    def points(self, ox, oy, n_points):
        pts = []
        for k in range(n_points):
            angle = 2 * math.pi * k / n_points
            r = min(self.radius, self.ray_stops[k])
            stopped = self.ray_stops[k] != float("inf") and self.radius >= self.ray_stops[k]
            pts.append((ox + math.cos(angle) * r, oy + math.sin(angle) * r, stopped))
        return pts

class MeshSimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mesh Routing Simulation")
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white", highlightthickness=0)
        self.canvas.pack()
        self.nodes = generate_nodes(NUM_NODES)
        self.ring_stops = precompute_ring_stops(self.nodes, WALL, RING_POINTS)
        self.can_reach = precompute_can_reach(self.nodes, WALL)
        self.sender_idx = 0
        self.reached = set()
        self.pulses = []
        self.rings = []
        self.fired_from = set()
        self._draw_frame()
        self.root.after(500, self._launch_from(self.sender_idx))

    def _launch_from(self, origin_idx):
        def _inner():
            self.fired_from.add(origin_idx)
            for j in range(len(self.nodes)):
                if j == origin_idx:
                    continue
                stop_dist = self.can_reach[(origin_idx, j)]
                self.pulses.append(Pulse(origin_idx, j, self.nodes, stop_dist))
            self.rings.append(Ring(origin_idx, self.ring_stops[origin_idx]))
            self._animate()
        return _inner

    def _draw_frame(self):
        self.canvas.delete("all")
        for w in WALL:
            self.canvas.create_line(
                w[0][0], w[0][1], w[1][0], w[1][1],
                fill="black", width=3
            )
        for ring in self.rings:
            ox, oy = self.nodes[ring.origin_idx]
            pts = ring.points(ox, oy, RING_POINTS)
            n = len(pts)
            for k in range(n):
                x1, y1, s1 = pts[k]
                x2, y2, s2 = pts[(k + 1) % n]
                if not s1 and not s2:
                    self.canvas.create_line(x1, y1, x2, y2, fill="black", width=1)
        for i, (x, y) in enumerate(self.nodes):
            fill = "black" if (i in self.reached or i == self.sender_idx) else "white"
            self.canvas.create_oval(
                x - NODE_RADIUS, y - NODE_RADIUS,
                x + NODE_RADIUS, y + NODE_RADIUS,
                fill=fill, outline="black", width=2
            )

    def _animate(self):
        newly_reached = []
        for pulse in self.pulses:
            pulse.step()
            if pulse.delivered and pulse.target_idx not in self.reached:
                self.reached.add(pulse.target_idx)
                newly_reached.append(pulse.target_idx)
        self.pulses = [p for p in self.pulses if not p.done]
        for ring in self.rings:
            ring.step()
        self.rings = [r for r in self.rings if not r.done]
        for idx in newly_reached:
            if idx not in self.fired_from:
                self.fired_from.add(idx)
                self.rings.append(Ring(idx, self.ring_stops[idx]))
                for j in range(len(self.nodes)):
                    if j == idx:
                        continue
                    if j not in self.reached and j != self.sender_idx:
                        stop_dist = self.can_reach[(idx, j)]
                        self.pulses.append(Pulse(idx, j, self.nodes, stop_dist))
        self._draw_frame()
        if self.pulses or self.rings:
            self.root.after(FRAME_MS, self._animate)

def main():
    root = tk.Tk()
    MeshSimApp(root)
    root.mainloop()

main()
