import tkinter as tk

WIDTH, HEIGHT = 700, 500
OUTPUT_FILE = "points.txt"
GRID = 50
HIT_RADIUS = 5

def snap(x, y):
    return round(x / GRID) * GRID, round(y / GRID) * GRID

def draw_point(canvas, x, y):
    r = 4
    canvas.create_oval(x - r, y - r, x + r, y + r, fill="#e74c3c", outline="#c0392b", width=1)
    canvas.create_text(x + 10, y - 10, text=f"({x}, {y})", fill="#333333", font=("Arial", 9))

def draw_line(canvas, x1, y1, x2, y2):
    canvas.create_line(x1, y1, x2, y2, fill="#2980b9", width=2)
    r = 4
    canvas.create_oval(x1 - r, y1 - r, x1 + r, y1 + r, fill="#2980b9", outline="#1a5276", width=1)
    canvas.create_oval(x2 - r, y2 - r, x2 + r, y2 + r, fill="#2980b9", outline="#1a5276", width=1)

def redraw(canvas, points, lines):
    canvas.delete("all")
    draw_grid(canvas)
    for x, y in points:
        draw_point(canvas, x, y)
    for (x1, y1), (x2, y2) in lines:
        draw_line(canvas, x1, y1, x2, y2)

def write_file(points, lines):
    with open(OUTPUT_FILE, "w") as f:
        for x, y in points:
            f.write(f"({x}, {y}),\n")
        for (x1, y1), (x2, y2) in lines:
            f.write(f"({x1}, {y1}), ({x2}, {y2}),\n")

def find_node(x, y, points, lines):
    for i, (px, py) in enumerate(points):
        if (px - x) ** 2 + (py - y) ** 2 <= HIT_RADIUS ** 2:
            return ("point", i)
    for i, ((x1, y1), (x2, y2)) in enumerate(lines):
        if (x1 - x) ** 2 + (y1 - y) ** 2 <= HIT_RADIUS ** 2:
            return ("line", i)
        if (x2 - x) ** 2 + (y2 - y) ** 2 <= HIT_RADIUS ** 2:
            return ("line", i)
    return None

def remove_node(hit, canvas, points, lines):
    kind, i = hit
    if kind == "point":
        points.pop(i)
    else:
        lines.pop(i)
    write_file(points, lines)
    redraw(canvas, points, lines)

def record_point(event, canvas, points, lines, space_held):
    x, y = event.x, event.y
    if space_held[0]:
        x, y = snap(x, y)
    hit = find_node(x, y, points, lines)
    if hit:
        remove_node(hit, canvas, points, lines)
        return
    points.append((x, y))
    write_file(points, lines)
    draw_point(canvas, x, y)

def record_line(event, canvas, points, lines, line_start, space_held):
    x, y = event.x, event.y
    if space_held[0]:
        x, y = snap(x, y)
    hit = find_node(x, y, points, lines)
    if hit:
        remove_node(hit, canvas, points, lines)
        return
    if line_start[0] is None:
        line_start[0] = (x, y)
        r = 4
        canvas.create_oval(x - r, y - r, x + r, y + r, fill="#f39c12", outline="#d68910", width=1)
        canvas.create_text(x + 10, y - 10, text=f"({x}, {y})?", fill="#f39c12", font=("Arial", 9))
    else:
        x1, y1 = line_start[0]
        line_start[0] = None
        lines.append(((x1, y1), (x, y)))
        write_file(points, lines)
        redraw(canvas, points, lines)

def on_undo(event, canvas, points, lines, line_start):
    line_start[0] = None
    if lines:
        lines.pop()
    elif points:
        points.pop()
    else:
        return
    write_file(points, lines)
    redraw(canvas, points, lines)

def draw_grid(canvas):
    for x in range(0, WIDTH, 50):
        canvas.create_line(x, 0, x, HEIGHT, fill="#eeeeee", width=1)
    for y in range(0, HEIGHT, 50):
        canvas.create_line(0, y, WIDTH, y, fill="#eeeeee", width=1)
    for x in range(0, WIDTH, 50):
        canvas.create_text(x + 2, 4, text=str(x), fill="#cccccc", font=("Arial", 7), anchor="nw")
    for y in range(0, HEIGHT, 50):
        canvas.create_text(2, y + 2, text=str(y), fill="#cccccc", font=("Arial", 7), anchor="nw")

def load_file():
    points = []
    lines = []
    try:
        with open(OUTPUT_FILE, "r") as f:
            for raw in f:
                line = raw.strip().rstrip(",")
                if not line:
                    continue
                parts = line.split("), (")
                if len(parts) == 1:
                    text = parts[0].strip("()")
                    x, y = map(int, text.split(","))
                    points.append((x, y))
                elif len(parts) == 2:
                    p1 = parts[0].strip("()")
                    p2 = parts[1].strip("()")
                    x1, y1 = map(int, p1.split(","))
                    x2, y2 = map(int, p2.split(","))
                    lines.append(((x1, y1), (x2, y2)))
    except FileNotFoundError:
        open(OUTPUT_FILE, "w").close()
    return points, lines

def main():
    points, lines = load_file()
    space_held = [False]
    line_start = [None]
    root = tk.Tk()
    root.title("Point Recorder — LMB: point/remove, RMB: line (2 clicks)/remove, Ctrl+Z: undo, Space: snap")
    root.resizable(False, False)
    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white", highlightthickness=0)
    canvas.pack()
    redraw(canvas, points, lines)
    canvas.bind("<Button-1>", lambda e: record_point(e, canvas, points, lines, space_held))
    canvas.bind("<Button-3>", lambda e: record_line(e, canvas, points, lines, line_start, space_held))
    root.bind("<Control-z>", lambda e: on_undo(e, canvas, points, lines, line_start))
    root.bind("<KeyPress-space>", lambda e: space_held.__setitem__(0, True))
    root.bind("<KeyRelease-space>", lambda e: space_held.__setitem__(0, False))
    root.bind("<Control-q>", lambda e: root.destroy())
    root.mainloop()

if __name__ == "__main__":
    main()
