import tkinter as tk

WIDTH, HEIGHT = 700, 500
OUTPUT_FILE = "points.txt"
GRID = 50

def record_point(event, canvas, points, space_held):
    x, y = event.x, event.y
    if space_held[0]:
        x = round(x / GRID) * GRID
        y = round(y / GRID) * GRID
    points.append((x, y))
    with open(OUTPUT_FILE, "a") as f:
        f.write(f"({x}, {y}),\n")
    r = 4
    canvas.create_oval(x - r, y - r, x + r, y + r, fill="#e74c3c", outline="#c0392b", width=1)
    canvas.create_text(x + 10, y - 10, text=f"({x}, {y})", fill="#333333", font=("Arial", 9))

def on_undo(event, canvas, points):
    if not points:
        return
    points.pop()
    with open(OUTPUT_FILE, "w") as f:
        for px, py in points:
            f.write(f"({px}, {py}),\n")
    canvas.delete("all")
    draw_grid(canvas)
    for px, py in points:
        r = 4
        canvas.create_oval(px - r, py - r, px + r, py + r, fill="#e74c3c", outline="#c0392b", width=1)
        canvas.create_text(px + 10, py - 10, text=f"({px}, {py})", fill="#333333", font=("Arial", 9))

def draw_grid(canvas):
    for x in range(0, WIDTH, 50):
        canvas.create_line(x, 0, x, HEIGHT, fill="#eeeeee", width=1)
    for y in range(0, HEIGHT, 50):
        canvas.create_line(0, y, WIDTH, y, fill="#eeeeee", width=1)
    for x in range(0, WIDTH, 50):
        canvas.create_text(x + 2, 4, text=str(x), fill="#cccccc", font=("Arial", 7), anchor="nw")
    for y in range(0, HEIGHT, 50):
        canvas.create_text(2, y + 2, text=str(y), fill="#cccccc", font=("Arial", 7), anchor="nw")

def main():
    points = []
    space_held = [False]
    open(OUTPUT_FILE, "w").close()
    root = tk.Tk()
    root.title("Point Recorder — click to record, Ctrl+Z to undo, Space to snap")
    root.resizable(False, False)
    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white", highlightthickness=0)
    canvas.pack()
    draw_grid(canvas)
    canvas.bind("<Button-1>", lambda e: record_point(e, canvas, points, space_held))
    root.bind("<Control-z>", lambda e: on_undo(e, canvas, points))
    root.bind("<KeyPress-space>", lambda e: space_held.__setitem__(0, True))
    root.bind("<KeyRelease-space>", lambda e: space_held.__setitem__(0, False))
    root.mainloop()

main()
