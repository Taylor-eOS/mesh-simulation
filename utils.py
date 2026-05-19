import math

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segment_intersection(p1, p2, p3, p4):
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4))

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def sigmoid(x):
    x = max(-40, min(40, x))
    return 1.0 / (1.0 + math.exp(-x))
