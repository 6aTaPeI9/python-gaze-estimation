import numpy as np

def ray_intersect_triangle(p0, p1, triangle):
    v0, v1, v2 = triangle
    u = v1 - v0
    v = v2 - v0
    normal = np.cross(u, v)
    b = np.inner(normal, p1 - p0)
    a = np.inner(normal, v0 - p0)

    if (b == 0.0):
        if a != 0.0:
            return 0
        else:
            rI = 0.0
    else:
        rI = a / b

    if rI < 0.0:
        return 0
    w = p0 + rI * (p1 - p0) - v0
    denom = np.inner(u, v) * np.inner(u, v) - \
        np.inner(u, u) * np.inner(v, v)
    si = (np.inner(u, v) * np.inner(w, v) - \
        np.inner(v, v) * np.inner(w, u)) / denom

    if (si < 0.0) | (si > 1.0):
        return 0
    ti = (np.inner(u, v) * np.inner(w, u) - \
        np.inner(u, u) * np.inner(w, v)) / denom

    if (ti < 0.0) | (si + ti > 1.0):
        return 0

    if (rI == 0.0):
        return 2
    return 1