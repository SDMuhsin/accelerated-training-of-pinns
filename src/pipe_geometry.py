"""
Pipe geometry extraction from battery cooling channel image.

Ported from partner team's get_boundaries.py (Feb 2026).
Reads Pipe6.png, extracts boundary/fluid pixels via BFS,
finds inlet/outlet openings, exports JSON + visualization PNG.

All logic identical to partner's original code.
"""

import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_binary_masks(image_path: str, white_thr: int = 250):
    """
    Returns:
      boundary_mask: (H,W) bool where True means boundary (white)
      fluid_mask:    (H,W) bool where True means fluid (non-white)
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)  # (H,W,3) uint8
    boundary_mask = (arr[..., 0] >= white_thr) & (arr[..., 1] >= white_thr) & (arr[..., 2] >= white_thr)
    fluid_mask = ~boundary_mask
    return boundary_mask, fluid_mask


def connected_components_in_strip(fluid_mask: np.ndarray, strip_w: int = 8, min_size: int = 20):
    """
    Find connected components (4-neighborhood) of fluid pixels restricted to left strip [0:strip_w).
    Returns: list of components; each component is a list of (y,x) tuples in full image coords.
    """
    H, W = fluid_mask.shape
    strip_w = min(strip_w, W)

    # Work on a small strip to keep BFS cheap
    strip = fluid_mask[:, :strip_w].copy()
    visited = np.zeros_like(strip, dtype=bool)

    comps = []
    for y in range(H):
        for x in range(strip_w):
            if not strip[y, x] or visited[y, x]:
                continue

            q = deque([(y, x)])
            visited[y, x] = True
            comp = []

            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))  # still in strip coords

                # 4-neighbors inside strip bounds
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < H and 0 <= nx < strip_w and strip[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))

            if len(comp) >= min_size:
                comps.append(comp)

    # Sort by size desc
    comps.sort(key=len, reverse=True)
    return comps


def pick_inlet_outlet(fluid_mask: np.ndarray, strip_w: int = 8):
    """
    Finds two left openings and returns inlet/outlet points as (x,y).

    Strategy:
      - Find connected components of fluid pixels in a left strip.
      - Take the two largest components.
      - For each component, pick a representative point near the leftmost x (min x in that component)
        and median y among those leftmost pixels.
      - Lower y (larger y value) => inlet; upper y => outlet.
    """
    comps = connected_components_in_strip(fluid_mask, strip_w=strip_w)

    if len(comps) < 2:
        raise RuntimeError(
            f"Could not find 2 openings in the left strip. "
            f"Try increasing strip_w or lowering min_size."
        )

    # Take two biggest components
    c1, c2 = comps[0], comps[1]

    def rep_point(comp):
        ys = np.array([p[0] for p in comp])
        xs = np.array([p[1] for p in comp])

        minx = xs.min()
        # focus on pixels at the leftmost x for stability
        sel = (xs == minx)
        ys_left = ys[sel]
        y_med = int(np.median(ys_left))
        x_rep = int(minx)
        return (x_rep, y_med)  # (x,y)

    p1 = rep_point(c1)
    p2 = rep_point(c2)

    # Determine inlet/outlet by y (image coords: y increases downward)
    if p1[1] > p2[1]:
        inlet, outlet = p1, p2
    else:
        inlet, outlet = p2, p1

    return inlet, outlet


def save_pipe_json(boundary_mask: np.ndarray, inlet, outlet, json_path: str):
    """
    Save JSON with:
      - width, height
      - inlet, outlet
      - points: list of [x,y,b] where b=1 boundary else 0
    """
    H, W = boundary_mask.shape

    # Build [x,y,b] list (row-major order)
    ys, xs = np.indices((H, W))
    b = boundary_mask.astype(np.uint8)
    points = np.stack([xs, ys, b], axis=-1).reshape(-1, 3).tolist()

    payload = {
        "width": int(W),
        "height": int(H),
        "inlet": {"x": int(inlet[0]), "y": int(inlet[1])},
        "outlet": {"x": int(outlet[0]), "y": int(outlet[1])},
        "points": points,
    }

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    print(f"[saved] {json_path}  (points={len(points)})")


def plot_from_json(json_path: str, save_png: str = None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    W = data["width"]
    H = data["height"]
    inlet = (data["inlet"]["x"], data["inlet"]["y"])
    outlet = (data["outlet"]["x"], data["outlet"]["y"])

    pts = np.array(data["points"], dtype=np.int32)  # (N,3) [x,y,b]
    x = pts[:, 0]
    y = pts[:, 1]
    b = pts[:, 2]  # 1 boundary, 0 fluid

    # Reconstruct boundary mask
    boundary_mask = np.zeros((H, W), dtype=np.uint8)
    boundary_mask[y, x] = b.astype(np.uint8)

    # Create black/white visualization: boundary=white(255), fluid=black(0)
    vis = (boundary_mask * 255).astype(np.uint8)

    plt.figure(figsize=(8, 6))
    plt.imshow(vis, cmap="gray", origin="upper")
    plt.scatter([inlet[0]], [inlet[1]], c="blue", s=60, marker="o", label="inlet")
    plt.scatter([outlet[0]], [outlet[1]], c="red", s=60, marker="o", label="outlet")
    plt.legend(loc="upper right")
    plt.title("Pipe reconstructed from JSON (boundary=white, fluid=black)")
    plt.axis("off")

    if save_png:
        plt.savefig(save_png, bbox_inches="tight", dpi=200)
        print(f"[saved] {save_png}")

    plt.close()


if __name__ == "__main__":
    # Paths updated for project structure
    image_path = "data/pipe_geometry/Pipe6.png"
    out_json = "results/pipe_geometry/pipe_points.json"
    out_plot = "results/pipe_geometry/pipe_from_json.png"

    boundary_mask, fluid_mask = load_binary_masks(image_path, white_thr=250)
    inlet, outlet = pick_inlet_outlet(fluid_mask, strip_w=10)

    print(f"[info] inlet (blue)  = {inlet}")
    print(f"[info] outlet (red)  = {outlet}")

    save_pipe_json(boundary_mask, inlet, outlet, out_json)
    plot_from_json(out_json, save_png=out_plot)
