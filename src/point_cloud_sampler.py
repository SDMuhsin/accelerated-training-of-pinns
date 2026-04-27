"""
Point Cloud Sampler — CAD-to-point-cloud geometry converter.

Ported from partner team V3 code (from_partner_team/partner_code_v3/PCS.py).
UI/browser code removed; geometry processing logic preserved exactly.
"""

import numpy as np
import trimesh
import time
import os
import json
from pathlib import Path
from matplotlib.path import Path as mpl_path
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from collections import deque
from PIL import Image


class PointCloudSampler:
    """
    Convert CAD files (STEP/STL) or images into 2D point cloud JSON
    with wall/inside/background classification.
    """

    def __init__(self, cad_path=None):
        """
        Initialize from a STEP, STL, or PNG file.

        Args:
            cad_path: Path to STEP, STL, or PNG file.
        """
        self.step_to_stl_tol = 0.1
        self.polygon = None
        self.uploaded_image_path = None

        if cad_path is None:
            raise ValueError("cad_path is required")

        cad_ext = Path(cad_path).suffix.lower()
        if cad_ext in {".png"}:
            self.mesh = None
            self.uploaded_image_path = cad_path
        elif cad_ext in {".step", ".stp"}:
            cad_path = PointCloudSampler.__convert_step_to_stl__(
                cad_path, tol=self.step_to_stl_tol
            )
            self.mesh = trimesh.load_mesh(cad_path)
        else:
            self.mesh = trimesh.load_mesh(cad_path)

    @staticmethod
    def __convert_step_to_stl__(file_dir, tol=0.1):
        import importlib

        cadquery = importlib.import_module("cadquery")
        result = cadquery.importers.importStep(file_dir)
        abs_path = Path("/".join(file_dir.split("/")[:-1])).expanduser().resolve()
        save_path = os.path.join(abs_path, "exported.stl")
        cadquery.exporters.export(result, save_path, tolerance=tol)
        time.sleep(3)

        print("STEP-->STL SUCCESSFUL")
        return save_path

    def get_2d_proj(self, normal_axis):
        """
        Project mesh triangles into a 2D plane and merge them.
        normal_axis: 'x', 'y', or 'z'
        """
        if normal_axis == "x":
            dims = [1, 2]
        elif normal_axis == "y":
            dims = [0, 2]
        else:
            dims = [0, 1]

        polygons = []
        for face in self.mesh.triangles:
            poly_2d = Polygon(face[:, dims])
            if poly_2d.is_valid and poly_2d.area > 1e-9:
                polygons.append(poly_2d)

        silhouette = unary_union(polygons)

        if not silhouette.is_valid:
            silhouette = silhouette.buffer(0)

        self.polygon = silhouette
        print(f"2D PROJ SUCCESSFUL FOR {normal_axis}")

    def generate_mask(self, res=1024):
        """
        Generate binary mask: 1 for inside fluid, 0 for outside.

        Returns:
            binary_mask: (H, W) array with correct aspect ratio
            bounds: (minx, maxx, miny, maxy) in physical units
        """
        if self.polygon is None:
            raise Exception("Please run get_2d_proj() first")

        minx, miny, maxx, maxy = self.polygon.bounds

        width_mm = maxx - minx
        height_mm = maxy - miny
        aspect_ratio = width_mm / height_mm

        if width_mm >= height_mm:
            res_x = res
            res_y = int(res / aspect_ratio)
        else:
            res_y = res
            res_x = int(res * aspect_ratio)

        x = np.linspace(minx, maxx, res_x)
        y = np.linspace(miny, maxy, res_y)
        grid_x, grid_y = np.meshgrid(x, y)
        coords = np.vstack((grid_x.flatten(), grid_y.flatten())).T

        main_path = mpl_path(np.array(self.polygon.exterior.coords))
        mask = main_path.contains_points(coords)

        for interior in self.polygon.interiors:
            hole_path = mpl_path(np.array(interior.coords))
            mask = mask & ~hole_path.contains_points(coords)

        binary_mask = mask.reshape((res_y, res_x)).astype(np.uint8)
        return binary_mask, (minx, maxx, miny, maxy)

    def sample_points(self, boundary_count, interior_count):
        boundary_points = []
        perimeter = self.polygon.exterior.length
        distances = np.linspace(0, perimeter, boundary_count)
        for d in distances:
            pt = self.polygon.exterior.interpolate(d)
            boundary_points.append([pt.x, pt.y])

        for ring in self.polygon.interiors:
            hole_perimeter = ring.length
            hole_distances = np.linspace(0, hole_perimeter, int(boundary_count / 2))
            for d in hole_distances:
                pt = ring.interpolate(d)
                boundary_points.append([pt.x, pt.y])

        interior_points = []
        min_x, min_y, max_x, max_y = self.polygon.bounds
        while len(interior_points) < interior_count:
            batch = np.random.uniform(
                [min_x, min_y], [max_x, max_y], (interior_count, 2)
            )
            for p in batch:
                if self.polygon.contains(Point(p)):
                    interior_points.append(p)
                if len(interior_points) >= interior_count:
                    break

        return np.array(boundary_points), np.array(interior_points)

    @staticmethod
    def _connected_components_in_strip_at_x(mask, x0, strip_w=12, min_size=5):
        """
        Find 4-neighborhood connected components in columns [x0 : x0 + strip_w).
        """
        H, W = mask.shape
        x0 = int(np.clip(x0, 0, W - 1))
        x1 = int(np.clip(x0 + strip_w, 0, W))

        strip = mask[:, x0:x1]
        visited = np.zeros_like(strip, dtype=bool)
        comps = []

        for y in range(H):
            for sx in range(x1 - x0):
                if not strip[y, sx] or visited[y, sx]:
                    continue

                q = deque([(y, sx)])
                visited[y, sx] = True
                comp = []

                while q:
                    cy, csx = q.popleft()
                    comp.append((cy, csx + x0))

                    for ny, nsx in (
                        (cy - 1, csx),
                        (cy + 1, csx),
                        (cy, csx - 1),
                        (cy, csx + 1),
                    ):
                        if 0 <= ny < H and 0 <= nsx < (x1 - x0):
                            if strip[ny, nsx] and not visited[ny, nsx]:
                                visited[ny, nsx] = True
                                q.append((ny, nsx))

                if len(comp) >= min_size:
                    comps.append(comp)

        comps.sort(key=len, reverse=True)
        return comps

    @classmethod
    def _pick_inlet_outlet(
        cls, pipe_inside_mask, strip_w_candidates=(8, 10, 12, 14, 16), min_size=5
    ):
        """
        Robustly detect two left-side openings using adaptive strip widths.
        """
        ys, xs = np.where(pipe_inside_mask)
        if xs.size == 0:
            raise RuntimeError("pipe_inside_mask is empty.")

        x_min = int(xs.min())
        best = None

        for strip_w in strip_w_candidates:
            comps = cls._connected_components_in_strip_at_x(
                pipe_inside_mask, x0=x_min, strip_w=strip_w, min_size=min_size
            )
            if len(comps) >= 2:
                best = comps
                break

        if best is None:
            for dx in range(1, 10):
                for strip_w in strip_w_candidates:
                    comps = cls._connected_components_in_strip_at_x(
                        pipe_inside_mask,
                        x0=x_min + dx,
                        strip_w=strip_w,
                        min_size=min_size,
                    )
                    if len(comps) >= 2:
                        best = comps
                        break
                if best is not None:
                    break

        if best is None:
            raise RuntimeError(
                "Could not detect 2 openings. Try smaller strip widths or lower min_size."
            )

        c1, c2 = best[0], best[1]

        def rep_point(comp):
            ys_local = np.array([p[0] for p in comp])
            xs_local = np.array([p[1] for p in comp])
            minx = xs_local.min()
            y_med = int(np.median(ys_local[xs_local == minx]))
            return (int(minx), y_med)

        p1 = rep_point(c1)
        p2 = rep_point(c2)
        inlet, outlet = (p1, p2) if p1[1] > p2[1] else (p2, p1)
        return inlet, outlet

    @staticmethod
    def _class_mask_from_projection(mask):
        """
        Build a 3-class mask: 0=background, 1=pipe_wall, 2=pipe_inside.
        """
        fluid = mask.astype(bool)
        bg = ~fluid

        bg_nbr = np.zeros_like(bg, dtype=bool)
        shifts = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0),           (1, 0),
            (-1, 1),  (0, 1),  (1, 1),
        ]

        H, W = bg.shape
        for dx, dy in shifts:
            xs0 = max(0, dx)
            xe0 = W + min(0, dx)
            ys0 = max(0, dy)
            ye0 = H + min(0, dy)

            xs1 = max(0, -dx)
            xe1 = W + min(0, -dx)
            ys1 = max(0, -dy)
            ye1 = H + min(0, -dy)

            bg_nbr[ys0:ye0, xs0:xe0] |= bg[ys1:ye1, xs1:xe1]

        class_mask = np.zeros(mask.shape, dtype=np.uint8)
        wall = fluid & bg_nbr
        inside = fluid & (~wall)

        class_mask[wall] = 1
        class_mask[inside] = 2
        return class_mask

    def convert_to_json(self, output_path, res=512, strip_w=10, white_thr=250):
        if self.uploaded_image_path is not None and self.mesh is None:
            return self.__convert_to_json_img__(
                output_path=output_path, strip_w=strip_w, white_thr=white_thr
            )
        return self.__convert_to_json_cad__(
            output_path=output_path, res=res, strip_w=strip_w, white_thr=white_thr
        )

    def __convert_to_json_img__(self, output_path, strip_w=10, white_thr=250):
        img = Image.open(self.uploaded_image_path).convert("RGB")
        arr = np.array(img)
        boundary_mask = (
            (arr[..., 0] >= white_thr)
            & (arr[..., 1] >= white_thr)
            & (arr[..., 2] >= white_thr)
        )
        fluid_mask = ~boundary_mask

        H, W = fluid_mask.shape
        strip_w = min(strip_w, W)
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
                    comp.append((cy, cx))
                    for ny, nx in (
                        (cy - 1, cx),
                        (cy + 1, cx),
                        (cy, cx - 1),
                        (cy, cx + 1),
                    ):
                        if (
                            0 <= ny < H
                            and 0 <= nx < strip_w
                            and strip[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            q.append((ny, nx))
                if len(comp) >= 20:
                    comps.append(comp)

        comps.sort(key=len, reverse=True)
        if len(comps) < 2:
            raise RuntimeError("Could not find 2 openings in the left strip.")

        def rep_point(comp):
            ys = np.array([p[0] for p in comp])
            xs = np.array([p[1] for p in comp])
            minx = xs.min()
            sel = xs == minx
            ys_left = ys[sel]
            y_med = int(np.median(ys_left))
            x_rep = int(minx)
            return (x_rep, y_med)

        p1 = rep_point(comps[0])
        p2 = rep_point(comps[1])

        if p1[1] > p2[1]:
            inlet, outlet = p1, p2
        else:
            inlet, outlet = p2, p1

        ys_idx, xs_idx = np.indices((H, W))
        b = boundary_mask.astype(np.uint8)
        points = np.stack([xs_idx, ys_idx, b], axis=-1).reshape(-1, 3).tolist()

        payload = {
            "width": int(W),
            "height": int(H),
            "inlet": {"x": int(inlet[0]), "y": int(inlet[1])},
            "outlet": {"x": int(outlet[0]), "y": int(outlet[1])},
            "points": points,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        print(f"Saved JSON: {output_path}")
        print(f"Resolution: {W}x{H}, Points: {len(points):,}")
        print(f"Inlet: x={inlet[0]}, y={inlet[1]}")
        print(f"Outlet: x={outlet[0]}, y={outlet[1]}")
        return payload

    def __convert_to_json_cad__(self, output_path, res=512, strip_w=10, white_thr=250):
        """Convert 2D projection to JSON format."""
        if self.polygon is None:
            self.get_2d_proj("z")

        mask, bounds = self.generate_mask(res=res)
        H, W = mask.shape

        class_mask = self._class_mask_from_projection(mask)
        pipe_inside_mask = class_mask == 2
        if not pipe_inside_mask.any():
            pipe_inside_mask = class_mask > 0

        inlet, outlet = self._pick_inlet_outlet(
            pipe_inside_mask=pipe_inside_mask,
            strip_w_candidates=(8, 10, 12, 14, 16, max(6, int(strip_w))),
            min_size=5,
        )

        ys, xs = np.indices((H, W))
        labels = class_mask.reshape(-1)
        points = np.stack([xs.reshape(-1), ys.reshape(-1), labels], axis=-1).tolist()

        minx, maxx, miny, maxy = bounds

        payload = {
            "width": int(W),
            "height": int(H),
            "bounds": {
                "x_min": float(minx),
                "x_max": float(maxx),
                "y_min": float(miny),
                "y_max": float(maxy),
            },
            "inlet": {"x": int(inlet[0]), "y": int(inlet[1])},
            "outlet": {"x": int(outlet[0]), "y": int(outlet[1])},
            "points": points,
            "legend": {"0": "background", "1": "pipe_wall", "2": "pipe_inside"},
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        print(f"Saved JSON: {output_path}")
        print(f"Resolution: {W}x{H}")
        print(f"Physical size: {maxx-minx:.2f} x {maxy-miny:.2f} mm")
        print(f"Points: {len(points):,}, Inlet: x={inlet[0]} y={inlet[1]}, Outlet: x={outlet[0]} y={outlet[1]}")
        return payload

    @staticmethod
    def plot_from_json(json_path, save_png=None):
        import matplotlib.pyplot as plt

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        W = data["width"]
        H = data["height"]
        inlet = (data["inlet"]["x"], data["inlet"]["y"])
        outlet = (data["outlet"]["x"], data["outlet"]["y"])

        pts = np.array(data["points"], dtype=np.int32)
        x = pts[:, 0]
        y = pts[:, 1]
        label = pts[:, 2]

        class_mask = np.zeros((H, W), dtype=np.uint8)
        class_mask[y, x] = label.astype(np.uint8)

        if np.max(class_mask) <= 1:
            vis = (class_mask * 255).astype(np.uint8)
            cmap = "gray"
        else:
            vis = np.zeros((H, W, 3), dtype=np.uint8)
            vis[class_mask == 0] = [255, 255, 255]
            vis[class_mask == 1] = [0, 0, 0]
            vis[class_mask == 2] = [180, 180, 180]
            cmap = None

        plt.figure(figsize=(8, 6))
        plt.imshow(vis, cmap=cmap, origin="upper")
        plt.scatter([inlet[0]], [inlet[1]], c="red", s=60, marker="o", label="inlet")
        plt.scatter([outlet[0]], [outlet[1]], c="blue", s=60, marker="o", label="outlet")
        plt.legend(loc="upper right")
        plt.title("Pipe from JSON (0=bg, 1=wall, 2=inside)")
        plt.axis("off")

        if save_png:
            plt.savefig(save_png, bbox_inches="tight", dpi=200)
            print(f"[saved] {save_png}")

        plt.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python point_cloud_sampler.py <path_to_step_or_stl>")
        sys.exit(1)
    pcs = PointCloudSampler(cad_path=sys.argv[1])
    pcs.convert_to_json("converted.json")
    PointCloudSampler.plot_from_json("converted.json", save_png="converted.png")
