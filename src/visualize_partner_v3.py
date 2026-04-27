import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

try:
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
except Exception:
    FuncAnimation = None
    FFMpegWriter = None
    PillowWriter = None


# ----------------------------
# IO
# ----------------------------
def load_geom(geom_path: str):
    """
    Geom format:
      obj["points"] = [[x,y,type], ...]
      type: 0=background, 1=pipe_wall, 2=pipe_inside

    Returns:
      W, H
      geom_type_grid: (H,W) int32
      inlet, outlet: (x,y) or None
      obj: full json
    """
    obj = json.loads(Path(geom_path).read_text())
    W = int(obj["width"])
    H = int(obj["height"])

    pts = np.asarray(obj["points"], dtype=np.int32)
    x = pts[:, 0]
    y = pts[:, 1]
    tp = pts[:, 2]

    geom_type_grid = np.zeros((H, W), dtype=np.int32)
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    geom_type_grid[y[valid], x[valid]] = tp[valid]

    inlet = None
    outlet = None
    if "inlet" in obj and obj["inlet"] is not None:
        inlet = (int(obj["inlet"]["x"]), int(obj["inlet"]["y"]))
    if "outlet" in obj and obj["outlet"] is not None:
        outlet = (int(obj["outlet"]["x"]), int(obj["outlet"]["y"]))

    return W, H, geom_type_grid, inlet, outlet, obj


def detect_pred_format(obj):
    """
    Supported formats:
      1) flat list:
         [ {x,y,time,temperature,type,...}, ... ]
         [ {x,y,time,u,type,...}, ... ]
         [ {x,y,time,v,type,...}, ... ]
         [ {x,y,time,p,type,...}, ... ]

      2) packed_points:
         {
           "timesteps":[...],
           "points":[
             {"x":..,"y":..,"type":..,"temperature":[...]},
             ...
           ]
         }

      3) grid-style output written by _write_inference_json:
         {
           "xy":[[x,y], ...],
           "times":[...],
           "values":[ [...], [...], ... ],
           ...
         }
         or maybe:
         {
           "xy":[[x,y], ...],
           "times":[...],
           "temperature":[ [...], [...], ... ],
           ...
         }

      4) steady fields output:
         {
           "xy":[[x,y], ...],
           "fields":{
             "u":[...],
             "v":[...],
             "p":[...]
           },
           "point_type":[...],   # optional
           "xy_raw":[[x,y], ...] # optional
         }
    """
    if isinstance(obj, list):
        return "flat_list"

    if isinstance(obj, dict):
        if "timesteps" in obj and "points" in obj:
            return "packed_points"
        if "xy" in obj and "times" in obj:
            return "xy_times_values"
        if "xy" in obj and "fields" in obj:
            return "xy_fields_steady"

    raise ValueError(
        "Unknown pred format. Expected one of: flat-list, packed-points, xy/times/values, or xy/fields steady format."
    )


def detect_scalar_field_name_from_record(record):
    """
    Detect which scalar field exists in a single record.
    Priority order is temperature, u, v, p, then first unknown numeric field.
    """
    priority = ["temperature", "T", "u", "v", "p"]
    for k in priority:
        if k in record:
            return k

    skip = {"x", "y", "time", "type"}
    for k, v in record.items():
        if k in skip:
            continue
        if isinstance(v, (int, float)):
            return k

    raise ValueError(f"Could not detect scalar field in record keys: {list(record.keys())}")


def detect_scalar_field_name_from_points_entry(p):
    priority = ["temperature", "T", "u", "v", "p"]
    for k in priority:
        if k in p:
            return k

    skip = {"x", "y", "type"}
    for k, v in p.items():
        if k in skip:
            continue
        if isinstance(v, list):
            return k

    raise ValueError(f"Could not detect scalar field in packed point keys: {list(p.keys())}")


def detect_scalar_field_name_from_xy_times_values(obj):
    """
    Try to infer the scalar field name in the xy/times format.
    """
    priority = ["values", "temperature", "T", "u", "v", "p"]
    for k in priority:
        if k in obj:
            return k

    skip = {"xy", "times", "norm", "point_type", "points", "timesteps"}
    for k, v in obj.items():
        if k in skip:
            continue
        if isinstance(v, list):
            return k

    raise ValueError(f"Could not detect scalar field in xy/times object keys: {list(obj.keys())}")


def detect_scalar_field_name_from_xy_fields(obj):
    """
    Detect field name from steady xy/fields JSON.
    """
    if "fields" not in obj or not isinstance(obj["fields"], dict):
        raise ValueError("Expected dict with key 'fields'.")

    fields = obj["fields"]
    priority = ["temperature", "T", "u", "v", "p"]
    for k in priority:
        if k in fields:
            return k

    for k, v in fields.items():
        if isinstance(v, list):
            return k

    raise ValueError(f"Could not detect scalar field in obj['fields'] keys: {list(fields.keys())}")


def load_pred(pred_path: str, field_name: str = None):
    """
    Returns unified representation:
      times: List[float]
      by_t: Dict[float, List[dict]] where each dict has at least {x,y,type,<field>}
      scalar_name: detected or requested scalar field name
      has_time: whether input truly contains multiple/explicit timesteps
    """
    obj = json.loads(Path(pred_path).read_text())
    fmt = detect_pred_format(obj)

    by_t = defaultdict(list)

    # -------------------------
    # flat_list format
    # -------------------------
    if fmt == "flat_list":
        if len(obj) == 0:
            raise ValueError("Prediction list is empty.")

        scalar_name = field_name or detect_scalar_field_name_from_record(obj[0])

        has_any_time = all("time" in r for r in obj)
        if has_any_time:
            for r in obj:
                t = float(r["time"])
                by_t[t].append(r)
            times = sorted(by_t.keys())
            has_time = len(times) > 1
        else:
            t = 0.0
            for r in obj:
                rr = dict(r)
                rr["time"] = t
                by_t[t].append(rr)
            times = [t]
            has_time = False

        return times, by_t, scalar_name, has_time

    # -------------------------
    # packed_points format
    # -------------------------
    if fmt == "packed_points":
        pts = obj["points"]
        if len(pts) == 0:
            raise ValueError("Packed-points prediction has no points.")

        scalar_name = field_name or detect_scalar_field_name_from_points_entry(pts[0])

        times = [float(t) for t in obj["timesteps"]]
        Tn = len(times)
        has_time = len(times) > 1

        for p in pts:
            x = p["x"]
            y = p["y"]
            tp = p.get("type", -1)

            val_list = p.get(scalar_name, None)
            if val_list is None:
                continue

            if len(val_list) != Tn:
                raise ValueError(
                    f"Packed pred mismatch at point (x={x},y={y}): "
                    f"len({scalar_name})={len(val_list)} != len(timesteps)={Tn}"
                )

            for ti, t in enumerate(times):
                by_t[t].append(
                    {
                        "x": x,
                        "y": y,
                        "type": tp,
                        scalar_name: float(val_list[ti]),
                    }
                )

        return times, by_t, scalar_name, has_time

    # -------------------------
    # xy/times/values format
    # -------------------------
    if fmt == "xy_times_values":
        xy = np.asarray(obj["xy"], dtype=np.float32)
        times = [float(t) for t in obj["times"]]
        scalar_name = field_name or detect_scalar_field_name_from_xy_times_values(obj)

        if scalar_name == "values":
            vals = np.asarray(obj["values"], dtype=np.float32)
        else:
            vals = np.asarray(obj[scalar_name], dtype=np.float32)

        # prefer xy_raw when available because it usually matches geometry pixels
        if "xy_raw" in obj and obj["xy_raw"] is not None:
            xy_use = np.asarray(obj["xy_raw"], dtype=np.float32)
        else:
            xy_use = xy

        point_type = obj.get("point_type", None)
        if point_type is not None:
            point_type = np.asarray(point_type, dtype=np.int32)
        else:
            point_type = np.full((xy_use.shape[0],), -1, dtype=np.int32)

        if vals.ndim == 1:
            vals = vals[None, :]
            times = [times[0] if len(times) > 0 else 0.0]
            has_time = False
        elif vals.ndim == 2:
            has_time = len(times) > 1
        else:
            raise ValueError(f"Expected scalar values shape [T,N] or [N], got {vals.shape}")

        if vals.shape[1] != xy_use.shape[0]:
            raise ValueError(
                f"Mismatch: values has N={vals.shape[1]} points but xy has N={xy_use.shape[0]}"
            )

        if vals.shape[0] != len(times):
            raise ValueError(
                f"Mismatch: values has T={vals.shape[0]} timesteps but times has {len(times)}"
            )

        for ti, t in enumerate(times):
            for i in range(xy_use.shape[0]):
                by_t[t].append(
                    {
                        "x": float(xy_use[i, 0]),
                        "y": float(xy_use[i, 1]),
                        "type": int(point_type[i]),
                        scalar_name: float(vals[ti, i]),
                    }
                )

        return times, by_t, scalar_name, has_time

    # -------------------------
    # xy/fields steady format
    # -------------------------
    if fmt == "xy_fields_steady":
        xy = np.asarray(obj["xy"], dtype=np.float32)

        if "xy_raw" in obj and obj["xy_raw"] is not None:
            xy_use = np.asarray(obj["xy_raw"], dtype=np.float32)
        else:
            xy_use = xy

        scalar_name = field_name or detect_scalar_field_name_from_xy_fields(obj)

        if scalar_name not in obj["fields"]:
            raise ValueError(
                f"Requested field '{scalar_name}' not found in fields={list(obj['fields'].keys())}"
            )

        vals = np.asarray(obj["fields"][scalar_name], dtype=np.float32)

        point_type = obj.get("point_type", None)
        if point_type is not None:
            point_type = np.asarray(point_type, dtype=np.int32)
        else:
            point_type = np.full((xy_use.shape[0],), -1, dtype=np.int32)

        if vals.ndim != 1:
            raise ValueError(
                f"Expected steady field '{scalar_name}' to have shape [N], got {vals.shape}"
            )

        if vals.shape[0] != xy_use.shape[0]:
            raise ValueError(
                f"Mismatch: field '{scalar_name}' has N={vals.shape[0]} but xy has N={xy_use.shape[0]}"
            )

        t = 0.0
        for i in range(xy_use.shape[0]):
            by_t[t].append(
                {
                    "x": float(xy_use[i, 0]),
                    "y": float(xy_use[i, 1]),
                    "type": int(point_type[i]),
                    scalar_name: float(vals[i]),
                }
            )

        return [t], by_t, scalar_name, False

    raise ValueError(f"Unsupported prediction format: {fmt}")


# ----------------------------
# Build scalar frames (NO POINT MATCHING)
# ----------------------------
def build_scalar_frames(times, by_t, W, H, geom_type_grid, scalar_name, strict_type=True):
    """
    Returns:
      scalar_frames: (T,H,W) float32, NaN where no value
    """
    Tn = len(times)
    scalar_frames = np.full((Tn, H, W), np.nan, dtype=np.float32)

    valid_mask = (geom_type_grid != 0)

    for ti, t in enumerate(times):
        frame = np.full((H, W), np.nan, dtype=np.float32)

        rows = by_t[t]
        for r in rows:
            tp = int(r.get("type", -1))
            x = int(np.rint(float(r["x"])))
            y = int(np.rint(float(r["y"])))

            if not (0 <= x < W and 0 <= y < H):
                continue

            if not valid_mask[y, x]:
                continue

            if strict_type:
                if tp not in (1, 2):
                    continue
                if geom_type_grid[y, x] != tp:
                    continue
            else:
                if geom_type_grid[y, x] not in (1, 2):
                    continue

            if scalar_name not in r:
                continue

            frame[y, x] = float(r[scalar_name])

        scalar_frames[ti] = frame

        missing = np.isnan(frame[valid_mask]).sum()
        total = int(valid_mask.sum())
        filled = total - int(missing)
        print(f"[t={t:.3f}] filled valid pixels: {filled}/{total}, missing={missing}")

    return scalar_frames


# ----------------------------
# Inlet/Outlet radius highlight (GRID)
# ----------------------------
def disk_mask(W, H, center_xy, radius):
    cx, cy = center_xy
    yy, xx = np.ogrid[:H, :W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def pretty_label(field_name: str) -> str:
    mapping = {
        "temperature": "Temperature (°C)",
        "T": "Temperature (°C)",
        "u": "Velocity u",
        "v": "Velocity v",
        "p": "Pressure p",
        "values": "Scalar Value",
    }
    return mapping.get(field_name, field_name)


def pretty_title(field_name: str) -> str:
    mapping = {
        "temperature": "Predicted Temperature",
        "T": "Predicted Temperature",
        "u": "Predicted Velocity u",
        "v": "Predicted Velocity v",
        "p": "Predicted Pressure p",
        "values": "Predicted Scalar Field",
    }
    return mapping.get(field_name, f"Predicted {field_name}")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", required=False, default="/home/ubuntu/research/PINN/done.json",
                    help="Geometry JSON (width/height/points with type).")
    ap.add_argument("--pred", required=False, default="/home/ubuntu/research/PINN/done_pred_flow.json",
                    help="Prediction JSON. Supports flat-list, packed-points, xy/times, and steady xy/fields formats.")
    ap.add_argument("--outdir", default="viz_out_2", help="Output directory.")
    ap.add_argument("--field", default=None,
                    help="Optional scalar field name to force. Example: temperature, T, u, v, p")

    ap.add_argument("--frames", action="store_true", help="Save PNG frame per time step.")
    ap.add_argument("--mp4", action="store_true", help="Save MP4 animation (requires ffmpeg).")
    ap.add_argument("--gif", action="store_true", help="Save GIF animation.")
    ap.add_argument("--dpi", type=int, default=150)

    ap.add_argument("--inlet_radius", type=float, default=1.0, help="Radius (px) to highlight inlet neighborhood.")
    ap.add_argument("--outlet_radius", type=float, default=1.0, help="Radius (px) to highlight outlet neighborhood.")
    ap.add_argument("--strict_type", action="store_true",
                    help="Require pred.type to match geom type at (x,y). If off, fill any geom type in {1,2}.")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    W, H, geom_type_grid, inlet, outlet, _ = load_geom(args.geom)
    times, by_t, scalar_name, has_time = load_pred(args.pred, field_name=args.field)

    scalar_frames = build_scalar_frames(
        times=times,
        by_t=by_t,
        W=W,
        H=H,
        geom_type_grid=geom_type_grid,
        scalar_name=scalar_name,
        strict_type=args.strict_type,
    )

    finite = np.isfinite(scalar_frames)
    if finite.any():
        vmin = float(np.nanmin(scalar_frames))
        vmax = float(np.nanmax(scalar_frames))
    else:
        vmin, vmax = 0.0, 1.0

    print(f"[range] {scalar_name} vmin={vmin:.4f}, vmax={vmax:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axis_off()

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(
        np.ma.masked_invalid(scalar_frames[0]),
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(pretty_label(scalar_name))

    overlay = np.zeros((H, W, 4), dtype=np.float32)
    if inlet is not None:
        inlet_m = disk_mask(W, H, inlet, args.inlet_radius)
        overlay[inlet_m] = [1.0, 0.0, 0.0, 0.35]
    if outlet is not None:
        outlet_m = disk_mask(W, H, outlet, args.outlet_radius)
        overlay[outlet_m] = [0.0, 0.0, 1.0, 0.35]
    ax.imshow(overlay, origin="upper", interpolation="nearest")

    if inlet is not None:
        ax.scatter([inlet[0]], [inlet[1]], c="red", s=60, marker="o", label="inlet")
    if outlet is not None:
        ax.scatter([outlet[0]], [outlet[1]], c="blue", s=60, marker="o", label="outlet")
    if inlet is not None or outlet is not None:
        ax.legend(loc="upper right")

    if has_time:
        ax.set_title(f"{pretty_title(scalar_name)} | t={times[0]:.3f}s")
    else:
        ax.set_title(pretty_title(scalar_name))

    def draw_frame(ti):
        im.set_data(np.ma.masked_invalid(scalar_frames[ti]))
        if has_time:
            ax.set_title(f"{pretty_title(scalar_name)} | t={times[ti]:.3f}s")
        else:
            ax.set_title(pretty_title(scalar_name))
        return (im,)

    if not has_time:
        png_path = outdir / f"{scalar_name}.png"
        draw_frame(0)
        fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        print(f"[OK] wrote png to: {png_path}")

        if not (args.frames or args.mp4 or args.gif):
            plt.show()
        return

    if args.frames:
        for ti in range(len(times)):
            draw_frame(ti)
            fpath = outdir / f"{scalar_name}_frame_{ti:04d}_t{times[ti]:.3f}.png"
            fig.savefig(fpath, dpi=args.dpi, bbox_inches="tight")
        print(f"[OK] wrote frames to: {outdir}")

    if (args.mp4 or args.gif) and FuncAnimation is None:
        raise RuntimeError("matplotlib animation not available in this environment.")

    anim = None
    if args.mp4 or args.gif:
        anim = FuncAnimation(fig, draw_frame, frames=len(times), interval=200, blit=False)

    if args.mp4:
        mp4_path = outdir / f"{scalar_name}.mp4"
        writer = FFMpegWriter(fps=5) if FFMpegWriter is not None else None
        if writer is None:
            raise RuntimeError("FFMpegWriter not available. Install ffmpeg or use --gif/--frames.")
        anim.save(mp4_path, writer=writer, dpi=args.dpi)
        print(f"[OK] wrote mp4 to: {mp4_path}")

    if args.gif:
        gif_path = outdir / f"{scalar_name}.gif"
        writer = PillowWriter(fps=5) if PillowWriter is not None else None
        if writer is None:
            raise RuntimeError("PillowWriter not available. Install pillow or use --mp4/--frames.")
        anim.save(gif_path, writer=writer, dpi=args.dpi)
        print(f"[OK] wrote gif to: {gif_path}")

    if not (args.frames or args.mp4 or args.gif):
        plt.show()


if __name__ == "__main__":
    main()