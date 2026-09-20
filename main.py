"""
Ego-vehicle trajectory & BEV mapping (Part A).

Uses the traffic light as a fixed world reference point:
  - The traffic light's 3D position is read from the per-frame point cloud
    (xyz/depthNNNNNN.npz) at the pixel given by its bounding box center
    (bbox_light.csv).
  - The README states camera axes as (X forward, Y right, Z up), but the
    actual .npz data contradicts this: sampling pixels across increasing
    image columns (physically left -> right) shows the Y value *decreasing*
    monotonically. That means +Y in the data already points left (same
    physical direction as the world frame's +Y), not right. We trust the
    verified data over the written spec and do NOT mirror Y when converting
    from camera to world axes.
  - Since we have no heading/IMU data, we assume the car's heading stays
    parallel to its heading at the first usable frame (t0): i.e. the camera
    axes are treated as a fixed translating frame. This lets us define the
    ground/world frame as specified: origin under the light, Z through the
    light, and at t0 the car->light line along +X.
  - The car's world position is then just the (rotated, sign-flipped)
    negative of the light's camera-frame position.
"""

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio_ffmpeg

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def make_ffmpeg_writer(fps=10):
    # +faststart moves the moov atom to the front of the file, which
    # matplotlib/ffmpeg don't do by default; without it some players/web
    # previews (e.g. VS Code's) can fail to play an otherwise-valid mp4.
    # Without an explicit CRF, ffmpeg's default quality for this codec was
    # extremely low (~10 kb/s for a 600x600 video), producing visible
    # blocking/ghosting artifacts; -crf 18 gives a visually lossless result.
    return animation.FFMpegWriter(
        fps=fps,
        extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart"],
    )

RGB_DIR = "rgb"
XYZ_DIR = "xyz"
BBOX_CSV = "bbox_light.csv"
PATCH = 3  # +/- pixels around bbox center to sample for depth

OUT_TRAJ_PNG = "trajectory.png"
OUT_TRAJ_MP4 = "trajectory.mp4"
OUT_BEV_MP4 = "bev_ego_frame.mp4"

# First frame that has a matching depth file; used as t0 for Part A's world
# frame, and as the seed frame for the Part B golf-cart tracker.
FIRST_FRAME = 65
CART_SEED_BBOX = (355, 610, 150, 130)  # (x, y, w, h) in left000065.png


def xyz_path(frame_idx):
    return os.path.join(XYZ_DIR, f"depth{frame_idx:06d}.npz")


def rgb_path(frame_idx):
    return os.path.join(RGB_DIR, f"left{frame_idx:06d}.png")


def sample_xyz_patch(data, cx, cy, patch=PATCH):
    """Median XYZ (camera coords) of valid points in a small patch around (cx, cy)."""
    h, w = data.shape[:2]
    u, v = int(round(cx)), int(round(cy))

    u0, u1 = max(u - patch, 0), min(u + patch + 1, w)
    v0, v1 = max(v - patch, 0), min(v + patch + 1, h)
    win = data[v0:v1, u0:u1]

    valid = win[..., 3] == 0
    if not np.any(valid):
        return None

    pts = win[..., :3][valid]
    return np.median(pts, axis=0)  # (X, Y, Z)


def light_position_camera_frame(frame_idx, cx, cy):
    path = xyz_path(frame_idx)
    if not os.path.exists(path):
        return None
    data = np.load(path)["xyz"]  # (H, W, 4): x, y, z, flag (0 == valid)
    return sample_xyz_patch(data, cx, cy)


def load_light_bboxes(csv_path):
    df = pd.read_csv(csv_path)
    bad = (df[["x1", "y1", "x2", "y2"]] == 0).all(axis=1)
    df = df[~bad]
    df["cx"] = (df.x1 + df.x2) / 2.0
    df["cy"] = (df.y1 + df.y2) / 2.0
    return df.set_index("frame")[["cx", "cy"]]


def build_trajectory(bboxes):
    frames, cam_xyz = [], []
    for frame_idx, row in bboxes.iterrows():
        pos = light_position_camera_frame(frame_idx, row.cx, row.cy)
        if pos is None:
            continue
        frames.append(frame_idx)
        cam_xyz.append(pos)

    order = np.argsort(frames)
    frames = np.array(frames)[order]
    cam_xyz = np.array(cam_xyz)[order]  # (N, 3): Xc, Yc, Zc (light in camera coords)

    # Rotation that aligns the car->light direction at t0 with world +X.
    x0, y0 = cam_xyz[0, 0], cam_xyz[0, 1]
    theta0 = np.arctan2(y0, x0)
    c, s = np.cos(theta0), np.sin(theta0)

    xc, yc = cam_xyz[:, 0], cam_xyz[:, 1]
    xr = xc * c + yc * s   # light position, rotated so t0 direction -> +X
    yr = -xc * s + yc * c

    # Car position = -(light position). No axis flip: empirically (see below)
    # the data's +Y already points left, matching the world frame's +Y (left),
    # so camera-Y and world-Y are the same physical direction.
    x_world = -xr
    y_world = -yr

    return frames, x_world, y_world


def plot_static(frames, x_world, y_world):
    plt.figure(figsize=(6, 6))
    plt.plot(x_world, y_world, "-o", ms=3, lw=1, color="tab:blue", label="ego trajectory")
    plt.scatter([0], [0], marker="*", s=200, color="orange", label="traffic light (origin)")
    plt.scatter([x_world[0]], [y_world[0]], marker="s", s=60, color="green", label="start")
    plt.scatter([x_world[-1]], [y_world[-1]], marker="X", s=60, color="red", label="end")
    plt.xlabel("X (m, forward at t0)")
    plt.ylabel("Y (m, left)")
    plt.title("Ego-vehicle trajectory in ground (BEV) frame")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_TRAJ_PNG, dpi=150)
    plt.close()


def animate_trajectory(frames, x_world, y_world):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("X (m, forward at t0)")
    ax.set_ylabel("Y (m, left)")
    ax.set_title("Ego-vehicle trajectory (ground frame)")
    ax.grid(True, alpha=0.3)
    ax.scatter([0], [0], marker="*", s=200, color="orange", label="traffic light")
    line, = ax.plot([], [], "-o", ms=3, lw=1, color="tab:blue", label="ego trajectory")
    car, = ax.plot([], [], "s", ms=8, color="green", label="car")
    ax.legend(loc="upper right")

    # Fix the view (and lock it) only after every artist has been added -
    # ax.axis("equal")/autoscale re-triggered by a later artist would
    # otherwise collapse these limits back down to the last-added point.
    # The bounds must include (0, 0) too, or the traffic-light star (plotted
    # at the origin) falls outside the view and never appears in the video.
    # With explicit xlim/ylim already set, set_aspect("equal") on a square
    # figure shrinks the drawn axes box to satisfy the 1:1 data scale instead
    # of adjusting the limits, which left large blank bars top/bottom (the
    # X span here is ~3x the Y span). Pad both axes out to the same span,
    # centered on their own data, so the equal-aspect box fills the square
    # figure with no letterboxing.
    pad = 2.0
    x_all = np.append(x_world, 0.0)
    y_all = np.append(y_world, 0.0)
    x_lo, x_hi = x_all.min() - pad, x_all.max() + pad
    y_lo, y_hi = y_all.min() - pad, y_all.max() + pad
    span = max(x_hi - x_lo, y_hi - y_lo)
    x_mid, y_mid = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
    ax.set_xlim(x_mid - span / 2, x_mid + span / 2)
    ax.set_ylim(y_mid - span / 2, y_mid + span / 2)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    fig.tight_layout()

    def update(i):
        line.set_data(x_world[: i + 1], y_world[: i + 1])
        car.set_data([x_world[i]], [y_world[i]])
        return line, car

    # blit=False: blitting only redraws the changed artists and is meant for
    # interactive backends - with the ffmpeg writer it left stale copies of
    # the "car" marker on screen and, on some backends, never composited the
    # static background (title/axes/star) into the saved frames at all.
    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, blit=False)
    try:
        anim.save(OUT_TRAJ_MP4, writer=make_ffmpeg_writer())
    except Exception as e:
        print(f"Could not write {OUT_TRAJ_MP4} ({e}); is ffmpeg installed?")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part B (bonus): golf cart (dynamic) + barrels (static) in an ego-centric BEV.
# The car sits fixed at the origin facing +X; every other object is placed by
# its raw camera-frame (X forward, Y left -- see note above) position for
# that frame, used directly with no sign flip, matching Part A.
# ---------------------------------------------------------------------------

def track_cart_centers(frame_ids):
    """Track the golf cart across every rgb frame with a simple appearance
    tracker (seeded once, by hand, on the first usable frame), returning the
    tracked box center for each requested frame_id."""
    tracker = cv2.TrackerMIL_create()
    seed_img = cv2.imread(rgb_path(FIRST_FRAME))
    tracker.init(seed_img, CART_SEED_BBOX)

    wanted = set(frame_ids)
    centers = {}
    for f in range(FIRST_FRAME, max(frame_ids) + 1):
        img = seed_img if f == FIRST_FRAME else cv2.imread(rgb_path(f))
        ok, box = tracker.update(img)
        if ok and f in wanted:
            x, y, w, h = box
            centers[f] = (x + w / 2.0, y + h / 2.0)
    return centers


def detect_barrel_centers(img):
    """Orange traffic-barrel blobs -> list of (cx, cy) pixel centroids."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (2, 90, 80), (22, 255, 255))
    mask[: img.shape[0] // 2, :] = 0  # sky / distant traffic-light housings

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 45))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    n, _, stats, cent = cv2.connectedComponentsWithStats(closed)
    return [tuple(cent[i]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 500]


def build_partB(frames, bboxes):
    cart_px = track_cart_centers(frames)

    ego_frame_data = {}  # frame -> dict(light=(x,y), cart=(x,y) or None, barrels=[(x,y),...])
    for f in frames:
        data = np.load(xyz_path(f))["xyz"]
        img = cv2.imread(rgb_path(f))

        light_xy = None
        row = bboxes.loc[f]
        pos = sample_xyz_patch(data, row.cx, row.cy)
        if pos is not None:
            light_xy = (pos[0], pos[1])

        cart_xy = None
        if f in cart_px:
            pos = sample_xyz_patch(data, *cart_px[f])
            if pos is not None:
                cart_xy = (pos[0], pos[1])

        barrels_xy = []
        for cx, cy in detect_barrel_centers(img):
            pos = sample_xyz_patch(data, cx, cy)
            if pos is not None:
                barrels_xy.append((pos[0], pos[1]))

        ego_frame_data[f] = dict(light=light_xy, cart=cart_xy, barrels=barrels_xy)

    return ego_frame_data


def animate_ego_frame(frames, ego_data):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-5, 50)
    ax.set_ylim(-15, 15)
    ax.invert_xaxis()  # forward "up" the screen reads better with X to the left here
    ax.set_xlabel("X (m, forward)")
    ax.set_ylabel("Y (m, left)")
    ax.set_title("Ego-centric BEV: light, golf cart, barrels")
    ax.grid(True, alpha=0.3)

    ax.plot([0], [0], "^", ms=14, color="black", label="ego car")
    light_pt, = ax.plot([], [], "*", ms=16, color="orange", label="traffic light")
    cart_pt, = ax.plot([], [], "o", ms=10, color="purple", label="golf cart")
    barrel_pts, = ax.plot([], [], "s", ms=6, color="tab:red", label="barrels")
    ax.legend(loc="upper right")
    fig.tight_layout()

    def update(i):
        f = frames[i]
        d = ego_data[f]
        light_pt.set_data(*([[d["light"][0]], [d["light"][1]]] if d["light"] else [[], []]))
        cart_pt.set_data(*([[d["cart"][0]], [d["cart"][1]]] if d["cart"] else [[], []]))
        if d["barrels"]:
            bx, by = zip(*d["barrels"])
        else:
            bx, by = [], []
        barrel_pts.set_data(bx, by)
        return light_pt, cart_pt, barrel_pts

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, blit=False)
    try:
        anim.save(OUT_BEV_MP4, writer=make_ffmpeg_writer())
    except Exception as e:
        print(f"Could not write {OUT_BEV_MP4} ({e}); is ffmpeg installed?")
    plt.close(fig)


def main():
    bboxes = load_light_bboxes(BBOX_CSV)

    frames, x_world, y_world = build_trajectory(bboxes)
    print(f"Usable frames: {len(frames)} (range {frames.min()}..{frames.max()})")
    plot_static(frames, x_world, y_world)
    animate_trajectory(frames, x_world, y_world)
    print(f"Wrote {OUT_TRAJ_PNG} and {OUT_TRAJ_MP4}")

    ego_data = build_partB(frames, bboxes)
    animate_ego_frame(frames, ego_data)
    print(f"Wrote {OUT_BEV_MP4}")


if __name__ == "__main__":
    main()
