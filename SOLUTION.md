# Solution — Ego Trajectory (Part A)

## Method
1. **Traffic light pixel** — for each frame, take the bbox center `(cx, cy)` from `bbox_light.csv`. Rows with an all-zero box (no detection) are dropped.
2. **3D position** — load `xyz/depthNNNNNN.npz` (`xyz` array, shape `(H, W, 4)`: X, Y, Z, flag). The 4th channel is `0` for valid points and `inf`/`nan` for invalid ones. Sample a 7x7 patch around the bbox center, keep only valid points, and take the **median** X/Y/Z as the light's position in camera coordinates for that frame (median is robust to a few invalid/noisy pixels at the box edge).
3. **World frame alignment** — only 198 of the 299 frames have a matching depth file, so the trajectory covers frames 65–298. There is no IMU/heading data, so the camera axes are treated as a fixed-orientation frame that only translates with the car (reasonable since the car doesn't appear to turn sharply). Frame 65 (first usable frame) is used as `t0`: the light's direction in that frame is rotated to align with world `+X`, per the spec ("at t=0 the line joining the car and the light is aligned with +X").
4. **Ego position** — the car's ground position is the negated, rotated light vector: `car = -Rot(-theta0) * light_camera`. **No Y-axis mirror is applied**, contrary to what the README's camera-axis description ("+Y → right") would suggest — see the note below.

## Assumptions
- Car heading is roughly constant (no rotation correction beyond the one-time `t0` alignment) — acceptable per the assignment's tolerance for a non-perfectly-stable trajectory.
- Frames without a depth file (0–64) or without a light detection are simply skipped rather than interpolated.
- Depth patch validity is determined by the array's 4th channel (`0` = valid), discovered empirically since it isn't documented in the challenge README.

## Data note: the actual `+Y` axis is left, not right
The main README states the camera axes as `+X forward, +Y right, +Z up`. The actual `.npz` data contradicts this: sampling a row of pixels at increasing image column `u` (physically left→right in the photo) gives **monotonically decreasing** `Y` values (e.g. `u=100 → Y=+5.55`, `u=1800 → Y=-5.64`). So `+Y` in the data points **left**, not right — the same physical direction as the world frame's `+Y (left)`.
An earlier version of this code trusted the README's text, mirrored `Y` when converting camera → world coordinates, and produced a trajectory mirrored left/right (and correspondingly mirrored golf-cart/barrel positions in Part B). Since camera-`Y` and world-`Y` are actually the same physical direction, the conversion only needs the heading-alignment rotation — no additional flip. This was fixed by trusting the verified data over the written spec, and all outputs (`trajectory.png`, `trajectory.mp4`, `bev_ego_frame.mp4`) were regenerated.

## Results
- `trajectory.png` — static BEV plot, start/end markers, light at the origin.
- `trajectory.mp4` — animated BEV trajectory (10 fps, ~20 s), point drawn per available frame.
- The recovered path is a smooth, gradually curving approach from ~30 m out down to ~7 m from the light, consistent with the sample plots in the main README.

---

# Part B (bonus) — Golf Cart & Barrels in an Ego-Centric BEV

Rendered in the car's own frame (allowed by the assignment), so no world-frame rotation is needed: the ego car sits fixed at the origin facing `+X`, and every other object is placed by its raw camera-frame `(X forward, Y left)` position for that frame, used as-is (see the Y-axis data note in Part A above).

## Method
- **Golf cart (dynamic)** — the cart's appearance (dark body, white canopy roof) is fairly consistent but not easily color-thresholded against the road, fences, and white barriers in the background, so it's tracked instead: seeded once with a hand-picked bounding box on frame 65 (`CART_SEED_BBOX`), then followed frame-to-frame with OpenCV's `TrackerMIL` across the full rgb sequence (so the tracker never has to jump a gap). The tracked box center is used to sample XYZ, same patch/median method as the traffic light.
- **Barrels (static)** — detected per frame via an HSV mask for the barrels' orange (`H 2–22, S>90, V>80`), restricted to the lower half of the image to exclude the sky and distant traffic-light housings (which are a similar hue). A morphological close merges each barrel's orange/white stripes into one blob per barrel cluster; blobs over ~500 px become detections, each sampled for XYZ the same way.
- Both are plotted every frame alongside the traffic light, so the animation shows the cart closing the distance while the (static) barrels drift past the fixed car frame as the vehicle approaches — matching the "ego-frame" sample animation in the main README.

## Assumptions / limitations
- The cart tracker is a classical appearance tracker with a fixed-size box; it doesn't grow with the cart as it gets closer, so its center can drift onto a sub-part of the cart (e.g. the seat) at close range — still a reasonable point estimate for "center of the visible region."
- Barrel detection returns one point per contiguous orange cluster, not per physical barrel (the barrels sit in tight rows, so a "cluster" is usually several barrels or a whole barricade section).
- No pedestrian/other-traffic-light tracking was implemented — not clearly visible/separable with simple thresholding in the available frames.

## Result
- `bev_ego_frame.mp4` — animated ego-centric BEV (10 fps): black triangle = ego car (fixed at origin), orange star = traffic light, purple circle = golf cart, red squares = barrel clusters.
