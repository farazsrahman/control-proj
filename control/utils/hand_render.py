"""Offline renderer for "the hand as it is" vs "the hand as the policy perceives it" (numpy + PIL only, no GPU / GL needed).

The data comes from scripts/diagnostics/dump_hand_viz.py (api/hand_viz.py): link-frame meshes, and per-frame link poses at the true joint
positions and at the measured ones. Typical use (notebook code cell):

    from utils import hand_render as hr
    M = hr.load_meshes(".cache/tmp/hand_viz/meshes.npz");  R = hr.load_rollout(".cache/tmp/hand_viz/rollout_N288.npz")
    img = hr.render_comparison(M, R, t=100, env=0, title="N = 288")      # PIL image: true | perceived | overlay

Conventions: poses are env-local, quaternions (w, x, y, z), meshes are in each link's frame. The rasterizer is a plain z-buffer with
two-sided Lambert shading, supersampled; about a second per hand.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

N_REF, H0_DEG = 15.0, 0.1  # MeasurementModel's backlash half-width: h = (N / N_REF) * H0_DEG degrees (checked against the recorded rollouts)


# ---------------------------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------------------------
def load_meshes(path):
    """{'hand': {body: (verts, faces)}, 'object': (verts, faces), 'joint_names', 'body_names', 'finger_bodies'}"""
    z = np.load(path, allow_pickle=False)
    bodies = {k[:-2]: (z[k], z[k[:-2] + "/f"]) for k in z.files if k.endswith("/v")}
    obj = bodies.pop("object")
    return dict(hand=bodies, object=obj, joint_names=list(z["joint_names"]), body_names=list(z["body_names"]), finger_bodies=list(z["finger_bodies"]))


def load_rollout(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def play_operator(q, h, q_init, fresh):
    """The backlash model of api.blind_repose.MeasurementModel (chaos off): the measurement stays put until the true joint moves h away.
    q [T, E, nd] true joint positions, h scalar or [nd] half-width (rad), q_init [E, nd] the first observation's state, fresh [T, E] bool:
    the env was reset at this frame (play re-centres on q). Returns the measured q, [T, E, nd]."""
    play, out = q_init.copy(), np.empty_like(q)
    for t in range(len(q)):
        play = np.where(fresh[t][:, None], q[t], play)
        play = np.maximum(np.minimum(play, q[t] + h), q[t] - h)
        out[t] = play
    return out


def half_width(n):
    return (n / N_REF) * math.radians(H0_DEG)


# ---------------------------------------------------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------------------------------------------------
def quat_to_rot(q):
    """[..., 4] (w, x, y, z) -> [..., 3, 3]."""
    w, x, y, z = (q[..., i] for i in range(4))
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def world_triangles(meshes, body_pos, body_quat):
    """All hand triangles in the env frame, [F, 3, 3], for link poses [B, 3], [B, 4] (ordered like meshes['body_names'])."""
    tris = []
    for i, name in enumerate(meshes["body_names"]):
        if name not in meshes["hand"]:
            continue
        v, f = meshes["hand"][name]
        R = quat_to_rot(body_quat[i].astype(np.float64))
        tris.append((v.astype(np.float64) @ R.T + body_pos[i])[f])
    return np.concatenate(tris)


def object_triangles(meshes, pos, quat):
    v, f = meshes["object"]
    R = quat_to_rot(quat.astype(np.float64))
    return (v.astype(np.float64) @ R.T + pos)[f]


def rot_angle(q1, q2):
    """Angle (rad) of the rotation between unit quaternions (w, x, y, z), [..., 4] each (the env's rotation_distance, robust near pi)."""
    w = np.abs((q1 * q2).sum(-1))
    return 2 * np.arccos(np.clip(w, 0, 1))


def goal_edges(meshes, pos, goal_quat):
    """The 12 edges [12, 2, 3] of the cube drawn with the GOAL orientation, centred where the cube is now (the env's goal is an orientation only)."""
    v, _ = meshes["object"]
    lo, hi = v.min(0).astype(np.float64), v.max(0).astype(np.float64)
    c, hh = (lo + hi) / 2, (hi - lo) / 2
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * hh + c
    w = corners @ quat_to_rot(goal_quat.astype(np.float64)).T + pos
    idx = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
    return np.stack([[w[i], w[j]] for i, j in idx])


def box_signed_distance(points, meshes, pos, quat):
    """Signed distance (m, negative inside) of world `points` [P, 3] to the cube, taken as the axis-aligned box of its mesh in its own frame."""
    v, _ = meshes["object"]
    lo, hi = v.min(0).astype(np.float64), v.max(0).astype(np.float64)
    c, hh = (lo + hi) / 2, (hi - lo) / 2
    p = (points - pos) @ quat_to_rot(quat.astype(np.float64)) - c  # world -> cube frame
    d = np.abs(p) - hh
    return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0)


# ---------------------------------------------------------------------------------------------------------------------------------
# rasterizer
# ---------------------------------------------------------------------------------------------------------------------------------
class Camera:
    def __init__(self, eye, target, up=(0, 0, 1), fov_deg=30.0):
        eye, target, up = (np.asarray(a, dtype=np.float64) for a in (eye, target, up))
        f = target - eye
        f /= np.linalg.norm(f)
        r = np.cross(f, up)
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.eye, self.R, self.fov = eye, np.stack([r, u, f]), math.radians(fov_deg)  # rows: right, up, forward

    def project(self, pts, W, H):
        c = (np.asarray(pts, dtype=np.float64) - self.eye) @ self.R.T
        fpx = 0.5 * H / math.tan(self.fov / 2)
        z = c[..., 2]
        return np.stack([W / 2 + fpx * c[..., 0] / z, H / 2 - fpx * c[..., 1] / z], -1), z


def rasterize(tris, cam, W, H, color, light=(0.35, 0.5, 1.0)):
    """z-buffer render of triangles [F, 3, 3] with a flat `color` (3,): returns (rgb float [H, W, 3] in 0..1, depth [H, W], inf = empty)."""
    rgb, depth = np.zeros((H, W, 3), np.float32), np.full((H, W), np.inf, np.float32)
    xy, z = cam.project(tris, W, H)  # [F, 3, 2], [F, 3]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    ldir = cam.R.T @ np.asarray(light, dtype=np.float64)  # light fixed relative to the camera
    ldir /= np.linalg.norm(ldir)
    shade = 0.3 + 0.7 * np.abs(n @ ldir)
    ok = (z > 1e-3).all(1)
    ys, xs = np.mgrid[0:H, 0:W]
    for i in np.flatnonzero(ok):
        (x0, y0), (x1, y1), (x2, y2) = xy[i]
        xa, xb = max(int(math.floor(min(x0, x1, x2))), 0), min(int(math.ceil(max(x0, x1, x2))) + 1, W)
        ya, yb = max(int(math.floor(min(y0, y1, y2))), 0), min(int(math.ceil(max(y0, y1, y2))) + 1, H)
        if xa >= xb or ya >= yb:
            continue
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-9:
            continue
        px, py = xs[ya:yb, xa:xb] + 0.5, ys[ya:yb, xa:xb] + 0.5
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        d = 1.0 / (w0 / z[i, 0] + w1 / z[i, 1] + w2 / z[i, 2])  # perspective-correct depth
        sub = depth[ya:yb, xa:xb]
        m = inside & (d < sub)
        sub[m] = d[m]
        rgb[ya:yb, xa:xb][m] = np.asarray(color, np.float32) * shade[i]
    return rgb, depth


def _outline(mask, width=1):
    m = mask.copy()
    for _ in range(width):
        er = m.copy()
        er[1:] &= m[:-1]; er[:-1] &= m[1:]; er[:, 1:] &= m[:, :-1]; er[:, :-1] &= m[:, 1:]  # noqa: E702
        m = er
    return mask & ~m


# palette (RGB 0..1)
GOAL = (80, 235, 120)  # 0..255, for PIL drawing
TRUE_HAND, PERCEIVED, CUBE, BG = (0.62, 0.68, 0.78), (1.0, 0.45, 0.1), (0.95, 0.8, 0.2), (0.07, 0.08, 0.1)


def render_layers(layers, cam, W=480, H=480, ss=2, bg=BG, outline_layers=()):
    """layers: list of dict(tris, color, alpha=1.0, outline=None (colour), tol=0.0). Drawn in order over a shared z-buffer of the opaque layers;
    a translucent layer (alpha < 1) is depth-tested against everything drawn so far with tolerance `tol` (m). Outlines are drawn on top, always
    visible. Returns a PIL image."""
    w, h = W * ss, H * ss
    img = np.empty((h, w, 3), np.float32)
    img[:] = bg
    zb = np.full((h, w), np.inf, np.float32)
    outlines = []
    for L in layers:
        rgb, d = rasterize(L["tris"], cam, w, h, L["color"])
        a = L.get("alpha", 1.0)
        vis = np.isfinite(d) & (d < zb + (L.get("tol", 0.0) if a < 1 else 0.0)) if a < 1 else np.isfinite(d) & (d < zb)
        img[vis] = (1 - a) * img[vis] + a * rgb[vis]
        if a >= 1:
            zb = np.where(vis, d, zb)
        if L.get("outline") is not None:
            outlines.append((_outline(np.isfinite(d), ss), L["outline"]))
    for m, c in outlines:
        img[m] = c
    out = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    return out.resize((W, H), Image.LANCZOS) if ss > 1 else out


def default_camera(meshes, R, env=0, t=0, azim=-35.0, elev=45.0, dist=0.40, fov=30.0):
    """Camera orbiting the middle of the palm, fingertips and cube (azimuth / elevation in degrees about +z)."""
    bi = {b: i for i, b in enumerate(meshes["body_names"])}
    pts = np.concatenate([R["fk_true_pos"][t, env][meshes["finger_bodies"]], R["fk_true_pos"][t, env][[bi["robot0_palm"]]], R["obj_pos"][t, env][None]])
    target = pts.mean(0)
    a, e = math.radians(azim), math.radians(elev)
    eye = target + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return Camera(eye, target, fov_deg=fov)


def _overlay_layers(meshes, R, t, env, key):
    pt = world_triangles(meshes, R["fk_true_pos"][t, env], R["fk_true_quat"][t, env])
    pm = world_triangles(meshes, R[f"fk_{key}_pos"][t, env], R[f"fk_{key}_quat"][t, env])
    layers = [dict(tris=pt, color=TRUE_HAND)]
    if "obj_pos" in R:  # scenes without a cube (e.g. the single-joint demo) just omit it
        layers.append(dict(tris=object_triangles(meshes, R["obj_pos"][t, env], R["obj_quat"][t, env]), color=CUBE))
    return layers + [dict(tris=pm, color=PERCEIVED, alpha=0.6, tol=0.004, outline=PERCEIVED)]


def _tip_marks(d, meshes, R, t, env, key, cam, size, x0, y0):
    fb = meshes["finger_bodies"]
    tt, tm = cam.project(R["fk_true_pos"][t, env][fb], size, size)[0], cam.project(R[f"fk_{key}_pos"][t, env][fb], size, size)[0]
    for a, b in zip(tt, tm):
        if np.hypot(*(a - b)) < 0.5:  # identical on screen: no marker (e.g. the fingers that are not driven)
            continue
        d.line([(x0 + a[0], y0 + a[1]), (x0 + b[0], y0 + b[1])], fill=(255, 255, 255), width=1)
        d.ellipse([x0 + a[0] - 3, y0 + a[1] - 3, x0 + a[0] + 3, y0 + a[1] + 3], outline=(255, 255, 255))
        d.ellipse([x0 + b[0] - 3, y0 + b[1] - 3, x0 + b[0] + 3, y0 + b[1] + 3], fill=(255, 115, 25))


def render_sweep(meshes, R, t, ns, env=0, size=420, cam=None):
    """One frame of the CLEAN trajectory R (rollout_N0.npz with post-hoc 'ph<N>' poses): the true hand + cube, then the overlay of what
    a policy with backlash half-width h(N) would perceive, one panel per gear ratio in `ns`."""
    cam = cam or default_camera(meshes, R, env, t)
    first = _overlay_layers(meshes, R, t, env, f"ph{ns[0]:g}")[:-1]
    panels = [(render_layers(first, cam, size, size), "true hand + cube", None)]
    for n in ns:
        panels.append((render_layers(_overlay_layers(meshes, R, t, env, f"ph{n:g}"), cam, size, size),
                       f"N = {n:g}: h = {math.degrees(half_width(n)):.2f} deg", f"ph{n:g}"))
    canvas = Image.new("RGB", (len(panels) * size, size + 30), tuple(int(255 * c) for c in BG))
    d = ImageDraw.Draw(canvas)
    for i, (im, label, key) in enumerate(panels):
        canvas.paste(im, (i * size, 30))
        d.text((i * size + 8, 9), label, fill=(235, 235, 235))
        if key:
            _tip_marks(d, meshes, R, t, env, key, cam, size, i * size, 30)
    return canvas


def render_comparison(meshes, R, t, env=0, key="meas", title="", cam=None, size=420, ss=2, counter=None):
    """Three panels at frame t: [the hand as it is, with the cube] | [the hand as the policy perceives it (no cube: it does not see it)] | [overlay].
    `key` selects the perceived pose set: 'meas' (recorded q_meas of this rollout) or e.g. 'ph288' (post-hoc play operator on the clean trajectory)."""
    cam = cam or default_camera(meshes, R, env, t)
    L = _overlay_layers(meshes, R, t, env, key)
    A = render_layers(L[:-1], cam, size, size, ss)
    B = render_layers([dict(tris=L[-1]["tris"], color=PERCEIVED)], cam, size, size, ss)
    C = render_layers(L, cam, size, size, ss)
    canvas = Image.new("RGB", (3 * size, size + 30), tuple(int(255 * c) for c in BG))
    d = ImageDraw.Draw(canvas)
    x_label = 8 if counter is None else 8 + 128  # the goal counter takes the top-left corner of the first panel's header
    for i, (im, s_) in enumerate(zip([A, B, C], ["true hand + cube" if "obj_pos" in R else "true hand", "perceived hand (policy's joint readings)", "overlay: orange = perceived"])):
        canvas.paste(im, (i * size, 30))
        d.text((i * size + (x_label if i == 0 else 8), 9), s_, fill=(235, 235, 235))
    if counter is not None:
        try:
            font = ImageFont.load_default(size=18)
        except TypeError:  # Pillow < 10.1: fixed-size bitmap font
            font = ImageFont.load_default()
        d.rectangle([4, 3, 124, 27], fill=(24, 60, 36), outline=GOAL)
        d.text((10, 5), f"GOALS: {int(counter)}", fill=GOAL, font=font)
    d.text((8, size + 30 - 16), title, fill=(255, 190, 120))  # bottom-left: the hand never reaches this corner
    _tip_marks(d, meshes, R, t, env, key, cam, size, 2 * size, 30)
    if "goal_quat" in R:  # goal orientation: green wireframe cube at the cube's position, over the panels that show the cube
        e = goal_edges(meshes, R["obj_pos"][t, env], R["goal_quat"][t, env])
        for x0 in (0, 2 * size):
            for a, b in e:
                pa, pb = cam.project(a, size, size)[0], cam.project(b, size, size)[0]
                d.line([(x0 + pa[0], 30 + pa[1]), (x0 + pb[0], 30 + pb[1])], fill=GOAL, width=2)
        d.text((3 * size - 8 - 6 * 28, size + 30 - 16), "green box = goal orientation", fill=GOAL)
    return canvas


# ---------------------------------------------------------------------------------------------------------------------------------
# single-joint demo
# ---------------------------------------------------------------------------------------------------------------------------------
def finger_view(meshes, R, env=0, bodies=("robot0_ffproximal", "robot0_ffdistal"), flip=False, tilt_deg=12.0):
    """(look-at centre, unit direction from the centre to the camera) for a camera looking square onto the flexion plane of the finger that moves in
    R: the normal of the plane spanned by the finger at rest and at its largest excursion, tilted slightly upwards. `bodies` = (proximal link, tip link)."""
    bi = {b: i for i, b in enumerate(meshes["body_names"])}
    P = R["fk_true_pos"][:, env]
    k = int(np.abs(R["q_true"][:, env] - R["q_true"][0, env]).sum(-1).argmax())
    pb, t0, t1 = P[0, bi[bodies[0]]], P[0, bi[bodies[1]]], P[k, bi[bodies[1]]]
    n = np.cross(t0 - pb, t1 - pb)
    n /= np.linalg.norm(n)
    n = -n if flip else n
    a = math.radians(tilt_deg)
    n = n * math.cos(a) + np.array([0, 0, 1.0]) * math.sin(a)
    return (pb + t0 + t1) / 3, n / np.linalg.norm(n)


def render_joint_frame(meshes, R, t, env=0, view=None, dist=0.30, zoom_dist=0.06, tip_body="robot0_ffdistal", title="", size=320, ss=2, **_):
    """Four panels for the single-joint demo at frame t: true hand | perceived hand | overlay (orange = perceived) | the overlay zoomed on the true
    fingertip (the camera follows the fingertip; the same view direction). `view` = finger_view(...)."""
    centre, n = view
    cam = Camera(centre + dist * n, centre, fov_deg=30.0)
    bi = {b: i for i, b in enumerate(meshes["body_names"])}
    tip = R["fk_true_pos"][t, env, bi[tip_body]]
    zcam = Camera(tip + zoom_dist * n, tip, fov_deg=30.0)
    L = _overlay_layers(meshes, R, t, env, "meas")
    ims = [render_layers(L[:-1], cam, size, size, ss), render_layers([dict(tris=L[-1]["tris"], color=PERCEIVED)], cam, size, size, ss),
           render_layers(L, cam, size, size, ss), render_layers(L, zcam, size, size, ss)]
    canvas = Image.new("RGB", (4 * size, size + 30), tuple(int(255 * c) for c in BG))
    d = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(zip(ims, ["true hand", "perceived hand (joint reading)", "overlay: orange = perceived", "overlay, zoomed on the true fingertip"])):
        canvas.paste(im, (i * size, 30))
        d.text((i * size + 8, 9), label, fill=(235, 235, 235))
    d.text((8, size + 30 - 16), title, fill=(255, 190, 120))
    for x0, c, sz in [(2 * size, cam, size), (3 * size, zcam, size)]:
        a = c.project(R["fk_true_pos"][t, env, bi[tip_body]], sz, sz)[0]
        b = c.project(R["fk_meas_pos"][t, env, bi[tip_body]], sz, sz)[0]
        d.line([(x0 + a[0], 30 + a[1]), (x0 + b[0], 30 + b[1])], fill=(255, 255, 255), width=1)
        d.ellipse([x0 + a[0] - 3, 30 + a[1] - 3, x0 + a[0] + 3, 30 + a[1] + 3], outline=(255, 255, 255))
        d.ellipse([x0 + b[0] - 3, 30 + b[1] - 3, x0 + b[0] + 3, 30 + b[1] + 3], fill=(255, 115, 25))
    return canvas


# ---------------------------------------------------------------------------------------------------------------------------------
# video frames (parallel over frames)
# ---------------------------------------------------------------------------------------------------------------------------------
_W = {}


def _init_worker(meshes, R, env, key, cam, size, ss, joint_kw):
    _W.update(meshes=meshes, R=R, env=env, key=key, cam=cam, size=size, ss=ss, joint_kw=joint_kw)


def _render_one(args):
    t, title, counter = args
    w = _W
    if w["joint_kw"] is not None:
        return np.asarray(render_joint_frame(w["meshes"], w["R"], t, w["env"], title=title, size=w["size"], ss=w["ss"], **w["joint_kw"]))
    return np.asarray(render_comparison(w["meshes"], w["R"], t, w["env"], w["key"], title, w["cam"], w["size"], w["ss"], counter))


def render_frames(meshes, R, ts, env=0, key="meas", title_fn=None, cam=None, size=320, ss=2, workers=8, counter_fn=None, joint_kw=None):
    """RGB uint8 frames [len(ts), H, W, 3] of render_comparison at frames `ts` of rollout R, rendered in `workers` processes with a fixed camera
    (default: the one at ts[0]). `title_fn(t) -> str` is the caption at the bottom of each frame, `counter_fn(t) -> int` the "GOALS: k" box in the top-left corner.
    With `joint_kw` (dict: view=..., dist, zoom_dist, tip_body) the frames are the four-panel single-joint demo (render_joint_frame) instead."""
    import multiprocessing as mp

    cam = cam or (None if joint_kw is not None else default_camera(meshes, R, env, ts[0]))
    jobs = [(t, title_fn(t) if title_fn else "", counter_fn(t) if counter_fn else None) for t in ts]
    with mp.get_context("spawn").Pool(workers, _init_worker, (meshes, R, env, key, cam, size, ss, joint_kw)) as pool:
        return np.stack(pool.map(_render_one, jobs, chunksize=2))
