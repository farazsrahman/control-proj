"""Videos that live only inside the notebook (no files left in the workspace)."""

from __future__ import annotations

import tempfile

import imageio
from IPython.display import Video


def show(frames, fps: float, width: int = 640) -> Video:
    """Encode RGB frames to mp4 via a temp file and return an embedded IPython Video."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        imageio.mimwrite(f.name, frames, fps=fps)
        data = open(f.name, "rb").read()
    return Video(data=data, embed=True, mimetype="video/mp4", width=width)


def draw_tracks(frames, uv, visible, colors, trail: int = 10, radius: int = 3) -> list:
    """Overlay point tracks on RGB frames: a dot per visible point plus a fading trail of its last `trail` steps.

    Args:
        frames: [T, H, W, 3] uint8.
        uv: [T, N, 2] pixel coords (u=col, v=row).
        visible: [T, N] bool; occluded/off-screen points get no dot and break their trail.
        colors: [N, 3] RGB in [0, 255].
    """
    import cv2
    import numpy as np

    out = []
    for t, frame in enumerate(frames):
        img = np.ascontiguousarray(frame).copy()
        for s in range(max(1, t - trail + 1), t + 1):
            overlay = img.copy()
            seg_vis = visible[s - 1] & visible[s]
            for n in np.nonzero(seg_vis)[0]:
                p0, p1 = uv[s - 1, n].round().astype(int), uv[s, n].round().astype(int)
                cv2.line(overlay, tuple(p0), tuple(p1), tuple(int(c) for c in colors[n]), 1, cv2.LINE_AA)
            alpha = (s - t + trail) / trail  # older segments fade out
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        for n in np.nonzero(visible[t])[0]:
            p = tuple(uv[t, n].round().astype(int))
            cv2.circle(img, p, radius + 1, (0, 0, 0), -1, cv2.LINE_AA)  # thin dark rim, drawn under the fill
            cv2.circle(img, p, radius, tuple(int(c) for c in colors[n]), -1, cv2.LINE_AA)
        out.append(img)
    return out


def track_colors(uv0, height: int):
    """CoTracker-style rainbow colors [N, 3] uint8 for points by their initial image row (`uv0`: [N, 2])."""
    import matplotlib.pyplot as plt

    return (plt.cm.gist_rainbow(uv0[:, 1] / height)[:, :3] * 255).astype("uint8")
