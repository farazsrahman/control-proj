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
