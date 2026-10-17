"""Matplotlib font helpers for the camera-ready figures.

    from utils.fonts import use_roboto

    use_roboto()                                    # once, before plotting
    plt.rcParams["font.family"] = "Roboto"
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

from matplotlib import font_manager as fm

from utils.path import CACHE_DIR

ROBOTO_URL = "https://github.com/googlefonts/roboto/releases/download/v2.138/roboto-unhinted.zip"
ROBOTO_FILES = ("Roboto-Regular.ttf", "Roboto-Medium.ttf", "Roboto-Bold.ttf")


def use_roboto() -> None:
    """Register Roboto with matplotlib: a system install if there is one, else a one-time download of the static TTFs into .cache/fonts/roboto."""
    if not any(f.name == "Roboto" for f in fm.fontManager.ttflist):
        d = CACHE_DIR / "fonts" / "roboto"
        if not d.exists():
            with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(ROBOTO_URL, timeout=60).read())) as z:
                d.mkdir(parents=True)
                for name in z.namelist():
                    if Path(name).name in ROBOTO_FILES:
                        (d / Path(name).name).write_bytes(z.read(name))
        for ttf in d.glob("*.ttf"):
            fm.fontManager.addfont(str(ttf))
    assert any(f.name == "Roboto" for f in fm.fontManager.ttflist), "Roboto is not available"
