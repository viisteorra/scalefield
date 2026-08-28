"""ScaleGen watcher — generated samples + morph, not the v1 codec board."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scalefield import save_png
from scalegen import Generator, grid

BG = (10, 8, 16)
ACCENT = (220, 140, 255)
MUTED = (160, 150, 175)
FONT_PATHS = (
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
)


def _font(size: int, bold: bool = False):
    paths = FONT_PATHS if bold else FONT_PATHS[::-1]
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _atomic_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.tmp.png"
    save_png(str(tmp), rgb)
    tmp.replace(path)


def _bar(width: int, lines: list[str]) -> np.ndarray:
    canvas = np.full((72, width, 3), BG, dtype=np.uint8)
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    d.text((16, 8), lines[0], font=_font(20, True), fill=ACCENT)
    if len(lines) > 1:
        d.text((16, 40), lines[1][:140], font=_font(13), fill=MUTED)
    return np.asarray(im)


def _label(tile: np.ndarray, text: str) -> np.ndarray:
    h, w = tile.shape[:2]
    bar = 28
    canvas = np.full((h + bar, w, 3), BG, dtype=np.uint8)
    canvas[bar:] = tile
    im = Image.fromarray(canvas)
    ImageDraw.Draw(im).text((8, 4), text, font=_font(14, True), fill=ACCENT)
    return np.asarray(im)


def _stack(*rows: np.ndarray, gap: int = 12) -> np.ndarray:
    w = max(r.shape[1] for r in rows)
    parts = []
    for i, r in enumerate(rows):
        if r.shape[1] < w:
            z = np.full((r.shape[0], w, 3), BG, dtype=np.uint8)
            z[:, : r.shape[1]] = r
            r = z
        if i:
            parts.append(np.full((gap, w, 3), BG, dtype=np.uint8))
        parts.append(r)
    return np.concatenate(parts, axis=0)


def write_board(
    gen: Generator,
    out_dir: Path,
    status: dict,
    tile: int = 160,
    n_samples: int = 8,
    seed: int = 0,
    backend: str = "cpu",
) -> Path:
    """SAMPLE grid + MORPH strip. Field is frozen; this is the transform."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = gen.field.bank
    samples, _ = gen.sample(n_samples, tile, tile, temp=0.15, seed=seed, backend=backend, cloud=True)
    samp = _label(grid(samples, cols=4), "SAMPLES  (mix c in z-space)")
    morph = None
    if bank is not None and len(bank) >= 2:
        i0, i1 = 0, min(4, len(bank) - 1)
        frames = gen.morph(bank[i0], bank[i1], frames=6, H=tile, W=tile, backend=backend)
        morph = _label(grid(frames, cols=6), "MORPH  lerp z")
    body = samp if morph is None else _stack(samp, morph, gap=10)
    head = _bar(
        body.shape[1],
        [
            f"SCALEGEN   ep {status.get('epoch','')}   GMM={status.get('n_comp','?')}   "
            f"nll={status.get('nll', 0):.3f}   k={status.get('k')}   {status.get('tag', '')}",
            status.get("formula", "")[:160],
        ],
    )
    gap = np.full((8, body.shape[1], 3), BG, dtype=np.uint8)
    sheet = np.concatenate([head, gap, body], axis=0)
    path = out_dir / "board.png"
    _atomic_png(path, sheet)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="2">
<title>ScaleGen</title>
<style>
body {{ margin:0; background:#0a0810; color:#e8d8ff; font-family:sans-serif; }}
header {{ padding:16px 20px; border-bottom:1px solid #3a2a4a; }}
h1 {{ margin:0; font-size:18px; color:#dc8cff; }}
p {{ margin:6px 0 0; color:#9a8aaa; font-size:13px; }}
img {{ max-width:100%; image-rendering:pixelated; }}
</style></head>
<body>
<header>
<h1>ScaleGen  ep {status.get('epoch','')}  GMM={status.get('n_comp','?')}  nll={status.get('nll',0):.3f}  {status.get('tag','')}</h1>
<p>{status.get('formula','')}</p>
</header>
<p style="padding:12px 20px"><img src="board.png?t={int(time.time())}" alt="board"></p>
</body></html>
"""
    (out_dir / "index.html").write_text(html)
    (out_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    return path


def write_placeholder(out_dir: Path, msg: str = "loading frozen field…") -> Path:
    canvas = np.full((240, 720, 3), BG, dtype=np.uint8)
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    d.text((24, 90), "SCALEGEN", font=_font(28, True), fill=ACCENT)
    d.text((24, 140), msg, font=_font(16), fill=MUTED)
    path = out_dir / "board.png"
    _atomic_png(path, np.asarray(im))
    return path


def open_viewer(board: Path) -> subprocess.Popen | None:
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return None
    imv = shutil.which("imv")
    if not imv:
        return None
    return subprocess.Popen(
        [imv, "-s", "shrink", "-u", "nearest_neighbour", "-w", "ScaleGen  SAMPLES | MORPH", str(board)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
