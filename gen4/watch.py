"""ScaleGen4 watcher — real clip vs predicted next frames."""

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


def grid(imgs, cols=8, pad=2):
    cols = max(1, min(cols, len(imgs)))
    rows = (len(imgs) + cols - 1) // cols
    h, w = imgs[0].shape[:2]
    H = rows * h + max(0, rows - 1) * pad
    W = cols * w + max(0, cols - 1) * pad
    canvas = np.zeros((H, W, 3), dtype=imgs[0].dtype)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        y = r * (h + pad)
        x = c * (w + pad)
        canvas[y : y + h, x : x + w] = im
    return canvas


BG = (6, 10, 18)
ACCENT = (80, 190, 255)
MUTED = (130, 160, 190)
FONT_PATHS = (
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
)


def _font(size: int, bold: bool = False):
    for p in FONT_PATHS if bold else FONT_PATHS[::-1]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _atomic_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.tmp.png"
    save_png(str(tmp), rgb)
    tmp.replace(path)


def _label_row(imgs: list[np.ndarray], title: str) -> np.ndarray:
    body = grid(imgs, cols=len(imgs), pad=2)
    head = np.full((28, body.shape[1], 3), BG, dtype=np.uint8)
    im = Image.fromarray(head)
    ImageDraw.Draw(im).text((8, 6), title, font=_font(13, True), fill=ACCENT)
    return np.concatenate([np.asarray(im), body], axis=0)


def _pad_w(a: np.ndarray, width: int) -> np.ndarray:
    if a.shape[1] == width:
        return a
    z = np.full((a.shape[0], width, 3), BG, dtype=np.uint8)
    z[:, : a.shape[1]] = a
    return z


def write_board(gen, out_dir: Path, status: dict, tile: int = 72, seed: int = 0, backend: str = "cpu") -> Path:
    nclips = len(gen.clip_slices)
    ci = int(seed) % max(nclips, 1)
    bank = gen.clip_bank(ci)
    name = gen.clip_slices[ci][0]
    n = min(8, len(bank))
    real = [gen.raster(bank[i], tile, tile, backend=backend) for i in range(n)]
    pred_c = gen.nf.rollout(bank[0], n_frames=n, temp=0.04, seed=seed)
    pred = [gen.raster(c, tile, tile, backend=backend) for c in pred_c]
    # teacher-forced one-step vs real next
    tf = []
    for i in range(min(4, len(bank) - 1)):
        hat = gen.nf.one_step(bank[i], bank[i + 1])
        tf.append(gen.raster(bank[i], tile, tile, backend=backend))
        tf.append(gen.raster(hat, tile, tile, backend=backend))
        tf.append(gen.raster(bank[i + 1], tile, tile, backend=backend))
    rows = [
        _label_row(real, f"REAL  clip={name}  t=0..{n-1}"),
        _label_row(pred, f"ROLLOUT  start from t=0, guess the rest  (identity + residual)"),
    ]
    if tf:
        rows.append(_label_row(tf, "ONE-STEP  repeating: real_t | pred_t+1 | real_t+1"))
    width = max(r.shape[1] for r in rows)
    body = np.concatenate([_pad_w(r, width) for r in rows], axis=0)
    head = np.full((58, width, 3), BG, dtype=np.uint8)
    im = Image.fromarray(head)
    d = ImageDraw.Draw(im)
    d.text(
        (12, 6),
        f"SCALEGEN4 NEXT-FRAME  ep {status.get('epoch','')}  {status.get('tag','')}  "
        f"pairs={status.get('n_pairs','')}",
        font=_font(18, True),
        fill=ACCENT,
    )
    d.text((12, 32), str(status.get("formula", ""))[:140], font=_font(11), fill=MUTED)
    sheet = np.concatenate([np.asarray(im), np.full((4, width, 3), BG, dtype=np.uint8), body], axis=0)
    path = out_dir / "board.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_png(path, sheet)
    (out_dir / "index.html").write_text(
        f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="3">
<title>ScaleGen4 next-frame</title></head><body style="background:#060a12;color:#50beff;font-family:sans-serif">
<h1>ScaleGen4 next-frame ep {status.get('epoch')} {status.get('tag')}</h1>
<img src="board.png?t={int(time.time())}" style="max-width:100%;image-rendering:pixelated">
</body></html>"""
    )
    (out_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    return path


def write_placeholder(out_dir: Path, msg: str) -> Path:
    canvas = np.full((220, 720, 3), BG, dtype=np.uint8)
    im = Image.fromarray(canvas)
    ImageDraw.Draw(im).text((20, 80), "SCALEGEN4  NEXT-FRAME", font=_font(26, True), fill=ACCENT)
    ImageDraw.Draw(im).text((20, 130), msg, font=_font(14), fill=MUTED)
    path = out_dir / "board.png"
    _atomic_png(path, np.asarray(im))
    return path


def open_viewer(board: Path):
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return None
    imv = shutil.which("imv")
    if not imv:
        return None
    return subprocess.Popen(
        [imv, "-s", "shrink", "-u", "nearest_neighbour", "-w", "ScaleGen4  NEXT-FRAME", str(board)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
