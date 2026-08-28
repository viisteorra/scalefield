"""ScaleGen2 watcher — real-image cloud, not the frozen trippy GMM."""

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


def grid(imgs, cols=4):
    cols = max(1, cols)
    rows = (len(imgs) + cols - 1) // cols
    h, w = imgs[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=imgs[0].dtype)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = im
    return canvas

BG = (8, 14, 12)
ACCENT = (120, 220, 160)
MUTED = (140, 170, 155)
FONT_PATHS = (
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
)


def _font(size: int, bold: bool = False):
    for p in (FONT_PATHS if bold else FONT_PATHS[::-1]):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _atomic_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.tmp.png"
    save_png(str(tmp), rgb)
    tmp.replace(path)


def write_board(gen, out_dir: Path, status: dict, tile: int = 128, seed: int = 0, backend: str = "cpu") -> Path:
    samples, Cs = gen.sample(8, tile, tile, seed=seed, backend=backend)
    samp = grid(samples, cols=4)
    bank = gen.field.bank
    morph_row = None
    if bank is not None and len(bank) >= 4:
        frames = gen.morph(bank[0], bank[min(5, len(bank) - 1)], frames=6, H=tile, W=tile, backend=backend)
        morph_row = grid(frames, cols=6)
    stage_row = None
    if hasattr(gen, "stages") and Cs is not None and len(Cs):
        st = gen.stages(Cs[0])
        # show a subset so the board stays small on this box
        pick = st[:: max(1, len(st) // 8)] if len(st) > 8 else st
        if pick[-1][0] != st[-1][0]:
            pick = list(pick) + [st[-1]]
        stage_row = grid(
            [gen.raster(part, min(tile, 64), min(tile, 64), backend=backend) for _, part in pick],
            cols=len(pick),
        )
    h, w = samp.shape[:2]
    if morph_row is not None:
        if morph_row.shape[1] < w:
            z = np.zeros((morph_row.shape[0], w, 3), dtype=samp.dtype)
            z[:, : morph_row.shape[1]] = morph_row
            morph_row = z
        elif morph_row.shape[1] > w:
            samp_pad = np.zeros((h, morph_row.shape[1], 3), dtype=samp.dtype)
            samp_pad[:, :w] = samp
            samp = samp_pad
            w = samp.shape[1]
        body = np.concatenate([samp, np.full((8, w, 3), BG, dtype=np.uint8), morph_row], axis=0)
    else:
        body = samp
    if stage_row is not None:
        if stage_row.shape[1] < body.shape[1]:
            z = np.zeros((stage_row.shape[0], body.shape[1], 3), dtype=body.dtype)
            z[:, : stage_row.shape[1]] = stage_row
            stage_row = z
        elif stage_row.shape[1] > body.shape[1]:
            z = np.zeros((body.shape[0], stage_row.shape[1], 3), dtype=body.dtype)
            z[:, : body.shape[1]] = body
            body = z
        body = np.concatenate([stage_row, np.full((8, body.shape[1], 3), BG, dtype=np.uint8), body], axis=0)
    head = np.full((64, body.shape[1], 3), BG, dtype=np.uint8)
    im = Image.fromarray(head)
    d = ImageDraw.Draw(im)
    hold = status.get("mean_hold_r2")
    perm = status.get("permute_r2")
    holdbit = f"  holdR2={hold:.2f}" if isinstance(hold, (int, float)) else ""
    if isinstance(perm, (int, float)):
        holdbit += f"  perm={perm:.2f}"
    if isinstance(status.get("mean_acc"), (int, float)):
        holdbit += f"  acc={status['mean_acc']:.2f}"
    formula = str(status.get("formula", ""))
    if status.get("kind") == "elem" or formula.startswith("ELEM"):
        kind = "ELEM"
    elif status.get("kind") == "thin" or formula.startswith("THIN"):
        kind = "THIN"
    elif status.get("kind") in ("vq", "embed") or formula.startswith("VQ") or formula.startswith("EMBED"):
        kind = "EMBED"
    elif status.get("kind") == "abstract" or "ABSTRACT" in formula:
        kind = "ABSTRACT"
    else:
        kind = "STRUCT"
    d.text(
        (14, 8),
        f"SCALEGEN2 {kind}  ep {status.get('epoch','')}  {status.get('tag','')}  "
        f"n={status.get('n_data','')}{holdbit}",
        font=_font(18, True),
        fill=ACCENT,
    )
    d.text((14, 36), str(status.get("formula", ""))[:120], font=_font(12), fill=MUTED)
    sheet = np.concatenate([np.asarray(im), np.full((6, body.shape[1], 3), BG, dtype=np.uint8), body], axis=0)
    path = out_dir / "board.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_png(path, sheet)
    (out_dir / "index.html").write_text(
        f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="2">
<title>ScaleGen2</title></head><body style="background:#080e0c;color:#9edcb4;font-family:sans-serif">
<h1>ScaleGen2 ep {status.get('epoch')} {status.get('tag')}</h1>
<img src="board.png?t={int(time.time())}" style="max-width:100%;image-rendering:pixelated">
</body></html>"""
    )
    (out_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    return path


def write_placeholder(out_dir: Path, msg: str) -> Path:
    canvas = np.full((220, 640, 3), BG, dtype=np.uint8)
    im = Image.fromarray(canvas)
    ImageDraw.Draw(im).text((20, 90), "SCALEGEN2", font=_font(26, True), fill=ACCENT)
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
        [imv, "-s", "shrink", "-u", "nearest_neighbour", "-w", "ScaleGen2  ELEM", str(board)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
