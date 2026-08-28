"""ScaleGen3 watcher — unfold filmstrip + next-frame morph, not a noise cloud."""

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


def grid(imgs, cols=4, pad=0):
    cols = max(1, cols)
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


BG = (16, 10, 6)
ACCENT = (255, 176, 72)
MUTED = (180, 150, 120)
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


def _label_row(imgs: list[np.ndarray], labels: list[str], title: str) -> np.ndarray:
    body = grid(imgs, cols=len(imgs), pad=2)
    head = np.full((36, body.shape[1], 3), BG, dtype=np.uint8)
    im = Image.fromarray(head)
    d = ImageDraw.Draw(im)
    d.text((8, 8), title, font=_font(13, True), fill=ACCENT)
    tile_w = imgs[0].shape[1] + 2
    for i, lab in enumerate(labels):
        d.text((8 + i * tile_w, 22), lab, font=_font(10), fill=MUTED)
    return np.concatenate([np.asarray(im), body], axis=0)


def _pad_w(a: np.ndarray, width: int) -> np.ndarray:
    if a.shape[1] == width:
        return a
    z = np.full((a.shape[0], width, 3), BG, dtype=np.uint8)
    z[:, : a.shape[1]] = a
    return z


def write_board(gen, out_dir: Path, status: dict, tile: int = 72, seed: int = 0, backend: str = "cpu") -> Path:
    bank = gen.field.bank
    rng_seed = int(seed)
    samples, Cs = gen.sample(4, tile, tile, temp=0.12, seed=rng_seed, backend=backend)

    # generation as video: one sample unfolding 1→31
    c_gen = Cs[0]
    st_gen = gen.unfold.stages(c_gen)
    gen_frames = [gen.raster(part, tile, tile, backend=backend) for _, part in st_gen]
    gen_labs = [str(p) for p, _ in st_gen]

    # same unfold from a populated real prefix — should look like the photo
    real_frames = None
    morph_frames = None
    walk_frames = None
    if bank is not None and len(bank) >= 2:
        i0 = int(rng_seed) % len(bank)
        i1 = (i0 + 7) % len(bank)
        c_real = bank[i0]
        c_hat = gen.from_real_prefix(c_real, temp=0.0, seed=rng_seed)
        st_real = gen.unfold.stages(c_hat)
        real_frames = [gen.raster(part, tile, tile, backend=backend) for _, part in st_real]
        # next-frame: lerp two prefixes, unfold
        morph_cs = gen.next_frames(bank[i0], bank[i1], frames=8, temp=0.0, seed=rng_seed)
        morph_frames = [gen.raster(c, tile, tile, backend=backend) for c in morph_cs]
        walk_cs = gen.walk_frames(bank[i0], frames=8, step=0.10, seed=rng_seed)
        walk_frames = [gen.raster(c, tile, tile, backend=backend) for c in walk_cs]
        samples = samples + [
            gen.raster(c_hat, tile, tile, backend=backend),
            gen.raster(c_real, tile, tile, backend=backend),
        ]

    row_unf = _label_row(gen_frames, gen_labs, "UNFOLD generated  (each next prime is the next frame)")
    rows = [row_unf]
    if real_frames is not None:
        rows.append(_label_row(real_frames, gen_labs, "UNFOLD real prefix  (fine primes guessed from 1,2,3)"))
    if morph_frames is not None:
        rows.append(_label_row(morph_frames, [f"t{i}" for i in range(len(morph_frames))], "NEXT FRAME  prefix lerp → unfold"))
    if walk_frames is not None:
        rows.append(_label_row(walk_frames, [f"+{i}" for i in range(len(walk_frames))], "NEXT FRAME  prefix walk → unfold"))
    rows.append(_label_row(
        [samples[i] for i in range(min(6, len(samples)))],
        ["g0", "g1", "g2", "g3", "from-real", "real"][: min(6, len(samples))],
        "samples  (last two = unfold(real prefix) | real)",
    ))

    width = max(r.shape[1] for r in rows)
    body = np.concatenate([_pad_w(r, width) for r in rows], axis=0)
    head = np.full((58, width, 3), BG, dtype=np.uint8)
    im = Image.fromarray(head)
    d = ImageDraw.Draw(im)
    d.text(
        (12, 6),
        f"SCALEGEN3 UNFOLD  ep {status.get('epoch','')}  {status.get('tag','')}  n={status.get('n_data','')}",
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
<title>ScaleGen3 unfold</title></head><body style="background:#100a06;color:#ffb048;font-family:sans-serif">
<h1>ScaleGen3 unfold ep {status.get('epoch')} {status.get('tag')}</h1>
<img src="board.png?t={int(time.time())}" style="max-width:100%;image-rendering:pixelated">
</body></html>"""
    )
    (out_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    return path


def write_placeholder(out_dir: Path, msg: str) -> Path:
    canvas = np.full((220, 720, 3), BG, dtype=np.uint8)
    im = Image.fromarray(canvas)
    ImageDraw.Draw(im).text((20, 80), "SCALEGEN3  UNFOLD", font=_font(26, True), fill=ACCENT)
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
        [imv, "-s", "shrink", "-u", "nearest_neighbour", "-w", "ScaleGen3  UNFOLD", str(board)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
