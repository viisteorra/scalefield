#!/usr/bin/env python3
"""Build the visual training set: synthetic fields first, then real thumbnails."""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def save_png(path: Path, rgb: np.ndarray) -> None:
    a = np.asarray(rgb)
    if a.dtype != np.uint8:
        a = np.clip(a, 0.0, 1.0)
        if a.max() <= 1.5:
            a = a * 255.0
        a = (a + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a, mode="RGB").save(path)


def load_rgb(path: Path, size: int | None = 64) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if size is not None:
        im = im.resize((size, size), Image.Resampling.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return a


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_folder_named(folder: Path, size: int = 64) -> list[tuple[str, np.ndarray]]:
    paths = list_images(folder)
    if not paths:
        raise FileNotFoundError(f"no images in {folder}")
    return [(p.name, load_rgb(p, size=size)) for p in paths]


def load_folder(folder: Path, size: int = 64) -> list[np.ndarray]:
    return [im for _, im in load_folder_named(folder, size=size)]


def resolve_splits(root: Path) -> tuple[Path, Path]:
    train, val = root / "train", root / "val"
    if train.is_dir() and val.is_dir() and list_images(train) and list_images(val):
        return train, val
    raise FileNotFoundError(f"expected {train} and {val} with images")


# ---------- synthetic (the v0 set) ----------

def _blob(H: int, W: int, cx: float, cy: float, s: float, color) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / max(s, 1.0))
    img = np.zeros((H, W, 3), dtype=np.float32)
    for c, v in enumerate(color):
        img[:, :, c] = float(v) * g
    return img


def _solid(H: int, W: int, color) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[:] = np.asarray(color, dtype=np.float32)
    return img


def _ramp(H: int, W: int, axis: int, color_a, color_b) -> np.ndarray:
    t = np.linspace(0.0, 1.0, W if axis == 0 else H, dtype=np.float32)
    if axis == 0:
        mix = t[None, :, None]
    else:
        mix = t[:, None, None]
    a = np.asarray(color_a, dtype=np.float32).reshape(1, 1, 3)
    b = np.asarray(color_b, dtype=np.float32).reshape(1, 1, 3)
    return a + (b - a) * mix


def _panel(H: int, W: int, color_a, color_b, split: float, axis: int) -> np.ndarray:
    img = _solid(H, W, color_a)
    if axis == 0:
        img[:, int(split * W) :] = np.asarray(color_b, dtype=np.float32)
    else:
        img[int(split * H) :, :] = np.asarray(color_b, dtype=np.float32)
    return img


def _rect(H: int, W: int, y0, x0, y1, x1, fg, bg=(0.12, 0.12, 0.14)) -> np.ndarray:
    img = _solid(H, W, bg)
    img[int(y0) : int(y1), int(x0) : int(x1)] = np.asarray(fg, dtype=np.float32)
    return img


def _ring(H: int, W: int, cx, cy, r0, r1, color, bg=(0.05, 0.05, 0.08)) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = (d >= r0) & (d <= r1)
    img = _solid(H, W, bg)
    img[mask] = np.asarray(color, dtype=np.float32)
    return img


def _plus(H: int, W: int, color, bg=(0.08, 0.08, 0.1)) -> np.ndarray:
    img = _solid(H, W, bg)
    t = max(2, H // 10)
    img[H // 2 - t : H // 2 + t, W // 6 : 5 * W // 6] = color
    img[H // 6 : 5 * H // 6, W // 2 - t : W // 2 + t] = color
    return img


def _window(H: int, W: int) -> np.ndarray:
    img = _solid(H, W, (0.18, 0.2, 0.28))
    img[6:16, 6:-6] = (0.35, 0.45, 0.75)
    img[20:-8, 8 : W // 2 - 2] = (0.75, 0.78, 0.82)
    img[20:-8, W // 2 + 2 : -8] = (0.55, 0.6, 0.7)
    return img


def make_synthetic(n: int, size: int, rng: np.random.Generator) -> list[np.ndarray]:
    H = W = size
    out: list[np.ndarray] = []

    def add(img):
        out.append(np.clip(img, 0.0, 1.0).astype(np.float32))

    palette = [
        (0.95, 0.25, 0.2),
        (0.2, 0.75, 0.35),
        (0.2, 0.4, 0.95),
        (0.95, 0.85, 0.2),
        (0.9, 0.4, 0.85),
        (0.2, 0.85, 0.85),
        (0.95, 0.95, 0.95),
        (0.15, 0.15, 0.18),
    ]

    # Solids, ramps, panels — const/lin bases should eat these.
    for c in palette:
        add(_solid(H, W, c))
    for i in range(8):
        add(_ramp(H, W, i % 2, palette[i % 8], palette[(i + 3) % 8]))
    for i in range(8):
        add(_panel(H, W, palette[i % 8], palette[(i + 4) % 8], 0.35 + 0.05 * (i % 4), i % 2))

    # Centered + offset blobs (v0 bar lives here).
    for i in range(24):
        cx = W * float(rng.uniform(0.2, 0.8))
        cy = H * float(rng.uniform(0.2, 0.8))
        if i < 6:
            cx, cy = W / 2, H / 2
        s = float(rng.uniform(40.0, 280.0))
        col = palette[i % 8]
        img = _blob(H, W, cx, cy, s, col)
        if i % 3 == 0:
            img = img + _blob(
                H,
                W,
                W * float(rng.uniform(0.2, 0.8)),
                H * float(rng.uniform(0.2, 0.8)),
                s * 0.7,
                palette[(i + 2) % 8],
            )
        bg = 0.08 + 0.05 * (i % 3)
        img = np.clip(img + bg, 0, 1)
        add(img)

    for i in range(8):
        add(
            _rect(
                H,
                W,
                rng.integers(4, H // 3),
                rng.integers(4, W // 3),
                rng.integers(2 * H // 3, H - 2),
                rng.integers(2 * W // 3, W - 2),
                palette[i % 8],
            )
        )
    for i in range(6):
        add(_ring(H, W, W / 2, H / 2, 8 + i * 2, 14 + i * 3, palette[i % 8]))
    add(_plus(H, W, (0.95, 0.9, 0.2)))
    add(_window(H, W))

    # Fill / shuffle down to n.
    while len(out) < n:
        i = len(out)
        add(
            _blob(
                H,
                W,
                W * float(rng.uniform(0.15, 0.85)),
                H * float(rng.uniform(0.15, 0.85)),
                float(rng.uniform(50, 220)),
                palette[i % 8],
            )
        )
    rng.shuffle(out)
    return out[:n]


# ---------- real thumbnails ----------

PICSUM = "https://picsum.photos/seed/scalefield{seed}/{size}/{size}"


def _http_get(url: str, timeout: float = 20.0) -> bytes:
    req = Request(url, headers={"User-Agent": "scalefield/0.1 (training-set)"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_picsum(out_dir: Path, n: int, size: int, start_seed: int = 1) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for i in range(n):
        path = out_dir / f"picsum_{start_seed + i:03d}.jpg"
        if path.exists() and path.stat().st_size > 500:
            saved.append(path)
            continue
        url = PICSUM.format(seed=start_seed + i, size=size)
        try:
            blob = _http_get(url)
            im = Image.open(io.BytesIO(blob)).convert("RGB")
            im = im.resize((size, size), Image.Resampling.BILINEAR)
            im.save(path, quality=92)
            saved.append(path)
            print(f"  downloaded {path.name}")
        except (URLError, HTTPError, OSError) as e:
            print(f"  skip seed {start_seed + i}: {e}")
    return saved


def write_split(images: list[np.ndarray], folder: Path, prefix: str) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        save_png(folder / f"{prefix}_{i:03d}.png", img)
    return len(images)


def build(
    root: Path,
    size: int = 64,
    n_synth_train: int = 80,
    n_synth_val: int = 20,
    n_real_train: int = 40,
    n_real_val: int = 10,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    train_dir, val_dir = root / "train", root / "val"
    if train_dir.exists():
        for p in list_images(train_dir):
            p.unlink()
    if val_dir.exists():
        for p in list_images(val_dir):
            p.unlink()

    synth = make_synthetic(n_synth_train + n_synth_val, size, rng)
    n_tr = write_split(synth[:n_synth_train], train_dir, "syn")
    n_va = write_split(synth[n_synth_train:], val_dir, "syn")

    real_ok = 0
    raw = root / "real_raw"
    print(f"downloading {n_real_train + n_real_val} real thumbnails…")
    paths = download_picsum(raw, n_real_train + n_real_val, size=max(size, 96), start_seed=11)
    random.Random(seed).shuffle(paths)
    for i, p in enumerate(paths):
        img = load_rgb(p, size=size)
        dest = train_dir if i < n_real_train else val_dir
        prefix = "real"
        idx = i if i < n_real_train else i - n_real_train
        save_png(dest / f"{prefix}_{idx:03d}.png", img)
        real_ok += 1

    manifest = {
        "size": size,
        "synthetic_train": n_tr,
        "synthetic_val": n_va,
        "real_total": real_ok,
        "train_images": len(list_images(train_dir)),
        "val_images": len(list_images(val_dir)),
        "seed": seed,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.out, size=args.size, seed=args.seed)


if __name__ == "__main__":
    main()
