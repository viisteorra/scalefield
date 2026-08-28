#!/usr/bin/env python3
"""Build the v1 contact sheets: prime shaders, no-hole coverage, 360p roundtrips."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from scalefield import (
    PrimeField,
    grid_line_overlay,
    save_png,
    to_float_rgb,
)

ROOT = Path(__file__).resolve().parents[1]
ART = Path("artifacts")
ART.mkdir(exist_ok=True)


def _up(img01: np.ndarray, h: int) -> np.ndarray:
    from PIL import Image

    u8 = (np.clip(img01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return np.asarray(Image.fromarray(u8, mode="RGB").resize((h, h), Image.Resampling.NEAREST))


def _label_row(imgs: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(imgs, axis=1)


def shader_sheet(out: Path, size: int = 256) -> None:
    """One panel per prime: that lattice only, a structured p×p pattern."""
    pf = PrimeField.preset("full")
    panels = []
    for p in pf.primes:
        g = np.zeros((p, p), dtype=np.float64)
        if p == 1:
            g[0, 0] = 0.85
        elif p == 2:
            # weight and balance: four corners pull different ways
            g[0, 0], g[0, 1] = 1.2, -0.2
            g[1, 0], g[1, 1] = 0.1, 0.9
        else:
            yy, xx = np.mgrid[0:p, 0:p]
            g = 0.9 * np.cos(np.pi * xx / max(p - 1, 1)) * np.cos(np.pi * yy / max(p - 1, 1))
            g[p // 2, p // 2] += 0.6
        layer = PrimeField(primes=pf.primes, foveal=pf.foveal)
        layer.c[0] = 0.25  # keep a floor so blacks aren't crushed
        layer.set_grid(p, g)
        panels.append(np.clip(layer.field(size, size), 0, 1))
    # two rows
    n = len(panels)
    mid = (n + 1) // 2
    row1 = _label_row(panels[:mid])
    row2 = _label_row(panels[mid:] + [np.zeros_like(panels[0])] * (mid - (n - mid)))
    save_png(str(out), np.concatenate([row1, row2], axis=0))
    print(f"wrote {out}  primes={pf.primes} N={pf.nparams()}")


def coverage_sheet(out: Path, size: int = 256) -> None:
    pf = PrimeField.preset("medium")
    w = pf.spatial_weight(size, size)
    w_n = w / max(w.max(), 1e-9)
    img = np.stack([w_n, w_n, w_n], axis=-1)
    # red where coverage would be a hole (none expected)
    holes = w < 0.5 * len(pf.primes)
    img[holes] = (1.0, 0.1, 0.1)
    lines = grid_line_overlay(size, size, pf.primes, pf.foveal)
    both = np.clip(img * 0.35 + lines, 0, 1)
    save_png(str(out), np.concatenate([img, lines, both], axis=1))
    print(f"wrote {out}  coverage min={w.min():.3f} max={w.max():.3f}")


def generate_sheet(out: Path, size: int = 360) -> None:
    pf = PrimeField.preset("medium")
    imgs = [pf.generate(size, size, seed=s, amp=1.1) for s in range(6)]
    row1 = np.concatenate(imgs[:3], axis=1)
    row2 = np.concatenate(imgs[3:], axis=1)
    save_png(str(out), np.concatenate([row1, row2], axis=0))
    print(f"wrote {out}  {pf.formula()}")


def roundtrip_sheet(out: Path, size: int = 64, out_h: int = 360) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from data import load_folder_named, resolve_splits

    train, val = resolve_splits(ROOT / "data")
    named = load_folder_named(val, size=size)
    pf = PrimeField.preset("medium")
    rows = []
    picks = []
    syn = [i for i, (n, _) in enumerate(named) if n.startswith("syn")]
    real = [i for i, (n, _) in enumerate(named) if n.startswith("real")]
    picks = syn[:3] + real[:3]
    if len(picks) < 6:
        picks = list(range(min(6, len(named))))
    for k in picks:
        tgt = to_float_rgb(named[k][1])
        pred, st = pf.reconstruct(tgt, lam_l2=1e-4)
        hi = pf.decode(out_h, out_h, stretch=False)
        row = np.concatenate([_up(tgt, out_h), _up(np.clip(pred, 0, 1), out_h), hi], axis=1)
        rows.append(row)
        print(f"  {named[k][0]} mse={st['mse']:.4f} fovea={st['fovea_mse']:.4f}")
    save_png(str(out), np.concatenate(rows, axis=0))
    print(f"wrote {out}")


def gpu_compare(out: Path, size: int = 360) -> None:
    try:
        from gpu import gpu_available, gpu_info, raster_gpu
    except Exception as e:
        print(f"gpu skip: {e}")
        return
    if not gpu_available():
        print(f"gpu skip: {gpu_info()}")
        return
    pf = PrimeField.preset("medium")
    pf.generate(size, size, seed=3, amp=1.0)
    t0 = time.time()
    cpu = pf.field(size, size)
    t_cpu = time.time() - t0
    t0 = time.time()
    gpu = raster_gpu(pf, size, size)
    t_gpu = time.time() - t0
    d = np.abs(cpu - gpu)
    print(
        f"gpu {gpu_info()}  cpu={t_cpu*1e3:.1f}ms gpu={t_gpu*1e3:.1f}ms  "
        f"maxΔ={d.max():.4f} meanΔ={d.mean():.5f}"
    )
    vis = np.clip(np.concatenate(
        [np.clip(cpu, 0, 1), np.clip(gpu, 0, 1), np.clip(d * 8.0, 0, 1)],
        axis=1,
    ), 0, 1)
    save_png(str(out), vis)
    print(f"wrote {out}")


def blob_360(out: Path) -> None:
    H = 64
    yy, xx = np.mgrid[0:H, 0:H]
    blob = np.zeros((H, H, 3))
    blob[:, :, 0] = np.exp(-((xx - 22) ** 2 + (yy - 24) ** 2) / 140)
    blob[:, :, 1] = np.exp(-((xx - 40) ** 2 + (yy - 38) ** 2) / 180)
    blob[:, :, 2] = 0.12 + 0.35 * xx / H
    pf = PrimeField.preset("medium")
    pred, st = pf.reconstruct(blob, lam_l2=1e-4)
    hi = pf.decode(360, 360, stretch=False)
    row = np.concatenate([_up(blob, 360), _up(np.clip(pred, 0, 1), 360), hi], axis=1)
    save_png(str(out), row)
    print(f"wrote {out}  mse={st['mse']:.4f} N={pf.nparams()}  {pf.formula()}")


def main() -> None:
    shader_sheet(ART / "prime_shaders.png")
    coverage_sheet(ART / "coverage_grids.png")
    generate_sheet(ART / "generate_360.png")
    blob_360(ART / "blob_roundtrip_360.png")
    roundtrip_sheet(ART / "val_roundtrip_360.png")
    gpu_compare(ART / "cpu_vs_gpu_360.png")


if __name__ == "__main__":
    main()
