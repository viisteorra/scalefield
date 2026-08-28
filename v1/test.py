#!/usr/bin/env python3
"""Smoke tests for ScaleField v1 (prime-grid color-phase shaders)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scalefield import (
    COLOR_PHASES,
    PRIMES,
    PrimeField,
    foveal_map,
    mse,
    nparams_of,
    save_png,
    to_float_rgb,
)


def _ok(name: str, cond: bool, detail: str = "") -> bool:
    mark = "PASS" if cond else "FAIL"
    extra = f"  {detail}" if detail else ""
    print(f"  [{mark}] {name}{extra}")
    return cond


def smoke() -> bool:
    print("== smoke ==")
    passed = True

    pf = PrimeField.preset("tiny", chord_gain=0.0)
    passed &= _ok("nparams tiny", pf.nparams() == 1 + 4 + 9 + 25 + 49, f"N={pf.nparams()}")
    tied = PrimeField.preset("tiny", src=8, chord_gain=0.0)
    passed &= _ok("tied nparams=src²", tied.nparams() == 64, f"N={tied.nparams()} src={tied.src}")
    rng2 = np.random.default_rng(2)
    tied.s = rng2.normal(0, 0.2, size=(8, 8))
    tied._sync_c()
    f0 = tied.field(24, 24)
    A = tied.design(24, 24)
    rec = (A @ tied.s.reshape(-1)).reshape(24, 24, 3)
    passed &= _ok("tied field=A@s", float(np.max(np.abs(rec - f0))) < 1e-8, f"maxΔ={np.max(np.abs(rec - f0)):.2e}")
    passed &= _ok("nparams=sum p²", pf.nparams() == nparams_of(pf.primes))

    threw = False
    try:
        PrimeField(primes=(1,) + PRIMES)  # 27 primes, way over cap
    except ValueError:
        threw = True
    passed &= _ok("cap", threw)

    # Linear in c when chords are off (DCT + edge boost is linear).
    rng = np.random.default_rng(0)
    pf.c = rng.normal(0.0, 0.2, size=pf.c.shape)
    f1 = pf.field(32, 32)
    c1 = pf.c.copy()
    pf.c = rng.normal(0.0, 0.2, size=pf.c.shape)
    f2 = pf.field(32, 32)
    c2 = pf.c.copy()
    pf.c = c1 + c2
    f12 = pf.field(32, 32)
    passed &= _ok("linear in c", float(np.max(np.abs(f12 - f1 - f2))) < 1e-9)

    # Design matrix matches field()
    pf.c = c1
    A = pf.design(32, 32)
    recon = (A @ pf.c).reshape(32, 32, 3)
    passed &= _ok("field=A@c", float(np.max(np.abs(recon - f1))) < 1e-8, f"maxΔ={np.max(np.abs(recon - f1)):.2e}")

    # In-span: an image the model produced encodes back.
    pf2 = PrimeField.preset("tiny", chord_gain=0.0)
    pred, stats = pf2.reconstruct(f1, lam_l2=1e-12)
    passed &= _ok("in-span mse<1e-8", stats["mse"] < 1e-8, f"mse={stats['mse']:.2e}")

    # Cosine grids cover the whole panel. p=1 DC is identically 1.
    w = PrimeField.preset("tiny").spatial_weight(64, 64)
    passed &= _ok("dc covers", float(w.min()) > 0.99 and float(w.max()) < 1.01, f"min={w.min():.4f} max={w.max():.4f}")
    pf2x = PrimeField(primes=(1, 2), chord_gain=0.0)
    pf2x.c[0] = 1.0
    img = pf2x.field(48, 48)
    passed &= _ok("2×2 covers", float(np.min(img)) > 0.5, f"min={np.min(img):.4f}")

    # Prime grid lines do not share interior ticks.
    def ticks(p):
        return {i / p for i in range(1, p)}

    clash = ticks(2) & ticks(3) & ticks(5)
    interior_clash = ticks(2) & ticks(3)
    passed &= _ok("primes never align", interior_clash == set(), f"2∩3={interior_clash} 2∩3∩5={clash}")

    # Foveation: |d chart / d screen| larger at center than at the edge.
    u = np.array([0.0, 0.9])
    v = np.array([0.0, 0.0])
    a = 1.15
    # numerical derivative along u
    eps = 1e-4
    c0, _ = foveal_map(u, v, a)
    c1, _ = foveal_map(u + eps, v, a)
    du = (c1 - c0) / eps
    passed &= _ok("outer pixels larger", float(du[0]) > float(du[1]), f"dcenter={du[0]:.3f} dedge={du[1]:.3f}")

    # Color phases span RGB (invertible 3×3).
    cond = float(np.linalg.cond(COLOR_PHASES))
    passed &= _ok("color phases span RGB", cond < 20.0, f"cond={cond:.2f}")

    # Blob bar: solid+blob should fit well under 0.02 at 64 with tiny.
    H = W = 64
    yy, xx = np.mgrid[0:H, 0:W]
    blob = np.zeros((H, W, 3))
    blob[:, :, 0] = np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / 180)
    blob[:, :, 1] = 0.25 * blob[:, :, 0]
    blob[:, :, 2] = 0.10
    sfb = PrimeField.preset("tiny", chord_gain=0.0)
    _, st = sfb.reconstruct(blob, lam_l2=1e-4)
    passed &= _ok("blob mse<0.02", st["mse"] < 0.02, f"mse={st['mse']:.4f} N={sfb.nparams()}")

    # Solid color — the 3 phases + brightness must reach a constant RGB.
    solid = np.zeros((32, 32, 3))
    solid[:] = (0.8, 0.2, 0.15)
    sfs = PrimeField.preset("tiny", chord_gain=0.0)
    _, ss = sfs.reconstruct(solid, lam_l2=1e-6)
    passed &= _ok("solid red mse<0.01", ss["mse"] < 0.01, f"mse={ss['mse']:.4f}")

    # Sharp step: high-k DCT + prime chords should lock an edge, not a ramp.
    step = np.zeros((64, 64, 3))
    step[:, :32] = (0.95, 0.95, 0.95)
    step[:, 32:] = (0.05, 0.05, 0.08)
    sfe = PrimeField.preset("tiny")
    pred_e, se = sfe.reconstruct(step, lam_l2=1e-4)
    # jump across the seam should stay large (not fully blurred)
    jump = float(np.mean(pred_e[:, 28, 0]) - np.mean(pred_e[:, 35, 0]))
    passed &= _ok("edge jump", jump > 0.25, f"jump={jump:.3f} mse={se['mse']:.4f}")

    # Save/load.
    path = Path("artifacts") / "smoke_roundtrip.npz"
    path.parent.mkdir(exist_ok=True)
    sfb.save(str(path))
    loaded = PrimeField.load(str(path))
    passed &= _ok("reload c", np.allclose(loaded.c, sfb.c), f"maxΔ={np.max(np.abs(loaded.c - sfb.c)):.2e}")
    passed &= _ok("reload primes", loaded.primes == sfb.primes)

    # Multi-res: same field, center of 64 vs 360 should be close.
    c64 = sfb.field(64, 64)[32, 32]
    c360 = sfb.field(360, 360)[180, 180]
    passed &= _ok(
        "multi-res center",
        float(np.max(np.abs(c64 - c360))) < 0.08,
        f"Δ={np.max(np.abs(c64 - c360)):.4f}",
    )

    print(sfb.formula())
    return passed


def gpu_smoke() -> bool:
    print("== gpu ==")
    try:
        from gpu import gpu_available, gpu_info, raster_gpu
    except Exception as e:
        print(f"  [SKIP] gpu import: {e}")
        return True
    if not gpu_available():
        print(f"  [SKIP] {gpu_info()}")
        return True
    print(f"  device {gpu_info()}")
    pf = PrimeField.preset("tiny")
    rng = np.random.default_rng(1)
    pf.c = rng.normal(0.0, 0.3, size=pf.c.shape)
    pf.c[0] = 0.4
    cpu = pf.field(64, 64)
    gpu = raster_gpu(pf, 64, 64)
    d = float(np.max(np.abs(cpu - gpu)))
    m = mse(cpu, gpu)
    ok = _ok("cpu≈gpu 64", d < 0.05, f"maxΔ={d:.4f} mse={m:.6f}")
    cpu360 = pf.field(360, 360)
    gpu360 = raster_gpu(pf, 360, 360)
    d360 = float(np.max(np.abs(cpu360 - gpu360)))
    ok &= _ok("cpu≈gpu 360", d360 < 0.08, f"maxΔ={d360:.4f}")
    return ok


def eval_ckpt(ckpt: Path, data: Path, out: Path, size: int) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data import load_folder_named, resolve_splits

    print(f"== checkpoint {ckpt} ==")
    pf = PrimeField.load(str(ckpt))
    train_dir, val_dir = resolve_splits(data)
    val_named = load_folder_named(val_dir, size=size)
    val = [im for _, im in val_named]
    val_s = pf.evaluate(val, lam_l2=1e-4)
    out.mkdir(parents=True, exist_ok=True)
    hold = pf.c.copy()
    rows = []
    from PIL import Image

    def up(img01, h):
        u8 = (np.clip(img01, 0, 1) * 255 + 0.5).astype(np.uint8)
        return np.asarray(Image.fromarray(u8, mode="RGB").resize((h, h), Image.Resampling.NEAREST))

    for k in range(min(6, len(val))):
        pf.c = val_s["bank"][k]
        tgt = to_float_rgb(val[k])
        rec = np.clip(pf.field(*tgt.shape[:2]), 0, 1)
        hi = pf.decode(256, 256, stretch=False)
        rows.append(np.concatenate([up(tgt, 256), up(rec, 256), hi], axis=1))
    pf.c = hold
    if rows:
        save_png(str(out / "val_grid.png"), np.concatenate(rows, axis=0))
    report = {
        "ckpt": str(ckpt),
        "formula": pf.formula(),
        "nparams": pf.nparams(),
        "val_mse": val_s["mse"],
        "val_fovea": val_s["fovea_mse"],
        "sparsity": val_s["sparsity"],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=Path("../data"))
    ap.add_argument("--out", type=Path, default=Path("runs/live/test"))
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()

    ok = smoke()
    ok = gpu_smoke() and ok
    if args.ckpt is not None:
        if not args.ckpt.exists():
            print(f"FAIL missing ckpt {args.ckpt}")
            return 2
        eval_ckpt(args.ckpt, args.data, args.out, args.size)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
