#!/usr/bin/env python3
"""
Score a ScaleField run.

Smoke tests always run (in-span roundtrip, nparams, save/load, blob bar).
If --ckpt is set, encode the val split from scratch and write a contact sheet.

    python test.py
    python test.py --ckpt runs/live/best.npz --data ./data --out runs/live/test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scalefield import ScaleField, pack_b, save_png, to_float_rgb, unpack_b


def _ok(name: str, cond: bool, detail: str = "") -> bool:
    mark = "PASS" if cond else "FAIL"
    extra = f"  {detail}" if detail else ""
    print(f"  [{mark}] {name}{extra}")
    return cond


def smoke() -> bool:
    print("== smoke ==")
    passed = True

    sf = ScaleField(x=3, i=4, r=0.5)
    passed &= _ok("nparams", sf.nparams() == 3 * 3 * 4, f"N={sf.nparams()}")

    threw = False
    try:
        ScaleField(x=33, i=24)  # 2376 > 2304
    except ValueError:
        threw = True
    passed &= _ok("cap", threw)

    # pack/unpack is the design-matrix layout.
    sf.b = np.arange(sf.nparams(), dtype=float).reshape(sf.b.shape)
    passed &= _ok("pack/unpack", np.allclose(unpack_b(pack_b(sf.b), sf.i, sf.x), sf.b))

    # In-span: an image the model itself produced must encode back almost exactly.
    rng = np.random.default_rng(0)
    sf.b = rng.normal(0.0, 0.12, size=sf.b.shape)
    target = sf.field(64, 64)
    A = sf.design(64, 64)
    recon_lin = (A @ pack_b(sf.b)).reshape(64, 64, 3)
    passed &= _ok("field=A@b", float(np.max(np.abs(recon_lin - target))) < 1e-12)
    sf2 = ScaleField(x=3, i=4, r=0.5)
    pred, stats = sf2.reconstruct(target, lam_l1=0.0, lam_l2=1e-12, ista_steps=0)
    passed &= _ok("in-span mse<1e-8", stats["mse"] < 1e-8, f"mse={stats['mse']:.2e}")
    # Bases overlap, so b is not unique; the recovered field must match.
    passed &= _ok(
        "in-span field-recover",
        float(np.max(np.abs(pred - target))) < 1e-6,
        f"maxΔ={np.max(np.abs(pred - target)):.2e}",
    )

    # Blob bar (v0): MSE < 0.02 at 64 with N=36.
    H = W = 64
    yy, xx = np.mgrid[0:H, 0:W]
    blob = np.zeros((H, W, 3))
    blob[:, :, 0] = np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / 180)
    blob[:, :, 1] = 0.25 * blob[:, :, 0]
    blob[:, :, 2] = 0.10
    sfb = ScaleField(x=3, i=4, r=0.5)
    _, st = sfb.reconstruct(blob, lam_l1=0.0, ista_steps=0)
    passed &= _ok("blob mse<0.02", st["mse"] < 0.02, f"mse={st['mse']:.4f} N={sfb.nparams()}")

    # Save/load bit-identical b.
    path = Path("artifacts") / "smoke_roundtrip.npz"
    path.parent.mkdir(exist_ok=True)
    sfb.save(str(path))
    loaded = ScaleField.load(str(path))
    passed &= _ok("reload b", np.allclose(loaded.b, sfb.b), f"maxΔ={np.max(np.abs(loaded.b - sfb.b)):.2e}")
    passed &= _ok("reload formula", loaded.x == sfb.x and loaded.i == sfb.i and abs(loaded.r - sfb.r) < 1e-12)

    # Resolution adapter: same field, two rasters — center value must match.
    c64 = sfb.field(64, 64)[32, 32]
    c256 = sfb.field(256, 256)[128, 128]
    passed &= _ok(
        "multi-res center",
        float(np.max(np.abs(c64 - c256))) < 0.05,
        f"Δ={np.max(np.abs(c64 - c256)):.4f}",
    )

    print(sfb.formula())
    return passed


def contact_sheet(
    sf: ScaleField,
    images: list[np.ndarray],
    bank: np.ndarray,
    path: Path,
    indices: list[int] | None = None,
    n: int = 8,
    out_h: int = 192,
) -> None:
    from PIL import Image

    def up(img01, H, W):
        u8 = (np.clip(img01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        return np.asarray(Image.fromarray(u8, mode="RGB").resize((W, H), Image.Resampling.NEAREST))

    idxs = indices if indices is not None else list(range(min(n, len(images))))
    rows = []
    hold = sf.b.copy()
    for k in idxs:
        tgt = to_float_rgb(images[k])
        sf.b = bank[k]
        rec64 = np.clip(sf.field(*tgt.shape[:2]), 0, 1)
        rec_hi = sf.decode(out_h, out_h, stretch=False)
        row = np.concatenate([up(tgt, out_h, out_h), up(rec64, out_h, out_h), rec_hi], axis=1)
        rows.append(row)
    sf.b = hold
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_png(str(path), np.concatenate(rows, axis=0))


def _subset_mse(names: list[str], per_image: list[float], prefix: str) -> dict | None:
    vals = [m for name, m in zip(names, per_image) if name.startswith(prefix)]
    if not vals:
        return None
    return {"mse": float(np.mean(vals)), "n": len(vals), "std": float(np.std(vals))}


def eval_ckpt(ckpt: Path, data: Path, out: Path, size: int) -> dict:
    from data import load_folder_named, resolve_splits

    print(f"== checkpoint {ckpt} ==")
    sf = ScaleField.load(str(ckpt))
    train_dir, val_dir = resolve_splits(data)
    val_named = load_folder_named(val_dir, size=size)
    train_named = load_folder_named(train_dir, size=size)
    val_names = [n for n, _ in val_named]
    train_names = [n for n, _ in train_named]
    val = [im for _, im in val_named]
    train = [im for _, im in train_named]
    val_s = sf.evaluate(val, lam_l1=1e-3, lam_l2=1e-4, ista_steps=8)
    train_s = sf.evaluate(train, lam_l1=1e-3, lam_l2=1e-4, ista_steps=8)

    out.mkdir(parents=True, exist_ok=True)
    syn_idx = [i for i, n in enumerate(val_names) if n.startswith("syn")][:4]
    real_idx = [i for i, n in enumerate(val_names) if n.startswith("real")][:4]
    mixed = syn_idx + real_idx
    contact_sheet(sf, val, val_s["bank"], out / "val_grid.png", indices=mixed or None)
    contact_sheet(sf, val, val_s["bank"], out / "val_syn.png", indices=syn_idx or None)
    contact_sheet(sf, val, val_s["bank"], out / "val_real.png", indices=real_idx or None)
    tr_syn = [i for i, n in enumerate(train_names) if n.startswith("syn")][:4]
    tr_real = [i for i, n in enumerate(train_names) if n.startswith("real")][:4]
    contact_sheet(sf, train, train_s["bank"], out / "train_grid.png", indices=tr_syn + tr_real)

    # Mean-field prototype (what a single shared b looks like).
    sf.b = train_s["bank"].mean(axis=0)
    save_png(str(out / "mean_field_256.png"), sf.decode(256, 256, stretch=False))
    save_png(str(out / "sample0_256.png"), sf.generate_from_cloud(256, 256, seed=0))
    save_png(str(out / "sample1_256.png"), sf.generate_from_cloud(256, 256, seed=1))

    val_syn = _subset_mse(val_names, val_s["per_image_mse"], "syn")
    val_real = _subset_mse(val_names, val_s["per_image_mse"], "real")
    train_syn = _subset_mse(train_names, train_s["per_image_mse"], "syn")
    train_real = _subset_mse(train_names, train_s["per_image_mse"], "real")
    syn_bar = val_syn is not None and val_syn["mse"] < 0.02
    report = {
        "ckpt": str(ckpt),
        "formula": sf.formula(),
        "nparams": sf.nparams(),
        "x": sf.x,
        "i": sf.i,
        "r": sf.r,
        "train_mse": train_s["mse"],
        "train_fovea": train_s["fovea_mse"],
        "val_mse": val_s["mse"],
        "val_fovea": val_s["fovea_mse"],
        "val_mse_std": val_s["mse_std"],
        "val_syn": val_syn,
        "val_real": val_real,
        "train_syn": train_syn,
        "train_real": train_real,
        "sparsity": val_s["sparsity"],
        "val_n": val_s["n"],
        "pixels": size * size * 3,
        "compression": (size * size * 3) / max(sf.nparams(), 1),
        "val_bar_0.02": val_s["mse"] < 0.02,
        "syn_bar_0.02": syn_bar,
        "previews": {
            "val_grid": str(out / "val_grid.png"),
            "val_syn": str(out / "val_syn.png"),
            "val_real": str(out / "val_real.png"),
            "train_grid": str(out / "train_grid.png"),
            "mean_field": str(out / "mean_field_256.png"),
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "formula.txt").write_text(sf.formula() + "\n")

    print(sf.formula())
    print(
        f"  train mse={train_s['mse']:.4f}  fovea={train_s['fovea_mse']:.4f}  n={train_s['n']}"
    )
    if train_syn and train_real:
        print(f"         syn={train_syn['mse']:.4f} n={train_syn['n']}  real={train_real['mse']:.4f} n={train_real['n']}")
    print(
        f"  val   mse={val_s['mse']:.4f}  fovea={val_s['fovea_mse']:.4f}  "
        f"std={val_s['mse_std']:.4f}  n={val_s['n']}"
    )
    if val_syn and val_real:
        bar = "OK" if syn_bar else "HIGH"
        print(
            f"         syn={val_syn['mse']:.4f} n={val_syn['n']} [{bar} <0.02]  "
            f"real={val_real['mse']:.4f} n={val_real['n']}"
        )
    print(f"  sparsity={val_s['sparsity']:.2f}  N={sf.nparams()}  px/param={report['compression']:.1f}")
    print(f"  grids {out / 'val_syn.png'}  {out / 'val_real.png'}  {out / 'val_grid.png'}")
    print(f"  report {out / 'report.json'}")

    metrics = ckpt.parent / "metrics.jsonl"
    if metrics.exists():
        lines = metrics.read_text().strip().splitlines()
        print(f"== last {min(8, len(lines))} train epochs ({metrics}) ==")
        for line in lines[-8:]:
            row = json.loads(line)
            mark = "*" if row.get("best") else " "
            print(
                f" {mark} ep {row['epoch']:4d} {row.get('phase','?'):<7} "
                f"val={row['val_mse']:.4f} train={row['train_mse']:.4f} "
                f"N={row['nparams']} {row.get('tag','')}"
            )
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("runs/live/test"))
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()

    ok = smoke()
    if args.ckpt is not None:
        if not args.ckpt.exists():
            print(f"FAIL missing ckpt {args.ckpt}")
            return 2
        eval_ckpt(args.ckpt, args.data, args.out, args.size)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
