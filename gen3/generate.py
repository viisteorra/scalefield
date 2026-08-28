#!/usr/bin/env python3
"""Dump unfold filmstrips and next-frame morphs from a fitted gen3 run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "v1"))
sys.path.insert(0, str(HERE))

from scalefield import PrimeField, save_png  # noqa: E402
from unfold import DiagGMM, GeneratorV3, Unfold  # noqa: E402
from watch import grid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", type=Path, default=HERE / "runs/live/scalegen")
    ap.add_argument("--out", type=Path, default=HERE / "runs/live")
    ap.add_argument("--tile", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    field = PrimeField.load(str(args.stem) + ".field.npz")
    unf = Unfold.load(str(args.stem) + ".unfold.npz")
    gmm_path = Path(str(args.stem) + ".gmm.npz")
    gmm = DiagGMM.load(str(gmm_path)) if gmm_path.exists() else None
    gen = GeneratorV3(field, unf, gmm)
    args.out.mkdir(parents=True, exist_ok=True)

    samples, Cs = gen.sample(8, args.tile, args.tile, temp=0.1, seed=args.seed)
    save_png(str(args.out / "samples.png"), grid(samples, cols=4, pad=2))

    st = unf.stages(Cs[0])
    frames = [gen.raster(part, args.tile, args.tile) for _, part in st]
    save_png(str(args.out / "unfold_gen.png"), grid(frames, cols=len(frames), pad=2))

    if field.bank is not None and len(field.bank) >= 2:
        c0, c1 = field.bank[0], field.bank[min(9, len(field.bank) - 1)]
        hat = gen.from_real_prefix(c0, temp=0.0, seed=args.seed)
        st_r = unf.stages(hat)
        real_f = [gen.raster(part, args.tile, args.tile) for _, part in st_r]
        save_png(str(args.out / "unfold_real.png"), grid(real_f, cols=len(real_f), pad=2))
        morph = [gen.raster(c, args.tile, args.tile) for c in gen.next_frames(c0, c1, frames=12)]
        save_png(str(args.out / "next_frame.png"), grid(morph, cols=12, pad=2))
        # png sequence for a cheap "video"
        seq = args.out / "frames"
        seq.mkdir(exist_ok=True)
        for i, im in enumerate(morph):
            save_png(str(seq / f"morph_{i:02d}.png"), im)
        for i, im in enumerate(frames):
            save_png(str(seq / f"unfold_{i:02d}.png"), im)
    print(gen.formula())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
