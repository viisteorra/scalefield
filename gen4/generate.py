#!/usr/bin/env python3
"""Dump a real clip vs predicted rollout from a fitted gen4 run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "v1"))
sys.path.insert(0, str(ROOT / "gen3"))
sys.path.insert(0, str(HERE))

from scalefield import PrimeField, save_png  # noqa: E402
from nextframe import GeneratorV4, NextFrame  # noqa: E402
from watch import grid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", type=Path, default=HERE / "runs/live/scalegen")
    ap.add_argument("--out", type=Path, default=HERE / "runs/live")
    ap.add_argument("--tile", type=int, default=96)
    ap.add_argument("--clip", type=int, default=0)
    args = ap.parse_args()

    field = PrimeField.load(str(args.stem) + ".field.npz")
    nf = NextFrame.load(str(args.stem) + ".nextframe.npz")
    z = np.load(str(args.stem) + ".clips.npz", allow_pickle=True)
    slices = [(str(n), int(s), int(l)) for n, s, l in zip(z["names"], z["starts"], z["lens"])]
    gen = GeneratorV4(field, nf, slices)
    args.out.mkdir(parents=True, exist_ok=True)

    bank = gen.clip_bank(args.clip)
    n = min(12, len(bank))
    real = [gen.raster(bank[i], args.tile, args.tile) for i in range(n)]
    pred = [gen.raster(c, args.tile, args.tile) for c in nf.rollout(bank[0], n, temp=0.0)]
    save_png(str(args.out / "real.png"), grid(real, cols=n, pad=2))
    save_png(str(args.out / "rollout.png"), grid(pred, cols=n, pad=2))
    seq = args.out / "frames"
    seq.mkdir(exist_ok=True)
    for i, im in enumerate(pred):
        save_png(str(seq / f"pred_{i:02d}.png"), im)
    for i, im in enumerate(real):
        save_png(str(seq / f"real_{i:02d}.png"), im)
    print(gen.formula())


if __name__ == "__main__":
    main()
