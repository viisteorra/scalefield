#!/usr/bin/env python3
"""
Sample / morph / walk a trained ScaleGen.

    python generate.py --stem ./runs/live/scalegen --n 16 --out samples.png
    python generate.py --stem ./runs/live/scalegen --morph 0,5 --frames 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scalegen import Generator, grid, save_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", type=Path, default=Path("runs/live/scalegen"))
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--morph", type=str, default="", help="i0,i1 indices into the field bank")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--backend", default="cpu")
    args = ap.parse_args()

    gen = Generator.load(str(args.stem))
    print(gen.formula(), flush=True)
    out = args.out or Path("samples.png")

    if args.morph:
        a, b = (int(x) for x in args.morph.split(","))
        bank = gen.field.bank
        if bank is None:
            raise SystemExit("no bank on field; remorph needs encoded stills")
        imgs = gen.morph(bank[a], bank[b], frames=args.frames, H=args.res, W=args.res, backend=args.backend)
        save_png(str(out), grid(imgs, cols=len(imgs)))
    else:
        imgs, _ = gen.sample(args.n, args.res, args.res, temp=args.temp, seed=args.seed, backend=args.backend)
        save_png(str(out), grid(imgs, cols=4))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
