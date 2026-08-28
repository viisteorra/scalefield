#!/usr/bin/env python3
"""
ScaleGen2 — real photos only. Frozen v1 field. Local mixes in z.

The trippy GMM is frozen at ../gen/runs/cool-gmm/. This is a new version.

Dataset: Picsum stills, no synthetic blobs (those were mixing with photos
and that's why v1 gen looked like oil-slick DCT).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "v1"))

from data import download_picsum, load_rgb, list_images  # noqa: E402
from scalefield import PrimeField  # noqa: E402
from scalegen import CoarseFine, DiagGMM, GeneratorV2, LatentPrior  # noqa: E402
from watch import open_viewer, write_board, write_placeholder  # noqa: E402

STOP = False


def _stop(signum, _frame):
    global STOP
    STOP = True
    print(f"\nsignal {signum}; save after epoch", flush=True)


def _assert_alone():
    import os

    me = os.getpid()
    others = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        if pid == me:
            continue
        try:
            raw = (p / "cmdline").read_bytes().split(b"\x00")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        args = [c.decode("utf-8", "replace") for c in raw if c]
        if args and "python" in args[0] and any(a.endswith("train.py") or a == "train.py" for a in args):
            others.append(f"pid={pid} {' '.join(args[:8])}")
    if others:
        raise SystemExit("another trainer is running — one at a time:\n  " + "\n  ".join(others))


def gather_reals(n: int, size: int) -> list[np.ndarray]:
    raw = HERE / "data" / "real_raw"
    print(f"picsum → {n} stills @ {size}…", flush=True)
    paths = download_picsum(raw, n, size=max(size, 128), start_seed=200)
    imgs = []
    for p in paths:
        try:
            imgs.append(load_rgb(p, size=size))
        except OSError:
            continue
    # also take v1 reals, not syn
    v1_train = ROOT / "data" / "train"
    for p in list_images(v1_train):
        if p.name.startswith("real"):
            imgs.append(load_rgb(p, size=size))
    print(f"  real cloud n={len(imgs)}", flush=True)
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", type=Path, default=ROOT / "v1/runs/live/best.npz")
    ap.add_argument("--out-dir", type=Path, default=HERE / "runs/live")
    ap.add_argument("--n-real", type=int, default=250)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--k-coarse", type=int, default=8)
    ap.add_argument("--max-comp", type=int, default=12)
    ap.add_argument("--split-every", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--board-every", type=int, default=8)
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument(
        "--from-bank",
        type=Path,
        default=None,
        help="load populated c-bank from a frozen field.npz (skip encode/download)",
    )
    args = ap.parse_args()

    _assert_alone()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    watch_dir = args.out_dir / "watch"
    watch_dir.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    board = write_placeholder(watch_dir, "loading populated bank / fitting prior…")
    viewer = None if args.no_viewer else open_viewer(board)
    if viewer:
        print(f"viewer pid={viewer.pid}  {board}", flush=True)

    if args.from_bank:
        pf = PrimeField.load(str(args.from_bank))
        if pf.bank is None or len(pf.bank) < 8:
            raise SystemExit(f"{args.from_bank} has no populated bank")
        bank = np.asarray(pf.bank, dtype=np.float64)
        print(
            f"resume bank {bank.shape} from {args.from_bank}  N={pf.nparams()}  "
            f"(no encode, no download)",
            flush=True,
        )
    else:
        if not args.field.exists():
            raise SystemExit(f"need frozen v1 field at {args.field}")
        imgs = gather_reals(args.n_real, args.size)
        pf = PrimeField.load(str(args.field))
        print(f"frozen field N={pf.nparams()}  encoding {len(imgs)} reals (new bank, v1 untouched)", flush=True)
        stats = pf.evaluate(imgs, lam_l2=args.lam, chord_steps=0)
        bank = stats["bank"]
        pf.bank = bank
        print(f"  encode mse={stats['mse']:.4f}  bank={bank.shape}", flush=True)

    prior = LatentPrior.fit(bank, k=args.k, energy=1.0)
    Z = np.stack([prior.encode(c) for c in bank])
    kc = min(args.k_coarse, prior.k)
    cf = CoarseFine.fit(Z, kc)
    Zc = Z[:, :kc]
    gmm = DiagGMM.from_data(Zc)
    gen = GeneratorV2(pf, prior, gmm, cf)
    print(f"PCA k={prior.k}  coarse={kc}  {gen.formula()}", flush=True)

    rng = np.random.default_rng(1)
    t0 = time.time()
    epoch = 0
    max_ep = args.epochs if args.epochs > 0 else 10**9
    metrics = args.out_dir / "metrics.jsonl"

    def snap(tag: str, force=False):
        nll = gmm.nll(Zc)
        status = {
            "epoch": epoch,
            "tag": tag,
            "nll": nll,
            "n_comp": gmm.m,
            "n_data": len(bank),
            "k": prior.k,
            "k_coarse": kc,
            "formula": gen.formula(),
        }
        with metrics.open("a") as f:
            f.write(json.dumps(status) + "\n")
        if force or epoch % args.board_every == 0 or epoch < 4:
            write_board(gen, watch_dir, status, tile=args.tile, seed=epoch, backend=args.backend)
        gen.field.save(str(args.out_dir / "scalegen.field.npz"))
        prior.save(str(args.out_dir / "scalegen.prior.npz"))
        gmm.save(str(args.out_dir / "scalegen.gmm.npz"))
        cf.save(str(args.out_dir / "scalegen.cf.npz"))
        return nll

    nll = snap("init", force=True)
    print(f"epoch {epoch:4d}  GMM={gmm.m}  nll={nll:.3f}  n={len(bank)} reals", flush=True)

    while not STOP and epoch < max_ep:
        epoch += 1
        nll = gmm.em_step(Zc)
        if epoch % args.split_every == 0 and gmm.m < args.max_comp:
            gmm.split(rng)
            nll = snap("split", force=True)
            print(f"epoch {epoch:4d}  SPLIT GMM={gmm.m}  nll={nll:.3f}", flush=True)
            continue
        if epoch % 15 == 0:
            print(f"epoch {epoch:4d}  em     GMM={gmm.m}  nll={nll:.3f}", flush=True)
        snap("em")

    snap("stop", force=True)
    print(f"stop  epochs={epoch}  GMM={gmm.m}  secs={time.time()-t0:.1f}", flush=True)
    print(f"cool trippy model still at {ROOT / 'gen/runs/cool-gmm'}", flush=True)


if __name__ == "__main__":
    main()
