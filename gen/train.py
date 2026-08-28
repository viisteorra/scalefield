#!/usr/bin/env python3
"""
Train ScaleGen on CPU. Field is frozen. Only the z-transform (PCA + GMM) moves.

    python train.py --field ../v1/runs/live/best.npz

Starts with one Gaussian on z, EM-fits, splits new components, writes a
SAMPLES | MORPH board as it learns. Ctrl-C saves.
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
V1 = ROOT / "v1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(V1))

from scalefield import PrimeField  # noqa: E402
from scalegen import DiagGMM, Generator, LatentPrior  # noqa: E402
from watch import open_viewer, write_board, write_placeholder  # noqa: E402

STOP = False


def _stop(signum, _frame):
    global STOP
    STOP = True
    print(f"\nsignal {signum}; saving after this epoch", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", type=Path, default=V1 / "runs/live/best.npz")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/live"))
    ap.add_argument("--k", type=int, default=32, help="PCA dim (frozen after fit)")
    ap.add_argument("--max-comp", type=int, default=16, help="max GMM components")
    ap.add_argument("--split-every", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=0, help="0 = until Ctrl-C")
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--board-every", type=int, default=5)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    watch_dir = args.out_dir / "watch"
    watch_dir.mkdir(parents=True, exist_ok=True)
    metrics = args.out_dir / "metrics.jsonl"

    if not args.field.exists():
        raise SystemExit(f"frozen field missing: {args.field}")

    pf = PrimeField.load(str(args.field))
    if pf.bank is None or pf.bank.shape[1] != pf.nparams():
        raise SystemExit("field npz needs a coefficient bank from v1 fit")

    print(f"frozen field  N={pf.nparams()}  bank={pf.bank.shape}  (c will not be updated)", flush=True)

    board = write_placeholder(watch_dir, "PCA on frozen c, then GMM on z…")
    viewer = None if args.no_viewer else open_viewer(board)
    if viewer:
        print(f"viewer  ScaleGen  pid={viewer.pid}  {board}", flush=True)

    prior = LatentPrior.fit(pf.bank, k=args.k, energy=1.0)
    Z = np.stack([prior.encode(c) for c in pf.bank])
    print(f"PCA k={prior.k}  resid={prior.resid_std:.4f}  Z={Z.shape}", flush=True)

    rng = np.random.default_rng(0)
    gmm = DiagGMM.from_data(Z)
    gen = Generator(pf, prior, gmm=gmm)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    t0 = time.time()
    best_nll = float("inf")
    epoch = 0
    max_ep = args.epochs if args.epochs > 0 else 10**9

    def snapshot(tag: str, nll: float, force_board: bool = False) -> None:
        status = {
            "epoch": epoch,
            "n_comp": gmm.m,
            "nll": nll,
            "k": prior.k,
            "n_field": pf.nparams(),
            "n_prior": prior.nparams() + gmm.nparams(),
            "resid": prior.resid_std,
            "tag": tag,
            "formula": gen.formula(),
        }
        with metrics.open("a") as f:
            f.write(json.dumps(status) + "\n")
        if force_board or epoch % args.board_every == 0 or epoch <= 3:
            write_board(
                gen, watch_dir, status,
                tile=args.tile, n_samples=8, seed=epoch + 7, backend=args.backend,
            )
        gen.save(str(args.out_dir / "scalegen"))
        (args.out_dir / "formula.txt").write_text(gen.formula() + "\n")

    nll = gmm.nll(Z)
    snapshot("init", nll, force_board=True)
    print(f"epoch {epoch:4d}  GMM={gmm.m}  nll={nll:.4f}  (one blob)", flush=True)

    while not STOP and epoch < max_ep:
        epoch += 1
        nll = gmm.em_step(Z)
        improved = nll < best_nll - 1e-5
        if improved:
            best_nll = nll
        if epoch % args.split_every == 0 and gmm.m < args.max_comp:
            gmm.split(rng)
            nll = gmm.nll(Z)
            print(f"epoch {epoch:4d}  SPLIT  GMM={gmm.m}  nll={nll:.4f}", flush=True)
            snapshot("split", nll, force_board=True)
            continue
        tag = "em*" if improved else "em"
        if epoch % 10 == 0 or improved and epoch < 20:
            print(f"epoch {epoch:4d}  {tag:<6} GMM={gmm.m}  nll={nll:.4f}", flush=True)
        snapshot(tag, nll)

    snapshot("stop", gmm.nll(Z), force_board=True)
    report = {
        "formula": gen.formula(),
        "n_field": pf.nparams(),
        "k": prior.k,
        "n_comp": gmm.m,
        "nll": gmm.nll(Z),
        "epochs": epoch,
        "secs": time.time() - t0,
        "board": str(watch_dir / "board.png"),
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print("field frozen. trained GMM on z only.", flush=True)


if __name__ == "__main__":
    main()
