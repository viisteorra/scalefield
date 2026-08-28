#!/usr/bin/env python3
"""
ScaleGen2 structure — residual unfold, not a training-set blender.

Frozen field + bank from good-reals. One trainer. Hold-out R² is the score.
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

from scalefield import PrimeField  # noqa: E402
from structure import DiagGMM, GeneratorStruct, StructUnfold  # noqa: E402
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
        if args and "python" in args[0] and any(a.endswith(".py") and "train" in a for a in args):
            others.append(f"pid={pid} {' '.join(args[:8])}")
    if others:
        raise SystemExit("another trainer is running — one at a time:\n  " + "\n  ".join(others))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--from-bank",
        type=Path,
        default=ROOT / "gen2/runs/good-reals/scalegen.field.npz",
    )
    ap.add_argument("--out-dir", type=Path, default=HERE / "runs/night")
    ap.add_argument("--prefix-last", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--hold", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-comp", type=int, default=8)
    ap.add_argument("--split-every", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--board-every", type=int, default=10)
    ap.add_argument("--tile", type=int, default=80)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()

    _assert_alone()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    watch_dir = args.out_dir / "watch"
    watch_dir.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    board = write_placeholder(watch_dir, "fitting structural unfold on hold-out split…")
    viewer = None if args.no_viewer else open_viewer(board)
    if viewer:
        print(f"viewer pid={viewer.pid}  {board}", flush=True)

    pf = PrimeField.load(str(args.from_bank))
    if pf.bank is None or len(pf.bank) < 32:
        raise SystemExit(f"{args.from_bank} needs a populated bank")
    bank = np.asarray(pf.bank, dtype=np.float64)
    print(f"bank {bank.shape}  N={pf.nparams()}  primes={pf.primes}  (good-reals frozen)", flush=True)

    unf, stats = StructUnfold.fit(
        bank, pf.primes, prefix_last=args.prefix_last, lam=args.lam, hold=args.hold
    )
    print(f"  mean hold R²={stats['mean_hold']:.3f}  maps={unf.nparams()}", flush=True)

    Zp = np.stack([unf.encode_prefix(c) for c in bank])
    gmm = DiagGMM.from_data(Zp)
    gen = GeneratorStruct(pf, unf, gmm)
    print(gen.formula(), flush=True)

    rng = np.random.default_rng(1)
    t0 = time.time()
    epoch = 0
    max_ep = args.epochs if args.epochs > 0 else 10**9
    metrics = args.out_dir / "metrics.jsonl"
    best_hold = stats["mean_hold"]
    best_W = [(s.W.copy(), s.b.copy()) for s in unf.steps]

    def snap(tag: str, force=False, extra=None):
        nll = gmm.nll(Zp)
        hold = {int(s.p): s.r2_hold for s in unf.steps}
        status = {
            "epoch": epoch,
            "tag": tag,
            "nll": nll,
            "n_comp": gmm.m,
            "n_data": int(len(bank)),
            "n_hold": stats["n_hold"],
            "mean_hold_r2": float(np.mean(list(hold.values()))),
            "r2_hold": hold,
            "r2_train": {int(s.p): s.r2_train for s in unf.steps},
            "sgd_loss": None if extra is None else extra,
            "formula": gen.formula(),
        }
        with metrics.open("a") as f:
            f.write(json.dumps(status) + "\n")
        if force or epoch % args.board_every == 0 or epoch < 3:
            write_board(gen, watch_dir, status, tile=args.tile, seed=epoch, backend=args.backend)
        gen.save(str(args.out_dir / "scalegen"))
        return nll, status["mean_hold_r2"]

    nll, mh = snap("struct-fit", force=True)
    print(f"epoch {epoch:4d}  holdR2={mh:.3f}  GMM={gmm.m}  nll={nll:.3f}", flush=True)

    while not STOP and epoch < max_ep:
        epoch += 1
        loss = unf.sgd_step(bank, lr=args.lr, batch=48, rng=rng)
        nll = gmm.em_step(Zp)
        if epoch % args.split_every == 0 and gmm.m < args.max_comp:
            gmm.split(rng)
        if epoch % 25 == 0:
            hold_map = unf.holdout_r2(bank[stats["te_idx"]])
            hold_now = float(np.mean(list(hold_map.values())))
            if hold_now > best_hold:
                best_hold = hold_now
                best_W = [(s.W.copy(), s.b.copy()) for s in unf.steps]
                for s, p in zip(unf.steps, hold_map):
                    s.r2_hold = hold_map[p]
            if epoch % 50 == 0:
                print(
                    f"epoch {epoch:4d}  sgd={loss:.4f}  holdR2={hold_now:.3f}  "
                    f"best={best_hold:.3f}  GMM={gmm.m}  nll={nll:.3f}",
                    flush=True,
                )
        snap("sgd", extra=loss)

    # restore best hold-out maps
    for s, (W, b) in zip(unf.steps, best_W):
        s.W, s.b = W, b
    snap("stop", force=True)
    print(f"stop  epochs={epoch}  best_holdR2={best_hold:.3f}  secs={time.time()-t0:.1f}", flush=True)
    print(f"blender freeze still at {ROOT / 'gen2/runs/good-reals'}", flush=True)


if __name__ == "__main__":
    main()
