#!/usr/bin/env python3
"""Shared abstract code. One trainer. Hold-out + permute R² are the score."""

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

from abstract import AbstractCore, DiagGMM, GeneratorAbs  # noqa: E402
from scalefield import PrimeField  # noqa: E402
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
        if args and "python" in args[0] and any("train" in a for a in args):
            others.append(f"pid={pid} {' '.join(args[:8])}")
    if others:
        raise SystemExit("another trainer is running — one at a time:\n  " + "\n  ".join(others))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-bank", type=Path, default=ROOT / "gen2/runs/good-reals/scalegen.field.npz")
    ap.add_argument("--out-dir", type=Path, default=HERE / "runs/night")
    ap.add_argument("--prefix-last", type=int, default=7)
    ap.add_argument("--k-a", type=int, default=16)
    ap.add_argument("--k-hid", type=int, default=24)
    ap.add_argument("--lam", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--max-comp", type=int, default=10)
    ap.add_argument("--split-every", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--board-every", type=int, default=12)
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

    board = write_placeholder(watch_dir, "fitting shared abstract code (coarse → a → all primes)…")
    viewer = None if args.no_viewer else open_viewer(board)
    if viewer:
        print(f"viewer pid={viewer.pid}  {board}", flush=True)

    pf = PrimeField.load(str(args.from_bank))
    if pf.bank is None or len(pf.bank) < 32:
        raise SystemExit(f"{args.from_bank} needs a populated bank")
    bank = np.asarray(pf.bank, dtype=np.float64)
    print(f"bank {bank.shape}  N={pf.nparams()}  (good-reals frozen)", flush=True)

    core, stats = AbstractCore.fit(
        bank,
        pf.primes,
        prefix_last=args.prefix_last,
        k_a=args.k_a,
        k_hid=args.k_hid,
        lam=args.lam,
    )
    te = stats["te_idx"]
    Cte = bank[te]
    A = core.encode_a(bank)
    gmm = DiagGMM.from_data(A)
    gen = GeneratorAbs(pf, core, gmm)
    print(gen.formula(), flush=True)

    rng = np.random.default_rng(1)
    t0 = time.time()
    epoch = 0
    max_ep = args.epochs if args.epochs > 0 else 10**9
    metrics = args.out_dir / "metrics.jsonl"
    best_hold = stats["mean_hold"]
    best_snap = (core.Wh.copy(), core.bh.copy(), core.Wa.copy(), core.ba.copy(),
                 [(h.U.copy(), h.d.copy()) for h in core.heads])

    def snap(tag: str, force=False, extra=None):
        nll = gmm.nll(A)
        hold = core.hold_r2(Cte)
        perm = core.permute_r2(Cte)
        mh = float(np.mean(list(hold.values())))
        mp = float(np.mean(list(perm.values())))
        for h in core.heads:
            h.r2_hold = hold[h.p]
        status = {
            "epoch": epoch,
            "tag": tag,
            "kind": "abstract",
            "nll": nll,
            "n_comp": gmm.m,
            "n_data": int(len(bank)),
            "n_hold": int(len(te)),
            "mean_hold_r2": mh,
            "permute_r2": mp,
            "r2_hold": {int(k): v for k, v in hold.items()},
            "sgd_loss": extra,
            "formula": gen.formula(),
        }
        with metrics.open("a") as f:
            f.write(json.dumps(status) + "\n")
        if force or epoch % args.board_every == 0 or epoch < 3:
            write_board(gen, watch_dir, status, tile=args.tile, seed=epoch, backend=args.backend)
        gen.save(str(args.out_dir / "scalegen"))
        return nll, mh, mp

    nll, mh, mp = snap("abs-fit", force=True)
    print(f"epoch {epoch:4d}  holdR2={mh:.3f}  permR2={mp:.3f}  GMM={gmm.m}  nll={nll:.3f}", flush=True)

    while not STOP and epoch < max_ep:
        epoch += 1
        loss = core.sgd_step(bank, lr=args.lr, l2=args.l2, rng=rng, batch=48)
        # refresh A for GMM occasionally
        if epoch % 20 == 0:
            A = core.encode_a(bank)
        nll = gmm.em_step(A)
        if epoch % args.split_every == 0 and gmm.m < args.max_comp:
            gmm.split(rng)
        if epoch % 25 == 0:
            hold = core.hold_r2(Cte)
            mh_now = float(np.mean(list(hold.values())))
            if mh_now > best_hold:
                best_hold = mh_now
                best_snap = (core.Wh.copy(), core.bh.copy(), core.Wa.copy(), core.ba.copy(),
                             [(h.U.copy(), h.d.copy()) for h in core.heads])
            if epoch % 50 == 0:
                mp_now = float(np.mean(list(core.permute_r2(Cte).values())))
                print(
                    f"epoch {epoch:4d}  sgd={loss:.4f}  holdR2={mh_now:.3f}  "
                    f"permR2={mp_now:.3f}  best={best_hold:.3f}  GMM={gmm.m}  nll={nll:.3f}",
                    flush=True,
                )
        snap("sgd", extra=loss)

    core.Wh, core.bh, core.Wa, core.ba, heads = best_snap
    for h, (U, d) in zip(core.heads, heads):
        h.U, h.d = U, d
    snap("stop", force=True)
    print(f"stop  epochs={epoch}  best_holdR2={best_hold:.3f}  secs={time.time()-t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
