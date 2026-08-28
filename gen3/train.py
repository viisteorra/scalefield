#!/usr/bin/env python3
"""
ScaleGen3 — one trainer. Frozen v1 field. Unfold primes 1→31.

Does not touch gen/runs/cool-gmm or gen2/runs/good-reals.
Reuses the populated c-bank from the frozen gen2 field (no second encode).
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
from unfold import DiagGMM, GeneratorV3, Unfold  # noqa: E402
from watch import open_viewer, write_board, write_placeholder  # noqa: E402

STOP = False


def _stop(signum, _frame):
    global STOP
    STOP = True
    print(f"\nsignal {signum}; save after epoch", flush=True)


def _assert_alone():
    """Refuse to start if another python train.py is already on this box."""
    import os

    me = os.getpid()
    others = []
    proc = Path("/proc")
    for p in proc.iterdir():
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
        if not args:
            continue
        if "python" not in args[0]:
            continue
        if any(a.endswith("train.py") or a == "train.py" for a in args):
            others.append(f"pid={pid} {' '.join(args[:8])}")
    if others:
        raise SystemExit(
            "another trainer is running — one at a time on this rig:\n  " + "\n  ".join(others)
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--field",
        type=Path,
        default=ROOT / "gen2/runs/good-reals/scalegen.field.npz",
        help="frozen populated field+bank (gen2 freeze). v1 best.npz also works.",
    )
    ap.add_argument("--out-dir", type=Path, default=HERE / "runs/live")
    ap.add_argument("--prefix-last", type=int, default=3)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--max-comp", type=int, default=10)
    ap.add_argument("--split-every", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--board-every", type=int, default=6)
    ap.add_argument("--tile", type=int, default=64)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()

    _assert_alone()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    watch_dir = args.out_dir / "watch"
    watch_dir.mkdir(parents=True, exist_ok=True)

    if not args.field.exists():
        raise SystemExit(f"need frozen field+bank at {args.field}")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    board = write_placeholder(watch_dir, "fitting unfold 1→31 on populated bank…")
    viewer = None if args.no_viewer else open_viewer(board)
    if viewer:
        print(f"viewer pid={viewer.pid}  {board}", flush=True)

    pf = PrimeField.load(str(args.field))
    if pf.bank is None or len(pf.bank) < 8:
        raise SystemExit(f"{args.field} has no populated bank — encode once, freeze, then unfold")
    bank = np.asarray(pf.bank, dtype=np.float64)
    print(
        f"frozen field N={pf.nparams()}  bank={bank.shape}  primes={pf.primes}  "
        f"(gen2/cool-gmm untouched)",
        flush=True,
    )

    t_fit = time.time()
    unf = Unfold.fit(bank, pf.primes, prefix_last=args.prefix_last, lam=args.lam)
    print(f"  maps nparams={unf.nparams()}  fit {time.time()-t_fit:.1f}s", flush=True)
    _, r2map = unf.predict_bank(bank)
    print("  per-prime R²:", {p: f"{d['r2']:.3f}" for p, d in r2map.items()}, flush=True)

    Zp = bank[:, : unf.n_prefix]
    gmm = DiagGMM.from_data(Zp)
    gen = GeneratorV3(pf, unf, gmm)
    print(gen.formula(), flush=True)

    rng = np.random.default_rng(1)
    t0 = time.time()
    epoch = 0
    max_ep = args.epochs if args.epochs > 0 else 10**9
    metrics = args.out_dir / "metrics.jsonl"

    def snap(tag: str, force=False):
        nll = gmm.nll(Zp)
        status = {
            "epoch": epoch,
            "tag": tag,
            "nll": nll,
            "n_comp": gmm.m,
            "n_data": int(len(bank)),
            "n_prefix": unf.n_prefix,
            "prefix_last": unf.prefix_last,
            "n_maps": unf.nparams(),
            "r2": {int(s.p): s.r2 for s in unf.steps},
            "formula": gen.formula(),
        }
        with metrics.open("a") as f:
            f.write(json.dumps(status) + "\n")
        if force or epoch % args.board_every == 0 or epoch < 3:
            write_board(gen, watch_dir, status, tile=args.tile, seed=epoch, backend=args.backend)
        gen.save(str(args.out_dir / "scalegen"))
        return nll

    nll = snap("unfold-fit", force=True)
    print(f"epoch {epoch:4d}  GMM={gmm.m}  nll={nll:.3f}  prefix={unf.n_prefix}", flush=True)

    while not STOP and epoch < max_ep:
        epoch += 1
        nll = gmm.em_step(Zp)
        if epoch % args.split_every == 0 and gmm.m < args.max_comp:
            gmm.split(rng)
            nll = snap("split", force=True)
            print(f"epoch {epoch:4d}  SPLIT GMM={gmm.m}  nll={nll:.3f}", flush=True)
            continue
        if epoch % 12 == 0:
            print(f"epoch {epoch:4d}  em     GMM={gmm.m}  nll={nll:.3f}", flush=True)
        snap("em")

    snap("stop", force=True)
    print(f"stop  epochs={epoch}  GMM={gmm.m}  secs={time.time()-t0:.1f}", flush=True)
    print(f"frozen stills still at {ROOT / 'gen2/runs/good-reals'}", flush=True)
    print(f"trippy GMM still at {ROOT / 'gen/runs/cool-gmm'}", flush=True)


if __name__ == "__main__":
    main()
