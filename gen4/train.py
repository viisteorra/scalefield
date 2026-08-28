#!/usr/bin/env python3
"""
ScaleGen4 — one trainer. Frozen v1 field. Next-frame residual unfold.

Does not touch gen3 unfold, gen2 stills, or the trippy GMM.
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
sys.path.insert(0, str(ROOT / "gen3"))

from scalefield import PrimeField  # noqa: E402
from nextframe import GeneratorV4, NextFrame  # noqa: E402
from video import gather_clips  # noqa: E402
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
    ap.add_argument("--field", type=Path, default=ROOT / "v1/runs/live/best.npz")
    ap.add_argument("--out-dir", type=Path, default=HERE / "runs/live")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--max-frames", type=int, default=28)
    ap.add_argument("--lam", type=float, default=3e-2)
    ap.add_argument("--encode-lam", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--board-every", type=int, default=1)
    ap.add_argument("--tile", type=int, default=72)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()

    _assert_alone()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    watch_dir = args.out_dir / "watch"
    watch_dir.mkdir(parents=True, exist_ok=True)
    if not args.field.exists():
        raise SystemExit(f"need frozen v1 field at {args.field}")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    board = write_placeholder(watch_dir, "downloading short motion clips…")
    viewer = None if args.no_viewer else open_viewer(board)
    if viewer:
        print(f"viewer pid={viewer.pid}  {board}", flush=True)

    seqs = gather_clips(HERE, size=args.size, fps=args.fps, max_frames=args.max_frames)
    if len(seqs) < 2:
        raise SystemExit("need at least 2 clips with frames")

    pf = PrimeField.load(str(args.field))
    # drop any stills bank — this run is video frames only
    pf.bank = None
    all_imgs = []
    clip_slices: list[tuple[str, int, int]] = []
    for name, imgs in seqs:
        clip_slices.append((name, len(all_imgs), len(imgs)))
        all_imgs.extend(imgs)
    print(
        f"frozen field N={pf.nparams()}  clips={len(seqs)}  frames={len(all_imgs)}  "
        f"(unfold/stills frozen)",
        flush=True,
    )
    write_placeholder(watch_dir, f"encoding {len(all_imgs)} frames into frozen field…")
    stats = pf.evaluate(all_imgs, lam_l2=args.encode_lam, chord_steps=0)
    bank = stats["bank"]
    pf.bank = bank
    print(f"  encode mse={stats['mse']:.4f}  bank={bank.shape}", flush=True)

    prev, nxt = [], []
    for name, s, n in clip_slices:
        c = bank[s : s + n]
        prev.append(c[:-1])
        nxt.append(c[1:])
        print(f"    {name}: {n} frames → {n-1} pairs", flush=True)
    prev = np.concatenate(prev, axis=0)
    nxt = np.concatenate(nxt, axis=0)

    t_fit = time.time()
    nf = NextFrame.fit(prev, nxt, pf.primes, lam=args.lam)
    print(f"  maps nparams={nf.nparams()}  fit {time.time()-t_fit:.1f}s", flush=True)
    gen = GeneratorV4(pf, nf, clip_slices)
    print(gen.formula(), flush=True)

    t0 = time.time()
    epoch = 0
    max_ep = args.epochs if args.epochs > 0 else 10**9
    metrics = args.out_dir / "metrics.jsonl"

    def snap(tag: str, force=False):
        status = {
            "epoch": epoch,
            "tag": tag,
            "n_pairs": int(len(prev)),
            "n_clips": len(clip_slices),
            "n_frames": int(len(bank)),
            "n_maps": nf.nparams(),
            "r2": {int(s.p): s.r2 for s in nf.steps},
            "formula": gen.formula(),
        }
        with metrics.open("a") as f:
            f.write(json.dumps(status) + "\n")
        if force or epoch % args.board_every == 0:
            write_board(gen, watch_dir, status, tile=args.tile, seed=epoch, backend=args.backend)
        gen.save(str(args.out_dir / "scalegen"))
        return status

    snap("fit", force=True)
    print(f"epoch {epoch:4d}  pairs={len(prev)}  clips={len(clip_slices)}", flush=True)

    # maps are closed-form. Loop redraws rollouts on different clips so the
    # watcher stays alive without a second trainer.
    while not STOP and epoch < max_ep:
        epoch += 1
        if epoch % 8 == 0:
            print(f"epoch {epoch:4d}  rollout clip={clip_slices[epoch % len(clip_slices)][0]}", flush=True)
        snap("rollout")
        time.sleep(0.4)

    snap("stop", force=True)
    print(f"stop  epochs={epoch}  secs={time.time()-t0:.1f}", flush=True)
    print(f"unfold stills frozen at {ROOT / 'gen3/runs/good-unfold'}", flush=True)


if __name__ == "__main__":
    main()
