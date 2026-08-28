#!/usr/bin/env python3
"""Generate in the thin field (low modes only). Frozen codec is just the rasterizer."""

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
sys.path.insert(0, str(ROOT / "gen"))

from data import download_picsum, list_images, load_rgb  # noqa: E402
from scalefield import PrimeField, save_png  # noqa: E402
from thin import (  # noqa: E402
    DiagGMM,
    GeneratorThin,
    LatentPrior,
    PhaseUnfold,
    keep_through,
    pack,
    residual_decompose,
)
from watch import grid, open_viewer, write_board, write_placeholder  # noqa: E402

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


def gather_reals(n: int, size: int) -> list:
    raw = HERE / "data" / "real_raw"
    paths = download_picsum(raw, n, size=max(size, 128), start_seed=200)
    imgs = []
    for p in paths:
        try:
            imgs.append(load_rgb(p, size=size))
        except OSError:
            continue
    v1_train = ROOT / "data" / "train"
    if v1_train.is_dir():
        for p in list_images(v1_train):
            if p.name.startswith("real"):
                imgs.append(load_rgb(p, size=size))
    print(f"  photos n={len(imgs)}", flush=True)
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-bank", type=Path, default=ROOT / "gen2/runs/picsum540.field.npz")
    ap.add_argument("--out-dir", type=Path, default=HERE / "runs/night")
    ap.add_argument(
        "--through",
        type=int,
        default=11,
        help="keep full p×p for primes <= this (elemental oscillators), zero larger",
    )
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--max-comp", type=int, default=12)
    ap.add_argument("--split-every", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--board-every", type=int, default=8)
    ap.add_argument("--tile", type=int, default=96)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--n-real", type=int, default=500)
    ap.add_argument("--size", type=int, default=96)
    args = ap.parse_args()

    _assert_alone()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    watch_dir = args.out_dir / "watch"
    watch_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    board = write_placeholder(watch_dir, "thin field: full small primes as basal sines…")
    viewer = None if args.no_viewer else open_viewer(board)
    if viewer:
        print(f"viewer pid={viewer.pid}  {board}", flush=True)

    pf = PrimeField.load(str(args.from_bank))
    mask = keep_through(pf.primes, args.through)
    if args.n_real > 0:
        write_placeholder(watch_dir, "peeling photos into residual primes 1 then 2 then 3…")
        imgs = gather_reals(args.n_real, args.size)
        print(f"decompose {len(imgs)} photos, one prime at a time (not joint LS)", flush=True)
        bank = residual_decompose(pf, imgs, args.through)
        pf.bank = bank
    else:
        if pf.bank is None:
            raise SystemExit("need bank or --n-real")
        bank = np.asarray(pf.bank, dtype=np.float64)
        print(f"reuse bank {bank.shape} (already peeled)", flush=True)
    thin = pack(bank, mask)
    print(
        f"residual bank {bank.shape}  thin {thin.shape}  keep {int(mask.sum())}  "
        f"primes ≤ {args.through} peeled in order",
        flush=True,
    )
    # show one photo split into phases so the breakup is visible
    pf._apply_params(bank[0])
    panels = []
    for p in pf.primes:
        if p > args.through:
            break
        layer = np.clip(pf.layer_field(p, args.size, args.size), 0.0, 1.0)
        panels.append((layer * 255 + 0.5).astype(np.uint8))
    try:
        save_png(str(watch_dir / "phases.png"), grid(panels, cols=len(panels)))
    except Exception:
        pass
    unf, _ust = PhaseUnfold.fit(thin, pf.primes, args.through, lam=0.05)
    pfx = thin[:, : unf.n_prefix]
    prior = LatentPrior.fit(pfx, k=min(args.k, pfx.shape[1] - 1), energy=1.0)
    Z = np.stack([prior.encode(t) for t in pfx])
    gmm = DiagGMM.from_data(Z)
    gen = GeneratorThin(pf, prior, gmm, mask, m=args.through, unf=unf)
    print(gen.formula(), flush=True)

    rng = np.random.default_rng(1)
    t0 = time.time()
    epoch = 0
    max_ep = args.epochs if args.epochs > 0 else 10**9
    metrics = args.out_dir / "metrics.jsonl"

    def snap(tag, force=False):
        nll = gmm.nll(Z)
        status = {
            "epoch": epoch,
            "tag": tag,
            "kind": "thin",
            "nll": nll,
            "n_comp": gmm.m,
            "n_data": int(len(bank)),
            "keep": int(mask.sum()),
            "k": prior.k,
            "formula": gen.formula(),
        }
        with metrics.open("a") as f:
            f.write(json.dumps(status) + "\n")
        if force or epoch % args.board_every == 0 or epoch < 4:
            write_board(gen, watch_dir, status, tile=args.tile, seed=epoch, backend=args.backend)
        gen.save(str(args.out_dir / "scalegen"))
        return nll

    nll = snap("thin-fit", force=True)
    print(f"epoch {epoch:4d}  GMM={gmm.m}  nll={nll:.3f}  n={len(bank)}", flush=True)

    while not STOP and epoch < max_ep:
        epoch += 1
        nll = gmm.em_step(Z)
        if epoch % args.split_every == 0 and gmm.m < args.max_comp:
            gmm.split(rng)
            nll = snap("split", force=True)
            print(f"epoch {epoch:4d}  SPLIT GMM={gmm.m}  nll={nll:.3f}", flush=True)
            continue
        if epoch % 20 == 0:
            print(f"epoch {epoch:4d}  em  GMM={gmm.m}  nll={nll:.3f}", flush=True)
        snap("em")

    snap("stop", force=True)
    print(f"stop  epochs={epoch}  GMM={gmm.m}  secs={time.time()-t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
