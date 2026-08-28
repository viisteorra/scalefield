#!/usr/bin/env python3
"""Prime-harmonic + phase (sin/cos) residual peel. No extra embed table."""

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
from elemental import (  # noqa: E402
    EL_PRIMES,
    GeneratorEl,
    n_el,
    offsets_el,
    raster_el,
    residual_fit,
)
from scalefield import PrimeField, save_png  # noqa: E402
from scalegen import DiagGMM, LatentPrior  # noqa: E402
from thin import PhaseUnfold, _r2, _ridge  # noqa: E402
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
    v1 = ROOT / "data" / "train"
    if v1.is_dir():
        for p in list_images(v1):
            if p.name.startswith("real"):
                imgs.append(load_rgb(p, size=size))
    print(f"  photos n={len(imgs)}", flush=True)
    return imgs


def fit_phase(thin: np.ndarray, n_prefix: int, lam: float, hold: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(thin)
    idx = rng.permutation(n)
    n_te = max(32, int(round(n * hold)))
    te, tr = idx[:n_te], idx[n_te:]
    Ttr, Tte = thin[tr], thin[te]
    steps = []
    print(f"  harmonic unfold  train={len(tr)} hold={len(te)}  prefix={n_prefix}", flush=True)
    for p, start, nn in offsets_el():
        if start < n_prefix:
            continue
        W, b = _ridge(Ttr[:, :start], Ttr[:, start : start + nn], lam=lam)
        ytr = Ttr[:, :start] @ W.T + b
        yte = Tte[:, :start] @ W.T + b
        r_tr = _r2(Ttr[:, start : start + nn], ytr)
        r_te = _r2(Tte[:, start : start + nn], yte)
        std = np.maximum(np.std(Ttr[:, start : start + nn] - ytr, axis=0), 1e-4)
        steps.append(
            {"p": p, "start": start, "n": nn, "W": W, "b": b, "std": std, "r2_train": r_tr, "r2_hold": r_te}
        )
        print(f"    p={p:2d}  {nn} amps | {start}  R² train={r_tr:.3f} hold={r_te:.3f}", flush=True)
    unf = PhaseUnfold((1,) + EL_PRIMES, 13, steps, thin.shape[1])
    unf.n_prefix = n_prefix
    return unf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", type=Path, default=ROOT / "v1/runs/live/best.npz")
    ap.add_argument("--out-dir", type=Path, default=HERE / "runs/night")
    ap.add_argument("--n-real", type=int, default=500)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--max-comp", type=int, default=10)
    ap.add_argument("--split-every", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--board-every", type=int, default=8)
    ap.add_argument("--tile", type=int, default=96)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()

    _assert_alone()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    watch_dir = args.out_dir / "watch"
    watch_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    board = write_placeholder(watch_dir, "1×1, 2×2, 3×3, 5×5, … both phases, no skip…")
    viewer = None if args.no_viewer else open_viewer(board)
    if viewer:
        print(f"viewer pid={viewer.pid}  {board}", flush=True)

    pf = PrimeField.load(str(args.codec))
    imgs = gather_reals(args.n_real, args.size)
    print(f"fit a 4-phase p×p matrix on every prime {EL_PRIMES}", flush=True)
    bank_path = args.out_dir / "scalegen.el.npz"
    bank = None
    if bank_path.is_file():
        prev = np.load(bank_path)
        if "bank" in prev.files and prev["bank"].shape[1] == n_el():
            bank = np.array(prev["bank"], dtype=np.float64)
            print(f"  reuse bank {bank.shape}  {bank_path}", flush=True)
    if bank is None:
        bank = residual_fit(pf, imgs)
    n_prefix = sum(n for p, _s, n in offsets_el() if p <= 3)
    unf = fit_phase(bank, n_prefix, lam=args.lam, hold=0.2)
    pfx = bank[:, :n_prefix]
    prior = LatentPrior.fit(pfx, k=min(24, pfx.shape[1] - 1), energy=1.0)
    Z = np.stack([prior.encode(t) for t in pfx])
    gmm = DiagGMM.from_data(Z)
    gen = GeneratorEl(pf, prior, gmm, unf, bank)
    print(gen.formula(), flush=True)

    # one photo as DC + each prime's harmonic field
    panels = []
    a0 = np.zeros(n_el())
    a0[0] = bank[0, 0]
    panels.append(raster_el(pf, a0, args.size, args.size))
    for p, start, n in offsets_el():
        if p == 1:
            continue
        one = np.zeros(n_el())
        one[start : start + n] = bank[0, start : start + n]
        panels.append(raster_el(pf, one, args.size, args.size))
    save_png(str(watch_dir / "phases.png"), grid(panels, cols=len(panels)))

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
            "kind": "elem",
            "nll": nll,
            "n_comp": gmm.m,
            "n_data": int(len(bank)),
            "formula": gen.formula(),
        }
        with metrics.open("a") as f:
            f.write(json.dumps(status) + "\n")
        if force or epoch % args.board_every == 0 or epoch < 3:
            write_board(gen, watch_dir, status, tile=args.tile, seed=epoch, backend=args.backend)
        gen.save(str(args.out_dir / "scalegen"))
        return nll

    nll = snap("el-fit", force=True)
    print(f"epoch {epoch:4d}  GMM={gmm.m}  nll={nll:.3f}", flush=True)

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
    print(f"stop  epochs={epoch}  secs={time.time()-t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
