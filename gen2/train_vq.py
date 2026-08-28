#!/usr/bin/env python3
"""Residual codebook unfold. Sample structure tokens, don't regress texture."""

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

from data import download_picsum, list_images, load_rgb  # noqa: E402
from scalefield import PrimeField  # noqa: E402
from vq import DiagGMM, GeneratorVQ, ScaleVQ  # noqa: E402
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


def gather_reals(n: int, size: int) -> list:
    raw = HERE / "data" / "real_raw"
    print(f"picsum stills → {n} @ {size} (reuse cache, seed 200)…", flush=True)
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
    print(f"  real cloud n={len(imgs)}", flush=True)
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-bank", type=Path, default=ROOT / "gen2/runs/good-reals/scalegen.field.npz")
    ap.add_argument("--codec", type=Path, default=ROOT / "v1/runs/live/best.npz")
    ap.add_argument("--out-dir", type=Path, default=HERE / "runs/night")
    ap.add_argument("--n-real", type=int, default=0, help="if >0, encode this many picsum stills (new bank)")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--prefix-last", type=int, default=5)
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--k-a", type=int, default=12)
    ap.add_argument("--k-z", type=int, default=24)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--max-comp", type=int, default=10)
    ap.add_argument("--split-every", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--board-every", type=int, default=10)
    ap.add_argument("--tile", type=int, default=80)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--resume", action="store_true", help="load scalegen.vq.npz instead of refitting books")
    args = ap.parse_args()

    _assert_alone()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    watch_dir = args.out_dir / "watch"
    watch_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    board = write_placeholder(watch_dir, "fitting residual codebooks (structure tokens)…")
    viewer = None if args.no_viewer else open_viewer(board)
    if viewer:
        print(f"viewer pid={viewer.pid}  {board}", flush=True)

    if args.n_real > 0:
        if not args.codec.exists():
            raise SystemExit(f"need frozen codec at {args.codec}")
        write_placeholder(watch_dir, f"encoding {args.n_real} stills into frozen field…")
        imgs = gather_reals(args.n_real, args.size)
        pf = PrimeField.load(str(args.codec))
        print(f"frozen codec N={pf.nparams()}  encoding {len(imgs)} (good-reals untouched)", flush=True)
        stats_enc = pf.evaluate(imgs, lam_l2=1e-4, chord_steps=0)
        bank = stats_enc["bank"]
        pf.bank = bank
        print(f"  encode mse={stats_enc['mse']:.4f}  bank={bank.shape}", flush=True)
    else:
        pf = PrimeField.load(str(args.from_bank))
        if pf.bank is None or len(pf.bank) < 32:
            raise SystemExit("need populated bank")
        bank = np.asarray(pf.bank, dtype=np.float64)
        print(f"bank {bank.shape}  (good-reals frozen, codec untouched)", flush=True)

    resume_path = args.out_dir / "scalegen.vq.npz"
    gmm_path = args.out_dir / "scalegen.gmm.npz"
    if args.resume and resume_path.exists():
        vq = ScaleVQ.load(str(resume_path))
        rng_split = np.random.default_rng(0)
        n = len(bank)
        idx = rng_split.permutation(n)
        n_te = max(32, int(round(n * 0.2)))
        te = idx[:n_te]
        Cte = bank[te]
        A = vq.encode_a(bank)
        gmm = DiagGMM.load(str(gmm_path)) if gmm_path.exists() else DiagGMM.from_data(A)
        sc0 = vq.hold_scores(Cte)
        stats = {"mean_acc": sc0["mean_acc"], "n_hold": int(len(te))}
        print(f"resume {resume_path}  holdAcc={sc0['mean_acc']:.3f}  clfR2={sc0['mean_clf']:.3f}", flush=True)
    else:
        vq, stats = ScaleVQ.fit(
            bank, pf.primes, prefix_last=args.prefix_last, K=args.K, k_a=args.k_a, k_z=args.k_z
        )
        te = stats["te_idx"]
        Cte = bank[te]
        A = vq.encode_a(bank)
        gmm = DiagGMM.from_data(A)
    gen = GeneratorVQ(pf, vq, gmm)
    print(gen.formula(), flush=True)

    rng = np.random.default_rng(1)
    t0 = time.time()
    epoch = 0
    max_ep = args.epochs if args.epochs > 0 else 10**9
    metrics = args.out_dir / "metrics.jsonl"
    best_acc = stats["mean_acc"]
    best_W = [(h.W.copy(), h.b.copy()) for h in vq.heads]

    def snap(tag, force=False, extra=None):
        sc = vq.hold_scores(Cte)
        for h in vq.heads:
            h.r2_vq = sc["r2_vq"][h.p]
            h.r2_clf = sc["r2_clf"][h.p]
            h.acc = sc["acc"][h.p]
        nll = gmm.nll(A)
        status = {
            "epoch": epoch,
            "tag": tag,
            "kind": "embed",
            "nll": nll,
            "n_comp": gmm.m,
            "n_data": int(len(bank)),
            "n_hold": int(len(te)),
            "mean_hold_r2": sc["mean_clf"],
            "vq_ceiling_r2": sc["mean_vq"],
            "mean_acc": sc["mean_acc"],
            "chance": 1.0 / vq.K,
            "acc": sc["acc"],
            "sgd_loss": extra,
            "formula": gen.formula(),
        }
        with metrics.open("a") as f:
            f.write(json.dumps(status) + "\n")
        if force or epoch % args.board_every == 0 or epoch < 3:
            write_board(gen, watch_dir, status, tile=args.tile, seed=epoch, backend=args.backend)
        gen.save(str(args.out_dir / "scalegen"))
        return sc

    sc = snap("vq-fit", force=True)
    print(
        f"epoch {epoch:4d}  holdAcc={sc['mean_acc']:.3f}  "
        f"clfR2={sc['mean_clf']:.3f}  vqR2={sc['mean_vq']:.3f}",
        flush=True,
    )

    while not STOP and epoch < max_ep:
        epoch += 1
        loss = vq.sgd_step(bank, lr=args.lr, l2=1e-3, rng=rng, batch=48)
        nll = gmm.em_step(A)
        if epoch % args.split_every == 0 and gmm.m < args.max_comp:
            gmm.split(rng)
        if epoch % 20 == 0:
            sc_now = vq.hold_scores(Cte)
            if sc_now["mean_acc"] > best_acc:
                best_acc = sc_now["mean_acc"]
                best_W = [(h.W.copy(), h.b.copy()) for h in vq.heads]
            if epoch % 40 == 0:
                print(
                    f"epoch {epoch:4d}  ce={loss:.3f}  holdAcc={sc_now['mean_acc']:.3f}  "
                    f"clfR2={sc_now['mean_clf']:.3f}  bestAcc={best_acc:.3f}  GMM={gmm.m}",
                    flush=True,
                )
        snap("sgd", extra=loss)

    for h, (W, b) in zip(vq.heads, best_W):
        h.W, h.b = W, b
    snap("stop", force=True)
    print(f"stop  epochs={epoch}  bestAcc={best_acc:.3f}  secs={time.time()-t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
