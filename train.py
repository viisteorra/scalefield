#!/usr/bin/env python3
"""
Train ScaleField on a folder of images.

Default is continuous: keep refining b, then r, then a few psi, then grow
x or i (never both at once) until you kill it. Ctrl-C writes latest+best.

    python train.py --data ./data --out-dir ./runs/live
    python test.py  --ckpt ./runs/live/best.npz --data ./data
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from pathlib import Path

import numpy as np

from data import load_folder_named, resolve_splits
from scalefield import MAX_PARAMS, ScaleField, save_png, to_float_rgb

STOP = False


def _request_stop(signum, _frame):
    global STOP
    STOP = True
    print(f"\nreceived signal {signum}; will save after this epoch", flush=True)


def hstack_u8(*imgs: np.ndarray) -> np.ndarray:
    hs = [im.shape[0] for im in imgs]
    if len(set(hs)) != 1:
        raise ValueError("hstack height mismatch")
    return np.concatenate(imgs, axis=1)


def vstack_u8(*imgs: np.ndarray) -> np.ndarray:
    return np.concatenate(imgs, axis=0)


def up8(img01: np.ndarray, H: int, W: int) -> np.ndarray:
    from PIL import Image

    u8 = (np.clip(img01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    im = Image.fromarray(u8, mode="RGB").resize((W, H), Image.Resampling.NEAREST)
    return np.asarray(im)


def preview_indices(names: list[str], n: int = 4) -> list[int]:
    syn = [i for i, name in enumerate(names) if name.startswith("syn")]
    real = [i for i, name in enumerate(names) if name.startswith("real")]
    half = max(1, n // 2)
    out = syn[:half] + real[: n - half]
    if len(out) < n:
        out = list(range(min(n, len(names))))
    return out[:n]


def write_codec_preview(
    sf: ScaleField,
    images: list[np.ndarray],
    bank: np.ndarray,
    path: Path,
    names: list[str] | None = None,
    n: int = 4,
    out_h: int = 256,
) -> None:
    idxs = preview_indices(names, n) if names else list(range(min(n, len(images))))
    rows = []
    hold = sf.b.copy()
    for k in idxs:
        tgt = to_float_rgb(images[k])
        sf.b = bank[k]
        H, W = tgt.shape[:2]
        rec64 = np.clip(sf.field(H, W), 0, 1)
        orig256 = up8(tgt, out_h, out_h)
        rec64_up = up8(rec64, out_h, out_h)
        rec256 = sf.decode(out_h, out_h, stretch=False)
        rows.append(hstack_u8(orig256, rec64_up, rec256))
    sf.b = hold
    if rows:
        save_png(str(path), vstack_u8(*rows))


def atomic_save(sf: ScaleField, path: Path, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.tmp.npz"
    sf.save(str(tmp), **extra)
    tmp.replace(path)


class Trainer:
    def __init__(
        self,
        sf: ScaleField,
        train_imgs: list[np.ndarray],
        val_imgs: list[np.ndarray],
        out_dir: Path,
        lam_l1: float = 1e-3,
        lam_l2: float = 1e-4,
        max_n: int = 288,
        patience: int = 3,
        grow: bool = True,
    ):
        self.sf = sf
        self.train_imgs = train_imgs
        self.val_imgs = val_imgs
        self.train_names = []
        self.val_names = []
        self.out_dir = out_dir
        self.lam_l1 = lam_l1
        self.lam_l2 = lam_l2
        self.max_n = min(max_n, MAX_PARAMS)
        self.patience = patience
        self.grow = grow
        self.epoch = 0
        self.best_val = math.inf
        self.stall = 0
        self.phase = "b"
        self.metrics_path = out_dir / "metrics.jsonl"
        self.best_path = out_dir / "best.npz"
        self.latest_path = out_dir / "latest.npz"
        self.preview_dir = out_dir / "previews"
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        # Ridge-only: ISTA on 768-wide A is too slow for a live loop.
        self._encode_kw = dict(lam_l2=lam_l2, lam_l1=lam_l1, ista_steps=0)
        self.rng = np.random.default_rng(0)

    def _eval(self, sf: ScaleField, images: list[np.ndarray]) -> dict:
        return sf.evaluate(images, **self._encode_kw)

    def _log(self, row: dict) -> None:
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def _save(self, path: Path, val_stats: dict, train_stats: dict) -> None:
        extra = dict(
            epoch=np.int64(self.epoch),
            phase=np.array(self.phase),
            val_mse=np.float64(val_stats["mse"]),
            train_mse=np.float64(train_stats["mse"]),
            formula=np.array(self.sf.formula()),
        )
        atomic_save(self.sf, path, **extra)

    def _scorecard(self, train_s: dict, val_s: dict, tag: str, best: bool) -> dict:
        row = {
            "epoch": self.epoch,
            "phase": self.phase,
            "tag": tag,
            "train_mse": train_s["mse"],
            "val_mse": val_s["mse"],
            "val_fovea": val_s["fovea_mse"],
            "sparsity": val_s["sparsity"],
            "nparams": self.sf.nparams(),
            "x": self.sf.x,
            "i": self.sf.i,
            "r": self.sf.r,
            "best": best,
        }
        mark = "  BEST" if best else ""
        print(
            f"epoch {self.epoch:4d}  {self.phase:<7} {tag:<10}  "
            f"train={train_s['mse']:.4f}  val={val_s['mse']:.4f}  "
            f"fovea={val_s['fovea_mse']:.4f}  sparse={val_s['sparsity']:.2f}  "
            f"N={self.sf.nparams()} x={self.sf.x} i={self.sf.i} r={self.sf.r:.3f}"
            f"{mark}",
            flush=True,
        )
        return row

    def consider(self, tag: str = "fit") -> dict:
        train_s = self._eval(self.sf, self.train_imgs)
        train_bank = train_s["bank"]
        val_s = self._eval(self.sf, self.val_imgs)
        # Keep the train cloud as the saved representation.
        self.sf.bank = train_bank
        self.sf.b = train_bank.mean(axis=0)
        improved = val_s["mse"] < self.best_val - 1e-12
        if improved:
            self.best_val = val_s["mse"]
            self.stall = 0
            self._save(self.best_path, val_s, train_s)
            write_codec_preview(
                self.sf,
                self.val_imgs,
                val_s["bank"],
                self.preview_dir / "best_val.png",
                names=self.val_names,
            )
            write_codec_preview(
                self.sf,
                self.train_imgs,
                train_s["bank"],
                self.preview_dir / "best_train.png",
                names=self.train_names,
            )
        else:
            self.stall += 1
        self._save(self.latest_path, val_s, train_s)
        if improved:
            write_codec_preview(
                self.sf,
                self.val_imgs,
                val_s["bank"],
                self.preview_dir / f"epoch_{self.epoch:04d}.png",
                names=self.val_names,
            )
        row = self._scorecard(train_s, val_s, tag, best=improved)
        self._log(row)
        (self.out_dir / "formula.txt").write_text(self.sf.formula() + "\n")
        return val_s

    def try_candidate(self, cand: ScaleField, tag: str, min_delta: float | None = None) -> bool:
        """Accept cand if val MSE improves; otherwise revert."""
        if min_delta is None:
            min_delta = 1e-8
        cur = self.sf
        self.sf = cand
        val_s = self._eval(cand, self.val_imgs)
        if val_s["mse"] < self.best_val - min_delta:
            self.consider(tag=tag)
            return True
        self.sf = cur
        return False

    def _max_x(self, i: int | None = None) -> int:
        i = self.sf.i if i is None else i
        return max(1, self.max_n // (3 * max(i, 1)))

    def _max_i(self, x: int | None = None) -> int:
        x = self.sf.x if x is None else x
        return max(1, self.max_n // (3 * max(x, 1)))

    def tune_r(self) -> bool:
        self.phase = "r"
        r0 = self.sf.r
        grid = [r0 - 0.04, r0 - 0.02, r0 + 0.02, r0 + 0.04]
        grid = [float(np.clip(r, 0.40, 0.95)) for r in grid]
        seen = set()
        for r in grid:
            r = round(r, 4)
            if r in seen or abs(r - r0) < 1e-6:
                continue
            seen.add(r)
            print(f"  probe r={r:.3f}", flush=True)
            cand = self.sf.copy()
            cand.r = r
            if self.try_candidate(cand, tag=f"r={r:.3f}"):
                return True
        return False

    def tune_psi(self, n: int = 8, scale: float = 0.10) -> bool:
        """Move a few random blob centers. Covers all x, not just the first six."""
        self.phase = "psi"
        from scalefield import _offset

        for _ in range(n):
            k = int(self.rng.integers(0, self.sf.x))
            du, dv = _offset(k, self.sf.psi)
            axis = "du" if self.rng.random() < 0.5 else "dv"
            cur = du if axis == "du" else dv
            delta = float(self.rng.choice([-1.0, 1.0])) * scale
            name = f"{axis}{k}"
            cand = self.sf.copy()
            cand.psi[name] = float(np.clip(cur + delta, -0.95, 0.95))
            print(f"  probe {name}{delta:+.2f}", flush=True)
            if self.try_candidate(cand, tag=f"{name}{delta:+.2f}"):
                return True
        return False

    def try_grow(self) -> bool:
        if not self.grow:
            return False
        self.phase = "grow"
        # Nested model: extra basis cannot hurt the LS fit. Take any drop.
        raise_x = True
        if math.isfinite(self.best_val):
            # Center mush → spend on x (spec). Only raise i if fovea is already better.
            # We don't have a fresh val here; use last printed fovea via a cheap eval.
            val_s = self._eval(self.sf, self.val_imgs)
            raise_x = val_s["fovea_mse"] >= 0.9 * val_s["mse"]
        order = [self._grow_x, self._grow_i] if raise_x else [self._grow_i, self._grow_x]
        for fn in order:
            if fn(1e-8):
                return True
        return False

    def try_reshape(self) -> bool:
        """Trade one scale for more surface when the fovea is still mush."""
        if not self.grow or self.sf.i <= 2:
            return False
        self.phase = "shape"
        i2 = self.sf.i - 1
        x2 = min(self.sf.x + 2, self._max_x(i2))
        if x2 <= self.sf.x:
            return False
        if x2 * 3 * i2 > self.max_n:
            return False
        cand = self.sf.with_capacity(x=x2, i=i2)
        return self.try_candidate(cand, tag=f"x={x2},i={i2}", min_delta=1e-8)

    def _grow_x(self, min_delta: float, step: int = 1) -> bool:
        x = min(self.sf.x + step, self._max_x())
        if x <= self.sf.x or x * 3 * self.sf.i > self.max_n:
            return False
        cand = self.sf.with_capacity(x=x)
        return self.try_candidate(cand, tag=f"x={x}", min_delta=min_delta)

    def _grow_i(self, min_delta: float) -> bool:
        i = min(self.sf.i + 1, self._max_i())
        if i <= self.sf.i or self.sf.x * 3 * i > self.max_n:
            return False
        cand = self.sf.with_capacity(i=i)
        return self.try_candidate(cand, tag=f"i={i}", min_delta=min_delta)

    def rotate_tag(self) -> bool:
        self.phase = "tags"
        from scalefield import BASIS_DEFAULTS, SPIRAL_TAGS, VISUAL_TAGS

        used = [
            self.sf.tags[k] if k < len(self.sf.tags) else BASIS_DEFAULTS[k % len(BASIS_DEFAULTS)]
            for k in range(self.sf.x)
        ]
        unused = [t for t in VISUAL_TAGS if t not in used and t not in SPIRAL_TAGS]
        if not unused:
            unused = [t for t in VISUAL_TAGS if t not in SPIRAL_TAGS]
        bank = self.sf.bank
        if bank is None:
            k_weak = int(self.rng.integers(0, self.sf.x))
        else:
            strength = np.mean(np.abs(bank), axis=(0, 1, 2))
            k_weak = int(np.argmin(strength))
        for tag in unused[:3]:
            cand = self.sf.copy()
            tags = list(cand.tags)
            while len(tags) <= k_weak:
                tags.append("exp")
            if tags[k_weak] == tag:
                continue
            tags[k_weak] = tag
            cand.tags = tags
            print(f"  probe tag{k_weak}={tag}", flush=True)
            if self.try_candidate(cand, tag=f"tag{k_weak}={tag}"):
                return True
        return False

    def step(self) -> None:
        self.epoch += 1
        print(
            f"epoch {self.epoch:4d}  search    best_val={self.best_val:.4f}  "
            f"N={self.sf.nparams()} r={self.sf.r:.3f}",
            flush=True,
        )
        if self.epoch == 1:
            self.phase = "b"
            self.consider(tag="init")
            return

        # A few probes per epoch so the loop stays live. Psi covers every blob, not just k<6.
        scale = 0.10 if self.stall < 4 else 0.05
        if self.tune_psi(n=8, scale=scale):
            return
        if self.epoch % 3 == 0 and self.tune_r():
            return
        if self.epoch % 4 == 0 and self.rotate_tag():
            return
        if self.grow and (self.try_grow() or self.try_reshape()):
            return

        self.phase = "explore"
        cand = self.sf.copy()
        cand.r = float(np.clip(cand.r + float(self.rng.choice([-0.02, 0.02])), 0.40, 0.95))
        print(f"  probe r={cand.r:.3f}", flush=True)
        if self.try_candidate(cand, tag=f"r={cand.r:.3f}"):
            return
        if self.tune_psi(n=8, scale=0.03):
            return
        self.stall += 1
        print(
            f"epoch {self.epoch:4d}  hold      N={self.sf.nparams()}  "
            f"best_val={self.best_val:.4f}  stall={self.stall}  {self.sf.formula()}",
            flush=True,
        )

    def run(self, max_epochs: int | None) -> None:
        print(self.sf.formula(), flush=True)
        print(
            f"train={len(self.train_imgs)} val={len(self.val_imgs)}  "
            f"size={self.train_imgs[0].shape[0]}  max_n={self.max_n}  "
            f"out={self.out_dir}",
            flush=True,
        )
        t0 = time.time()
        while not STOP:
            if max_epochs is not None and self.epoch >= max_epochs:
                break
            self.step()
        print(
            f"stop  epochs={self.epoch}  best_val={self.best_val:.4f}  "
            f"secs={time.time() - t0:.1f}  formula={self.sf.formula()}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--preset", choices=["tiny", "small", "medium", "textured"], default="tiny")
    ap.add_argument("--x", type=int, default=None)
    ap.add_argument("--i", type=int, default=None)
    ap.add_argument("--r", type=float, default=None)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/live"))
    ap.add_argument("--epochs", type=int, default=0, help="0 = run until Ctrl-C")
    ap.add_argument("--max-n", type=int, default=288, help="capacity cap while growing")
    ap.add_argument("--no-grow", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    ap.add_argument("--lam", type=float, default=1e-3)
    args = ap.parse_args()

    train_dir, val_dir = resolve_splits(args.data)
    train_named = load_folder_named(train_dir, size=args.size)
    val_named = load_folder_named(val_dir, size=args.size)
    train_imgs = [im for _, im in train_named]
    val_imgs = [im for _, im in val_named]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        metrics = args.out_dir / "metrics.jsonl"
        if metrics.exists():
            metrics.unlink()
    latest = args.out_dir / "latest.npz"
    resumed = False
    if latest.exists() and not args.fresh:
        sf = ScaleField.load(str(latest))
        resumed = True
        print(f"resume {latest}  {sf.formula()}", flush=True)
    elif args.x is not None or args.i is not None:
        kw = dict(x=args.x or 3, i=args.i or 4)
        if args.r is not None:
            kw["r"] = args.r
        sf = ScaleField(**kw)
    else:
        sf = ScaleField.preset(args.preset)

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    trainer = Trainer(
        sf,
        train_imgs,
        val_imgs,
        args.out_dir,
        lam_l1=args.lam,
        max_n=args.max_n,
        grow=not args.no_grow,
    )
    trainer.train_names = [n for n, _ in train_named]
    trainer.val_names = [n for n, _ in val_named]
    if resumed:
        z = np.load(latest, allow_pickle=True)
        if "epoch" in z.files:
            trainer.epoch = int(z["epoch"])
        if "val_mse" in z.files:
            trainer.best_val = float(z["val_mse"])
        if "phase" in z.files:
            trainer.phase = str(z["phase"])
        print(
            f"resumed epoch={trainer.epoch} best_val={trainer.best_val:.4f} phase={trainer.phase}",
            flush=True,
        )
    trainer.run(None if args.epochs <= 0 else args.epochs)


if __name__ == "__main__":
    main()
