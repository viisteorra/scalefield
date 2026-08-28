#!/usr/bin/env python3
"""
Fit a PrimeField on a folder of images, with a live INPUT | OUTPUT board.

Primes come online one lattice at a time so you can watch the field fill in.

    python train.py --data ../data --preset full --out-dir ./runs/live
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scalefield import PrimeField, mse, save_png, to_float_rgb

STOP = False


def _request_stop(signum, _frame):
    global STOP
    STOP = True
    print(f"\nreceived signal {signum}; will save after this step", flush=True)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data import load_folder_named, resolve_splits  # noqa: E402

FONT_PATHS = (
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
)
BG = (16, 16, 20)
IN_COLOR = (120, 220, 140)
OUT_COLOR = (120, 190, 255)
MUTED = (180, 180, 190)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    paths = FONT_PATHS if bold else FONT_PATHS[1:] + FONT_PATHS[:1]
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def hstack(*imgs: np.ndarray) -> np.ndarray:
    return np.concatenate(imgs, axis=1)


def vstack(*imgs: np.ndarray) -> np.ndarray:
    return np.concatenate(imgs, axis=0)


def up8(img01: np.ndarray, H: int, W: int, nearest: bool = True) -> np.ndarray:
    u8 = (np.clip(img01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    filt = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    im = Image.fromarray(u8, mode="RGB").resize((W, H), filt)
    return np.asarray(im)


def preview_indices(names: list[str], n: int = 6) -> list[int]:
    syn = [i for i, name in enumerate(names) if name.startswith("syn")]
    real = [i for i, name in enumerate(names) if name.startswith("real")]
    half = max(1, n // 2)
    out = syn[:half] + real[: n - half]
    if len(out) < n:
        out = list(range(min(n, len(names))))
    return out[:n]


def _caption(tile: np.ndarray, title: str, sub: str, color: tuple[int, int, int]) -> np.ndarray:
    h, w = tile.shape[:2]
    bar = 36
    canvas = np.zeros((h + bar, w, 3), dtype=np.uint8)
    canvas[:, :] = BG
    canvas[bar:] = tile
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    d.text((10, 4), title, font=_font(16, True), fill=color)
    if sub:
        tw = d.textlength(title, font=_font(16, True))
        d.text((14 + tw, 8), sub, font=_font(13), fill=MUTED)
    return np.asarray(im)


def _pad_row(*tiles: np.ndarray, gap: int = 10) -> np.ndarray:
    h = max(t.shape[0] for t in tiles)
    parts = []
    for i, t in enumerate(tiles):
        if t.shape[0] < h:
            z = np.full((h, t.shape[1], 3), BG, dtype=np.uint8)
            z[: t.shape[0]] = t
            t = z
        if i:
            parts.append(np.full((h, gap, 3), BG, dtype=np.uint8))
        parts.append(t)
    return np.concatenate(parts, axis=1)


def _stack_rows(*rows: np.ndarray, gap: int = 10) -> np.ndarray:
    w = max(r.shape[1] for r in rows)
    parts = []
    for i, r in enumerate(rows):
        if r.shape[1] < w:
            z = np.full((r.shape[0], w, 3), BG, dtype=np.uint8)
            z[:, : r.shape[1]] = r
            r = z
        if i:
            parts.append(np.full((gap, w, 3), BG, dtype=np.uint8))
        parts.append(r)
    return np.concatenate(parts, axis=0)


def _header(width: int, lines: list[str]) -> np.ndarray:
    canvas = np.full((64, width, 3), BG, dtype=np.uint8)
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    d.text((14, 8), lines[0], font=_font(18, True), fill=(240, 240, 245))
    if len(lines) > 1:
        d.text((14, 34), lines[1], font=_font(13), fill=MUTED)
    return np.asarray(im)


def _atomic_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.tmp.png"
    save_png(str(tmp), rgb)
    tmp.replace(path)


def write_html(out_dir: Path, rows: list[dict], status: dict, stamp: int) -> None:
    cards = []
    for r in rows:
        cards.append(
            f"""
<div class="pair">
  <figure>
    <img src="{r['in_rel']}?t={stamp}" alt="input">
    <figcaption>INPUT  {r['name']}</figcaption>
  </figure>
  <figure>
    <img src="{r['out_rel']}?t={stamp}" alt="output">
    <figcaption>OUTPUT  mse={r['mse']:.4f}</figcaption>
  </figure>
  <figure>
    <img src="{r['hi_rel']}?t={stamp}" alt="field">
    <figcaption>FIELD  {r.get('hi_label', '360')}</figcaption>
  </figure>
</div>"""
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="2">
<title>ScaleField v1 watch</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{
    margin: 0; background: #111318; color: #ececf1;
    font-family: ui-sans-serif, system-ui, sans-serif;
  }}
  header {{
    position: sticky; top: 0; padding: 18px 24px 14px;
    background: #111318ee; border-bottom: 1px solid #2a2d36;
  }}
  header h1 {{ margin: 0 0 6px; font-size: 18px; font-weight: 650; }}
  header p {{ margin: 0; color: #9aa0ae; font-size: 13px; }}
  .mse {{ color: #8fd99b; }}
  main {{
    padding: 22px 24px 48px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 28px 32px;
  }}
  .pair {{ display: flex; gap: 10px; align-items: flex-start; margin: 0; }}
  figure {{ margin: 0; }}
  figcaption {{
    padding: 8px 2px 0; font-size: 12px; letter-spacing: .08em;
    color: #9ec4ff; text-transform: uppercase;
  }}
  img {{
    width: 220px; height: 220px; object-fit: cover; background: #000;
    image-rendering: pixelated; border-radius: 4px;
  }}
</style>
</head>
<body>
<header>
  <h1>ScaleField v1  <span class="mse">val={status.get('val_mse', float('nan')):.4f}</span>
      N={status.get('nparams','?')}  stage {status.get('stage','?')}</h1>
  <p>{status.get('formula','')}</p>
</header>
<main>
{''.join(cards)}
</main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html)


def write_watch_board(
    pf: PrimeField,
    images: list[np.ndarray],
    names: list[str],
    bank: np.ndarray,
    out_dir: Path,
    idxs: list[int],
    status: dict,
    tile: int = 220,
    hi: int = 360,
    backend: str = "cpu",
) -> Path:
    """INPUT | OUTPUT (train res) | FIELD (hi-res raster)."""
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    row_tiles = []
    html_rows = []
    hold = pf.s.copy() if pf.src > 0 else pf.c.copy()
    mses = []
    for j, k in enumerate(idxs):
        tgt = to_float_rgb(images[k])
        pf._apply_params(bank[k])
        rec = np.clip(pf.field(*tgt.shape[:2]), 0, 1)
        err = mse(rec, tgt)
        mses.append(err)
        rec_hi = pf.decode(hi, hi, stretch=False, backend=backend)
        if rec_hi.dtype != np.uint8:
            rec_hi = (np.clip(rec_hi, 0, 1) * 255).astype(np.uint8)
        inn = up8(tgt, tile, tile, nearest=True)
        out = up8(rec, tile, tile, nearest=True)
        hi_u = up8(rec_hi / 255.0, tile, tile, nearest=False)
        inn_c = _caption(inn, "INPUT", names[k], IN_COLOR)
        out_c = _caption(out, "OUTPUT", f"mse={err:.4f}", OUT_COLOR)
        hi_c = _caption(hi_u, f"FIELD {hi}", f"{hi}×{hi}", OUT_COLOR)
        row_tiles.append(_pad_row(inn_c, out_c, hi_c))

        in_path = pairs_dir / f"{j:02d}_input.png"
        out_path = pairs_dir / f"{j:02d}_output.png"
        hi_path = pairs_dir / f"{j:02d}_field.png"
        _atomic_png(in_path, inn)
        _atomic_png(out_path, out)
        _atomic_png(hi_path, hi_u)
        html_rows.append(
            dict(
                name=names[k],
                mse=err,
                in_rel=f"pairs/{in_path.name}",
                out_rel=f"pairs/{out_path.name}",
                hi_rel=f"pairs/{hi_path.name}",
                hi_label=str(hi),
            )
        )
    if pf.src > 0:
        pf.s = hold
        pf._sync_c()
    else:
        pf.c = hold
    status = dict(status)
    status["val_mse"] = float(np.mean(mses)) if mses else status.get("val_mse")
    if len(row_tiles) <= 3:
        board = _pad_row(*row_tiles)
    else:
        board = _stack_rows(
            _pad_row(*row_tiles[:3], gap=28),
            _pad_row(*row_tiles[3:], gap=28),
            gap=14,
        )
    head = _header(
        board.shape[1],
        [
            f"INPUT | OUTPUT | FIELD {hi}     stage {status.get('stage','?')}   "
            f"mse={status['val_mse']:.4f}   N={pf.nparams()}   "
            f"p={','.join(str(p) for p in pf.primes)}",
            pf.formula(),
        ],
    )
    gap = np.full((8, board.shape[1], 3), BG, dtype=np.uint8)
    sheet = np.concatenate([head, gap, board], axis=0)
    board_path = out_dir / "board.png"
    _atomic_png(board_path, sheet)
    write_html(out_dir, html_rows, status, stamp=int(time.time()))
    (out_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    return board_path


def write_placeholder(out_dir: Path, msg: str = "starting…") -> Path:
    canvas = np.full((280, 720, 3), BG, dtype=np.uint8)
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    d.text((24, 110), "ScaleField v1  INPUT | OUTPUT | FIELD", font=_font(22, True), fill=(240, 240, 245))
    d.text((24, 150), msg, font=_font(16), fill=MUTED)
    path = out_dir / "board.png"
    _atomic_png(path, np.asarray(im))
    return path


def open_viewer(board: Path) -> subprocess.Popen | None:
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("no display; watch the board file / html instead", flush=True)
        return None
    imv = shutil.which("imv")
    if not imv:
        print(f"no imv; open {board} or {board.parent / 'index.html'}", flush=True)
        return None
    proc = subprocess.Popen(
        [imv, "-s", "shrink", "-u", "nearest_neighbour", "-w", "ScaleField v1  INPUT | OUTPUT | FIELD", str(board)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"viewer imv pid={proc.pid}  {board}", flush=True)
    return proc


def prime_ladder(primes: tuple[int, ...]) -> list[tuple[int, ...]]:
    return [tuple(primes[:k]) for k in range(1, len(primes) + 1)]


class Trainer:
    """Fixed primes. Tweak shader knobs + coefficients until you kill it."""

    def __init__(
        self,
        pf: PrimeField,
        train_imgs,
        val_imgs,
        train_names,
        val_names,
        val_idxs,
        out_dir: Path,
        lam: float,
        backend: str,
        out_res: int,
        tile: int,
    ):
        self.pf = pf
        self.train_imgs = train_imgs
        self.val_imgs = val_imgs
        self.train_names = train_names
        self.val_names = val_names
        self.val_idxs = val_idxs
        self.out_dir = out_dir
        self.watch_dir = out_dir / "watch"
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.lam = lam
        self.chord_steps = 1
        self.backend = backend
        self.out_res = out_res
        self.tile = tile
        self.epoch = 0
        self.best_val = float("inf")
        self.stall = 0
        self.rng = np.random.default_rng(0)
        self.metrics_path = out_dir / "metrics.jsonl"
        self.t0 = time.time()

    def _status(self, tag: str, train_s: dict | None, val_s: dict, best: bool) -> dict:
        return {
            "epoch": self.epoch,
            "stage": f"ep {self.epoch} {tag}",
            "tag": tag,
            "primes": list(self.pf.primes),
            "nparams": self.pf.nparams(),
            "formula": self.pf.formula(),
            "train_mse": None if train_s is None else train_s["mse"],
            "val_mse": val_s["mse"],
            "val_fovea": val_s["fovea_mse"],
            "sparsity": val_s["sparsity"],
            "foveal": self.pf.foveal,
            "edge_gain": self.pf.edge_gain,
            "chord_gain": self.pf.chord_gain,
            "lam": self.lam,
            "chord_steps": self.chord_steps,
            "best": best,
        }

    def _log(self, status: dict) -> None:
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(status) + "\n")

    def _board(self, val_s: dict, status: dict) -> None:
        print(f"  writing board {self.out_res}px {self.backend}…", flush=True)
        write_watch_board(
            self.pf,
            self.val_imgs,
            self.val_names,
            val_s["bank"],
            self.watch_dir,
            self.val_idxs,
            status,
            tile=self.tile,
            hi=self.out_res,
            backend=self.backend,
        )

    def _save(self, path: Path, extra: dict) -> None:
        payload = dict(
            epoch=np.int64(self.epoch),
            best_val=np.float64(self.best_val),
            formula=np.array(self.pf.formula()),
        )
        payload.update(extra)
        self.pf.save(str(path), **payload)

    def consider(self, tag: str, train_s: dict, val_s: dict) -> bool:
        self.pf.bank = train_s["bank"]
        self.pf._apply_params(train_s["bank"].mean(axis=0))
        improved = val_s["mse"] < self.best_val - 1e-12
        if improved:
            self.best_val = val_s["mse"]
            self.stall = 0
            self._save(self.out_dir / "best.npz", dict(val_mse=np.float64(val_s["mse"])))
        else:
            self.stall += 1
        self._save(self.out_dir / "latest.npz", dict(val_mse=np.float64(val_s["mse"])))
        status = self._status(tag, train_s, val_s, best=improved)
        self._log(status)
        self._board(val_s, status)
        (self.out_dir / "formula.txt").write_text(self.pf.formula() + "\n")
        mark = "  BEST" if improved else ""
        print(
            f"epoch {self.epoch:4d}  {tag:<14}  "
            f"train={train_s['mse']:.5f}  val={val_s['mse']:.5f}  "
            f"fovea={val_s['fovea_mse']:.5f}  "
            f"edge={self.pf.edge_gain:.3f} chord={self.pf.chord_gain:.3f}  "
            f"fov={self.pf.foveal:.3f}  N={self.pf.nparams()}"
            f"{mark}",
            flush=True,
        )
        return improved

    def fit(self, chord_steps: int | None = None) -> tuple[dict, dict]:
        steps = self.chord_steps if chord_steps is None else chord_steps
        train_s = self.pf.evaluate(self.train_imgs, lam_l2=self.lam, chord_steps=steps)
        val_s = self.pf.evaluate(self.val_imgs, lam_l2=self.lam, chord_steps=steps)
        return train_s, val_s

    def try_candidate(self, cand: PrimeField, tag: str) -> bool:
        cur = self.pf
        self.pf = cand
        val_s = cand.evaluate(self.val_imgs, lam_l2=self.lam, chord_steps=self.chord_steps)
        if val_s["mse"] < self.best_val - 1e-8:
            train_s = cand.evaluate(self.train_imgs, lam_l2=self.lam, chord_steps=self.chord_steps)
            self.consider(tag, train_s, val_s)
            return True
        self.pf = cur
        return False

    def probe_knob(self, name: str, values: list[float]) -> bool:
        cur = float(getattr(self.pf, name))
        for v in values:
            v = float(v)
            if abs(v - cur) < 1e-9:
                continue
            print(f"  probe {name}={v:.4f} (was {cur:.4f})", flush=True)
            cand = self.pf.copy()
            setattr(cand, name, v)
            cand._A_cache.clear()
            if self.try_candidate(cand, tag=f"{name}={v:.3f}"):
                return True
        return False

    def tweak_coeffs(self, n_images: int = 10, n_dirs: int = 3, scale: float = 0.02) -> bool:
        """Random walks on per-image c. Helps the quadratic chord leftover."""
        if self.pf.bank is None:
            return False
        bank = self.pf.bank
        H, W = self.train_imgs[0].shape[:2]
        improved_any = False
        hold = self.pf.c.copy()
        for _ in range(n_images):
            k = int(self.rng.integers(0, len(self.train_imgs)))
            c = bank[k].copy()
            self.pf.c = c
            base = mse(self.pf.field(H, W), self.train_imgs[k])
            for _d in range(n_dirs):
                d = self.rng.normal(0.0, scale, size=c.shape)
                hit = False
                for s in (1.0, -1.0, 0.4):
                    self.pf.c = c + s * d
                    m = mse(self.pf.field(H, W), self.train_imgs[k])
                    if m < base - 1e-8:
                        c = self.pf.c.copy()
                        base = m
                        hit = True
                        improved_any = True
                        break
                if not hit:
                    break
            bank[k] = c
        self.pf.c = hold
        self.pf.bank = bank
        if not improved_any:
            return False
        # Re-score val with the tweaked train cloud only as a mean-field? No —
        # val has its own bank. Tweaks are train-only; re-encode val with same knobs
        # is a no-op. Report train mse drop via a cheap val re-eval of current knobs
        # (unchanged) so we still refresh the board from latest train mean.
        train_s = {
            "mse": float(
                np.mean(
                    [
                        mse(self.pf.field(*im.shape[:2], c=bank[i]), im)
                        for i, im in enumerate(self.train_imgs[:24])
                    ]
                )
            ),
            "fovea_mse": 0.0,
            "sparsity": float(np.mean(np.abs(bank) < 1e-3)),
            "bank": bank,
            "n": len(self.train_imgs),
        }
        val_s = self.pf.evaluate(self.val_imgs, lam_l2=self.lam, chord_steps=1)
        return self.consider("c-walk", train_s, val_s)

    def step(self) -> None:
        self.epoch += 1
        print(
            f"epoch {self.epoch:4d}  search    best_val={self.best_val:.5f}  "
            f"stall={self.stall}  {self.pf.formula()}",
            flush=True,
        )
        e = float(np.clip(self.pf.edge_gain, 0.0, 2.5))
        if self.probe_knob(
            "edge_gain",
            [np.clip(e + d, 0.0, 2.5) for d in (0.08, -0.08, 0.16, -0.16, 0.04, -0.04)],
        ):
            return
        g = float(np.clip(self.pf.chord_gain, 0.0, 0.8))
        if self.probe_knob(
            "chord_gain",
            [np.clip(g + d, 0.0, 0.8) for d in (0.04, -0.04, 0.08, -0.08)],
        ):
            return
        f = float(np.clip(self.pf.foveal, 0.0, 2.2))
        if self.probe_knob(
            "foveal",
            [np.clip(f + d, 0.0, 2.2) for d in (0.06, -0.06, 0.12, -0.12)],
        ):
            return

        for lam in (self.lam * 0.5, self.lam * 2.0, 3e-5, 3e-4):
            lam = float(lam)
            if abs(np.log(lam + 1e-18) - np.log(self.lam + 1e-18)) < 0.05:
                continue
            print(f"  probe lam={lam:.2e} (was {self.lam:.2e})", flush=True)
            hold = self.lam
            self.lam = lam
            val_s = self.pf.evaluate(self.val_imgs, lam_l2=lam, chord_steps=self.chord_steps)
            if val_s["mse"] < self.best_val - 1e-8:
                train_s = self.pf.evaluate(self.train_imgs, lam_l2=lam, chord_steps=self.chord_steps)
                self.consider(f"lam={lam:.1e}", train_s, val_s)
                return
            self.lam = hold

        if self.chord_steps < 4:
            nxt = self.chord_steps + 1
            print(f"  probe chord_steps={nxt}", flush=True)
            hold = self.chord_steps
            self.chord_steps = nxt
            train_s, val_s = self.fit()
            if val_s["mse"] < self.best_val - 1e-8:
                self.consider(f"gn={nxt}", train_s, val_s)
                return
            self.chord_steps = hold

        self.stall += 1
        # Keep the board alive even on a hold.
        if self.epoch % 4 == 0:
            train_s, val_s = self.fit()
            self.consider("refit", train_s, val_s)
        print(
            f"epoch {self.epoch:4d}  hold      best_val={self.best_val:.5f}  "
            f"stall={self.stall}",
            flush=True,
        )

    def run(self, max_epochs: int | None, skip_init: bool = False) -> None:
        print(self.pf.formula(), flush=True)
        print(
            f"perpetual  train={len(self.train_imgs)} val={len(self.val_imgs)}  "
            f"size={self.train_imgs[0].shape[0]}  N={self.pf.nparams()}  "
            f"out={self.out_dir}  Ctrl-C to stop",
            flush=True,
        )
        if not skip_init:
            train_s, val_s = self.fit()
            self.consider("init", train_s, val_s)
        while not STOP:
            if max_epochs is not None and self.epoch >= max_epochs:
                break
            self.step()
        train_s, val_s = self.fit(chord_steps=1)
        self.consider("stop", train_s, val_s)
        print(
            f"stop  epochs={self.epoch}  best_val={self.best_val:.5f}  "
            f"secs={time.time() - self.t0:.1f}  {self.pf.formula()}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("../data"))
    ap.add_argument("--preset", choices=list(PrimeField.PRESETS), default="sharp")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--out-res", type=int, default=360)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/live"))
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--foveal", type=float, default=1.15)
    ap.add_argument("--tile", type=int, default=220)
    ap.add_argument("--n-preview", type=int, default=6)
    ap.add_argument("--grow", action="store_true", help="ladder primes on first run (watch them come in)")
    ap.add_argument("--perpetual", action="store_true", help="keep probing shader knobs after the LS fit")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoints")
    ap.add_argument("--epochs", type=int, default=0, help="with --perpetual: 0 = until Ctrl-C")
    ap.add_argument(
        "--dwell",
        type=float,
        default=0.0,
        help="seconds to hold each grow stage (grow mode only)",
    )
    args = ap.parse_args()

    train_dir, val_dir = resolve_splits(args.data)
    train_named = load_folder_named(train_dir, size=args.size)
    val_named = load_folder_named(val_dir, size=args.size)
    train_imgs = [im for _, im in train_named]
    val_imgs = [im for _, im in val_named]
    train_names = [n for n, _ in train_named]
    val_names = [n for n, _ in val_named]
    val_idxs = preview_indices(val_names, args.n_preview)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    watch_dir = args.out_dir / "watch"
    watch_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.jsonl"
    if args.fresh and metrics_path.exists():
        metrics_path.unlink()

    backend = args.backend
    if backend == "gpu":
        from gpu import gpu_available, gpu_info

        if not gpu_available():
            print(f"gpu unavailable ({gpu_info()}); falling back to cpu", flush=True)
            backend = "cpu"
        else:
            print(f"gpu {gpu_info()}", flush=True)

    board = write_placeholder(watch_dir, "perpetual trainer — fixed primes, tweaking…")
    viewer = None if args.no_viewer else open_viewer(board)

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    latest = args.out_dir / "latest.npz"
    best = args.out_dir / "best.npz"
    resumed = False
    target = PrimeField.preset(args.preset, foveal=args.foveal)
    if not args.fresh and latest.exists():
        pf = PrimeField.load(str(latest))
        if pf.primes == target.primes:
            resumed = True
            print(f"resume {latest}  {pf.formula()}", flush=True)
        else:
            print(f"checkpoint primes {pf.primes} != {target.primes}; starting fresh", flush=True)
            pf = target
    else:
        pf = target

    print(pf.formula(), flush=True)
    print(
        f"train={len(train_imgs)} val={len(val_imgs)}  size={args.size}  "
        f"out={args.out_res}  N={pf.nparams()}  board={board}",
        flush=True,
    )
    print(f"html  {watch_dir / 'index.html'}", flush=True)

    if args.grow and not resumed:
        for si, primes in enumerate(prime_ladder(target.primes), start=1):
            if STOP:
                break
            pf = PrimeField(
                primes=primes,
                foveal=pf.foveal,
                edge_gain=pf.edge_gain,
                chord_gain=pf.chord_gain,
            )
            print(f"grow {si}  p={pf.primes}  N={pf.nparams()}", flush=True)
            train_s = pf.evaluate(train_imgs, lam_l2=args.lam)
            val_s = pf.evaluate(val_imgs, lam_l2=args.lam)
            pf.bank = train_s["bank"]
            pf.c = train_s["bank"].mean(axis=0)
            write_watch_board(
                pf, val_imgs, val_names, val_s["bank"], watch_dir, val_idxs,
                dict(stage=f"grow {si}", val_mse=val_s["mse"], nparams=pf.nparams(), formula=pf.formula()),
                tile=args.tile, hi=args.out_res, backend=backend,
            )
            print(
                f"grow {si}  train={train_s['mse']:.4f}  val={val_s['mse']:.4f}  N={pf.nparams()}",
                flush=True,
            )
            if args.dwell > 0:
                time.sleep(args.dwell)

    board_backend = "cpu" if max(pf.primes) >= 29 else backend
    board_hi = min(args.out_res, 256) if max(pf.primes) >= 29 else args.out_res
    trainer = Trainer(
        pf, train_imgs, val_imgs, train_names, val_names, val_idxs,
        args.out_dir, args.lam, board_backend, board_hi, args.tile,
    )
    if resumed:
        z = np.load(latest, allow_pickle=True)
        if "epoch" in z.files:
            trainer.epoch = int(z["epoch"])
        if "best_val" in z.files:
            trainer.best_val = float(z["best_val"])
        elif "val_mse" in z.files:
            trainer.best_val = float(z["val_mse"])
        elif best.exists():
            zb = np.load(best, allow_pickle=True)
            if "val_mse" in zb.files:
                trainer.best_val = float(zb["val_mse"])
        print(
            f"resumed epoch={trainer.epoch} best_val={trainer.best_val:.5f}",
            flush=True,
        )
    try:
        print("fitting fixed primes (one-shot LS)…", flush=True)
        train_s, val_s = trainer.fit()
        trainer.consider("fit", train_s, val_s)
        trainer._save(args.out_dir / "best.npz", dict(val_mse=np.float64(val_s["mse"])))
        if args.perpetual:
            trainer.run(None if args.epochs <= 0 else args.epochs, skip_init=True)
        else:
            print(
                f"done  train={train_s['mse']:.5f}  val={val_s['mse']:.5f}  "
                f"N={trainer.pf.nparams()}  {trainer.pf.formula()}",
                flush=True,
            )
    finally:
        previews = args.out_dir / "previews"
        previews.mkdir(exist_ok=True)
        if (watch_dir / "board.png").exists():
            shutil.copy2(watch_dir / "board.png", previews / "val_board.png")
        report = {
            "formula": trainer.pf.formula(),
            "nparams": trainer.pf.nparams(),
            "primes": list(trainer.pf.primes),
            "foveal": trainer.pf.foveal,
            "edge_gain": trainer.pf.edge_gain,
            "chord_gain": trainer.pf.chord_gain,
            "best_val": trainer.best_val,
            "epoch": trainer.epoch,
            "size": args.size,
            "out_res": args.out_res,
            "board": str(watch_dir / "board.png"),
            "html": str(watch_dir / "index.html"),
        }
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"board  {watch_dir / 'board.png'}", flush=True)
        print(f"html   {watch_dir / 'index.html'}", flush=True)
        if viewer is not None and viewer.poll() is not None:
            print("viewer closed", flush=True)


if __name__ == "__main__":
    main()
