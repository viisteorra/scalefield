#!/usr/bin/env python3
"""Rebuild an INPUT | OUTPUT image board whenever the checkpoint changes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data import load_folder_named, resolve_splits
from scalefield import ScaleField, mse, save_png, to_float_rgb

FONT_PATHS = (
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
)
BG = (16, 16, 20)
IN_COLOR = (120, 220, 140)
OUT_COLOR = (120, 190, 255)
MUTED = (180, 180, 190)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    paths = FONT_PATHS
    if bold:
        paths = ("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",) + FONT_PATHS
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _up(img01: np.ndarray, h: int, w: int) -> np.ndarray:
    u8 = (np.clip(img01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return np.asarray(
        Image.fromarray(u8, mode="RGB").resize((w, h), Image.Resampling.NEAREST)
    )


def _atomic_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.tmp.png"
    save_png(str(tmp), rgb)
    tmp.replace(path)


def _caption(tile: np.ndarray, title: str, sub: str, color: tuple[int, int, int]) -> np.ndarray:
    h, w = tile.shape[:2]
    bar = 36
    canvas = np.zeros((h + bar, w, 3), dtype=np.uint8)
    canvas[:, :] = BG
    canvas[bar:] = tile
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    d.text((10, 4), title, font=_font(16, bold=True), fill=color)
    if sub:
        tw = d.textlength(title, font=_font(16, bold=True))
        d.text((14 + tw, 8), sub, font=_font(13), fill=MUTED)
    return np.asarray(im)


def _pad_row(*tiles: np.ndarray, gap: int = 10) -> np.ndarray:
    h = max(t.shape[0] for t in tiles)
    parts = []
    for i, t in enumerate(tiles):
        if t.shape[0] < h:
            z = np.zeros((h, t.shape[1], 3), dtype=np.uint8)
            z[:, :] = BG
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


def _board_grid(groups: list[np.ndarray], groups_per_row: int = 2, gap: int = 10, group_gap: int = 28) -> np.ndarray:
    """Pack sample triplets into a landscape grid (default 6 tiles wide × 3 tall)."""
    if not groups:
        raise ValueError("no groups")
    rows = []
    for i in range(0, len(groups), groups_per_row):
        chunk = list(groups[i : i + groups_per_row])
        while len(chunk) < groups_per_row:
            h, w, _ = chunk[0].shape
            chunk.append(np.full((h, w, 3), BG, dtype=np.uint8))
        row = chunk[0]
        for t in chunk[1:]:
            spacer = np.full((row.shape[0], group_gap, 3), BG, dtype=np.uint8)
            row = np.concatenate([row, spacer, t], axis=1)
        rows.append(row)
    return _stack_rows(*rows, gap=gap)


def _header(width: int, lines: list[str]) -> np.ndarray:
    h = 64
    canvas = np.full((h, width, 3), BG, dtype=np.uint8)
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    d.text((14, 8), lines[0], font=_font(18, bold=True), fill=(240, 240, 245))
    if len(lines) > 1:
        d.text((14, 34), lines[1], font=_font(13), fill=MUTED)
    return np.asarray(im)


def _last_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    lines = metrics_path.read_text().strip().splitlines()
    if not lines:
        return {}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {}


def pick_indices(names: list[str], n: int) -> list[int]:
    syn = [i for i, name in enumerate(names) if name.startswith("syn")]
    real = [i for i, name in enumerate(names) if name.startswith("real")]
    half = max(1, n // 2)
    out = syn[:half] + real[: n - half]
    if len(out) < n:
        out = list(range(min(n, len(names))))
    return out[:n]


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
    <img src="{r['hi_rel']}?t={stamp}" alt="output hi-res">
    <figcaption>OUTPUT 256</figcaption>
  </figure>
</div>"""
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="4">
<title>ScaleField watch</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{
    margin: 0; background: #111318; color: #ececf1;
    font-family: ui-sans-serif, system-ui, sans-serif;
  }}
  header {{
    position: sticky; top: 0; padding: 18px 24px 14px;
    background: #111318ee; border-bottom: 1px solid #2a2d36;
    backdrop-filter: blur(8px);
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
  .pair {{
    display: flex; gap: 10px; align-items: flex-start;
    margin: 0; flex-wrap: nowrap;
  }}
  figure {{ margin: 0; }}
  figcaption {{
    padding: 8px 2px 0; font-size: 12px; letter-spacing: .08em;
    color: #9ec4ff; text-transform: uppercase;
  }}
  img {{
    width: 256px; height: 256px; object-fit: cover; background: #000;
    image-rendering: pixelated; border-radius: 4px;
  }}
</style>
</head>
<body>
<header>
  <h1>ScaleField  <span class="mse">val={status.get('val_mse', float('nan')):.4f}</span>
      N={status.get('nparams','?')}  ep {status.get('epoch','?')}</h1>
  <p>{status.get('formula','')}</p>
</header>
<main>
{''.join(cards)}
</main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html)


def render_once(
    ckpt: Path,
    data: Path,
    out_dir: Path,
    size: int = 64,
    tile: int = 220,
    n: int = 6,
    hi: int = 256,
) -> dict:
    if not ckpt.exists():
        alt = ckpt.parent / "latest.npz"
        if alt.exists():
            ckpt = alt
        else:
            raise FileNotFoundError(f"no checkpoint at {ckpt}")

    sf = ScaleField.load(str(ckpt))
    _, val_dir = resolve_splits(data)
    named = load_folder_named(val_dir, size=size)
    names = [n_ for n_, _ in named]
    images = [im for _, im in named]
    idxs = pick_indices(names, n)

    subset = [images[i] for i in idxs]
    bank = sf.encode_many(subset, lam_l2=1e-4, lam_l1=1e-3, ista_steps=8)

    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    row_tiles = []
    html_rows = []
    hold = sf.b.copy()
    mses = []
    for j, k in enumerate(idxs):
        tgt = to_float_rgb(images[k])
        sf.b = bank[j]
        rec = np.clip(sf.field(*tgt.shape[:2]), 0, 1)
        err = mse(rec, tgt)
        mses.append(err)
        rec_hi = sf.decode(hi, hi, stretch=False)
        inn = _up(tgt, tile, tile)
        out = _up(rec, tile, tile)
        hi_u = rec_hi if rec_hi.shape[0] == tile else _up(rec_hi / 255.0, tile, tile)
        inn_c = _caption(inn, "INPUT", names[k], IN_COLOR)
        out_c = _caption(out, "OUTPUT", f"mse={err:.4f}", OUT_COLOR)
        hi_c = _caption(hi_u, "OUTPUT 256", f"{hi}×{hi}", OUT_COLOR)
        row_tiles.append(_pad_row(inn_c, out_c, hi_c))

        in_path = pairs_dir / f"{j:02d}_input.png"
        out_path = pairs_dir / f"{j:02d}_output.png"
        hi_path = pairs_dir / f"{j:02d}_output256.png"
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
            )
        )
    sf.b = hold

    last = _last_metrics(ckpt.parent / "metrics.jsonl")
    status = {
        "formula": sf.formula(),
        "nparams": sf.nparams(),
        "x": sf.x,
        "i": sf.i,
        "r": sf.r,
        "epoch": last.get("epoch"),
        "val_mse": float(np.mean(mses)) if mses else last.get("val_mse"),
        "ckpt_val": last.get("val_mse"),
        "ckpt": str(ckpt),
        "mtime": ckpt.stat().st_mtime,
    }
    board = _board_grid(row_tiles, groups_per_row=2)
    head = _header(
        board.shape[1],
        [
            f"INPUT | OUTPUT     ep {status['epoch']}   "
            f"preview mse={status['val_mse']:.4f}   "
            f"ckpt val={last.get('val_mse', float('nan')):.4f}   "
            f"N={sf.nparams()} x={sf.x} i={sf.i}",
            sf.formula(),
        ],
    )
    gap = np.full((8, board.shape[1], 3), BG, dtype=np.uint8)
    sheet = np.concatenate([head, gap, board], axis=0)
    _atomic_png(out_dir / "board.png", sheet)
    write_html(out_dir, html_rows, status, stamp=int(time.time()))
    (out_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def loop(ckpt: Path, data: Path, out_dir: Path, interval: float, **kw) -> None:
    last = None
    print(f"watching {ckpt}  ->  {out_dir}/board.png", flush=True)
    while True:
        path = ckpt if ckpt.exists() else ckpt.parent / "latest.npz"
        mtime = path.stat().st_mtime if path.exists() else None
        if mtime != last:
            try:
                st = render_once(path, data, out_dir, **kw)
                last = path.stat().st_mtime
                print(
                    f"refresh  ep {st.get('epoch')}  preview_mse={st.get('val_mse'):.4f}  "
                    f"N={st.get('nparams')}  {st.get('formula')}",
                    flush=True,
                )
            except Exception as e:
                print(f"watch render failed: {e}", flush=True)
        time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("runs/live/best.npz"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("runs/live/watch"))
    ap.add_argument("--interval", type=float, default=4.0)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--tile", type=int, default=220)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    kw = dict(size=args.size, tile=args.tile, n=args.n)
    if args.once:
        st = render_once(args.ckpt, args.data, args.out, **kw)
        print(json.dumps({k: st[k] for k in ("epoch", "val_mse", "nparams", "formula")}, indent=2))
        return
    loop(args.ckpt, args.data, args.out, args.interval, **kw)


if __name__ == "__main__":
    main()
