#!/usr/bin/env python3
"""One-parameter walk of the band-interference field. No training.

    python wallpaper.py
    python wallpaper.py --size 384 --period 3 --fps 30 --fullscreen
    kill -USR1 $(cat runs/wallpaper/wallpaper.pid)   # pause / resume
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
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

from paper_field import (  # noqa: E402
    EL_PRIMES,
    GeneratorEl,
    LiveField,
    axis_p,
    n_el,
    offsets_el,
)
STOP = False
PAUSED = False


def _stop(signum, _frame):
    global STOP
    STOP = True


def _toggle_pause(signum, _frame):
    global PAUSED
    PAUSED = not PAUSED
    print(("paused" if PAUSED else "resume") + "  (USR1)", flush=True)


def _atomic_ppm(path: Path, rgb: np.ndarray) -> None:
    """Uncompressed P6 to tmpfs. ~0.1ms; PNG was 10–20ms and capped the fps."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    payload = b"P6\n%d %d\n255\n" % (w, h) + rgb.tobytes()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def param_label(idx: int) -> str:
    for p, start, n in offsets_el():
        if start <= idx < start + n:
            if p == 1:
                return "DC"
            k = idx - start
            i, j = divmod(k, p)
            F = axis_p(p)
            return f"p={p}  ({i},{j})  {F[i]:g}×{F[j]:g}"
    return f"#{idx}"


def playlist(bank: np.ndarray, all_params: bool) -> list[int]:
    std = np.std(np.asarray(bank, dtype=np.float64), axis=0)
    if all_params:
        order = np.argsort(-std)
        return [int(i) for i in order if std[i] > 1e-5]
    n_prefix = sum(n for p, _s, n in offsets_el() if p <= 3)
    # DC last: pulsing brightness is a dull first 10s
    out: list[int] = list(range(1, n_prefix)) + [0]
    for p, start, n in offsets_el():
        if p <= 3:
            continue
        F = axis_p(p)
        ranked = []
        for i in range(p):
            for j in range(p):
                if F[i] < 8.0 and F[j] < 8.0:
                    idx = start + i * p + j
                    ranked.append((float(std[idx]), idx))
        ranked.sort(reverse=True)
        out.extend(idx for _, idx in ranked[:6])
    seen: set[int] = set()
    uniq = []
    for i in out:
        if i not in seen and std[i] > 1e-6:
            seen.add(i)
            uniq.append(i)
    return uniq


def new_scene(gen: GeneratorEl, rng: np.random.Generator) -> np.ndarray:
    if gen.gmm is not None:
        z = gen.gmm.sample(1, rng)[0]
        pfx = gen.prior.decode(z)
    else:
        pfx = rng.normal(size=gen.unf.n_prefix) * 0.2
    if pfx.size > gen.unf.n_prefix:
        pfx = pfx[: gen.unf.n_prefix]
    seed = int(rng.integers(1, 2**31 - 1))
    return gen.unf.from_prefix(pfx, temp=0.2, seed=seed)


def open_paper(path: Path, fullscreen: bool):
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return None
    imv = shutil.which("imv")
    if not imv:
        return None
    cmd = [imv, "-s", "full", "-u", "nearest_neighbour", "-w", "ScaleField  wallpaper", str(path)]
    if fullscreen:
        cmd.insert(1, "-f")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", type=Path, default=HERE / "runs/wallpaper/scalegen")
    ap.add_argument("--codec", type=Path, default=ROOT / "v1/runs/live/best.npz")
    ap.add_argument("--out", type=Path, default=Path("/dev/shm/scalefield-wallpaper.ppm"))
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--period", type=float, default=3.0, help="seconds per parameter")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--nice", type=int, default=15, help="os.nice so we never fight a trainer")
    ap.add_argument("--all", action="store_true", help="walk every coeff, loudest first")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGUSR1, _toggle_pause)
    if args.nice:
        try:
            os.nice(int(args.nice))
        except OSError:
            pass

    gen = GeneratorEl.load(str(args.stem), args.codec)
    bank = gen.field.bank
    std = np.std(bank, axis=0)
    mean_abs = np.mean(np.abs(bank), axis=0)
    order = playlist(bank, args.all)
    print(
        f"wallpaper  n={n_el()}  primes={EL_PRIMES}  walking {len(order)} params  "
        f"{args.size}px  {args.period:g}s each",
        flush=True,
    )

    live = LiveField(gen.pf, args.size, args.size)
    rng = np.random.default_rng()
    base = new_scene(gen, rng)
    live.set_all(base)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rgb0 = live.rgb()
    _atomic_ppm(args.out, rgb0)
    # folder copy so a file manager / second imv can see it
    _atomic_ppm(HERE / "runs/wallpaper/live.ppm", rgb0)
    pid_path = HERE / "runs/wallpaper/wallpaper.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()) + "\n")

    viewer = None if args.no_viewer else open_paper(args.out, args.fullscreen)
    if viewer:
        print(f"viewer pid={viewer.pid}  {args.out}", flush=True)
    print(f"pause/resume: kill -USR1 {os.getpid()}", flush=True)

    t0 = time.perf_counter()
    rgb = live.rgb()
    _atomic_ppm(args.out, rgb)
    dt_ms = (time.perf_counter() - t0) * 1000
    print(f"raster+blit {args.size}px  {dt_ms:.1f} ms", flush=True)

    frame = 0
    pi = 0
    period = max(0.6, float(args.period))
    dt = 1.0 / max(8.0, float(args.fps))
    t_param = 0.0
    next_tick = time.perf_counter()

    def pace():
        nonlocal next_tick
        next_tick += dt
        sl = next_tick - time.perf_counter()
        if sl > 0:
            time.sleep(sl)
        else:
            next_tick = time.perf_counter()

    while not STOP:
        if PAUSED:
            time.sleep(0.15)
            next_tick = time.perf_counter()
            continue
        idx = order[pi % len(order)]
        amp = max(1.25 * float(std[idx]), 0.35 * float(mean_abs[idx]), 1e-3)
        phase = np.sin(2.0 * np.pi * (t_param / period))
        live.set_amp(idx, float(base[idx]) + amp * phase)
        rgb = live.rgb()
        _atomic_ppm(args.out, rgb)
        _atomic_ppm(HERE / "runs/wallpaper/live.ppm", rgb)
        if frame % max(1, int(args.fps)) == 0:
            print(f"  {param_label(idx)}  phase={phase:+.2f}", flush=True)
        frame += 1
        t_param += dt
        if t_param >= period:
            live.set_amp(idx, float(base[idx]))
            t_param = 0.0
            pi += 1
            if pi % len(order) == 0:
                nxt = new_scene(gen, rng)
                steps = max(8, int(1.2 / dt))
                c0 = live.amps.copy()
                for s in range(steps):
                    if STOP:
                        break
                    while PAUSED and not STOP:
                        time.sleep(0.15)
                    t = (s + 1) / steps
                    live.set_all((1.0 - t) * c0 + t * nxt)
                    _atomic_ppm(args.out, live.rgb())
                    pace()
                base = nxt
                print("  new scene", flush=True)
        pace()

    print("stop wallpaper", flush=True)
    try:
        pid_path.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    main()
