"""Download short Commons clips and extract consecutive 96² frames."""

from __future__ import annotations

import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
from PIL import Image

from clips import CLIPS

UA = "ScaleField/gen4 (local research; unfold next-frame)"


def _get(url: str, dest: Path | None = None, timeout: int = 180, tries: int = 6) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                blob = r.read()
            if dest is not None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(blob)
            return blob
        except HTTPError as e:
            last = e
            if e.code in (429, 503, 502) and i < tries - 1:
                wait = 8 * (i + 1)
                print(f"    {e.code} backoff {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
        except (URLError, TimeoutError, OSError) as e:
            last = e
            if i < tries - 1:
                time.sleep(4 * (i + 1))
                continue
            raise
    raise last  # type: ignore[misc]


def download_clips(raw_dir: Path) -> list[tuple[str, Path]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out: list[tuple[str, Path]] = []
    for i, (name, url) in enumerate(CLIPS):
        existing = list(raw_dir.glob(f"{name}.*"))
        if existing and existing[0].stat().st_size > 10_000:
            print(f"  have {existing[0].name}", flush=True)
            out.append((name, existing[0]))
            continue
        ext = Path(urllib.parse.urlparse(url).path).suffix or ".webm"
        dest = raw_dir / f"{name}{ext}"
        try:
            print(f"  get {name}{ext}", flush=True)
            _get(url, dest=dest)
            out.append((name, dest))
        except HTTPError as e:
            print(f"    skip {name}: {e}", flush=True)
            if dest.exists():
                dest.unlink()
            if e.code == 429:
                print("    Wikimedia 429 — stopping all remaining downloads.", flush=True)
                break
        except Exception as e:
            print(f"    skip {name}: {e}", flush=True)
            if dest.exists():
                dest.unlink()
        if i + 1 < len(CLIPS):
            time.sleep(2.5)
    return out


def extract_frames(
    clip: Path,
    out_dir: Path,
    size: int = 96,
    fps: float = 6.0,
    max_frames: int = 32,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    have = sorted(out_dir.glob("f_*.png"))
    if len(have) >= 4:
        return have[:max_frames]
    vf = (
        f"fps={fps},"
        f"scale={size}:{size}:force_original_aspect_ratio=increase,"
        f"crop={size}:{size}"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(clip),
        "-vf",
        vf,
        "-frames:v",
        str(max_frames),
        str(out_dir / "f_%03d.png"),
    ]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("f_*.png"))[:max_frames]


def load_rgb(path: Path, size: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _crop_pan(im: Image.Image, size: int, ox: int, oy: int) -> np.ndarray:
    w, h = im.size
    side = max(size + 8, min(w, h))
    x0 = max(0, (w - side) // 2 + ox)
    y0 = max(0, (h - side) // 2 + oy)
    x0 = min(x0, max(0, w - side))
    y0 = min(y0, max(0, h - side))
    crop = im.crop((x0, y0, x0 + side, y0 + side)).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(crop, dtype=np.float32) / 255.0


def pan_clips_from_stills(still_dirs: list[Path], size: int, n_clips: int = 12) -> list[tuple[str, list]]:
    """Tiny camera-pan sequences from stills — extra motion pairs, no extra download."""
    paths: list[Path] = []
    for d in still_dirs:
        if not d.is_dir():
            continue
        paths.extend(sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}))
    seqs: list[tuple[str, list]] = []
    # offsets in source pixels: a slow pan + a slight zoom-feel
    track = [(0, 0), (3, 0), (6, 1), (9, 2), (12, 2), (9, 4), (6, 5), (3, 6)]
    for i, p in enumerate(paths[:n_clips]):
        try:
            im = Image.open(p).convert("RGB")
        except OSError:
            continue
        if min(im.size) < size + 16:
            continue
        frames = [_crop_pan(im, size, ox, oy) for ox, oy in track]
        seqs.append((f"pan_{p.stem}", frames))
    if seqs:
        print(f"  pan-stills: {len(seqs)} clips × {len(track)} frames", flush=True)
    return seqs


def gather_clips(
    root: Path,
    size: int = 96,
    fps: float = 6.0,
    max_frames: int = 32,
) -> list[tuple[str, list]]:
    raw = root / "data" / "raw"
    frames_root = root / "data" / "frames"
    clips = download_clips(raw)
    seqs: list[tuple[str, list]] = []
    for name, path in clips:
        try:
            frames = extract_frames(path, frames_root / name, size=size, fps=fps, max_frames=max_frames)
        except subprocess.CalledProcessError as e:
            print(f"  ffmpeg fail {name}: {e}", flush=True)
            continue
        imgs = []
        for p in frames:
            try:
                imgs.append(load_rgb(p, size))
            except OSError:
                continue
        if len(imgs) >= 4:
            print(f"  clip {name}: {len(imgs)} frames", flush=True)
            seqs.append((name, imgs))
        else:
            print(f"  clip {name}: too short ({len(imgs)})", flush=True)
    parent = root.parent
    still_dirs = [
        parent / "gen2" / "data" / "real_raw",
        parent / "data" / "train",
    ]
    seqs.extend(pan_clips_from_stills(still_dirs, size=size, n_clips=10))
    return seqs
