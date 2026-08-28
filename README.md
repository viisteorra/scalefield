# ScaleField

Internal state is **not** H×W×3. It is one scalar per cell of each **prime lattice** (1×1, 2×2, 3×3, 5×5, …). Color is three incommensurate phases of that scalar. Raster is an adapter.

This repo is the v1 **codec** plus the generative experiments that came out of it. The codec works. Generation is still an exploration: a few frozen looks, a live wallpaper, and a ceiling on this machine (i5-6500 / 15 GB).

## Frozen artifacts (do not overwrite)

| stem | what it is |
|---|---|
| [`v1/runs/live`](v1/runs/live) | **Codec.** Prime DCT through 31, N=3359, val MSE ≈ 0.0004. |
| [`gen/runs/cool-gmm`](gen/runs/cool-gmm) | First generator that looked like *something* — trippy GMM in the 3359-D field. |
| [`gen2/runs/good-reals`](gen2/runs/good-reals) | Stills from a real-image cloud (local mixes + coarse GMM). Closest to “photos.” Also the retrieval failure mode if you push it. |
| [`gen3/runs/good-unfold`](gen3/runs/good-unfold) | Ancestral prime unfold 1→31. High hold-R² on later lattices; still a stills model. |
| [`gen2/runs/wallpaper`](gen2/runs/wallpaper) | **Animated wallpaper.** 1/p→p² cosine band, image = (Σ F)². Tartan / oil-slick interference. Lightweight. |
| [`gen2/runs/matrices-4ph`](gen2/runs/matrices-4ph) | Last scene-blob generator: a real **p×p × 4-phase** matrix on every prime through 13 (n=1509). Soft almost-images. GMM is maxed on this box. |

## Run

Python 3.12+, `numpy`, `pillow`. GPU optional (`moderngl`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Codec round-trip / board** (frozen `v1/runs/live/best.npz`):

```bash
cd v1
python demo.py   # if present, else: python train.py --help
```

**Wallpaper** (no training, ~2 ms/frame, writes `/dev/shm/scalefield-wallpaper.ppm`):

```bash
cd gen2
python wallpaper.py --size 320 --period 3 --fps 30
# pause / resume
kill -USR1 $(cat runs/wallpaper/wallpaper.pid)
```

**Last elemental generator** (blurry scenes, not tartan):

```bash
cd gen2
python -c "
from elemental import GeneratorEl
from pathlib import Path
from scalefield import save_png
g = GeneratorEl.load('runs/matrices-4ph/scalegen', '../v1/runs/live/best.npz')
imgs, _ = g.sample(4, 96, 96, seed=1)
from watch import grid
save_png('runs/matrices-4ph/sample.png', grid(imgs, cols=4))
print(g.formula())
"
```

## What actually worked

- **Prime lattices as a codec.** Fit a photo onto concatenated p×p DCTs. Sharp. That is v1.
- **Peel, don’t joint-LS, for generation.** Fit 1×1, subtract, fit 2×2, subtract, … so later primes don’t steal the same standing wave. “Peel” means *keep that matrix and remove it from the leftover*, not drop the prime.
- **Don’t skip 2 and 3.** 1×1 then 5×5 is a hole; the ripples were missing scales and missing sin/cos phases.
- **Wallpaper.** Off-phase prime bands interfering with themselves. Looks like a field, not a photo. Cheap to animate one coefficient at a time.

## What did not

- Inventing high-k DCT cells → oil-slick standing waves.
- Copying them from the train set → memorization / 1-NN.
- mix4 of nearest photos → rainbow noise on memorized stills.
- Linear unfold of 11×11 (and up) from 1+2+3 → hold R² ~0.1; sampling it is speckle. The 4-phase run leaves those lattices at 0.
- A big embed table on 540 photos → a lookup table, not a language.

The codec already *represents* images. The generator on this box is a 10-component GMM on 24 numbers of 1×1+2×2+3×3 plus a linear guess of 5 and 7. That prior is fitted. More EM will not sharpen it. A small **per-prime codebook** (tens of codes, not 540 image vectors) is the next honest try; a giant joint embed wants more data and more machine.

## Layout

```
v1/          codec (PrimeField, train, raster)
gen/         first GMM on the 3359-D field
gen2/        residual peel, thin lattices, elemental bands, wallpaper
gen3/        ancestral unfold
gen4/        next-frame / video sketches (code only)
```

Hardware this was run on: Intel i5-6500, 15 GB RAM, HD 530. One trainer at a time.
