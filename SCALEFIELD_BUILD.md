# ScaleField — full design + implementation

One sheet. Philosophy, architecture, train loop, knobs, honesty.
Code already started: `scalefield.py`, `train.py`. Sibling: `calcnet.py`.

**Win condition:** better *display* scaling with less compute. \(N = x \times 3 \times i \le 1000\). Panel size is not model size.

Not a photo codec. Not AGI. A tiny visual field you can train on a laptop and raster at 360p+.

---

## 0. Why this exists

Raster models live in \(H \times W \times 3\). That’s absurd as *state*. A 360p frame is ~7×10⁵ numbers before anyone learns anything.

ScaleField state is a **3-D amplitude surface** sampled by \(x\) bases, copied across 3 channels, stacked on \(i\) scales. Pixels are how data enters and how you look. Work happens in the field.

Same knife as CalcNet: printed calculation, frozen \(W\), trained offsets, no dense matrix.

---

## 1. Two objects (do not convolve)

| object | symbol | knob |
|---|---|---|
| complexity / vocabulary | \(x\) | how wrinkled the surface is; **foveal sharpness** |
| scale / spacing / lenses | \(i\) | how many quantizations of the same picture; **where** sharpness sits |
| color | \(3\) | three copies of the same surface |

Gaps that look like pixel pitch are **spacing**, not complexity.

Growth:

\[
N_{\text{params}} \approx x \times 3 \times i \le 1000
\]

Raise \(x\) if the *center* is mush. Raise \(i\) if you need another ruler / more foveation. Never both in the same experiment.

---

## 2. The field

Chart at scale \(\ell\):

\[
u^{(\ell)} = r^{\ell}\, u^{(0)}
\qquad r = \ln 2 \text{ by default (log growth)}
\]

Basis \(W_k(u)\) frozen in \(\{\mathrm{const},\mathrm{lin},\mathrm{exp},\mathrm{quad},\mathrm{logsin},\mathrm{logcos},\mathrm{rip},\mathrm{gabor}\}\).
Log tags use frequency \(\sim 2^k / \ln 2\).

Per scale, per channel:

\[
C_\ell^{c}(u)=\sum_{k=1}^{x} b_{\ell,c,k}\, W_k(u^{(\ell)})
\]

Total field:

\[
C(u)=\sum_{\ell=0}^{i-1} C_\ell(u^{(\ell)})
\]

`b.shape = (i, 3, x)`. That tensor **is** the model.

Optional later: shared mixer in the center, \(h\mapsto\sigma(Ah+c)\), one \(A\) for all scales, width \(x\). Count \(A\) against the 1k cap.

---

## 3. Fovea / periphery

Fine \(\ell\) + most of \(|b|\) mass → **center should get sharp**.
Coarse \(\ell\) only → **periphery stays soft**.

That is intended. Enough scale-quantizations of the dataset make the *middle* clear. The sides can stay blurry. If the middle is mush, \(x\) or \(W\) is wrong, not \(i\).

---

## 4. Codec loop

```
pixels (any H×W) → resize train res (64)
                 → encode / train  →  b
                 → transform b     (optional next-frame)
                 → decode(360 or 640×360)
```

Train small. Paint big. Never store a framebuffer as source of truth.

---

## 5. Frozen vs trained

| thing | train? |
|---|---|
| `b` | **yes** |
| tags / \(W\) shape | no (alphabet) |
| `r` | optional after `b` works |
| `psi` (centers, widths) | last resort, few floats |
| shared mixer \(A\) | optional, after stills work |

No per-pixel weights. No U-Net.

---

## 6. Presets (already in code)

| name | \(x\) | \(i\) | \(N\) | use |
|---|---|---|---|---|
| tiny | 3 | 4 | 36 | blobs, debug |
| small | 8 | 6 | 144 | first 360p |
| medium | 12 | 8 | 288 | more surface |
| textured | 16 | 16 | 768 | log texture alphabet, under cap |

Spend leftover budget on \(x\) for sharpness:

- `i=16` → \(x\le 20\) (960)
- `i=8` → \(x\le 41\) (984)
- `i=4` → \(x\le 83\) (996)

Smaller \(N\) that looks good beats a full 1000 that doesn’t.

---

## 7. Train (start today)

```bash
python train.py --data ./pics --preset textured --size 64 --out-res 360 --wide --out run.npz
```

or sharpness pass:

```bash
python train.py --data ./pics --x 20 --i 16 --size 64 --out-res 360 --wide
```

Loss: \(\mathrm{MSE}(\mathrm{field}, y) + \lambda\|b\|_1\), \(\lambda\sim 10^{-3}\).
Optimizer: projection `encode` (in repo) or Adam on `b` if you add autograd.
Print `formula()`, sparsity, preview 360p every few epochs.

**v0 bar:** synthetic blobs MSE ≲ 0.02 at 64².
**v1 bar:** icons / one object recognizable in the *center* at 360p.
**Not a bar:** DSLR-sharp arbitrary scenes from 768 floats.

Data: 20–100 images is enough to wrestle. Not ImageNet.

---

## 8. Generative (only after stills are decent)

Blurry still → next frame is a **dynamics** model on `b`, not a sharpener.

- encode \(t \to b_t\), \(t+1 \to b_{t+1}\)
- tiny map \(b_t \mapsto b_{t+1}\) (same \(N\))
- roll `b`, decode any res

If the still can’t draw an edge in the fovea, video will be a moving mush. Fix the still first.

Sampling: `generate(H,W,seed)` jitter-fills `b` and rasters. After training, sample from the empirical cloud of `b` instead of noise.

---

## 9. Implementation map (what exists / what to add)

**Have**

- `scalefield.py` — field, log/texture \(W\), encode, train, save/load, decode, generate, presets, 1k cap
- `train.py` — folder → npz + preview; `--preset textured`, `--out-res 360 --wide`
- demos: `scalefield_360p_tex.png` (640×360, N=768)

**Add when you wrestle**

- `train.py` logging of center-crop MSE vs full-frame MSE (fovea metric)
- optional shared mixer `A` (count toward 1k)
- `b_t → b_{t+1}` script on frame folders
- autograd `b` if projection stalls

**Do not add**

- a CNN backbone
- pixel-space diffusion
- extra params past 1000

---

## 10. Failure table

| see | turn |
|---|---|
| center mush | raise \(x\) or enrich \(W\) |
| center ok, no hierarchy | raise \(i\) |
| everything looks like 8 pixels | charts too coarse; `r` / more \(i\) |
| color wrong, shape ok | inspect per-channel `b`, don’t raise \(x\) |
| N exploding | you added a conv; undo |
| video mush | stills weren’t clear; don’t train next-frame yet |

---

## 11. Philosophy (short)

- Neuron output = raw height of a slice.
- \(W(u)\) = formula on an address chart (linear / exp / log-texture).
- Bias `b` = the parameters; weights are not a matrix.
- Expand = more slices / jet; contract = integral / pool.
- Incommensurate / log scales = vernier detail from little input.
- Periphery may stay blurry. That’s foveation, not failure.
- The product is the **printed field** + `b`, then a raster adapter.
- Fit ≠ law. A good 360p icon is a win. A claimed universal codec is not.

CalcNet taught: script the map, print the calculation, train only unlocked slots.
ScaleField is that map in 2-D color.

---

## 12. Definition of done (build weekend)

- [ ] `ScaleField(x,i)` refuses \(x*3*i > 1000`
- [ ] `preset("textured").generate(360, 640)` writes a 360p PNG
- [ ] `train.py` on a real folder produces `run.npz` + 360p preview
- [ ] center of preview is judged by eye (not just global MSE)
- [ ] `formula()` sits next to the image
- [ ] reload checkpoint, same `b`

Then it’s a tool. Fun starts when you fight \(x\) vs \(i\) on *your* pictures.

Hell ya. Less compute per displayed pixel is the win. Go build.
