# ScaleField — developer handoff

Build this, then train. Not a foundation model. A **small visual field** you can fit on a laptop.

Philosophy: as small as possible, as useful as possible. State is \(x\times 3\times i\) numbers, not \(H\times W\times 3\). Pixels are how data enters and how you look. Work happens in the field.

Code already started: `scalefield.py`. Design: `SCALEFIELD.md`. Calculus sibling: `calcnet.py`.

---

## 1. What you are shipping

A class that:

1. Holds `b` with shape `(i, 3, x)` — **the parameters**.
2. Holds frozen (or few) \(\psi, r\) that define charts and basis \(W\).
3. `encode` / `train` — images → `b`.
4. `decode(H,W)` — `b` → RGB array.
5. `formula()` — printed calculation (the model card).

Default: `x=3`, `i=4`, `r=0.5` → **36 params**.

If a feature needs a dense conv net, you are off-spec. Add \(x\) or \(i\), don’t add a U-Net.

---

## 2. Frozen vs trained

| name | shape | train? |
|---|---|---|
| `b` | `(i, 3, x)` | **yes** — always |
| `r` | scalar | optional, after `b` works |
| `psi` (W widths / centers) | ≤ 8 floats | optional, last |
| `tags` | `exp,lin,const` | hand-set, not trained |

No \(W_{hwc}\) tensor. No per-pixel weights.

---

## 3. Data

Anything that becomes `float32` RGB in `[0,1]`, shape `(H,W,3)`.

**Start set (do this first)**

- 20–100 synthetic images: blobs, ramps, two-color UI panels, 64×64.
- Then 20–100 real thumbnails (faces, icons, rooms) at 64×64.

You do **not** need ImageNet. If it cannot fit blobs, stop.

Loader sketch:

```
for path in folder:
    im = load_rgb(path)          # any H,W
    im = resize(im, 64, 64)
    im = im.astype(f32) / 255
    yield im
```

Train at 64. Decode at 256 when you demo. That *is* the point.

---

## 4. Loss

\[
L = \mathrm{MSE}(\mathrm{field}(H,W),\; y) + \lambda \|b\|_1
\]

- MSE on the **field** at train resolution (not after percentile stretch).
- \(\lambda \sim 10^{-3}\) so unused basis go to 0.
- Optional: small TV on the raster if edges ring. Don’t start there.

Metric to print every epoch:

- train MSE
- \(N=\texttt{b.size}\)
- sparsity: fraction of `|b| < 1e-3`
- decode a 256×256 preview

Success for v0 (synthetic blobs): MSE < 0.02 at 64×64 with \(N=36\).
Success for v1 (icons): recognizable at 256 after training at 64. Not pixel-perfect.

---

## 5. Train loop (copy this)

```
sf = ScaleField(x=3, i=4, r=0.5)
opt on sf.b only          # Adam lr=0.05 or the projection encode() already in scalefield.py
for epoch in 1..200:
    for y in batch:       # batch 8 is enough
        pred = sf.field(y.H, y.W)
        loss = mse(pred, y) + 1e-3 * abs(sf.b).mean()
        backprop → sf.b
    every 10 epochs: save preview decode(256,256), print formula()
```

**Two implementations, both allowed**

1. **Projection encode** (already in `scalefield.py.encode`) — no autograd. Good enough to start today.
2. **Autograd** (tiny PyTorch or JAX) — `b` is a tensor, `field()` written with the same ops. Use when you want `r` / `psi` trained.

Do (1) first. If MSE stalls above 0.05 on blobs, check charts, don’t jump to a CNN.

**Schedule**

- Epochs 0–50: train `b` only.
- If needed: unfreeze `r` in (0.3, 0.7).
- Last resort: unfreeze 2–4 `psi` widths.

Never unfreeze a pixel grid.

---

## 6. API the developer must keep

```python
sf = ScaleField(x=3, i=4, r=0.5)
sf.train(images)              # or encode(one_image)
rgb = sf.decode(256, 256)     # uint8 HWC
print(sf.formula())           # the model
sf.save("run.npz")            # b, r, x, i, psi, tags
ScaleField.load("run.npz")
```

`train` on a list/folder is the missing wrapper — add it next to `encode`. One image `encode` is for debugging.

---

## 7. File layout

```
scalefield/
  scalefield.py      # model + encode + raster
  train.py           # folder → npz + previews
  SCALEFIELD.md      # philosophy
  SCALEFIELD_DEV.md  # this file
previews/
checkpoints/
```

`train.py` CLI:

```
python train.py --data ./icons --x 3 --i 4 --out ./run.npz --preview ./previews
```

---

## 8. How to wrestle it into usefulness

| failure | knob |
|---|---|
| blob OK, icons mush | raise \(x\) (5 or 7), keep \(i=4\) |
| edges exist, no hierarchy | raise \(i\), keep \(x=3\) |
| color wrong, shape OK | don’t raise \(x\); check per-channel \(b\) |
| looks like 8 giant pixels | charts too coarse; shrink \(r\) or add a scale |
| N exploding | you added a conv. undo |

Complexity \(x\) ≠ spacing \(i\). Don’t turn both knobs at once.

---

## 9. What “powerful and small” means here

- **Small:** \(N=x\times 3\times i\) on the order of 10–200, not 10⁷.
- **Powerful:** one field, any output size; edits are `b` sliders; formula is readable.
- **Not powerful:** photoreal 4K, video, open-world vision.

Consumer angle: a 36-number “picture” you can train on a phone-class CPU, then zoom. That is the accessibility play. Large-model lesson: if a 36-param field captures structure, a big net is wasting capacity on the raster.

---

## 10. Definition of done (v0)

- [ ] `ScaleField` constructs, `nparams()==x*3*i`
- [ ] `decode(256,256)` from hand-set `b` writes a PNG
- [ ] `encode` / `train` on 32 synthetic 64×64 images finishes in minutes on CPU
- [ ] preview at 256 is recognizably the same blobs
- [ ] `formula()` printed beside the preview
- [ ] checkpoint reloads bit-identical `b`

Then you start the real fight: icons, then photos. That fight is the product, not a bigger spec.
