# ScaleField — project design sheet

Sibling of CalcNet, not a replacement.
CalcNet = printable 1-D / ODE maps.
ScaleField = printable **multi-scale color fields**; pixels are an output adapter only.

Not a photo codec. Not AGI. A visual interface + tiny generative field.

---

## Contract

- Internal state is **not** \(H\times W\times 3\).
- Internal state is a few **scales**, each with **9 numbers** (3 channels × 3 basis neurons).
- Those numbers parameterize a field \(C(u)\) on an address chart. Rasterize only to paint a panel.
- Off-beat / incommensurate charts are allowed (vernier detail).
- Product of the tool is still a **printed calculation** plus a short bias list.

Param fence: well under CalcNet’s 1000. Default experiment still \(x=3\), \(i=4\) → \(3\times 3\times 4=36\).

---

## Two objects (do not convolve)

| object | symbol | what it is |
|---|---|---|
| **complexity** | \(x\) | neurons needed to map the amplitude *surface* (3-D shape → 2-D field) at one scale |
| **scale / spacing** | \(i\) | how many lenses / iterations of that surface at tied charts \(r^\ell u\) |
| **color** | \(3\) | independent copies of the same surface |

Gaps between samples can *look* like pixel pitch. That is spacing \(i\) + chart, not complexity \(x\). Raising \(x\) buys a more wrinkled surface. Raising \(i\) buys more vernier rulers. Different knobs.

**Growth law**

\[
N_{\mathrm{params}} \approx x \times 3 \times i
\]

plus a few \(\psi\) for \(W\) and \(r\). Compare to raster \(H\times W\times 3\). \(N\) grows with *model* complexity and lens count, not with panel resolution.

Start with smallest \(x\) that recognizes amplitude at all (\(x=1\) flat, \(x=3\) a soft blob). Promote \(x\) only when the surface is too simple. Promote \(i\) when you need another octave, not more wrinkles.

---

## Codec loop

```
raw pixels  →  encode (fit b, ψ)  →  field C(u)   [work here]
            →  transform / PC / Euler on b
            →  raster C(u) onto H×W×3   [adapter only]
```

Input can still be a photo. Compute lives in \(x\times 3\times i\), not in the framebuffer. Output res is free.

---

## Per-scale object (\(x\) neurons × 3 channels)

One scale \(\ell\):

| slot | count | meaning |
|---|---|---|
| R basis | 3 | heights \(b^R_{1,2,3}\) |
| G basis | 3 | \(b^G_{1,2,3}\) |
| B basis | 3 | \(b^B_{1,2,3}\) |

Field on that scale’s chart \(u=(x,y)\) or log-polar \((\rho,\theta)\):

\[
C_\ell(u)
=
\begin{pmatrix}
\sum_{k=1}^{3} b^R_{k,\ell}\,W_k(u;\psi_\ell) \\
\sum_{k=1}^{3} b^G_{k,\ell}\,W_k(u;\psi_\ell) \\
\sum_{k=1}^{3} b^B_{k,\ell}\,W_k(u;\psi_\ell)
\end{pmatrix}
\]

\(W_k\in\{\mathrm{const},\mathrm{lin},\mathrm{exp}\}\) unless promoted. Same \(W\) family as CalcNet. Output of a “neuron” is the raw height \(b\), not a pixel.

**3-D shape → 2-D amplitude:** the 9-list is a point in \(\mathbb{R}^3\otimes\mathbb{R}^3\). Evaluating \(C_\ell(u)\) at many \(u\) is the 2-D map. Translation = sample the same shape on a display grid.

---

## Scale stack

Charts tied by one ratio \(r\) (start \(r=1/2\) or a single trained \(r\)):

\[
u^{(\ell)}=r^\ell\,u^{(0)}
\]

Optional second lattice at each scale, rotated by an angle that **does not** tile with the first (off-beat / vernier). Do not start with that; add when one lattice looks locked.

Total field:

\[
C(u)=\sum_{\ell=0}^{L-1} C_\ell(u^{(\ell)})
\]

Default \(L=4\). More scales = more lenses, not more pixels.

---

## Display adapter (the only place pixels exist)

```
C(u)  →  sample on panel grid  →  clip to [0,1]  →  RGB framebuffer
```

Changing window size does **not** change the 36 numbers. It only changes how densely you sample \(C\).

---

## Interface

- **Pointer / “dah”:** an address \(u_*\) on the coarsest chart. Zoom = switch to \(r u_*\).
- **Edit:** move nearby \(b_{k,\ell}\), not stamp pixels.
- **Readout:** `instrument()` prints \(W\), \(r\), and the 9-lists.

---

## Training

Transform data → the 9-vectors (and maybe \(\psi_{scale}, r\)).

- Forward: 9-list → field → raster (or → compare in field space if the target is already a field).
- Update: PC or WeirdTrainer on **biases only**. \(W\) stays a formula.
- Loss: reconstruction on a downsample, plus \(\lambda\|b\|_1\) so unused basis die.

Do not backprop into a learned \(H\times W\) weight tensor.

---

## First experiment (must pass before chrome)

1. 1-D analog: two charts, periods \(1\) and \(\ln 2\), 3 heights each. Plot the beat. Eye-test: more structure than \(6\) samples on one grid.
2. 2-D: \(L=2\), no second lattice, \(W=\exp\) or const. Fit a **solid color + one soft blob**. Printed formula must stay short.
3. Only then: a second rotated lattice, or a moving blob as Euler on the \(b\)’s (physics overlay).

Fail = enrich \(W\) or add one scale. Not “add pixels to the model.”

---

## Relation to CalcNet

| CalcNet | ScaleField |
|---|---|
| `Expr`, `deriv`, Euler | same devices for \(W\) and for time |
| 1-D \(x\) | 2-D address \(u\) |
| one height | 9 heights / scale |
| ODE engine | field + raster adapter |

Share: locked vs custom split, address chart, \(W\in\{\mathrm{const},\mathrm{lin},\mathrm{exp}\}\), printed calculation, bias-only train, no dense matrices.

Implement as `scalefield.py` importing `calcnet.py`, or a folder `scalefield/` with a thin `Field` type.

---

## Mantra

9 numbers are a 3-D shape. The chart slices it. Scales are lenses. Pixels are glass. Off-beat charts buy vernier detail, not a lossless photo.

## Do not

- Claim arbitrary image compression.
- Store a framebuffer as the source of truth.
- Train \(W_{ij}\) per pixel.
- Call this vision-complete or AGI.
- Start at 4K “because raster is free.” First experiment is a blob.
