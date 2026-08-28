# CalcNet — actionable plan

Not AGI. A cyber knife: tiny printable maps that *are* calculus.
Exist today: `/home/workdir/artifacts/calcnet.py` (devices, Expr, instrument, WeirdTrainer).

## Contract

1. Script calculus as a tree of named devices. Output is the **calculation** (`instrument` / `formula()`).
2. Optional: fit a tiny tree to data; still output the calculation.
3. Frozen known laws. Free numbers only where the equation says so.
4. Cap **1000 params** (fence, not a target). Start at **2–32 units**.

AI blob → printed math → smaller blob. Fit ≠ law.

## Parameter split

| thing | role | trained? |
|---|---|---|
| address chart \(u_i=\psi_{scale} i+\psi_0\) | where slices sit | optional, few \(\psi\) |
| weight field \(W(u;\psi)\) | height rule on the chart | only \(\psi\), never a matrix |
| bias \(b_i\) | vertical offset of slice \(i\) | yes (main params) |
| custom affine | only if the written eq needs \(kx+\phi\) | marked `fit` |

Default leaves: identity `x`, locked `sin/exp/log/tanh(x)` with **no** free \(w,b\).
Unlock affine only by name.

## Weight / address (start generic)

Chart always: \(u_i=\psi_{scale} i+\psi_0\).

\(W(u)\) promote only when needed:

- const (1 \(\psi\)) — start here
- linear \(au+c\) (2)
- exp \(a e^{bu}\) (2) — octave / scale stacks
- quadratic (3) — first maybe
- higher = an `Expr` for \(W\), cap \(|\psi|\le 6\)

Address scale and \(W\) share \(u\). Change \(\psi_{scale}\), the strip stretches. No raw \(W_{ij}\).

## Layers

Each layer \(\ell\) may have its own shape tag (const/lin/exp).
**Tie the charts:** \(\psi_{scale}^{(\ell+1)}=r\,\psi_{scale}^{(\ell)}\) with one \(r\) (try \(1/2\) or a single trained \(r\)).

Printable stack: one \(r\), a list of shape tags, bias vectors.

## What a neuron is

- **Output:** raw value (height of the slice). Do not emit gradients as the value.
- **Jet (description only):** \(\partial y/\partial\psi\), \(\partial y/\partial b\) exist via `deriv`. Observation, not a new output socket unless you add a debug device.
- Addressable like an LED: index \(i\), \(W(u_i)\) known from formula, \(b_i\) stored.

## Calculus devices (already / to lock)

Have: Neuron1D, Compose, Sum, Product, Expr (+ * / ** compose scale id), FiniteDiff, Trapezoid, EulerStep, instrument, WeirdTrainer.

Add next:

- locked `sin`, `exp`, `log`, `tanh` leaves (no \(w,b\))
- explicit `affine(k,φ)` node (the only custom slot)
- `WeightField(W_tag, ψ, b[])` layer
- `jet` debug (optional)
- `formula()` already prints the map

Rules that must stay true: chain, product, quotient, power, \(\int f'\approx f+C\), Euler = discrete \(\dot x=v(x)\).

## Learning

Not backprop-religion.

- **WeirdTrainer (have):** finite-diff SGD on marked leaves; returns `instrument(expr)`.
- **Predictive coding (next):** state per unit, predict down, update **biases** (and maybe \(\psi_{scale}\)) from local error. Weights stay \(W(u;\psi)\).

Score: MSE + \(\lambda\cdot n_{params}\). Halt at the Pareto knee, not at 1000.

## First build (neural physics engine)

One dimension. One process you already know the law of (particle, RC, pendulum, logistic).

1. Freeze \(W\in\{\mathrm{const},\mathrm{lin},\mathrm{exp}\}\), pick \(N\le 16\).
2. Train only \(b_i\) (and optionally \(\psi_{scale}\)).
3. Time = `EulerStep` on the field.
4. Accept iff `instrument()` looks like the law (or a short residual), not a hairball.

If that works, add a second layer with tied \(r\). If it fails, change \(W\) tag, don’t add a matrix.

## Mantra

Slices from \(W(u)\). Height is the neuron. Bias lifts the slice. Expand = more slices / jet. Contract = sum / integral. Layers = stacked charts with one scale ratio \(r\).

## Do not

- Call this AGI or a brain.
- Train a dense \(W_{ij}\).
- Treat a good fit as a discovered law.
- Count to 1000 params “because the cap exists.”
- Collapse value and jet into one output.

## Done looks like

A script that prints something like:

`layer0: exp(u=0.5*i), b=[...] ; ẋ = -tanh(x)`

and that string *is* the model you run.
