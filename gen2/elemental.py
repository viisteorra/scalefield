"""
A p×p matrix on every prime. Both phases. No skipped primes.

Peel is not delete: fit this prime's matrix to the leftover photo, *keep*
the matrix, subtract that reconstruction so the next prime sees what this
one could not explain. Raster uses every matrix.

Each cell (i, j) of p×p is the four products of {cos,sin}(F_i v) with
{cos,sin}(F_j u). F is length p, no holes: 1/p, q/p, 1, then 2,3,…,p.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v1"))
sys.path.insert(0, str(ROOT / "gen"))
sys.path.insert(0, str(ROOT / "gen2"))

from scalefield import COLOR_PHASES, PrimeField, pixel_uv, foveal_map  # noqa: E402
from scalegen import DiagGMM, LatentPrior  # noqa: E402
from thin import PhaseUnfold  # noqa: E402

EL_PRIMES = (2, 3, 5, 7, 11, 13)
PRIMES_ALL = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
N_PHASE = 4  # cu cv, su sv, cu sv, su cv
FEAT = None


def _uniq(xs: list[float]) -> list[float]:
    out: list[float] = []
    seen: set[float] = set()
    for x in xs:
        k = round(float(x), 10)
        if k not in seen:
            seen.add(k)
            out.append(float(x))
    return out


@lru_cache(maxsize=16)
def axis_p(p: int) -> np.ndarray:
    """Exactly p frequencies. 1/p, q/p, 1, then 2,3,…,p. Always ends at p."""
    fracs = [1.0 / p] + [q / float(p) for q in PRIMES_ALL if q < p]
    F = _uniq(fracs + [1.0, float(p)])
    if len(F) > p:
        F = _uniq(fracs + [float(p)])
        if len(F) > p:
            mid = F[1:-1]
            need = p - 2
            idx = np.unique(np.round(np.linspace(0, max(len(mid) - 1, 0), need)).astype(int))
            pick = [mid[int(i)] for i in idx if len(mid)]
            F = [F[0]] + pick[:need] + [F[-1]]
    if len(F) < p:
        for q in PRIMES_ALL:
            if len(F) >= p:
                break
            if q < p:
                F = _uniq(F + [float(q)])
    while len(F) < p:
        F.append(float(p))
    return np.asarray(sorted(F[:p]), dtype=np.float64)


def n_feat(p: int) -> int:
    return N_PHASE * p * p


def _uhv(pf: PrimeField, H: int, W: int):
    u, v = pixel_uv(H, W)
    u, v = foveal_map(u, v, pf.foveal)
    return (u + 1.0) * 0.5, (v + 1.0) * 0.5


def basis_p(p: int, uh: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """(H, W, 4 p²)  every (F_i, F_j) with all four phase products."""
    F = axis_p(p)
    cu = np.cos(np.pi * F[:, None, None] * uh[None, :, :])
    su = np.sin(np.pi * F[:, None, None] * uh[None, :, :])
    cv = np.cos(np.pi * F[:, None, None] * vh[None, :, :])
    sv = np.sin(np.pi * F[:, None, None] * vh[None, :, :])
    feats = []
    for i in range(p):
        for j in range(p):
            feats.extend(
                [
                    cv[i] * cu[j],
                    sv[i] * su[j],
                    cv[i] * su[j],
                    sv[i] * cu[j],
                ]
            )
    return np.stack(feats, axis=-1)


def hue_of(p: int, idx: int) -> np.ndarray:
    if p <= 1:
        return np.ones(3, dtype=np.float64)
    return COLOR_PHASES[idx % 3]


def design_p(pf: PrimeField, p: int, H: int, W: int, pidx: int) -> np.ndarray:
    uh, vh = _uhv(pf, H, W)
    hue = hue_of(p, pidx)
    n_pix = H * W
    if p == 1:
        A = np.ones((n_pix, 1), dtype=np.float64)
        block = A[:, None, :] * hue[None, :, None]
        return block.reshape(n_pix * 3, 1)
    n = n_feat(p)
    B = basis_p(p, uh, vh).reshape(n_pix, n)
    block = B[:, None, :] * hue[None, :, None]
    return block.reshape(n_pix * 3, n)


def n_el() -> int:
    return 1 + sum(n_feat(p) for p in EL_PRIMES)


def offsets_el() -> list[tuple[int, int, int]]:
    out = [(1, 0, 1)]
    s = 1
    for p in EL_PRIMES:
        n = n_feat(p)
        out.append((p, s, n))
        s += n
    return out


def _ls(A: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    k = A.shape[1]
    ATA = A.T @ A
    ATA.flat[:: k + 1] += float(lam) * Y.shape[1]
    return np.linalg.solve(ATA, (Y @ A).T).T


def residual_fit(pf: PrimeField, images: list, lam: float = 1e-4) -> np.ndarray:
    """Fit each prime's matrix to leftover; keep the matrix; subtract; next."""
    imgs = [np.asarray(im, dtype=np.float64) for im in images]
    if imgs[0].max() > 1.5:
        imgs = [im / 255.0 for im in imgs]
    imgs = [np.clip(im, 0.0, 1.0) for im in imgs]
    H, W = imgs[0].shape[:2]
    Y = np.stack([im.reshape(-1) for im in imgs])
    C = np.zeros((len(imgs), n_el()), dtype=np.float64)
    pidx = 0
    for p, start, n in offsets_el():
        if p > 1:
            ax = axis_p(p)
            print(
                f"    p={p}  {p}×{p}×{N_PHASE}={n}  leftover={float(np.mean(Y * Y)):.4f}  "
                f"F=[{', '.join(f'{x:g}' for x in ax)}]",
                flush=True,
            )
        else:
            print(f"    p={p}  1×1  leftover={float(np.mean(Y * Y)):.4f}", flush=True)
        A = design_p(pf, p, H, W, pidx if p > 1 else 0)
        cp = _ls(A, Y, lam)
        Y = Y - cp @ A.T
        C[:, start : start + n] = cp
        if p > 1:
            pidx += 1
        del A
    print(f"  leftover after all matrices={float(np.mean(Y * Y)):.4f}", flush=True)
    n_chk = min(32, len(imgs))
    acc = 0.0
    for i in range(n_chk):
        rec = raster_el(pf, C[i], H, W).astype(np.float64) / 255.0
        acc += float(np.mean((rec - imgs[i]) ** 2))
    print(f"  recon MSE vs photo n={n_chk}: {acc / n_chk:.4f}", flush=True)
    return C


def raster_el(pf: PrimeField, amps: np.ndarray, H: int, W: int) -> np.ndarray:
    amps = np.asarray(amps, dtype=np.float64).reshape(-1)
    uh, vh = _uhv(pf, H, W)
    acc = np.zeros((H, W, 3), dtype=np.float64)
    pidx = 0
    for p, start, n in offsets_el():
        hue = hue_of(p, pidx if p > 1 else 0)
        coef = amps[start : start + n]
        if p == 1:
            acc += float(coef[0]) * hue
        else:
            B = basis_p(p, uh, vh)
            acc += (B @ coef)[:, :, None] * hue
            pidx += 1
    img = np.clip(acc, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)


class _Bag:
    def __init__(self, bank):
        self.bank = bank


class GeneratorEl:
    def __init__(self, pf: PrimeField, prior: LatentPrior, gmm: DiagGMM | None, unf: PhaseUnfold, bank: np.ndarray):
        self.pf = pf
        self.field = _Bag(bank)
        self.prior = prior
        self.gmm = gmm
        self.unf = unf

    def raster(self, c, H=96, W=96, backend="cpu"):
        return raster_el(self.pf, c, H, W)

    def stages(self, c):
        c = np.asarray(c, dtype=np.float64).reshape(-1)
        acc = np.zeros_like(c)
        out = []
        for p, start, n in offsets_el():
            acc[start : start + n] = c[start : start + n]
            out.append((p, acc.copy()))
        return out

    def sample(self, n=1, H=96, W=96, temp=0.22, seed=0, backend="cpu", cloud=True):
        rng = np.random.default_rng(seed)
        Cs = []
        for i in range(n):
            if self.gmm is not None:
                z = self.gmm.sample(1, rng)[0]
                pfx = self.prior.decode(z)
            else:
                pfx = rng.normal(size=self.unf.n_prefix) * 0.2
            if pfx.size > self.unf.n_prefix:
                pfx = pfx[: self.unf.n_prefix]
            Cs.append(self.unf.from_prefix(pfx, temp=temp, seed=seed + 11 * i))
        Cs = np.stack(Cs)
        return [self.raster(c, H, W, backend=backend) for c in Cs], Cs

    def morph(self, c0, c1, frames=8, H=96, W=96, backend="cpu"):
        n0 = self.unf.n_prefix
        out = []
        for i in range(frames):
            t = i / max(frames - 1, 1)
            pfx = (1.0 - t) * np.asarray(c0)[:n0] + t * np.asarray(c1)[:n0]
            out.append(self.raster(self.unf.from_prefix(pfx, temp=0.12, seed=i), H, W, backend=backend))
        return out

    def formula(self) -> str:
        g = f"GMM={self.gmm.m}×N({self.gmm.d})" if self.gmm else "no-gmm"
        ph = ",".join(f"{s['p']}:{s['r2_hold']:.2f}" for s in self.unf.steps)
        mats = "+".join(f"{p}×{p}" for p in EL_PRIMES)
        return f"ELEM  1×1+{mats} ×{N_PHASE}ph  n={n_el()}  {g}  phase {ph}"

    def save(self, stem: str) -> None:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(stem) + ".el.npz", bank=self.field.bank)
        self.prior.save(str(stem) + ".prior.npz")
        if self.gmm is not None:
            self.gmm.save(str(stem) + ".gmm.npz")
        payload = {}
        self.unf.save_into(payload)
        payload["primes"] = np.array(self.unf.primes, np.int32)
        np.savez_compressed(str(stem) + ".unf.npz", **payload)
        Path(str(stem) + ".json").write_text(json.dumps({"formula": self.formula()}, indent=2) + "\n")

    @classmethod
    def load(cls, stem: str, codec: str | Path) -> "GeneratorEl":
        stem = Path(stem)
        pf = PrimeField.load(str(codec))
        bank = np.array(np.load(str(stem) + ".el.npz")["bank"], dtype=np.float64)
        prior = LatentPrior.load(str(stem) + ".prior.npz")
        gmm_path = Path(str(stem) + ".gmm.npz")
        gmm = DiagGMM.load(str(gmm_path)) if gmm_path.is_file() else None
        z = np.load(str(stem) + ".unf.npz")
        primes = tuple(int(p) for p in z["primes"])
        unf = PhaseUnfold.load_from(z, primes)
        unf.n_prefix = int(z["n_prefix"])
        return cls(pf, prior, gmm, unf, bank)
