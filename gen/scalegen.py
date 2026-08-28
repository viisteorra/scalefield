"""
ScaleGen — sample and transform PrimeField latents.

Visual decoder lives in ../v1 (PrimeField). This file is the prior on c:
a PCA Gaussian fit to the encoded training cloud. Generate = draw z,
c = μ + B z, raster at any resolution. Transforms happen in z (or c).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

V1 = Path(__file__).resolve().parents[1] / "v1"
if str(V1) not in sys.path:
    sys.path.insert(0, str(V1))

from scalefield import PrimeField, save_png  # noqa: E402


def _pca(X: np.ndarray, k: int | None = None, energy: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X (K, N) already centered. Returns components (k, N), values (k,), k used."""
    K, N = X.shape
    # economy SVD; rank ≤ min(K-1, N)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # drop the last if it's numerical dust
    tot = float(np.sum(S * S)) + 1e-18
    cum = np.cumsum(S * S) / tot
    if k is None:
        k = int(np.searchsorted(cum, energy) + 1)
    k = max(1, min(k, Vt.shape[0], K - 1 if K > 1 else 1))
    return Vt[:k].copy(), (S[:k] ** 2) / max(K - 1, 1), k


@dataclass
class LatentPrior:
    """c ≈ mean + components.T @ (sqrt(values) * z), z ~ N(0, I)."""

    mean: np.ndarray
    components: np.ndarray  # (k, N)
    values: np.ndarray  # (k,)
    resid_std: float = 0.0

    @property
    def k(self) -> int:
        return int(self.components.shape[0])

    @property
    def n(self) -> int:
        return int(self.mean.size)

    def nparams(self) -> int:
        return int(self.mean.size + self.components.size + self.values.size + 1)

    def truncated(self, k: int) -> "LatentPrior":
        k = max(1, min(int(k), self.k))
        return LatentPrior(
            mean=self.mean,
            components=self.components[:k],
            values=self.values[:k],
            resid_std=self.resid_std,
        )

    def encode(self, c: np.ndarray) -> np.ndarray:
        d = np.asarray(c, dtype=np.float64).reshape(-1) - self.mean
        # z in whitened PCA coords
        scale = np.sqrt(np.maximum(self.values, 1e-12))
        return (self.components @ d) / scale

    def decode(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        scale = np.sqrt(np.maximum(self.values, 1e-12))
        return self.mean + self.components.T @ (scale * z[: self.k])

    def sample_c(self, n: int = 1, temp: float = 0.4, seed: int = 0, resid: bool = False) -> np.ndarray:
        rng = np.random.default_rng(seed)
        z = rng.normal(0.0, temp, size=(n, self.k))
        scale = np.sqrt(np.maximum(self.values, 1e-12))
        C = self.mean + (z * scale) @ self.components
        if resid and self.resid_std > 0:
            C = C + rng.normal(0.0, temp * self.resid_std, size=C.shape)
        return C

    def sample_cloud(
        self,
        bank: np.ndarray,
        n: int = 1,
        n_mix: int = 2,
        temp: float = 0.12,
        seed: int = 0,
    ) -> np.ndarray:
        """Barycentric mix of training latents + a little PCA noise. Stays on the cloud."""
        rng = np.random.default_rng(seed)
        bank = np.asarray(bank, dtype=np.float64)
        K = len(bank)
        n_mix = max(1, min(n_mix, K))
        out = np.empty((n, bank.shape[1]), dtype=np.float64)
        for i in range(n):
            idx = rng.choice(K, size=n_mix, replace=False)
            w = rng.dirichlet(np.ones(n_mix))
            c = w @ bank[idx]
            z = self.encode(c)
            z = z + rng.normal(0.0, temp, size=z.shape)
            out[i] = self.decode(z)
        return out

    def mix(self, c0: np.ndarray, c1: np.ndarray, t: float) -> np.ndarray:
        """Lerp in PCA space (not raw c) so midpoints stay on the manifold."""
        z0, z1 = self.encode(c0), self.encode(c1)
        return self.decode((1.0 - t) * z0 + t * z1)

    def walk(self, z: np.ndarray, step: float, rng: np.random.Generator) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) + rng.normal(0.0, step, size=self.k)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            mean=self.mean,
            components=self.components,
            values=self.values,
            resid_std=np.float64(self.resid_std),
        )

    @classmethod
    def load(cls, path: str) -> "LatentPrior":
        z = np.load(path)
        return cls(
            mean=np.array(z["mean"], dtype=np.float64),
            components=np.array(z["components"], dtype=np.float64),
            values=np.array(z["values"], dtype=np.float64),
            resid_std=float(z["resid_std"]) if "resid_std" in z.files else 0.0,
        )

    @classmethod
    def fit(cls, bank: np.ndarray, k: int | None = 48, energy: float = 0.95) -> "LatentPrior":
        X = np.asarray(bank, dtype=np.float64)
        mean = X.mean(axis=0)
        xc = X - mean
        comp, vals, k_use = _pca(xc, k=k, energy=energy)
        recon = mean + xc @ comp.T @ comp
        resid = float(np.std(X - recon))
        return cls(mean=mean, components=comp, values=vals, resid_std=resid)


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis) + 1e-18)


class DiagGMM:
    """Diagonal Gaussian mixture on z. The thing we actually train."""

    def __init__(self, means: np.ndarray, vars_: np.ndarray, weights: np.ndarray):
        self.means = np.asarray(means, dtype=np.float64)
        self.vars_ = np.maximum(np.asarray(vars_, dtype=np.float64), 1e-4)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights = self.weights / self.weights.sum()

    @property
    def m(self) -> int:
        return int(self.means.shape[0])

    @property
    def d(self) -> int:
        return int(self.means.shape[1])

    def nparams(self) -> int:
        return int(self.means.size + self.vars_.size + self.weights.size)

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        """(n, d) → (n, m) log p(x | component)."""
        X = np.asarray(X, dtype=np.float64)
        diff = X[:, None, :] - self.means[None, :, :]
        inv = 1.0 / self.vars_
        quad = np.sum(diff * diff * inv[None, :, :], axis=-1)
        logdet = np.sum(np.log(self.vars_), axis=-1)
        return -0.5 * (self.d * np.log(2.0 * np.pi) + logdet[None, :] + quad)

    def nll(self, X: np.ndarray) -> float:
        lp = self.log_prob(X) + np.log(self.weights + 1e-18)
        return float(-np.mean(_logsumexp(lp, axis=1)))

    def e_step(self, X: np.ndarray) -> np.ndarray:
        lp = self.log_prob(X) + np.log(self.weights + 1e-18)
        lse = _logsumexp(lp, axis=1)[:, None]
        return np.exp(lp - lse)

    def m_step(self, X: np.ndarray, r: np.ndarray, var_floor: float = 1e-3) -> None:
        X = np.asarray(X, dtype=np.float64)
        nk = r.sum(axis=0) + 1e-8
        self.weights = nk / r.shape[0]
        self.means = (r.T @ X) / nk[:, None]
        diff = X[:, None, :] - self.means[None, :, :]
        self.vars_ = np.maximum(np.sum(r[:, :, None] * diff * diff, axis=0) / nk[:, None], var_floor)

    def em_step(self, X: np.ndarray, var_floor: float = 1e-3) -> float:
        r = self.e_step(X)
        self.m_step(X, r, var_floor=var_floor)
        return self.nll(X)

    def split(self, rng: np.random.Generator) -> None:
        j = int(np.argmax(self.weights))
        bump = 0.25 * np.sqrt(self.vars_[j]) * rng.normal(size=self.d)
        self.means = np.vstack([self.means, self.means[j] + bump])
        self.vars_ = np.vstack([self.vars_, self.vars_[j]])
        w = self.weights[j] * 0.5
        self.weights[j] = w
        self.weights = np.append(self.weights, w)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        comp = rng.choice(self.m, size=n, p=self.weights)
        z = np.empty((n, self.d), dtype=np.float64)
        for i, j in enumerate(comp):
            z[i] = rng.normal(self.means[j], np.sqrt(self.vars_[j]))
        return z

    @classmethod
    def from_data(cls, X: np.ndarray) -> "DiagGMM":
        X = np.asarray(X, dtype=np.float64)
        return cls(X.mean(axis=0, keepdims=True), X.var(axis=0, keepdims=True) + 1e-3, np.array([1.0]))

    def save(self, path: str) -> None:
        np.savez_compressed(path, means=self.means, vars_=self.vars_, weights=self.weights)

    @classmethod
    def load(cls, path: str) -> "DiagGMM":
        z = np.load(path)
        return cls(z["means"], z["vars_"], z["weights"])


class Generator:
    def __init__(self, field: PrimeField, prior: LatentPrior, gmm: DiagGMM | None = None):
        if prior.n != field.nparams():
            raise ValueError(f"prior N={prior.n} != field N={field.nparams()}")
        self.field = field
        self.prior = prior
        self.gmm = gmm

    def _set_c(self, c: np.ndarray) -> None:
        self.field._apply_params(c)

    def raster(self, c: np.ndarray, H: int = 256, W: int = 256, backend: str = "cpu") -> np.ndarray:
        self._set_c(c)
        return self.field.decode(H, W, stretch=False, backend=backend)

    def sample(self, n: int = 1, H: int = 256, W: int = 256, temp: float = 0.4, seed: int = 0, backend: str = "cpu", cloud: bool = True):
        rng = np.random.default_rng(seed)
        if self.gmm is not None:
            z = self.gmm.sample(n, rng)
            Cs = np.stack([self.prior.decode(zi) for zi in z])
        elif cloud and self.field.bank is not None:
            Cs = self.prior.sample_cloud(self.field.bank, n=n, temp=min(temp, 0.2), seed=seed)
        else:
            Cs = self.prior.sample_c(n, temp=temp, seed=seed)
        return [self.raster(c, H, W, backend=backend) for c in Cs], Cs

    def morph(self, c0, c1, frames: int = 8, H: int = 256, W: int = 256, backend: str = "cpu"):
        out = []
        for i in range(frames):
            t = i / max(frames - 1, 1)
            out.append(self.raster(self.prior.mix(c0, c1, t), H, W, backend=backend))
        return out

    def from_z(self, z: np.ndarray, H: int = 256, W: int = 256, backend: str = "cpu") -> np.ndarray:
        return self.raster(self.prior.decode(z), H, W, backend=backend)

    def formula(self) -> str:
        g = f"  GMM={self.gmm.m}×N({self.gmm.d})" if self.gmm is not None else ""
        return (
            f"ScaleGen  c = μ + B z{g}  "
            f"N_field={self.field.nparams()}  k={self.prior.k}  "
            f"N_prior={self.prior.nparams()}  resid={self.prior.resid_std:.4f}  "
            f"| {self.field.formula()}"
        )

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.field.save(str(path.with_suffix(".field.npz")))
        self.prior.save(str(path.with_suffix(".prior.npz")))
        if self.gmm is not None:
            self.gmm.save(str(path.with_suffix(".gmm.npz")))
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "k": self.prior.k,
                    "n": self.prior.n,
                    "n_prior": self.prior.nparams(),
                    "resid_std": self.prior.resid_std,
                    "formula": self.formula(),
                },
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def load(cls, stem: str) -> "Generator":
        stem = Path(stem)
        if stem.suffix:
            stem = stem.with_suffix("")
        field = PrimeField.load(str(stem) + ".field.npz")
        prior = LatentPrior.load(str(stem) + ".prior.npz")
        gmm_path = Path(str(stem) + ".gmm.npz")
        gmm = DiagGMM.load(str(gmm_path)) if gmm_path.exists() else None
        return cls(field, prior, gmm=gmm)


def grid(imgs: list[np.ndarray], cols: int = 4) -> np.ndarray:
    if not imgs:
        raise ValueError("no images")
    cols = max(1, cols)
    rows = (len(imgs) + cols - 1) // cols
    h, w = imgs[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=imgs[0].dtype)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = im
    return canvas
