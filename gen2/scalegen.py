"""ScaleGen v2: GMM on coarse z, linear map to fine z. Field frozen."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v1"))

import importlib.util

_spec = importlib.util.spec_from_file_location("scalegen_v1", ROOT / "gen" / "scalegen.py")
_g1 = importlib.util.module_from_spec(_spec)
sys.modules["scalegen_v1"] = _g1
_spec.loader.exec_module(_g1)
DiagGMM = _g1.DiagGMM
Generator = _g1.Generator
LatentPrior = _g1.LatentPrior


class CoarseFine:
    """z_fine = W @ z_coarse + b. Only this + GMM are trained."""

    def __init__(self, W: np.ndarray, b: np.ndarray, k_coarse: int):
        self.W = np.asarray(W, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.k_coarse = int(k_coarse)

    def predict_fine(self, zc: np.ndarray) -> np.ndarray:
        zc = np.asarray(zc, dtype=np.float64)
        if zc.ndim == 1:
            return self.W @ zc + self.b
        return zc @ self.W.T + self.b

    def full_z(self, zc: np.ndarray) -> np.ndarray:
        zc = np.atleast_2d(zc)
        zf = self.predict_fine(zc)
        return np.hstack([zc, zf])

    @classmethod
    def fit(cls, Z: np.ndarray, k_coarse: int, ridge: float = 1e-2) -> "CoarseFine":
        Z = np.asarray(Z, dtype=np.float64)
        zc, zf = Z[:, :k_coarse], Z[:, k_coarse:]
        # min ||zc W.T + b - zf||
        X = np.hstack([zc, np.ones((len(Z), 1))])
        xtx = X.T @ X
        xtx.flat[:: xtx.shape[0] + 1] += ridge
        wb = np.linalg.solve(xtx, X.T @ zf)
        W = wb[:-1].T
        b = wb[-1]
        return cls(W, b, k_coarse)

    def save(self, path: str) -> None:
        np.savez_compressed(path, W=self.W, b=self.b, k_coarse=np.int32(self.k_coarse))

    @classmethod
    def load(cls, path: str) -> "CoarseFine":
        z = np.load(path)
        return cls(z["W"], z["b"], int(z["k_coarse"]))


class GeneratorV2(Generator):
    def __init__(self, field, prior: LatentPrior, gmm: DiagGMM | None, cf: CoarseFine):
        super().__init__(field, prior, gmm=gmm)
        self.cf = cf

    def sample(self, n=1, H=256, W=256, temp=0.4, seed=0, backend="cpu", cloud=True):
        rng = np.random.default_rng(seed)
        bank = self.field.bank
        if bank is None:
            return super().sample(n, H, W, temp, seed, backend, cloud)
        Z = np.stack([self.prior.encode(c) for c in bank])
        kc = self.cf.k_coarse
        Cs = []
        for _ in range(n):
            # Local mixes of nearby real z's are the look that worked.
            # GMM is a small side door, not the main generator.
            if self.gmm is not None and rng.random() < 0.12:
                zc = self.gmm.sample(1, rng)[0]
                z = self.cf.full_z(zc)[0]
                z[:kc] += rng.normal(0.0, 0.03, size=kc)
            else:
                j = int(rng.integers(0, len(bank)))
                d = np.sum((Z - Z[j]) ** 2, axis=1)
                nn = np.argsort(d)[:6]
                take = nn[: int(rng.integers(2, 4))]
                w = rng.dirichlet(np.ones(len(take)) * 4.0)
                z = w @ Z[take]
                z[:kc] += rng.normal(0.0, 0.035, size=kc)
            Cs.append(self.prior.decode(z))
        Cs = np.stack(Cs)
        return [self.raster(c, H, W, backend=backend) for c in Cs], Cs

    def formula(self) -> str:
        g = f"GMM={self.gmm.m}×N({self.cf.k_coarse})" if self.gmm else "no-gmm"
        return (
            f"ScaleGen2  z_coarse~{g}  z_fine=W z_c+b  "
            f"k={self.prior.k} kc={self.cf.k_coarse}  N_field={self.field.nparams()}"
        )
