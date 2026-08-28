"""
Structural unfold: each next prime is resample(previous) + a small residual
map from coarser band-PCA. Maps are tiny so 290 stills can actually fit them.

Sampling is ancestral from a prefix GMM — no mixing of training latents.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v1"))

import importlib.util

from scalefield import PrimeField, resample_op  # noqa: E402

_spec = importlib.util.spec_from_file_location("scalegen_v1", ROOT / "gen" / "scalegen.py")
_g1 = importlib.util.module_from_spec(_spec)
sys.modules["scalegen_v1"] = _g1
_spec.loader.exec_module(_g1)
DiagGMM = _g1.DiagGMM


def offsets_of(primes: tuple[int, ...]) -> list[tuple[int, int, int]]:
    out = []
    s = 0
    for p in primes:
        n = p * p
        out.append((p, s, n))
        s += n
    return out


def _pca_fit(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    xc = X - mean
    _U, S, Vt = np.linalg.svd(xc, full_matrices=False)
    k = max(1, min(int(k), Vt.shape[0], max(1, X.shape[0] - 1)))
    vals = (S[:k] ** 2) / max(X.shape[0] - 1, 1)
    return mean, Vt[:k].copy(), vals


def _pca_encode(X: np.ndarray, mean: np.ndarray, comp: np.ndarray, vals: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.maximum(vals, 1e-12))
    z = (X - mean) @ comp.T
    return z / scale


def _pca_decode(z: np.ndarray, mean: np.ndarray, comp: np.ndarray, vals: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.maximum(vals, 1e-12))
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        return mean + (scale * z) @ comp
    return mean + (z * scale) @ comp


def _ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, d = X.shape
    X1 = np.hstack([X, np.ones((n, 1))])
    dd = X1.shape[1]
    if dd <= n:
        xtx = X1.T @ X1
        xtx.flat[:: dd + 1] += float(lam)
        xtx[-1, -1] -= float(lam)
        Wb = np.linalg.solve(xtx, X1.T @ Y)
    else:
        K = X1 @ X1.T
        K.flat[:: n + 1] += float(lam)
        Wb = X1.T @ np.linalg.solve(K, Y)
    return np.ascontiguousarray(Wb[:-1].T), np.ascontiguousarray(Wb[-1])


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean(axis=0)) ** 2)) + 1e-18
    return 1.0 - ss_res / ss_tot


def k_of(p: int, n_train: int) -> int:
    if p <= 1:
        return 1
    return max(2, min(12, p * p - 1, max(2, n_train // 12)))


@dataclass
class Band:
    p: int
    start: int
    n: int
    mean: np.ndarray
    comp: np.ndarray
    vals: np.ndarray


@dataclass
class Step:
    p: int
    start: int
    n: int
    p_prev: int
    scale: float
    W: np.ndarray
    b: np.ndarray
    resid_std: np.ndarray
    r2_train: float
    r2_hold: float


class StructUnfold:
    """c_p = scale * resample(c_{p_prev}) + decode(W z_<p + b)."""

    def __init__(
        self,
        primes: tuple[int, ...],
        prefix_last: int,
        bands: list[Band],
        steps: list[Step],
        D: dict[int, np.ndarray],
        prefix_mean: np.ndarray,
        prefix_comp: np.ndarray,
        prefix_vals: np.ndarray,
    ):
        self.primes = tuple(int(p) for p in primes)
        self.prefix_last = int(prefix_last)
        self.bands = bands
        self.steps = steps
        self.D = D
        self.prefix_mean = prefix_mean
        self.prefix_comp = prefix_comp
        self.prefix_vals = prefix_vals
        self.n = int(sum(p * p for p in self.primes))
        self.n_prefix = int(sum(p * p for p in self.primes if p <= self.prefix_last))
        self.k_prefix = int(self.prefix_comp.shape[0])
        self._band = {b.p: b for b in bands}

    def nparams(self) -> int:
        return int(sum(s.W.size + s.b.size for s in self.steps) + self.prefix_comp.size)

    def _slice(self, c: np.ndarray, p: int) -> np.ndarray:
        b = self._band[p]
        return np.asarray(c[..., b.start : b.start + b.n], dtype=np.float64)

    def encode_prefix(self, c: np.ndarray) -> np.ndarray:
        x = np.asarray(c, dtype=np.float64)[..., : self.n_prefix]
        return _pca_encode(x, self.prefix_mean, self.prefix_comp, self.prefix_vals)

    def decode_prefix(self, z: np.ndarray) -> np.ndarray:
        return _pca_decode(z, self.prefix_mean, self.prefix_comp, self.prefix_vals)

    def _ctx_z(self, c: np.ndarray, p: int) -> np.ndarray:
        """Band-PCA of every coarser prime, plus a few modes of resampled previous."""
        bits = [self.encode_prefix(c)]
        prev = None
        for b in self.bands:
            if b.p >= p:
                break
            if b.p > self.prefix_last:
                g = self._slice(c, b.p)
                bits.append(_pca_encode(g, b.mean, b.comp, b.vals))
            prev = b.p
        if prev is not None:
            up = self.D[p] @ self._slice(c, prev)
            # 6 PCA modes of the upsampled previous grid (or fewer)
            k_up = min(6, up.size)
            # cheap: take the largest-energy DCT-ish by just using leading entries after centering
            # real PCA of upsample would need the fit; use a fixed DCT-like: first k of centered up
            u = up - up.mean()
            bits.append(u[:k_up] / (np.linalg.norm(u) + 1e-8))
        z = np.concatenate(bits, axis=-1)
        return z

    def ctx_dim(self, p: int) -> int:
        # match _ctx_z
        d = self.k_prefix
        prev = None
        for b in self.bands:
            if b.p >= p:
                break
            if b.p > self.prefix_last:
                d += b.comp.shape[0]
            prev = b.p
        if prev is not None:
            d += min(6, p * p)
        return d

    def fill(self, c: np.ndarray, temp: float, rng: np.random.Generator) -> np.ndarray:
        c = np.asarray(c, dtype=np.float64).copy()
        for st in self.steps:
            prev = self._slice(c, st.p_prev)
            up = self.D[st.p] @ prev
            zc = self._ctx_z(c, st.p)
            zr = st.W @ zc + st.b
            if temp > 0:
                zr = zr + temp * st.resid_std * rng.normal(size=st.W.shape[0])
            band = self._band[st.p]
            resid = _pca_decode(zr, band.mean, band.comp, band.vals)
            c[st.start : st.start + st.n] = st.scale * up + resid
        return c

    def from_prefix_z(self, zp: np.ndarray, temp: float = 0.15, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        c = np.zeros(self.n, dtype=np.float64)
        c[: self.n_prefix] = self.decode_prefix(zp)
        return self.fill(c, temp, rng)

    def from_prefix_c(self, c_or_prefix: np.ndarray, temp: float = 0.0, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        c = np.zeros(self.n, dtype=np.float64)
        src = np.asarray(c_or_prefix, dtype=np.float64).reshape(-1)
        c[: self.n_prefix] = src[: self.n_prefix]
        return self.fill(c, temp, rng)

    def stages(self, c: np.ndarray) -> list[tuple[int, np.ndarray]]:
        full = np.asarray(c, dtype=np.float64)
        out = []
        filled = 0
        for p, start, n in offsets_of(self.primes):
            filled = start + n
            part = np.zeros_like(full)
            part[:filled] = full[:filled]
            out.append((p, part))
        return out

    def predict_c(self, c_true: np.ndarray) -> np.ndarray:
        """Teacher-forced: real coarser bands, predict this p. For hold-out R²."""
        c_true = np.asarray(c_true, dtype=np.float64)
        if c_true.ndim == 1:
            c_true = c_true[None, :]
        hat = c_true.copy()
        for st in self.steps:
            pb = self._band[st.p_prev]
            prev = c_true[:, pb.start : pb.start + pb.n]
            up = prev @ self.D[st.p].T
            Zc = np.stack([self._ctx_z(c_true[i], st.p) for i in range(len(c_true))])
            Zr = Zc @ st.W.T + st.b
            band = self._band[st.p]
            resid = _pca_decode(Zr, band.mean, band.comp, band.vals)
            hat[:, st.start : st.start + st.n] = st.scale * up + resid
        return hat

    def holdout_r2(self, bank: np.ndarray) -> dict[int, float]:
        hat = self.predict_c(bank)
        out = {}
        for st in self.steps:
            y = bank[:, st.start : st.start + st.n]
            yhat = hat[:, st.start : st.start + st.n]
            out[int(st.p)] = _r2(y, yhat)
        return out

    def formula(self) -> str:
        last = ",".join(f"{s.p}:{s.r2_hold:.2f}" for s in self.steps[-4:])
        return (
            f"STRUCT  resample(p-1)+PCA-resid  prefix≤{self.prefix_last} "
            f"kpfx={self.k_prefix}  maps={self.nparams()}  holdR2[{last}]"
        )

    def sgd_step(self, bank: np.ndarray, lr: float = 3e-3, batch: int = 48, rng: np.random.Generator | None = None) -> float:
        """Tiny CPU step on residual z. Returns train residual mse."""
        rng = rng or np.random.default_rng()
        n = len(bank)
        idx = rng.choice(n, size=min(batch, n), replace=False)
        C = bank[idx]
        loss = 0.0
        n_st = 0
        for st in self.steps:
            band = self._band[st.p]
            prev = C[:, self._band[st.p_prev].start : self._band[st.p_prev].start + self._band[st.p_prev].n]
            up = prev @ self.D[st.p].T
            Y = C[:, st.start : st.start + st.n]
            R = Y - st.scale * up
            Zt = _pca_encode(R, band.mean, band.comp, band.vals)
            Zc = np.stack([self._ctx_z(C[i], st.p) for i in range(len(C))])
            pred = Zc @ st.W.T + st.b
            err = pred - Zt
            loss += float(np.mean(err * err))
            n_st += 1
            # dW: (k, d)  err (B,k)  Zc (B,d)
            gW = (err.T @ Zc) / len(C)
            gb = err.mean(axis=0)
            st.W -= lr * gW
            st.b -= lr * gb
        return loss / max(n_st, 1)

    def save(self, path: str) -> None:
        payload: dict = {
            "primes": np.array(self.primes, dtype=np.int32),
            "prefix_last": np.int32(self.prefix_last),
            "prefix_mean": self.prefix_mean,
            "prefix_comp": self.prefix_comp.astype(np.float32),
            "prefix_vals": self.prefix_vals,
            "n_bands": np.int32(len(self.bands)),
            "n_steps": np.int32(len(self.steps)),
        }
        for i, b in enumerate(self.bands):
            payload[f"bp_{i}"] = np.int32(b.p)
            payload[f"bs_{i}"] = np.int32(b.start)
            payload[f"bn_{i}"] = np.int32(b.n)
            payload[f"bm_{i}"] = b.mean.astype(np.float64)
            payload[f"bc_{i}"] = b.comp.astype(np.float32)
            payload[f"bv_{i}"] = b.vals
        for i, st in enumerate(self.steps):
            payload[f"sp_{i}"] = np.int32(st.p)
            payload[f"ss_{i}"] = np.int32(st.start)
            payload[f"sn_{i}"] = np.int32(st.n)
            payload[f"sprev_{i}"] = np.int32(st.p_prev)
            payload[f"sscale_{i}"] = np.float64(st.scale)
            payload[f"sW_{i}"] = st.W.astype(np.float32)
            payload[f"sb_{i}"] = st.b
            payload[f"sstd_{i}"] = st.resid_std
            payload[f"str_{i}"] = np.float64(st.r2_train)
            payload[f"sho_{i}"] = np.float64(st.r2_hold)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "StructUnfold":
        z = np.load(path)
        primes = tuple(int(p) for p in z["primes"].tolist())
        offs = {p: (s, n) for p, s, n in offsets_of(primes)}
        bands = []
        for i in range(int(z["n_bands"])):
            p = int(z[f"bp_{i}"])
            bands.append(
                Band(
                    p=p,
                    start=int(z[f"bs_{i}"]),
                    n=int(z[f"bn_{i}"]),
                    mean=np.array(z[f"bm_{i}"], dtype=np.float64),
                    comp=np.array(z[f"bc_{i}"], dtype=np.float64),
                    vals=np.array(z[f"bv_{i}"], dtype=np.float64),
                )
            )
        steps = []
        D = {}
        for i in range(int(z["n_steps"])):
            p = int(z[f"sp_{i}"])
            p_prev = int(z[f"sprev_{i}"])
            D[p] = resample_op(p_prev, p)
            steps.append(
                Step(
                    p=p,
                    start=int(z[f"ss_{i}"]),
                    n=int(z[f"sn_{i}"]),
                    p_prev=p_prev,
                    scale=float(z[f"sscale_{i}"]),
                    W=np.array(z[f"sW_{i}"], dtype=np.float64),
                    b=np.array(z[f"sb_{i}"], dtype=np.float64),
                    resid_std=np.array(z[f"sstd_{i}"], dtype=np.float64),
                    r2_train=float(z[f"str_{i}"]),
                    r2_hold=float(z[f"sho_{i}"]),
                )
            )
        return cls(
            primes=primes,
            prefix_last=int(z["prefix_last"]),
            bands=bands,
            steps=steps,
            D=D,
            prefix_mean=np.array(z["prefix_mean"], dtype=np.float64),
            prefix_comp=np.array(z["prefix_comp"], dtype=np.float64),
            prefix_vals=np.array(z["prefix_vals"], dtype=np.float64),
        )

    @classmethod
    def fit(
        cls,
        bank: np.ndarray,
        primes: tuple[int, ...],
        prefix_last: int = 3,
        lam: float = 0.3,
        hold: float = 0.2,
        seed: int = 0,
    ) -> tuple["StructUnfold", dict]:
        bank = np.asarray(bank, dtype=np.float64)
        rng = np.random.default_rng(seed)
        n = len(bank)
        idx = rng.permutation(n)
        n_te = max(32, int(round(n * hold)))
        te, tr = idx[:n_te], idx[n_te:]
        Ctr, Cte = bank[tr], bank[te]
        offs = offsets_of(primes)
        n_prefix = sum(p * p for p, _s, _n in offs if p <= prefix_last)
        k_pfx = min(8, n_prefix, max(2, len(tr) // 12))
        pmean, pcomp, pvals = _pca_fit(Ctr[:, :n_prefix], k_pfx)

        bands: list[Band] = []
        D: dict[int, np.ndarray] = {}
        prev_p = None
        for p, start, nn in offs:
            kp = k_of(p, len(tr))
            mean, comp, vals = _pca_fit(Ctr[:, start : start + nn], kp)
            bands.append(Band(p=p, start=start, n=nn, mean=mean, comp=comp, vals=vals))
            if prev_p is not None and p > prefix_last:
                D[p] = resample_op(prev_p, p)
            if p >= 1:
                prev_p = p

        tmp = cls(
            primes=primes,
            prefix_last=prefix_last,
            bands=bands,
            steps=[],
            D=D,
            prefix_mean=pmean,
            prefix_comp=pcomp,
            prefix_vals=pvals,
        )

        steps: list[Step] = []
        print(
            f"  struct fit  train={len(tr)} hold={len(te)}  prefix≤{prefix_last} ({n_prefix}→{k_pfx})  lam={lam:g}",
            flush=True,
        )
        band_by_p = {b.p: b for b in bands}
        prev_p = None
        for p, start, nn in offs:
            if prev_p is None:
                prev_p = p
                continue
            if p <= prefix_last:
                prev_p = p
                continue
            Dp = D[p]
            prev_b = band_by_p[prev_p]
            Ytr = Ctr[:, start : start + nn]
            Yte = Cte[:, start : start + nn]
            Utr = Ctr[:, prev_b.start : prev_b.start + prev_b.n] @ Dp.T
            Ute = Cte[:, prev_b.start : prev_b.start + prev_b.n] @ Dp.T
            denom = float(np.sum(Utr * Utr)) + 1e-12
            scale = float(np.sum(Ytr * Utr) / denom)
            Rtr = Ytr - scale * Utr
            Rte = Yte - scale * Ute
            base_te = _r2(Yte, scale * Ute)
            band = band_by_p[p]
            # residual PCA was fit on raw grids; refit on residual of train
            rmean, rcomp, rvals = _pca_fit(Rtr, k_of(p, len(tr)))
            band.mean, band.comp, band.vals = rmean, rcomp, rvals
            Ztr = _pca_encode(Rtr, rmean, rcomp, rvals)
            Zte = _pca_encode(Rte, rmean, rcomp, rvals)
            Xtr = np.stack([tmp._ctx_z(Ctr[i], p) for i in range(len(Ctr))])
            Xte = np.stack([tmp._ctx_z(Cte[i], p) for i in range(len(Cte))])
            W, b = _ridge(Xtr, Ztr, lam=lam)
            pred_tr = Xtr @ W.T + b
            pred_te = Xte @ W.T + b
            Yhat_tr = scale * Utr + _pca_decode(pred_tr, rmean, rcomp, rvals)
            Yhat_te = scale * Ute + _pca_decode(pred_te, rmean, rcomp, rvals)
            r_tr = _r2(Ytr, Yhat_tr)
            r_te = _r2(Yte, Yhat_te)
            resid = Ztr - pred_tr
            std = np.maximum(np.std(resid, axis=0), 1e-4)
            steps.append(
                Step(
                    p=p,
                    start=start,
                    n=nn,
                    p_prev=prev_p,
                    scale=scale,
                    W=W,
                    b=b,
                    resid_std=std,
                    r2_train=r_tr,
                    r2_hold=r_te,
                )
            )
            print(
                f"    p={p:2d}←{prev_p:<2d}  ctx={Xtr.shape[1]} k={W.shape[0]}  "
                f"scale={scale:.3f}  R² resamp-hold={base_te:.3f}  "
                f"train={r_tr:.3f} hold={r_te:.3f}",
                flush=True,
            )
            prev_p = p

        model = cls(
            primes=primes,
            prefix_last=prefix_last,
            bands=bands,
            steps=steps,
            D=D,
            prefix_mean=pmean,
            prefix_comp=pcomp,
            prefix_vals=pvals,
        )
        stats = {
            "n_train": int(len(tr)),
            "n_hold": int(len(te)),
            "te_idx": te.astype(np.int32),
            "tr_idx": tr.astype(np.int32),
            "r2_train": {int(s.p): s.r2_train for s in steps},
            "r2_hold": {int(s.p): s.r2_hold for s in steps},
            "mean_hold": float(np.mean([s.r2_hold for s in steps])),
        }
        return model, stats


class GeneratorStruct:
    def __init__(self, field: PrimeField, unf: StructUnfold, gmm: DiagGMM | None):
        self.field = field
        self.unf = unf
        self.gmm = gmm

    def raster(self, c: np.ndarray, H: int = 96, W: int = 96, backend: str = "cpu") -> np.ndarray:
        self.field._apply_params(c)
        return self.field.decode(H, W, stretch=False, backend=backend)

    def sample(self, n=1, H=96, W=96, temp=0.2, seed=0, backend="cpu", cloud=True):
        rng = np.random.default_rng(seed)
        Cs = []
        for i in range(n):
            if self.gmm is not None:
                zp = self.gmm.sample(1, rng)[0]
            else:
                zp = rng.normal(size=self.unf.k_prefix)
            Cs.append(self.unf.from_prefix_z(zp, temp=temp, seed=seed + 17 * i + 1))
        Cs = np.stack(Cs)
        return [self.raster(c, H, W, backend=backend) for c in Cs], Cs

    def morph(self, c0, c1, frames: int = 8, H: int = 96, W: int = 96, backend: str = "cpu"):
        z0 = self.unf.encode_prefix(c0)
        z1 = self.unf.encode_prefix(c1)
        out = []
        for i in range(frames):
            t = i / max(frames - 1, 1)
            zp = (1.0 - t) * z0 + t * z1
            c = self.unf.from_prefix_z(zp, temp=0.05, seed=i)
            out.append(self.raster(c, H, W, backend=backend))
        return out

    def stages(self, c: np.ndarray):
        return self.unf.stages(c)

    def formula(self) -> str:
        g = f"  pfxGMM={self.gmm.m}×N({self.gmm.d})" if self.gmm is not None else ""
        return self.unf.formula() + g

    def save(self, stem: str) -> None:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        self.field.save(str(stem) + ".field.npz")
        self.unf.save(str(stem) + ".struct.npz")
        if self.gmm is not None:
            self.gmm.save(str(stem) + ".gmm.npz")
        Path(str(stem) + ".json").write_text(
            json.dumps(
                {
                    "formula": self.formula(),
                    "n": self.unf.n,
                    "n_prefix": self.unf.n_prefix,
                    "nparams": self.unf.nparams(),
                    "r2_hold": {int(s.p): s.r2_hold for s in self.unf.steps},
                    "r2_train": {int(s.p): s.r2_train for s in self.unf.steps},
                },
                indent=2,
            )
            + "\n"
        )
