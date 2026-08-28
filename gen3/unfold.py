"""
Prime unfold: c_p | c_<p. Generation is filling 1×1 → 31×31.

The frozen PrimeField is the decoder. This file is only the transform
that populates the matrices from coarsest to finest.
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

from scalefield import PrimeField  # noqa: E402

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


def prefix_len(primes: tuple[int, ...], last: int) -> int:
    n = 0
    for p in primes:
        n += p * p
        if p == last:
            return n
        if p > last:
            break
    raise ValueError(f"prefix last {last} not in primes {primes}")


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float = 1e-2) -> tuple[np.ndarray, np.ndarray]:
    """Y ≈ X W.T + b. Dual form when features > samples (this bank)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, d = X.shape
    X1 = np.hstack([X, np.ones((n, 1))])
    dd = X1.shape[1]
    ny = Y.shape[1]
    if dd <= n:
        xtx = X1.T @ X1
        xtx.flat[:: dd + 1] += float(lam)
        xtx[-1, -1] -= float(lam)  # don't ridge the bias
        Wb = np.linalg.solve(xtx, X1.T @ Y)
    else:
        K = X1 @ X1.T
        K.flat[:: n + 1] += float(lam)
        Wb = X1.T @ np.linalg.solve(K, Y)
    W = np.ascontiguousarray(Wb[:-1].T)  # (ny, d)
    b = np.ascontiguousarray(Wb[-1])
    if W.shape != (ny, d) or b.shape != (ny,):
        raise RuntimeError(f"ridge shape W{W.shape} b{b.shape} expected ({ny},{d})")
    return W, b


@dataclass
class Step:
    p: int
    start: int
    n: int
    W: np.ndarray
    b: np.ndarray
    resid_std: np.ndarray
    r2: float


class Unfold:
    """Ancestral maps c_p | c_<p for every prime after the prefix."""

    def __init__(
        self,
        primes: tuple[int, ...],
        prefix_last: int,
        steps: list[Step],
        x_mean: np.ndarray,
        prefix_mean: np.ndarray,
        prefix_std: np.ndarray,
    ):
        self.primes = tuple(int(p) for p in primes)
        self.prefix_last = int(prefix_last)
        self.steps = steps
        self.x_mean = np.asarray(x_mean, dtype=np.float64)
        self.prefix_mean = np.asarray(prefix_mean, dtype=np.float64)
        self.prefix_std = np.asarray(prefix_std, dtype=np.float64)
        self.n = int(sum(p * p for p in self.primes))
        self.n_prefix = int(self.prefix_mean.size)

    def nparams(self) -> int:
        return int(sum(s.W.size + s.b.size + s.resid_std.size for s in self.steps))

    def fill(self, c: np.ndarray, temp: float, rng: np.random.Generator, from_idx: int = 0) -> np.ndarray:
        """In-place unfold of primes after prefix. Returns c."""
        c = np.asarray(c, dtype=np.float64)
        for i, st in enumerate(self.steps):
            if i < from_idx:
                continue
            x = c[: st.start]
            y = st.W @ x + st.b
            if temp > 0:
                y = y + temp * st.resid_std * rng.normal(size=st.n)
            c[st.start : st.start + st.n] = y
        return c

    def from_prefix(self, prefix: np.ndarray, temp: float = 0.0, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        c = np.zeros(self.n, dtype=np.float64)
        pfx = np.asarray(prefix, dtype=np.float64).reshape(-1)
        if pfx.size != self.n_prefix:
            raise ValueError(f"prefix {pfx.size} != {self.n_prefix}")
        c[: self.n_prefix] = pfx
        return self.fill(c, temp, rng)

    def complete(self, c: np.ndarray, temp: float = 0.0, seed: int = 0) -> np.ndarray:
        """Keep whatever is already in c (usually the prefix) and unfold the rest."""
        rng = np.random.default_rng(seed)
        out = np.array(c, dtype=np.float64, copy=True)
        out[self.n_prefix :] = 0.0
        return self.fill(out, temp, rng)

    def stages(self, c: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Partial fields: only primes ≤ p populated. The generation filmstrip."""
        full = np.asarray(c, dtype=np.float64)
        out = []
        filled = 0
        for p, start, n in offsets_of(self.primes):
            filled = start + n
            part = np.zeros_like(full)
            part[:filled] = full[:filled]
            out.append((p, part))
        return out

    def predict_bank(self, bank: np.ndarray) -> tuple[np.ndarray, dict]:
        """Teacher-forced per-prime R² on a populated bank."""
        bank = np.asarray(bank, dtype=np.float64)
        hat = bank.copy()
        stats = {}
        for st in self.steps:
            X = bank[:, : st.start]
            Y = bank[:, st.start : st.start + st.n]
            Yh = X @ st.W.T + st.b
            hat[:, st.start : st.start + st.n] = Yh
            ss_res = float(np.sum((Y - Yh) ** 2))
            ss_tot = float(np.sum((Y - Y.mean(axis=0)) ** 2)) + 1e-18
            stats[int(st.p)] = {
                "r2": 1.0 - ss_res / ss_tot,
                "rmse": float(np.sqrt(np.mean((Y - Yh) ** 2))),
            }
        return hat, stats

    def formula(self) -> str:
        bits = "→".join(str(p) for p in self.primes)
        r2s = ",".join(f"{s.p}:{s.r2:.2f}" for s in self.steps[-4:])
        return (
            f"UNFOLD  {bits}  prefix≤{self.prefix_last} ({self.n_prefix})  "
            f"N={self.n}  maps={self.nparams()}  lastR2[{r2s}]"
        )

    def save(self, path: str) -> None:
        payload = {
            "primes": np.array(self.primes, dtype=np.int32),
            "prefix_last": np.int32(self.prefix_last),
            "x_mean": self.x_mean,
            "prefix_mean": self.prefix_mean,
            "prefix_std": self.prefix_std,
            "n_steps": np.int32(len(self.steps)),
        }
        for i, st in enumerate(self.steps):
            payload[f"p_{i}"] = np.int32(st.p)
            payload[f"start_{i}"] = np.int32(st.start)
            payload[f"n_{i}"] = np.int32(st.n)
            payload[f"W_{i}"] = st.W.astype(np.float32)
            payload[f"b_{i}"] = st.b.astype(np.float64)
            payload[f"resid_{i}"] = st.resid_std.astype(np.float64)
            payload[f"r2_{i}"] = np.float64(st.r2)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "Unfold":
        z = np.load(path)
        primes = tuple(int(p) for p in z["primes"].tolist())
        n_steps = int(z["n_steps"])
        steps = []
        for i in range(n_steps):
            steps.append(
                Step(
                    p=int(z[f"p_{i}"]),
                    start=int(z[f"start_{i}"]),
                    n=int(z[f"n_{i}"]),
                    W=np.array(z[f"W_{i}"], dtype=np.float64),
                    b=np.array(z[f"b_{i}"], dtype=np.float64),
                    resid_std=np.array(z[f"resid_{i}"], dtype=np.float64),
                    r2=float(z[f"r2_{i}"]),
                )
            )
        return cls(
            primes=primes,
            prefix_last=int(z["prefix_last"]),
            steps=steps,
            x_mean=np.array(z["x_mean"], dtype=np.float64),
            prefix_mean=np.array(z["prefix_mean"], dtype=np.float64),
            prefix_std=np.array(z["prefix_std"], dtype=np.float64),
        )

    @classmethod
    def fit(
        cls,
        bank: np.ndarray,
        primes: tuple[int, ...],
        prefix_last: int = 3,
        lam: float = 1e-2,
    ) -> "Unfold":
        bank = np.asarray(bank, dtype=np.float64)
        offs = offsets_of(primes)
        n_prefix = prefix_len(primes, prefix_last)
        steps: list[Step] = []
        print(
            f"  unfold fit  bank={bank.shape}  prefix≤{prefix_last} ({n_prefix})  "
            f"ridge={lam:g}",
            flush=True,
        )
        for p, start, n in offs:
            if p <= prefix_last:
                continue
            X = bank[:, :start]
            Y = bank[:, start : start + n]
            print(f"    p={p:2d}  X={X.shape} Y={Y.shape}  dual={X.shape[1]+1 > X.shape[0]}", flush=True)
            W, b = ridge_fit(X, Y, lam=lam)
            Yh = X @ W.T + b
            resid = Y - Yh
            ss_res = float(np.sum(resid * resid))
            ss_tot = float(np.sum((Y - Y.mean(axis=0)) ** 2)) + 1e-18
            r2 = 1.0 - ss_res / ss_tot
            std = np.std(resid, axis=0)
            std = np.maximum(std, 1e-4)
            steps.append(Step(p=p, start=start, n=n, W=W, b=b, resid_std=std, r2=r2))
            print(f"      r2={r2:.3f}  rmse={np.sqrt(np.mean(resid * resid)):.4f}", flush=True)
        pfx = bank[:, :n_prefix]
        return cls(
            primes=primes,
            prefix_last=prefix_last,
            steps=steps,
            x_mean=bank.mean(axis=0),
            prefix_mean=pfx.mean(axis=0),
            prefix_std=np.maximum(pfx.std(axis=0), 1e-4),
        )


class GeneratorV3:
    """Sample prefix, unfold primes, raster. Morph = lerp prefixes, unfold frames."""

    def __init__(self, field: PrimeField, unfold: Unfold, gmm: DiagGMM | None = None):
        if field.nparams() != unfold.n:
            raise ValueError(f"field N={field.nparams()} != unfold N={unfold.n}")
        self.field = field
        self.unfold = unfold
        self.gmm = gmm

    def raster(self, c: np.ndarray, H: int = 96, W: int = 96, backend: str = "cpu") -> np.ndarray:
        self.field._apply_params(c)
        return self.field.decode(H, W, stretch=False, backend=backend)

    def sample_prefix(self, rng: np.random.Generator, bank: bool = True) -> np.ndarray:
        n_p = self.unfold.n_prefix
        bank_c = self.field.bank
        # Prefer populated prefixes from the bank — that's the point of unfold.
        # GMM on 14-D was blowing out DC (white fog samples).
        if bank and bank_c is not None and (self.gmm is None or rng.random() < 0.85):
            j = int(rng.integers(0, len(bank_c)))
            pfx = bank_c[j, :n_p].copy()
            pfx = pfx + rng.normal(0.0, 0.04, size=n_p) * self.unfold.prefix_std
            lo = self.unfold.prefix_mean - 3.0 * self.unfold.prefix_std
            hi = self.unfold.prefix_mean + 3.0 * self.unfold.prefix_std
            return np.clip(pfx, lo, hi)
        if self.gmm is not None:
            pfx = self.gmm.sample(1, rng)[0]
            lo = self.unfold.prefix_mean - 2.5 * self.unfold.prefix_std
            hi = self.unfold.prefix_mean + 2.5 * self.unfold.prefix_std
            return np.clip(pfx, lo, hi)
        return self.unfold.prefix_mean + rng.normal(size=n_p) * self.unfold.prefix_std * 0.25

    def sample(self, n=1, H=96, W=96, temp=0.15, seed=0, backend="cpu"):
        rng = np.random.default_rng(seed)
        Cs = []
        for i in range(n):
            pfx = self.sample_prefix(rng, bank=True)
            Cs.append(self.unfold.from_prefix(pfx, temp=temp, seed=seed + 17 * i + 3))
        Cs = np.stack(Cs)
        return [self.raster(c, H, W, backend=backend) for c in Cs], Cs

    def from_real_prefix(self, c_real: np.ndarray, temp: float = 0.0, seed: int = 0) -> np.ndarray:
        return self.unfold.complete(c_real, temp=temp, seed=seed)

    def next_frames(
        self,
        c0: np.ndarray,
        c1: np.ndarray,
        frames: int = 8,
        temp: float = 0.0,
        seed: int = 0,
    ) -> list[np.ndarray]:
        """Guess frames between two populated fields: lerp prefix, unfold each tick."""
        n_p = self.unfold.n_prefix
        z0 = np.asarray(c0, dtype=np.float64)[:n_p]
        z1 = np.asarray(c1, dtype=np.float64)[:n_p]
        out = []
        for i in range(frames):
            t = i / max(frames - 1, 1)
            pfx = (1.0 - t) * z0 + t * z1
            out.append(self.unfold.from_prefix(pfx, temp=temp, seed=seed + i))
        return out

    def walk_frames(self, c0: np.ndarray, frames: int = 8, step: float = 0.12, seed: int = 0) -> list[np.ndarray]:
        """One scene drifting: walk the prefix, unfold. That's next-frame without a clip."""
        rng = np.random.default_rng(seed)
        n_p = self.unfold.n_prefix
        pfx = np.asarray(c0, dtype=np.float64)[:n_p].copy()
        out = []
        for i in range(frames):
            out.append(self.unfold.from_prefix(pfx, temp=0.05, seed=seed + 100 + i))
            pfx = pfx + step * self.unfold.prefix_std * rng.normal(size=n_p)
        return out

    def formula(self) -> str:
        g = f"  prefixGMM={self.gmm.m}×N({self.gmm.d})" if self.gmm is not None else ""
        return self.unfold.formula() + g

    def save(self, stem: str) -> None:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        self.field.save(str(stem) + ".field.npz")
        self.unfold.save(str(stem) + ".unfold.npz")
        if self.gmm is not None:
            self.gmm.save(str(stem) + ".gmm.npz")
        Path(str(stem) + ".json").write_text(
            json.dumps(
                {
                    "primes": list(self.unfold.primes),
                    "prefix_last": self.unfold.prefix_last,
                    "n": self.unfold.n,
                    "n_prefix": self.unfold.n_prefix,
                    "n_maps": self.unfold.nparams(),
                    "formula": self.formula(),
                    "step_r2": {int(s.p): s.r2 for s in self.unfold.steps},
                },
                indent=2,
            )
            + "\n"
        )
