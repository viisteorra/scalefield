"""
Next-frame residual unfold.

c_{t+1,p} = c_{t,p} + W_p [c_{t+1,<p} ; c_t through p] + b
Identity is the prior (Δ=0). Maps only have to learn motion.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v1"))
sys.path.insert(0, str(ROOT / "gen3"))

from scalefield import PrimeField  # noqa: E402
from unfold import offsets_of, ridge_fit  # noqa: E402


@dataclass
class Step:
    p: int
    start: int
    n: int
    W: np.ndarray
    b: np.ndarray
    resid_std: np.ndarray
    r2: float


class NextFrame:
    def __init__(self, primes: tuple[int, ...], steps: list[Step]):
        self.primes = tuple(int(p) for p in primes)
        self.steps = steps
        self.n = int(sum(p * p for p in self.primes))

    def nparams(self) -> int:
        return int(sum(s.W.size + s.b.size + s.resid_std.size for s in self.steps))

    def _ctx(self, c_next: np.ndarray, c_prev: np.ndarray, start: int, n: int) -> np.ndarray:
        return np.concatenate([c_next[:start], c_prev[: start + n]])

    def step(self, c_prev: np.ndarray, temp: float, rng: np.random.Generator) -> np.ndarray:
        c_prev = np.asarray(c_prev, dtype=np.float64).reshape(-1)
        c = np.zeros(self.n, dtype=np.float64)
        for st in self.steps:
            x = self._ctx(c, c_prev, st.start, st.n)
            d = st.W @ x + st.b
            if temp > 0:
                d = d + temp * st.resid_std * rng.normal(size=st.n)
            c[st.start : st.start + st.n] = c_prev[st.start : st.start + st.n] + d
        return c

    def rollout(self, c0: np.ndarray, n_frames: int, temp: float = 0.0, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        out = [np.asarray(c0, dtype=np.float64).reshape(-1)]
        c = out[0]
        for _ in range(max(0, n_frames - 1)):
            c = self.step(c, temp, rng)
            out.append(c)
        return np.stack(out)

    def one_step(self, c_prev: np.ndarray, c_next_true: np.ndarray | None = None) -> np.ndarray:
        """Teacher-forced one-step if c_next_true given; else ancestral."""
        if c_next_true is None:
            return self.step(c_prev, 0.0, np.random.default_rng(0))
        c_prev = np.asarray(c_prev, dtype=np.float64).reshape(-1)
        c_true = np.asarray(c_next_true, dtype=np.float64).reshape(-1)
        c = np.zeros(self.n, dtype=np.float64)
        for st in self.steps:
            x = np.concatenate([c_true[: st.start], c_prev[: st.start + st.n]])
            d = st.W @ x + st.b
            c[st.start : st.start + st.n] = c_prev[st.start : st.start + st.n] + d
        return c

    def formula(self) -> str:
        r2s = ",".join(f"{s.p}:{s.r2:.2f}" for s in self.steps[-4:])
        return (
            f"NEXTFRAME  c_t+1 = c_t + Δ_p(c_<p, c_t)  "
            f"N={self.n}  maps={self.nparams()}  lastR2[{r2s}]"
        )

    def save(self, path: str) -> None:
        payload = {
            "primes": np.array(self.primes, dtype=np.int32),
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
    def load(cls, path: str) -> "NextFrame":
        z = np.load(path)
        primes = tuple(int(p) for p in z["primes"].tolist())
        steps = []
        for i in range(int(z["n_steps"])):
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
        return cls(primes, steps)

    @classmethod
    def fit(cls, prev: np.ndarray, nxt: np.ndarray, primes: tuple[int, ...], lam: float = 3e-2) -> "NextFrame":
        prev = np.asarray(prev, dtype=np.float64)
        nxt = np.asarray(nxt, dtype=np.float64)
        if prev.shape != nxt.shape:
            raise ValueError("prev/next banks must match")
        steps: list[Step] = []
        print(f"  next-frame fit  pairs={len(prev)}  N={prev.shape[1]}  ridge={lam:g}", flush=True)
        for p, start, n in offsets_of(primes):
            X = np.hstack([nxt[:, :start], prev[:, : start + n]])
            Y = nxt[:, start : start + n] - prev[:, start : start + n]
            print(f"    p={p:2d}  X={X.shape} Δ={Y.shape}  dual={X.shape[1] + 1 > X.shape[0]}", flush=True)
            W, b = ridge_fit(X, Y, lam=lam)
            Yh = X @ W.T + b
            resid = Y - Yh
            ss_res = float(np.sum(resid * resid))
            ss_tot = float(np.sum((Y - Y.mean(axis=0)) ** 2)) + 1e-18
            r2 = 1.0 - ss_res / ss_tot
            std = np.maximum(np.std(resid, axis=0), 1e-5)
            steps.append(Step(p=p, start=start, n=n, W=W, b=b, resid_std=std, r2=r2))
            print(
                f"      r2={r2:.3f}  d-rmse={np.sqrt(np.mean(resid * resid)):.5f}  "
                f"|Δ|={np.sqrt(np.mean(Y * Y)):.5f}",
                flush=True,
            )
        return cls(primes, steps)


class GeneratorV4:
    def __init__(self, field: PrimeField, nf: NextFrame, clip_slices: list[tuple[str, int, int]]):
        self.field = field
        self.nf = nf
        self.clip_slices = clip_slices  # (name, start, length) into field.bank

    def raster(self, c: np.ndarray, H: int = 96, W: int = 96, backend: str = "cpu") -> np.ndarray:
        self.field._apply_params(c)
        return self.field.decode(H, W, stretch=False, backend=backend)

    def clip_bank(self, i: int) -> np.ndarray:
        name, s, n = self.clip_slices[i % len(self.clip_slices)]
        return self.field.bank[s : s + n]

    def formula(self) -> str:
        nclips = len(self.clip_slices)
        nframes = 0 if self.field.bank is None else len(self.field.bank)
        return self.nf.formula() + f"  clips={nclips} frames={nframes}"

    def save(self, stem: str) -> None:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        self.field.save(str(stem) + ".field.npz")
        self.nf.save(str(stem) + ".nextframe.npz")
        names = np.array([c[0] for c in self.clip_slices], dtype=object)
        starts = np.array([c[1] for c in self.clip_slices], dtype=np.int32)
        lens = np.array([c[2] for c in self.clip_slices], dtype=np.int32)
        np.savez_compressed(str(stem) + ".clips.npz", names=names, starts=starts, lens=lens)
        Path(str(stem) + ".json").write_text(
            json.dumps(
                {
                    "primes": list(self.nf.primes),
                    "n": self.nf.n,
                    "n_maps": self.nf.nparams(),
                    "clips": [{"name": n, "start": int(s), "len": int(l)} for n, s, l in self.clip_slices],
                    "formula": self.formula(),
                    "step_r2": {int(s.p): s.r2 for s in self.nf.steps},
                },
                indent=2,
            )
            + "\n"
        )
