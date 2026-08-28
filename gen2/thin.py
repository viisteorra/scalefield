"""
Thin generative field.

The codec stores every cell of every p×p DCT (N=3359). That is a *codec*.
Generation does not belong there: high-k modes are full-screen standing
waves. Inventing them is oil-slick. Copying them is memorization.

Original contract: a few numbers per scale (about 3×3). We keep only the
low modes of each prime — 95 numbers — and generate in that space.
Raster is the frozen PrimeField with high-k zeroed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v1"))
sys.path.insert(0, str(ROOT / "gen"))
sys.path.insert(0, str(ROOT / "gen2"))

from scalefield import PrimeField, _mode_weights  # noqa: E402
from scalegen import DiagGMM, LatentPrior  # noqa: E402


def low_mask(primes: tuple[int, ...], m: int = 3) -> np.ndarray:
    """Old: top-left m×m of every prime. Repeats k=0,1,2 on every lattice."""
    bits = []
    for p in primes:
        g = np.zeros((p, p), dtype=bool)
        mm = min(int(p), int(m))
        g[:mm, :mm] = True
        bits.append(g.reshape(-1))
    return np.concatenate(bits)


def keep_through(primes: tuple[int, ...], last: int) -> np.ndarray:
    """Keep the entire p×p for primes <= last, zero the rest.

    Each small prime is an elemental oscillator at new frequencies
    k=0..p-1. That is the sine-wave page without a new embed table:
    p=11 already owns k=7..10, which 3×3-of-31 never had.
    """
    bits = []
    for p in primes:
        if p <= last:
            bits.append(np.ones(p * p, dtype=bool))
        else:
            bits.append(np.zeros(p * p, dtype=bool))
    return np.concatenate(bits)


def _ridge(X: np.ndarray, Y: np.ndarray, lam: float = 1e-2) -> tuple[np.ndarray, np.ndarray]:
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


def _r2(y, yhat) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean(axis=0)) ** 2)) + 1e-18
    return 1.0 - ss_res / ss_tot


def thin_offsets(primes: tuple[int, ...], last: int) -> list[tuple[int, int, int]]:
    out = []
    s = 0
    for p in primes:
        if p > last:
            continue
        n = p * p
        out.append((p, s, n))
        s += n
    return out


class PhaseUnfold:
    """c_p | c_<p on the thin primes only. Each p×p is its own pattern."""

    def __init__(self, primes, last, steps, n_thin):
        self.primes = tuple(p for p in primes if p <= last)
        self.last = int(last)
        self.steps = steps  # list of dicts p,start,n,W,b,std,r2_hold,r2_train
        self.n_thin = int(n_thin)
        self.n_prefix = sum(p * p for p in self.primes if p <= 3)

    def fill(self, t: np.ndarray, temp: float, rng: np.random.Generator) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64).copy()
        for st in self.steps:
            y = st["W"] @ t[: st["start"]] + st["b"]
            # Don't invent a lattice we cannot predict (11×11×4 hold was 0.12).
            if st["p"] >= 17 or float(st.get("r2_hold", 1.0)) < 0.35:
                continue
            tp = float(temp) * (0.3 if st["p"] >= 13 else 1.0)
            if tp > 0:
                y = y + tp * st["std"] * rng.normal(size=st["n"])
            t[st["start"] : st["start"] + st["n"]] = y
        return t

    def from_prefix(self, pfx: np.ndarray, temp: float = 0.35, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        t = np.zeros(self.n_thin, dtype=np.float64)
        pfx = np.asarray(pfx, dtype=np.float64).reshape(-1)
        t[: self.n_prefix] = pfx[: self.n_prefix]
        return self.fill(t, temp, rng)

    @classmethod
    def fit(cls, thin: np.ndarray, primes, last: int, lam: float = 0.05, hold: float = 0.2, seed: int = 0):
        thin = np.asarray(thin, dtype=np.float64)
        rng = np.random.default_rng(seed)
        n = len(thin)
        idx = rng.permutation(n)
        n_te = max(32, int(round(n * hold)))
        te, tr = idx[:n_te], idx[n_te:]
        Ttr, Tte = thin[tr], thin[te]
        offs = thin_offsets(primes, last)
        steps = []
        print(f"  phase unfold  train={len(tr)} hold={len(te)}  primes≤{last}", flush=True)
        for p, start, nn in offs:
            if p <= 3:
                continue
            Xtr, Ytr = Ttr[:, :start], Ttr[:, start : start + nn]
            Xte, Yte = Tte[:, :start], Tte[:, start : start + nn]
            W, b = _ridge(Xtr, Ytr, lam=lam)
            ytr = Xtr @ W.T + b
            yte = Xte @ W.T + b
            r_tr, r_te = _r2(Ytr, ytr), _r2(Yte, yte)
            std = np.maximum(np.std(Ytr - ytr, axis=0), 1e-4)
            steps.append(
                {"p": p, "start": start, "n": nn, "W": W, "b": b, "std": std, "r2_train": r_tr, "r2_hold": r_te}
            )
            print(f"    p={p:2d}  {nn} cells | {start}  R² train={r_tr:.3f} hold={r_te:.3f}", flush=True)
        return cls(primes, last, steps, thin.shape[1]), {"te_idx": te, "tr_idx": tr}

    def save_into(self, payload: dict) -> None:
        payload["n_steps"] = np.int32(len(self.steps))
        payload["n_prefix"] = np.int32(self.n_prefix)
        payload["n_thin"] = np.int32(self.n_thin)
        payload["last"] = np.int32(self.last)
        for i, st in enumerate(self.steps):
            payload[f"p_{i}"] = np.int32(st["p"])
            payload[f"start_{i}"] = np.int32(st["start"])
            payload[f"n_{i}"] = np.int32(st["n"])
            payload[f"W_{i}"] = st["W"].astype(np.float32)
            payload[f"b_{i}"] = st["b"]
            payload[f"std_{i}"] = st["std"]
            payload[f"tr_{i}"] = np.float64(st["r2_train"])
            payload[f"ho_{i}"] = np.float64(st["r2_hold"])

    @classmethod
    def load_from(cls, z, primes) -> "PhaseUnfold":
        steps = []
        for i in range(int(z["n_steps"])):
            steps.append(
                {
                    "p": int(z[f"p_{i}"]),
                    "start": int(z[f"start_{i}"]),
                    "n": int(z[f"n_{i}"]),
                    "W": np.array(z[f"W_{i}"], np.float64),
                    "b": np.array(z[f"b_{i}"], np.float64),
                    "std": np.array(z[f"std_{i}"], np.float64),
                    "r2_train": float(z[f"tr_{i}"]),
                    "r2_hold": float(z[f"ho_{i}"]),
                }
            )
        return cls(primes, int(z["last"]), steps, int(z["n_thin"]))


def design_prime(pf: PrimeField, p: int, H: int, W: int) -> np.ndarray:
    """Linear map (H*W*3, p²) for one prime lattice only."""
    n_pix = H * W
    u, v = pf._uv(H, W)
    uh = ((u + 1.0) * 0.5).reshape(n_pix)
    vh = ((v + 1.0) * 0.5).reshape(n_pix)
    hue = pf._hues[list(pf.primes).index(p)][0, 0]
    n_loc = p * p
    if p == 1:
        A_sp = np.ones((n_pix, 1), dtype=np.float64)
    else:
        k = np.arange(p, dtype=np.float64)
        cu = np.cos(np.pi * np.outer(uh, k))
        cv = np.cos(np.pi * np.outer(vh, k))
        w = _mode_weights(p, pf.edge_gain)
        A_sp = (cv[:, :, None] * cu[:, None, :] * w[None, :, :]).reshape(n_pix, n_loc)
    block = A_sp[:, None, :] * hue[None, :, None]
    return block.reshape(n_pix * 3, n_loc)


def residual_decompose(pf: PrimeField, images: list, last: int, lam: float = 1e-4) -> np.ndarray:
    """Peel each photo into primes, one lattice at a time.

    leftover = photo
    for p in 1,2,3,5,...:
        fit this p×p to leftover
        leftover -= that field
    So 5×5 sees what 1,2,3 could not explain — not a joint grab of the same pixels.
    """
    imgs = [np.asarray(im, dtype=np.float64) for im in images]
    if imgs[0].max() > 1.5:
        imgs = [im / 255.0 for im in imgs]
    H, W = imgs[0].shape[:2]
    Y = np.stack([im.reshape(-1) for im in imgs])
    N = pf.nparams()
    C = np.zeros((len(imgs), N), dtype=np.float64)
    s = 0
    for p in pf.primes:
        n_loc = p * p
        if p > last:
            break
        print(f"    decompose p={p}  leftover → {p}×{p}  Y={Y.shape}", flush=True)
        A = design_prime(pf, p, H, W)
        ATA = A.T @ A
        ATA.flat[:: n_loc + 1] += float(lam) * Y.shape[1]
        BT = Y @ A
        cp = np.linalg.solve(ATA, BT.T).T
        Y = Y - cp @ A.T
        C[:, s : s + n_loc] = cp
        s += n_loc
        del A
    print(f"  residual energy left={float(np.mean(Y * Y)):.4f}", flush=True)
    return C


def pack(c: np.ndarray, mask: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return c[..., mask]


def unpack(thin: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    thin = np.asarray(thin, dtype=np.float64)
    if thin.ndim == 1:
        out = np.zeros(n, dtype=np.float64)
        out[mask] = thin
        return out
    out = np.zeros((len(thin), n), dtype=np.float64)
    out[:, mask] = thin
    return out


class GeneratorThin:
    def __init__(
        self,
        field: PrimeField,
        prior: LatentPrior,
        gmm: DiagGMM | None,
        mask: np.ndarray,
        m: int = 3,
        unf: PhaseUnfold | None = None,
    ):
        self.field = field
        self.prior = prior
        self.gmm = gmm
        self.mask = np.asarray(mask, dtype=bool)
        self.m = int(m)
        self.unf = unf

    def to_c(self, thin: np.ndarray) -> np.ndarray:
        return unpack(thin, self.mask, self.field.nparams())

    def raster(self, c, H=96, W=96, backend="cpu"):
        c = np.asarray(c, dtype=np.float64).reshape(-1)
        nthin = int(self.mask.sum())
        nfull = self.field.nparams()
        if c.size == nthin:
            c = self.to_c(c)
        elif c.size != nfull:
            raise ValueError(f"c size {c.size} not thin {nthin} or full {nfull}")
        self.field._apply_params(c)
        return self.field.decode(H, W, stretch=False, backend=backend)

    def sample(self, n=1, H=96, W=96, temp=0.45, seed=0, backend="cpu", cloud=True):
        rng = np.random.default_rng(seed)
        Th = []
        for i in range(n):
            if self.unf is not None:
                if self.gmm is not None:
                    z = self.gmm.sample(1, rng)[0]
                    pfx = self.prior.decode(z)
                else:
                    pfx = rng.normal(size=self.unf.n_prefix) * 0.3
                # prior on prefix is n_prefix-D; if PCA prior, decode to n_prefix
                if pfx.size != self.unf.n_prefix:
                    pfx = pfx[: self.unf.n_prefix]
                Th.append(self.unf.from_prefix(pfx, temp=temp, seed=seed + 17 * i))
            else:
                if self.gmm is not None:
                    z = self.gmm.sample(1, rng)[0]
                else:
                    z = rng.normal(size=self.prior.k) * 0.5
                Th.append(self.prior.decode(z))
        Th = np.stack(Th)
        Cs = unpack(Th, self.mask, self.field.nparams())
        return [self.raster(c, H, W, backend=backend) for c in Cs], Cs

    def morph(self, c0, c1, frames=8, H=96, W=96, backend="cpu"):
        t0 = pack(c0, self.mask)
        t1 = pack(c1, self.mask)
        out = []
        for i in range(frames):
            u = i / max(frames - 1, 1)
            if self.unf is not None:
                pfx = (1.0 - u) * t0[: self.unf.n_prefix] + u * t1[: self.unf.n_prefix]
                th = self.unf.from_prefix(pfx, temp=0.15, seed=i)
            else:
                z0, z1 = self.prior.encode(t0), self.prior.encode(t1)
                th = self.prior.decode((1.0 - u) * z0 + u * z1)
            out.append(self.raster(th, H, W, backend=backend))
        return out

    def formula(self) -> str:
        g = f"GMM={self.gmm.m}×N({self.gmm.d})" if self.gmm is not None else "no-gmm"
        ph = ""
        if self.unf is not None and self.unf.steps:
            ph = "  phase " + ",".join(f"{s['p']}:{s['r2_hold']:.2f}" for s in self.unf.steps)
        return (
            f"THIN  keep {int(self.mask.sum())}/{self.field.nparams()}  "
            f"primes≤{self.m}  {g}{ph}"
        )

    def save(self, stem: str) -> None:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        self.field.save(str(stem) + ".field.npz")
        self.prior.save(str(stem) + ".prior.npz")
        if self.gmm is not None:
            self.gmm.save(str(stem) + ".gmm.npz")
        payload = {"mask": self.mask, "m": np.int32(self.m)}
        if self.unf is not None:
            self.unf.save_into(payload)
        np.savez_compressed(str(stem) + ".thin.npz", **payload)
        Path(str(stem) + ".json").write_text(
            json.dumps({"formula": self.formula(), "keep": int(self.mask.sum())}, indent=2) + "\n"
        )
