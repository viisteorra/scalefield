"""
Shared abstract code.

Overnight linear unfold never had a scene representation: each prime had its
own map from growing context. Here every finer lattice is decoded from one
small code `a` inferred from the coarse primes only.

  x = PCA(primes ≤ prefix_last)
  h = tanh(Wh x + bh)
  a = tanh(Wa h + ba)          # THE abstraction
  c_p = scale·resample(c_{p-1}) + decode_p(U_p a + d_p)

Hold-out score is fine primes from coarse-only `a`. If permuting `a` kills
R², the code is carrying image-specific structure.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v1"))
sys.path.insert(0, str(ROOT / "gen2"))

from scalefield import PrimeField, resample_op  # noqa: E402
from structure import (  # noqa: E402
    DiagGMM,
    _pca_decode,
    _pca_encode,
    _pca_fit,
    _r2,
    _ridge,
    k_of,
    offsets_of,
)


def _tanh(x):
    return np.tanh(x)


def _dtanh(y):
    return 1.0 - y * y


@dataclass
class Band:
    p: int
    start: int
    n: int
    mean: np.ndarray
    comp: np.ndarray
    vals: np.ndarray


@dataclass
class Head:
    p: int
    start: int
    n: int
    p_prev: int
    scale: float
    U: np.ndarray
    d: np.ndarray
    resid_std: np.ndarray
    r2_hold: float
    r2_train: float


class AbstractCore:
    def __init__(
        self,
        primes,
        prefix_last,
        k_in,
        k_hid,
        k_a,
        pfx_mean,
        pfx_comp,
        pfx_vals,
        Wh,
        bh,
        Wa,
        ba,
        bands,
        heads,
        D,
        pfx_U,
        pfx_b,
    ):
        self.primes = tuple(int(p) for p in primes)
        self.prefix_last = int(prefix_last)
        self.k_in = int(k_in)
        self.k_hid = int(k_hid)
        self.k_a = int(k_a)
        self.pfx_mean = pfx_mean
        self.pfx_comp = pfx_comp
        self.pfx_vals = pfx_vals
        self.Wh = np.asarray(Wh, dtype=np.float64)
        self.bh = np.asarray(bh, dtype=np.float64)
        self.Wa = np.asarray(Wa, dtype=np.float64)
        self.ba = np.asarray(ba, dtype=np.float64)
        self.bands = bands
        self.heads = heads
        self.D = D
        self.pfx_U = np.asarray(pfx_U, dtype=np.float64)
        self.pfx_b = np.asarray(pfx_b, dtype=np.float64)
        self.n = int(sum(p * p for p in self.primes))
        self.n_prefix = int(sum(p * p for p in self.primes if p <= self.prefix_last))
        self._band = {b.p: b for b in bands}

    def nparams(self) -> int:
        return int(
            self.Wh.size
            + self.bh.size
            + self.Wa.size
            + self.ba.size
            + sum(h.U.size + h.d.size for h in self.heads)
        )

    def x_prefix(self, c: np.ndarray) -> np.ndarray:
        x = np.asarray(c, dtype=np.float64)[..., : self.n_prefix]
        return _pca_encode(x, self.pfx_mean, self.pfx_comp, self.pfx_vals)

    def encode_a(self, c: np.ndarray) -> np.ndarray:
        x = self.x_prefix(c)
        one = x.ndim == 1
        if one:
            x = x[None, :]
        h = _tanh(x @ self.Wh.T + self.bh)
        a = _tanh(h @ self.Wa.T + self.ba)
        return a[0] if one else a

    def forward_a(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = _tanh(x @ self.Wh.T + self.bh)
        a = _tanh(h @ self.Wa.T + self.ba)
        return x, h, a

    def decode_from_a(self, a: np.ndarray, c_prefix: np.ndarray | None = None) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        c = np.zeros(self.n, dtype=np.float64)
        if c_prefix is not None:
            c[: self.n_prefix] = np.asarray(c_prefix, dtype=np.float64).reshape(-1)[: self.n_prefix]
        else:
            zp = self.pfx_U @ a + self.pfx_b
            c[: self.n_prefix] = _pca_decode(zp, self.pfx_mean, self.pfx_comp, self.pfx_vals)
        for h in self.heads:
            prev = c[self._band[h.p_prev].start : self._band[h.p_prev].start + self._band[h.p_prev].n]
            up = self.D[h.p] @ prev
            zr = h.U @ a + h.d
            band = self._band[h.p]
            resid = _pca_decode(zr, band.mean, band.comp, band.vals)
            c[h.start : h.start + h.n] = h.scale * up + resid
        return c

    def stages(self, c: np.ndarray):
        full = np.asarray(c, dtype=np.float64)
        out = []
        filled = 0
        for p, start, n in offsets_of(self.primes):
            filled = start + n
            part = np.zeros_like(full)
            part[:filled] = full[:filled]
            out.append((p, part))
        return out

    def predict_fine(self, C: np.ndarray) -> np.ndarray:
        """Teacher-forced resample, a from real prefix only."""
        C = np.asarray(C, dtype=np.float64)
        A = self.encode_a(C)
        hat = C.copy()
        for h in self.heads:
            pb = self._band[h.p_prev]
            prev = C[:, pb.start : pb.start + pb.n]
            up = prev @ self.D[h.p].T
            zr = A @ h.U.T + h.d
            band = self._band[h.p]
            resid = _pca_decode(zr, band.mean, band.comp, band.vals)
            hat[:, h.start : h.start + h.n] = h.scale * up + resid
        return hat

    def hold_r2(self, C: np.ndarray) -> dict[int, float]:
        hat = self.predict_fine(C)
        out = {}
        for h in self.heads:
            out[int(h.p)] = _r2(C[:, h.start : h.start + h.n], hat[:, h.start : h.start + h.n])
        return out

    def permute_r2(self, C: np.ndarray) -> dict[int, float]:
        A = self.encode_a(C)
        A = np.roll(A, 1, axis=0)
        hat = C.copy()
        for h in self.heads:
            pb = self._band[h.p_prev]
            prev = C[:, pb.start : pb.start + pb.n]
            up = prev @ self.D[h.p].T
            zr = A @ h.U.T + h.d
            band = self._band[h.p]
            resid = _pca_decode(zr, band.mean, band.comp, band.vals)
            hat[:, h.start : h.start + h.n] = h.scale * up + resid
        out = {}
        for h in self.heads:
            out[int(h.p)] = _r2(C[:, h.start : h.start + h.n], hat[:, h.start : h.start + h.n])
        return out

    def sgd_step(self, C: np.ndarray, lr: float, l2: float, rng: np.random.Generator, batch: int = 48) -> float:
        n = len(C)
        idx = rng.choice(n, size=min(batch, n), replace=False)
        Cb = C[idx]
        x = self.x_prefix(Cb)
        _, h, a = self.forward_a(x)
        B = len(Cb)
        da = np.zeros_like(a)
        loss = 0.0
        n_h = 0
        dU = []
        dd = []
        for hd in self.heads:
            pb = self._band[hd.p_prev]
            prev = Cb[:, pb.start : pb.start + pb.n]
            up = prev @ self.D[hd.p].T
            Y = Cb[:, hd.start : hd.start + hd.n]
            R = Y - hd.scale * up
            band = self._band[hd.p]
            Zt = _pca_encode(R, band.mean, band.comp, band.vals)
            Zhat = a @ hd.U.T + hd.d
            err = Zhat - Zt
            loss += float(np.mean(err * err))
            n_h += 1
            gZ = err / B
            dU.append(gZ.T @ a)
            dd.append(gZ.sum(axis=0))
            da += gZ @ hd.U
        loss /= max(n_h, 1)
        # encoder
        da_pre = da * _dtanh(a)
        dWa = da_pre.T @ h + l2 * self.Wa
        dba = da_pre.sum(axis=0)
        dh = da_pre @ self.Wa
        dh_pre = dh * _dtanh(h)
        dWh = dh_pre.T @ x + l2 * self.Wh
        dbh = dh_pre.sum(axis=0)
        self.Wa -= lr * dWa
        self.ba -= lr * dba
        self.Wh -= lr * dWh
        self.bh -= lr * dbh
        for hd, gU, gd in zip(self.heads, dU, dd):
            hd.U -= lr * (gU + l2 * hd.U)
            hd.d -= lr * gd
        return loss

    def formula(self) -> str:
        last = ",".join(f"{h.p}:{h.r2_hold:.2f}" for h in self.heads[-4:])
        return (
            f"ABSTRACT  a={self.k_a}←h={self.k_hid}←pfx≤{self.prefix_last}  "
            f"maps={self.nparams()}  holdR2[{last}]"
        )

    def save(self, path: str) -> None:
        payload = {
            "primes": np.array(self.primes, np.int32),
            "prefix_last": np.int32(self.prefix_last),
            "k_in": np.int32(self.k_in),
            "k_hid": np.int32(self.k_hid),
            "k_a": np.int32(self.k_a),
            "pfx_mean": self.pfx_mean,
            "pfx_comp": self.pfx_comp.astype(np.float32),
            "pfx_vals": self.pfx_vals,
            "Wh": self.Wh.astype(np.float32),
            "bh": self.bh,
            "Wa": self.Wa.astype(np.float32),
            "ba": self.ba,
            "pfx_U": self.pfx_U.astype(np.float32),
            "pfx_b": self.pfx_b,
            "n_bands": np.int32(len(self.bands)),
            "n_heads": np.int32(len(self.heads)),
        }
        for i, b in enumerate(self.bands):
            payload[f"bp_{i}"] = np.int32(b.p)
            payload[f"bs_{i}"] = np.int32(b.start)
            payload[f"bn_{i}"] = np.int32(b.n)
            payload[f"bm_{i}"] = b.mean
            payload[f"bc_{i}"] = b.comp.astype(np.float32)
            payload[f"bv_{i}"] = b.vals
        for i, h in enumerate(self.heads):
            payload[f"hp_{i}"] = np.int32(h.p)
            payload[f"hs_{i}"] = np.int32(h.start)
            payload[f"hn_{i}"] = np.int32(h.n)
            payload[f"hprev_{i}"] = np.int32(h.p_prev)
            payload[f"hsc_{i}"] = np.float64(h.scale)
            payload[f"hU_{i}"] = h.U.astype(np.float32)
            payload[f"hd_{i}"] = h.d
            payload[f"hstd_{i}"] = h.resid_std
            payload[f"htr_{i}"] = np.float64(h.r2_train)
            payload[f"hho_{i}"] = np.float64(h.r2_hold)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "AbstractCore":
        z = np.load(path)
        primes = tuple(int(p) for p in z["primes"].tolist())
        bands = []
        for i in range(int(z["n_bands"])):
            bands.append(
                Band(
                    p=int(z[f"bp_{i}"]),
                    start=int(z[f"bs_{i}"]),
                    n=int(z[f"bn_{i}"]),
                    mean=np.array(z[f"bm_{i}"], np.float64),
                    comp=np.array(z[f"bc_{i}"], np.float64),
                    vals=np.array(z[f"bv_{i}"], np.float64),
                )
            )
        heads = []
        D = {}
        for i in range(int(z["n_heads"])):
            p = int(z[f"hp_{i}"])
            p_prev = int(z[f"hprev_{i}"])
            D[p] = resample_op(p_prev, p)
            heads.append(
                Head(
                    p=p,
                    start=int(z[f"hs_{i}"]),
                    n=int(z[f"hn_{i}"]),
                    p_prev=p_prev,
                    scale=float(z[f"hsc_{i}"]),
                    U=np.array(z[f"hU_{i}"], np.float64),
                    d=np.array(z[f"hd_{i}"], np.float64),
                    resid_std=np.array(z[f"hstd_{i}"], np.float64),
                    r2_train=float(z[f"htr_{i}"]),
                    r2_hold=float(z[f"hho_{i}"]),
                )
            )
        return cls(
            primes=primes,
            prefix_last=int(z["prefix_last"]),
            k_in=int(z["k_in"]),
            k_hid=int(z["k_hid"]),
            k_a=int(z["k_a"]),
            pfx_mean=np.array(z["pfx_mean"], np.float64),
            pfx_comp=np.array(z["pfx_comp"], np.float64),
            pfx_vals=np.array(z["pfx_vals"], np.float64),
            Wh=np.array(z["Wh"], np.float64),
            bh=np.array(z["bh"], np.float64),
            Wa=np.array(z["Wa"], np.float64),
            ba=np.array(z["ba"], np.float64),
            pfx_U=np.array(z["pfx_U"], np.float64),
            pfx_b=np.array(z["pfx_b"], np.float64),
            bands=bands,
            heads=heads,
            D=D,
        )

    @classmethod
    def fit(
        cls,
        bank: np.ndarray,
        primes: tuple[int, ...],
        prefix_last: int = 7,
        k_in: int = 20,
        k_hid: int = 24,
        k_a: int = 16,
        lam: float = 0.4,
        hold: float = 0.2,
        seed: int = 0,
    ) -> tuple["AbstractCore", dict]:
        bank = np.asarray(bank, dtype=np.float64)
        rng = np.random.default_rng(seed)
        n = len(bank)
        idx = rng.permutation(n)
        n_te = max(32, int(round(n * hold)))
        te, tr = idx[:n_te], idx[n_te:]
        Ctr, Cte = bank[tr], bank[te]
        offs = offsets_of(primes)
        n_prefix = sum(p * p for p, _s, _n in offs if p <= prefix_last)
        k_in = min(k_in, n_prefix, max(4, len(tr) // 8))
        pmean, pcomp, pvals = _pca_fit(Ctr[:, :n_prefix], k_in)
        k_in = int(pcomp.shape[0])
        k_hid = max(k_a, min(k_hid, k_in + 8))
        print(
            f"  abstract fit  train={len(tr)} hold={len(te)}  prefix≤{prefix_last} "
            f"({n_prefix}→{k_in})  a={k_a} h={k_hid}",
            flush=True,
        )

        bands = []
        D = {}
        prev = None
        for p, start, nn in offs:
            mean, comp, vals = _pca_fit(Ctr[:, start : start + nn], k_of(p, len(tr)))
            bands.append(Band(p=p, start=start, n=nn, mean=mean, comp=comp, vals=vals))
            if prev is not None and p > prefix_last:
                D[p] = resample_op(prev, p)
            prev = p
        band_by = {b.p: b for b in bands}

        Xtr = _pca_encode(Ctr[:, :n_prefix], pmean, pcomp, pvals)
        Xte = _pca_encode(Cte[:, :n_prefix], pmean, pcomp, pvals)
        # linear init: a ≈ first k_a of prefix PCA
        Atr0 = Xtr[:, :k_a]
        Ate0 = Xte[:, :k_a]

        heads = []
        prev = None
        for p, start, nn in offs:
            if prev is None or p <= prefix_last:
                prev = p
                continue
            Dp = D[p]
            pb = band_by[prev]
            Ytr = Ctr[:, start : start + nn]
            Yte = Cte[:, start : start + nn]
            Utr = Ctr[:, pb.start : pb.start + pb.n] @ Dp.T
            Ute = Cte[:, pb.start : pb.start + pb.n] @ Dp.T
            scale = float(np.sum(Ytr * Utr) / (np.sum(Utr * Utr) + 1e-12))
            Rtr = Ytr - scale * Utr
            Rte = Yte - scale * Ute
            rmean, rcomp, rvals = _pca_fit(Rtr, k_of(p, len(tr)))
            band_by[p].mean, band_by[p].comp, band_by[p].vals = rmean, rcomp, rvals
            Ztr = _pca_encode(Rtr, rmean, rcomp, rvals)
            Zte = _pca_encode(Rte, rmean, rcomp, rvals)
            U, d = _ridge(Atr0, Ztr, lam=lam)
            pred_tr = Atr0 @ U.T + d
            pred_te = Ate0 @ U.T + d
            Yh_tr = scale * Utr + _pca_decode(pred_tr, rmean, rcomp, rvals)
            Yh_te = scale * Ute + _pca_decode(pred_te, rmean, rcomp, rvals)
            r_tr = _r2(Ytr, Yh_tr)
            r_te = _r2(Yte, Yh_te)
            std = np.maximum(np.std(Ztr - pred_tr, axis=0), 1e-4)
            heads.append(
                Head(
                    p=p, start=start, n=nn, p_prev=prev, scale=scale,
                    U=U, d=d, resid_std=std, r2_train=r_tr, r2_hold=r_te,
                )
            )
            print(
                f"    p={p:2d}←{prev:<2d}  k={U.shape[0]}  linear-a hold R²={r_te:.3f} train={r_tr:.3f}",
                flush=True,
            )
            prev = p

        # tanh encoder init: small map so a ≈ 0.5 * x[:k_a]
        Wh = np.zeros((k_hid, k_in), dtype=np.float64)
        Wa = np.zeros((k_a, k_hid), dtype=np.float64)
        m = min(k_hid, k_in)
        Wh[np.arange(m), np.arange(m)] = 0.8
        m2 = min(k_a, k_hid)
        Wa[np.arange(m2), np.arange(m2)] = 0.8
        bh = np.zeros(k_hid)
        ba = np.zeros(k_a)

        pfx_U0 = np.eye(k_in, k_a)
        pfx_b0 = np.zeros(k_in)
        core = cls(
            primes=primes,
            prefix_last=prefix_last,
            k_in=k_in,
            k_hid=k_hid,
            k_a=k_a,
            pfx_mean=pmean,
            pfx_comp=pcomp,
            pfx_vals=pvals,
            Wh=Wh,
            bh=bh,
            Wa=Wa,
            ba=ba,
            bands=list(band_by[p] for p, _s, _n in offs),
            heads=heads,
            D=D,
            pfx_U=pfx_U0,
            pfx_b=pfx_b0,
        )
        # re-fit U from actual tanh-a so heads match encoder
        _, _, Atr = core.forward_a(Xtr)
        _, _, Ate = core.forward_a(Xte)
        core.pfx_U, core.pfx_b = _ridge(Atr, Xtr, lam=lam)
        prev = None
        for hd in heads:
            pb = band_by[hd.p_prev]
            Ytr = Ctr[:, hd.start : hd.start + hd.n]
            Yte = Cte[:, hd.start : hd.start + hd.n]
            Utr = Ctr[:, pb.start : pb.start + pb.n] @ D[hd.p].T
            Ute = Cte[:, pb.start : pb.start + pb.n] @ D[hd.p].T
            Rtr = Ytr - hd.scale * Utr
            Rte = Yte - hd.scale * Ute
            band = band_by[hd.p]
            Ztr = _pca_encode(Rtr, band.mean, band.comp, band.vals)
            U, d = _ridge(Atr, Ztr, lam=lam)
            hd.U, hd.d = U, d
            Zte = _pca_encode(Rte, band.mean, band.comp, band.vals)
            hd.r2_train = _r2(Ytr, hd.scale * Utr + _pca_decode(Atr @ U.T + d, band.mean, band.comp, band.vals))
            hd.r2_hold = _r2(Yte, hd.scale * Ute + _pca_decode(Ate @ U.T + d, band.mean, band.comp, band.vals))
            print(f"    p={hd.p:2d}  tanh-init hold R²={hd.r2_hold:.3f}", flush=True)
        stats = {
            "n_train": int(len(tr)),
            "n_hold": int(len(te)),
            "te_idx": te.astype(np.int32),
            "tr_idx": tr.astype(np.int32),
            "r2_hold": {int(h.p): h.r2_hold for h in heads},
            "r2_train": {int(h.p): h.r2_train for h in heads},
            "mean_hold": float(np.mean([h.r2_hold for h in heads])),
            "perm_hold": float(np.mean(list(core.permute_r2(Cte).values()))),
        }
        print(
            f"  mean hold R²={stats['mean_hold']:.3f}  permute R²={stats['perm_hold']:.3f}  maps={core.nparams()}",
            flush=True,
        )
        return core, stats


class GeneratorAbs:
    def __init__(self, field: PrimeField, core: AbstractCore, gmm: DiagGMM | None):
        self.field = field
        self.core = core
        self.gmm = gmm
        self.unf = core  # watch.py stages() looks at gen.stages

    def raster(self, c, H=96, W=96, backend="cpu"):
        self.field._apply_params(c)
        return self.field.decode(H, W, stretch=False, backend=backend)

    def sample(self, n=1, H=96, W=96, temp=0.25, seed=0, backend="cpu", cloud=True):
        rng = np.random.default_rng(seed)
        Cs = []
        for i in range(n):
            if self.gmm is not None:
                a = self.gmm.sample(1, rng)[0]
            else:
                a = rng.normal(size=self.core.k_a) * 0.5
            if temp > 0:
                a = a + rng.normal(size=a.shape) * (0.08 * temp)
            Cs.append(self.core.decode_from_a(a))
        Cs = np.stack(Cs)
        return [self.raster(c, H, W, backend=backend) for c in Cs], Cs

    def morph(self, c0, c1, frames=8, H=96, W=96, backend="cpu"):
        a0 = self.core.encode_a(c0)
        a1 = self.core.encode_a(c1)
        out = []
        for i in range(frames):
            t = i / max(frames - 1, 1)
            a = (1.0 - t) * a0 + t * a1
            out.append(self.raster(self.core.decode_from_a(a), H, W, backend=backend))
        return out

    def stages(self, c):
        return self.core.stages(c)

    def formula(self):
        g = f"  aGMM={self.gmm.m}×N({self.gmm.d})" if self.gmm is not None else ""
        return self.core.formula() + g

    def save(self, stem: str) -> None:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        self.field.save(str(stem) + ".field.npz")
        self.core.save(str(stem) + ".abs.npz")
        if self.gmm is not None:
            self.gmm.save(str(stem) + ".gmm.npz")
        Path(str(stem) + ".json").write_text(
            json.dumps(
                {
                    "formula": self.formula(),
                    "k_a": self.core.k_a,
                    "nparams": self.core.nparams(),
                    "r2_hold": {int(h.p): h.r2_hold for h in self.core.heads},
                },
                indent=2,
            )
            + "\n"
        )
