"""
Residual codebook unfold.

The codec (PrimeField) works. Generation failed because we *regressed*
the next prime — that predicts average texture and throws away objects.

Here the unpredictable part of each scale is a discrete token:

  r_p = grid_p - resample(grid_{p-1})
  z_p = PCA(r_p)
  code_p = nearest codebook entry          # structure atom
  P(code_p | a, code_{p-1})                # tiny softmax
  sample codes, decode, raster

`a` is PCA of the coarse prefix (a 5×5 already looks like a thumbnail).
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


def _softmax(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)


def _kmeans(Z: np.ndarray, k: int, rng: np.random.Generator, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    Z = np.asarray(Z, dtype=np.float64)
    k = max(2, min(int(k), len(Z)))
    centers = Z[rng.choice(len(Z), size=k, replace=False)].copy()
    assign = np.zeros(len(Z), dtype=np.int32)
    for _ in range(iters):
        d = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assign = d.argmin(axis=1).astype(np.int32)
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = Z[m].mean(axis=0)
            else:
                centers[j] = Z[int(rng.integers(0, len(Z)))]
    return centers, assign


def _onehot(idx: np.ndarray, k: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int32)
    oh = np.zeros((idx.size, k), dtype=np.float64)
    oh[np.arange(idx.size), np.clip(idx.reshape(-1), 0, k - 1)] = 1.0
    return oh


@dataclass
class Band:
    p: int
    start: int
    n: int
    mean: np.ndarray
    comp: np.ndarray
    vals: np.ndarray


@dataclass
class VQHead:
    p: int
    start: int
    n: int
    p_prev: int
    scale: float
    book: np.ndarray  # (K, kz)
    W: np.ndarray  # (K, feat)
    b: np.ndarray  # (K,)
    r2_vq: float  # hold recon with true codes
    r2_clf: float  # hold recon with argmax codes from a
    acc: float


class ScaleVQ:
    def __init__(
        self,
        primes,
        prefix_last,
        pfx_mean,
        pfx_comp,
        pfx_vals,
        pfx_U,
        pfx_b,
        bands,
        heads,
        D,
        K,
        scene_book=None,
    ):
        self.primes = tuple(int(p) for p in primes)
        self.prefix_last = int(prefix_last)
        self.pfx_mean = pfx_mean
        self.pfx_comp = pfx_comp
        self.pfx_vals = pfx_vals
        self.pfx_U = np.asarray(pfx_U, dtype=np.float64)
        self.pfx_b = np.asarray(pfx_b, dtype=np.float64)
        self.bands = bands
        self.heads = heads
        self.D = D
        self.K = int(K)
        self.scene_book = None if scene_book is None else np.asarray(scene_book, dtype=np.float64)
        self.n = int(sum(p * p for p in self.primes))
        self.n_prefix = int(sum(p * p for p in self.primes if p <= self.prefix_last))
        self.k_a = int(self.pfx_comp.shape[0])
        self._band = {b.p: b for b in bands}

    def nparams(self) -> int:
        return int(sum(h.book.size + h.W.size + h.b.size for h in self.heads) + self.pfx_U.size)

    def encode_a(self, c: np.ndarray) -> np.ndarray:
        x = np.asarray(c, dtype=np.float64)[..., : self.n_prefix]
        return _pca_encode(x, self.pfx_mean, self.pfx_comp, self.pfx_vals)

    def decode_prefix(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        zp = a if a.ndim == 1 else a
        if self.pfx_U.shape == (self.k_a, self.k_a):
            zp = (self.pfx_U @ a + self.pfx_b) if a.ndim == 1 else (a @ self.pfx_U.T + self.pfx_b)
        return _pca_decode(zp, self.pfx_mean, self.pfx_comp, self.pfx_vals)

    def _feat(self, a: np.ndarray, prev_code: np.ndarray) -> np.ndarray:
        a = np.atleast_2d(a)
        return np.hstack([a, _onehot(prev_code, self.K)])

    def logits(self, head: VQHead, a: np.ndarray, prev_code: np.ndarray) -> np.ndarray:
        X = self._feat(a, prev_code)
        return X @ head.W.T + head.b

    def encode_codes(self, C: np.ndarray) -> np.ndarray:
        """Teacher-forced true codes from residuals. (n, n_heads)"""
        C = np.atleast_2d(np.asarray(C, dtype=np.float64))
        codes = np.zeros((len(C), len(self.heads)), dtype=np.int32)
        for i, h in enumerate(self.heads):
            pb = self._band[h.p_prev]
            prev = C[:, pb.start : pb.start + pb.n]
            up = prev @ self.D[h.p].T
            R = C[:, h.start : h.start + h.n] - h.scale * up
            band = self._band[h.p]
            z = _pca_encode(R, band.mean, band.comp, band.vals)
            d = ((z[:, None, :] - h.book[None, :, :]) ** 2).sum(-1)
            codes[:, i] = d.argmin(axis=1)
        return codes

    def decode_codes(self, a: np.ndarray, codes: np.ndarray, c_prefix: np.ndarray | None = None) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        one = a.ndim == 1
        if one:
            a = a[None, :]
            codes = np.atleast_2d(codes)
        n = len(a)
        C = np.zeros((n, self.n), dtype=np.float64)
        if c_prefix is not None:
            C[:, : self.n_prefix] = np.asarray(c_prefix, dtype=np.float64).reshape(n, -1)[:, : self.n_prefix]
        else:
            C[:, : self.n_prefix] = self.decode_prefix(a)
        for i, h in enumerate(self.heads):
            pb = self._band[h.p_prev]
            prev = C[:, pb.start : pb.start + pb.n]
            up = prev @ self.D[h.p].T
            z = h.book[codes[:, i]]
            band = self._band[h.p]
            resid = _pca_decode(z, band.mean, band.comp, band.vals)
            C[:, h.start : h.start + h.n] = h.scale * up + resid
        return C[0] if one else C

    def sample_codes(self, a: np.ndarray, rng: np.random.Generator, temp: float = 1.0) -> np.ndarray:
        a = np.atleast_2d(a)
        n = len(a)
        codes = np.zeros((n, len(self.heads)), dtype=np.int32)
        prev = np.zeros(n, dtype=np.int32)
        t = max(float(temp), 1e-3)
        for i, h in enumerate(self.heads):
            lg = self.logits(h, a, prev) / t
            p = _softmax(lg)
            u = rng.random(n)
            cdf = np.cumsum(p, axis=1)
            codes[:, i] = (u[:, None] < cdf).argmax(axis=1)
            prev = codes[:, i]
        return codes

    def argmax_codes(self, a: np.ndarray) -> np.ndarray:
        a = np.atleast_2d(a)
        n = len(a)
        codes = np.zeros((n, len(self.heads)), dtype=np.int32)
        prev = np.zeros(n, dtype=np.int32)
        for i, h in enumerate(self.heads):
            lg = self.logits(h, a, prev)
            codes[:, i] = lg.argmax(axis=1)
            prev = codes[:, i]
        return codes

    def graft_fine(self, C: np.ndarray, donor: np.ndarray, pmin: int = 17) -> np.ndarray:
        """Keep our coarse/mid grids. Paste a real photo's high-k residual.

        Fine primes are full-screen standing waves. A 16-entry codebook mean
        cannot cancel into edges (that's the dream that never locks). A real
        encoded photo's leftover at p>=pmin *can*, because LS fit those modes
        to actual pixels.
        """
        C = np.array(C, dtype=np.float64, copy=True)
        donor = np.asarray(donor, dtype=np.float64)
        one = C.ndim == 1
        if one:
            C = C[None, :]
            donor = donor.reshape(1, -1)
        for h in self.heads:
            if h.p < pmin:
                continue
            pb = self._band[h.p_prev]
            prev_us = C[:, pb.start : pb.start + pb.n]
            prev_d = donor[:, pb.start : pb.start + pb.n]
            up_us = prev_us @ self.D[h.p].T
            up_d = prev_d @ self.D[h.p].T
            r_d = donor[:, h.start : h.start + h.n] - h.scale * up_d
            C[:, h.start : h.start + h.n] = h.scale * up_us + r_d
        return C[0] if one else C

    def graft_fine_mix(
        self,
        C: np.ndarray,
        donors: np.ndarray,
        weights: np.ndarray,
        pmin: int = 17,
    ) -> np.ndarray:
        """Same as graft_fine but barycentric mix of several real leftovers.

        1-NN graft copies a training photo's edges. Mixing 3–4 nearby
        donors in a-space is the smallest step onto new images.
        """
        C = np.array(C, dtype=np.float64, copy=True)
        donors = np.atleast_2d(np.asarray(donors, dtype=np.float64))
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        w = w / (w.sum() + 1e-12)
        one = C.ndim == 1
        if one:
            C = C[None, :]
        for h in self.heads:
            if h.p < pmin:
                continue
            pb = self._band[h.p_prev]
            up_us = C[:, pb.start : pb.start + pb.n] @ self.D[h.p].T
            r = np.zeros((C.shape[0], h.n), dtype=np.float64)
            for d, wi in zip(donors, w):
                d = d.reshape(1, -1)
                up_d = d[:, pb.start : pb.start + pb.n] @ self.D[h.p].T
                r_d = d[:, h.start : h.start + h.n] - h.scale * up_d
                r = r + wi * r_d
            C[:, h.start : h.start + h.n] = h.scale * up_us + r
        return C[0] if one else C

    def recon_from_a(self, C: np.ndarray, mode: str = "argmax") -> np.ndarray:
        A = self.encode_a(C)
        if mode == "true":
            codes = self.encode_codes(C)
        else:
            codes = self.argmax_codes(A)
        return self.decode_codes(A, codes, c_prefix=C[:, : self.n_prefix])

    def hold_scores(self, C: np.ndarray) -> dict:
        A = self.encode_a(C)
        true_c = self.encode_codes(C)
        hat_true = self.decode_codes(A, true_c, c_prefix=C[:, : self.n_prefix])
        hat_clf = self.recon_from_a(C, mode="argmax")
        r2_vq, r2_clf, acc = {}, {}, {}
        pred_c = self.argmax_codes(A)
        for i, h in enumerate(self.heads):
            sl = slice(h.start, h.start + h.n)
            r2_vq[int(h.p)] = _r2(C[:, sl], hat_true[:, sl])
            r2_clf[int(h.p)] = _r2(C[:, sl], hat_clf[:, sl])
            acc[int(h.p)] = float(np.mean(pred_c[:, i] == true_c[:, i]))
        return {
            "r2_vq": r2_vq,
            "r2_clf": r2_clf,
            "acc": acc,
            "mean_vq": float(np.mean(list(r2_vq.values()))),
            "mean_clf": float(np.mean(list(r2_clf.values()))),
            "mean_acc": float(np.mean(list(acc.values()))),
        }

    def sgd_step(self, C: np.ndarray, lr: float, l2: float, rng: np.random.Generator, batch: int = 64) -> float:
        n = len(C)
        idx = rng.choice(n, size=min(batch, n), replace=False)
        Cb = C[idx]
        A = self.encode_a(Cb)
        true_c = self.encode_codes(Cb)
        prev = np.zeros(len(Cb), dtype=np.int32)
        nll = 0.0
        for i, h in enumerate(self.heads):
            y = true_c[:, i]
            X = self._feat(A, prev)
            lg = X @ h.W.T + h.b
            p = _softmax(lg)
            nll += float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))
            g = p
            g[np.arange(len(y)), y] -= 1.0
            g /= len(y)
            h.W -= lr * (g.T @ X + l2 * h.W)
            h.b -= lr * g.sum(axis=0)
            prev = y  # teacher-forced
        # move embed rows toward the residuals they won (organize the table)
        decay = 0.95
        for i, h in enumerate(self.heads):
            pb = self._band[h.p_prev]
            prevg = Cb[:, pb.start : pb.start + pb.n]
            up = prevg @ self.D[h.p].T
            R = Cb[:, h.start : h.start + h.n] - h.scale * up
            band = self._band[h.p]
            z = _pca_encode(R, band.mean, band.comp, band.vals)
            y = true_c[:, i]
            for j in np.unique(y):
                m = y == j
                h.book[j] = decay * h.book[j] + (1.0 - decay) * z[m].mean(axis=0)
        return nll / max(len(self.heads), 1)

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

    def formula(self) -> str:
        acc = ",".join(f"{h.p}:{h.acc:.2f}" for h in self.heads[-4:])
        return (
            f"EMBED  prefix≤{self.prefix_last} a={self.k_a} K={self.K}  "
            f"maps={self.nparams()}  holdAcc[{acc}]"
        )

    def save(self, path: str) -> None:
        payload = {
            "primes": np.array(self.primes, np.int32),
            "prefix_last": np.int32(self.prefix_last),
            "K": np.int32(self.K),
            "scene_book": np.zeros((1, self.k_a)) if self.scene_book is None else self.scene_book.astype(np.float32),
            "pfx_mean": self.pfx_mean,
            "pfx_comp": self.pfx_comp.astype(np.float32),
            "pfx_vals": self.pfx_vals,
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
            payload[f"hbook_{i}"] = h.book.astype(np.float32)
            payload[f"hW_{i}"] = h.W.astype(np.float32)
            payload[f"hb_{i}"] = h.b
            payload[f"hvq_{i}"] = np.float64(h.r2_vq)
            payload[f"hclf_{i}"] = np.float64(h.r2_clf)
            payload[f"hacc_{i}"] = np.float64(h.acc)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "ScaleVQ":
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
        heads, D = [], {}
        K = int(z["K"])
        for i in range(int(z["n_heads"])):
            p = int(z[f"hp_{i}"])
            p_prev = int(z[f"hprev_{i}"])
            D[p] = resample_op(p_prev, p)
            heads.append(
                VQHead(
                    p=p,
                    start=int(z[f"hs_{i}"]),
                    n=int(z[f"hn_{i}"]),
                    p_prev=p_prev,
                    scale=float(z[f"hsc_{i}"]),
                    book=np.array(z[f"hbook_{i}"], np.float64),
                    W=np.array(z[f"hW_{i}"], np.float64),
                    b=np.array(z[f"hb_{i}"], np.float64),
                    r2_vq=float(z[f"hvq_{i}"]),
                    r2_clf=float(z[f"hclf_{i}"]),
                    acc=float(z[f"hacc_{i}"]),
                )
            )
        return cls(
            primes=primes,
            prefix_last=int(z["prefix_last"]),
            pfx_mean=np.array(z["pfx_mean"], np.float64),
            pfx_comp=np.array(z["pfx_comp"], np.float64),
            pfx_vals=np.array(z["pfx_vals"], np.float64),
            pfx_U=np.array(z["pfx_U"], np.float64),
            pfx_b=np.array(z["pfx_b"], np.float64),
            bands=bands,
            heads=heads,
            D=D,
            K=K,
            scene_book=np.array(z["scene_book"], np.float64)
            if "scene_book" in z.files and z["scene_book"].ndim == 2 and z["scene_book"].shape[0] > 1
            else None,
        )

    @classmethod
    def fit(
        cls,
        bank: np.ndarray,
        primes: tuple[int, ...],
        prefix_last: int = 5,
        K: int = 16,
        k_a: int = 12,
        k_z: int = 8,
        hold: float = 0.2,
        seed: int = 0,
    ) -> tuple["ScaleVQ", dict]:
        bank = np.asarray(bank, dtype=np.float64)
        rng = np.random.default_rng(seed)
        n = len(bank)
        idx = rng.permutation(n)
        n_te = max(32, int(round(n * hold)))
        te, tr = idx[:n_te], idx[n_te:]
        Ctr, Cte = bank[tr], bank[te]
        offs = offsets_of(primes)
        n_prefix = sum(p * p for p, _s, _n in offs if p <= prefix_last)
        k_a = min(k_a, n_prefix, max(4, len(tr) // 10))
        pmean, pcomp, pvals = _pca_fit(Ctr[:, :n_prefix], k_a)
        k_a = int(pcomp.shape[0])
        Xtr = _pca_encode(Ctr[:, :n_prefix], pmean, pcomp, pvals)
        pfx_U, pfx_b = _ridge(Xtr, Xtr, lam=0.1)  # identity-ish
        # actually decode_prefix uses pfx_U @ a; identity:
        pfx_U = np.eye(k_a)
        pfx_b = np.zeros(k_a)
        n_scene = min(32, max(8, len(tr) // 12))
        scene_book, _ = _kmeans(Xtr, n_scene, rng)

        print(
            f"  VQ fit  train={len(tr)} hold={len(te)}  prefix≤{prefix_last} "
            f"({n_prefix}→{k_a})  K={K} kz={k_z}",
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

        heads: list[VQHead] = []
        prev = None
        kz = k_z
        for p, start, nn in offs:
            if prev is None or p <= prefix_last:
                prev = p
                continue
            Dp = D[p]
            pb = band_by[prev]
            Ytr = Ctr[:, start : start + nn]
            Utr = Ctr[:, pb.start : pb.start + pb.n] @ Dp.T
            scale = float(np.sum(Ytr * Utr) / (np.sum(Utr * Utr) + 1e-12))
            Rtr = Ytr - scale * Utr
            rmean, rcomp, rvals = _pca_fit(Rtr, min(kz, Rtr.shape[1] - 1, len(tr) - 1))
            band_by[p].mean, band_by[p].comp, band_by[p].vals = rmean, rcomp, rvals
            Ztr = _pca_encode(Rtr, rmean, rcomp, rvals)
            book, assign = _kmeans(Ztr, K, rng)
            # init classifier from (a, ones) via one-vs-rest-ish: mean a per code
            Atr = Xtr
            feat_d = k_a + K
            W = np.zeros((K, feat_d))
            b = np.zeros(K)
            # start near uniform; a small bias toward empirical code freq
            freq = np.bincount(assign, minlength=K).astype(np.float64) + 1.0
            b = np.log(freq) - np.log(freq.mean())
            heads.append(
                VQHead(
                    p=p, start=start, n=nn, p_prev=prev, scale=scale,
                    book=book, W=W, b=b, r2_vq=0.0, r2_clf=0.0, acc=0.0,
                )
            )
            print(
                f"    p={p:2d}←{prev:<2d}  kz={book.shape[1]}  "
                f"used={int((np.bincount(assign, minlength=K)>0).sum())}/{K} codes",
                flush=True,
            )
            prev = p

        model = cls(
            primes=primes,
            prefix_last=prefix_last,
            pfx_mean=pmean,
            pfx_comp=pcomp,
            pfx_vals=pvals,
            pfx_U=pfx_U,
            pfx_b=pfx_b,
            bands=list(band_by[p] for p, _s, _n in offs),
            heads=heads,
            D=D,
            K=K,
            scene_book=scene_book,
        )
        # a few CE steps so argmax isn't random
        for _ in range(80):
            model.sgd_step(Ctr, lr=0.15, l2=1e-3, rng=rng, batch=min(64, len(Ctr)))
        sc_te = model.hold_scores(Cte)
        sc_tr = model.hold_scores(Ctr)
        for h in heads:
            h.r2_vq = sc_te["r2_vq"][h.p]
            h.r2_clf = sc_te["r2_clf"][h.p]
            h.acc = sc_te["acc"][h.p]
        print(
            f"  hold VQ-ceiling R²={sc_te['mean_vq']:.3f}  "
            f"argmax-from-a R²={sc_te['mean_clf']:.3f}  "
            f"code acc={sc_te['mean_acc']:.3f} (chance={1/K:.3f})  "
            f"train acc={sc_tr['mean_acc']:.3f}",
            flush=True,
        )
        stats = {
            "n_train": int(len(tr)),
            "n_hold": int(len(te)),
            "te_idx": te.astype(np.int32),
            "tr_idx": tr.astype(np.int32),
            **sc_te,
        }
        return model, stats


class GeneratorVQ:
    def __init__(self, field: PrimeField, vq: ScaleVQ, gmm: DiagGMM | None):
        self.field = field
        self.vq = vq
        self.gmm = gmm

    def raster(self, c, H=96, W=96, backend="cpu"):
        self.field._apply_params(c)
        return self.field.decode(H, W, stretch=False, backend=backend)

    def sample(self, n=1, H=96, W=96, temp=0.9, seed=0, backend="cpu", cloud=True):
        rng = np.random.default_rng(seed)
        Cs = []
        book = self.vq.scene_book
        for i in range(n):
            if book is not None and rng.random() < 0.65:
                a = book[int(rng.integers(0, len(book)))].copy()
                a = a + rng.normal(size=a.shape) * 0.2
            elif self.gmm is not None:
                a = self.gmm.sample(1, rng)[0]
            else:
                a = rng.normal(size=self.vq.k_a) * 0.5
            codes = self.vq.sample_codes(a, rng, temp=temp)
            Cs.append(self.vq.decode_codes(a, codes[0]))
        Cs = np.stack(Cs)
        return [self.raster(c, H, W, backend=backend) for c in Cs], Cs

    def morph(self, c0, c1, frames=8, H=96, W=96, backend="cpu"):
        a0 = self.vq.encode_a(c0)
        a1 = self.vq.encode_a(c1)
        rng = np.random.default_rng(0)
        out = []
        for i in range(frames):
            t = i / max(frames - 1, 1)
            a = (1.0 - t) * a0 + t * a1
            codes = self.vq.sample_codes(a, rng, temp=0.85)
            out.append(self.raster(self.vq.decode_codes(a, codes[0]), H, W, backend=backend))
        return out

    def stages(self, c):
        return self.vq.stages(c)

    def formula(self):
        g = f"  aGMM={self.gmm.m}×N({self.gmm.d})" if self.gmm is not None else ""
        return self.vq.formula() + g

    def save(self, stem: str) -> None:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        self.field.save(str(stem) + ".field.npz")
        self.vq.save(str(stem) + ".vq.npz")
        if self.gmm is not None:
            self.gmm.save(str(stem) + ".gmm.npz")
        Path(str(stem) + ".json").write_text(
            json.dumps({"formula": self.formula(), "K": self.vq.K, "nparams": self.vq.nparams()}, indent=2) + "\n"
        )
