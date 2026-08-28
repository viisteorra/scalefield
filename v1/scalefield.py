"""
ScaleField v1 — prime-grid color-phase field.

Internal state is not H×W×3. It is one scalar per cell of each prime lattice
(1×1 brightness, then 2×2, 3×3, 5×5, …). Those lattices are basal 2-D
frequency grids (a cosine / DCT transform whose periods never align).
Color is three incommensurate phases of the scalar. Raster is a full-screen
shader. Edges come from the same matrices: high-k modes (Laplacian boost)
plus the product of consecutive AC fields (prime-chord beats).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# 27th prime is 103; that's the 1240p ladder.
# This box (i5-6500 / 15GB) is happy through 31. 53 is optional "huge".
PRIMES = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
    53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
)
MAX_PARAMS = 14000  # 1²+…+53² = 13276. Default sharp is 31²-ladder = 3359.

# Three color phases. Off-diagonals are the harmonic leak of natural spectra.
# The 3×3 is invertible, so neighboring phases span RGB.
COLOR_PHASES = np.array(
    [
        [1.00, 0.16, 0.05],
        [0.08, 1.00, 0.14],
        [0.06, 0.12, 1.00],
    ],
    dtype=np.float64,
)


def _ok_primes(primes: tuple[int, ...]) -> tuple[int, ...]:
    if not primes or primes[0] != 1:
        raise ValueError("primes must start with 1 (brightness 1×1)")
    seen = {1}
    for p in primes[1:]:
        if p not in PRIMES:
            raise ValueError(f"{p} is not in the prime ladder")
        if p in seen:
            raise ValueError(f"duplicate prime {p}")
        seen.add(p)
    return tuple(int(p) for p in primes)


def nparams_of(primes: tuple[int, ...]) -> int:
    return int(sum(p * p for p in primes))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2))


def fovea_mse(pred: np.ndarray, tgt: np.ndarray, frac: float = 0.5) -> float:
    h, w = tgt.shape[:2]
    dh = max(1, int(h * frac))
    dw = max(1, int(w * frac))
    y0 = (h - dh) // 2
    x0 = (w - dw) // 2
    return mse(pred[y0 : y0 + dh, x0 : x0 + dw], tgt[y0 : y0 + dh, x0 : x0 + dw])


def sparsity(c: np.ndarray, eps: float = 1e-3) -> float:
    return float(np.mean(np.abs(c) < eps))


def to_float_rgb(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb, dtype=float)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("rgb must be H,W,3")
    if a.max() > 1.5:
        a = a / 255.0
    return a


def pixel_uv(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Image-space chart. u right, v down, origin at the panel center, range [-1, 1]."""
    ys = (np.arange(H, dtype=np.float64) + 0.5) / H * 2.0 - 1.0
    xs = (np.arange(W, dtype=np.float64) + 0.5) / W * 2.0 - 1.0
    v, u = np.meshgrid(ys, xs, indexing="ij")
    return u, v


def foveal_map(u: np.ndarray, v: np.ndarray, a: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Separable tanh chart. Center is expanded, periphery compressed, so a uniform
    p×p lattice has larger screen pixels at the outside. a=0 is off (identity).
    """
    if a <= 1e-8:
        return u, v
    ta = math.tanh(a)
    return np.tanh(a * u) / ta, np.tanh(a * v) / ta


def kernel_name(p: int) -> str:
    if p <= 1:
        return "dc"
    if p == 2:
        return "balance"
    return "dct"


def _mode_weights(p: int, edge_gain: float) -> np.ndarray:
    """Laplacian-style boost on high-k DCT modes. (0,0) stays 1 — that's DC."""
    if p <= 1:
        return np.ones((1, 1), dtype=np.float64)
    k = np.arange(p, dtype=np.float64)
    den = float(max(p - 1, 1) ** 2)
    return 1.0 + float(edge_gain) * (k[None, :] ** 2 + k[:, None] ** 2) / den


def _dct_scalar(
    coeff: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p: int,
    edge_gain: float,
    skip_dc: bool = False,
) -> np.ndarray:
    """
    Separable 2-D cosine grid. û,v̂ in [0,1] from foveated chart u,v in [-1,1].
    Full-screen standing waves — every mode covers the panel. No holes.
    """
    g = np.asarray(coeff, dtype=np.float64).reshape(p, p)
    if p == 1:
        return np.full(u.shape, float(g[0, 0]), dtype=np.float64)
    uh = (u + 1.0) * 0.5
    vh = (v + 1.0) * 0.5
    k = np.arange(p, dtype=np.float64)
    cu = np.cos(np.pi * k[:, None, None] * uh[None, :, :])
    cv = np.cos(np.pi * k[:, None, None] * vh[None, :, :])
    cw = g * _mode_weights(p, edge_gain)
    if skip_dc:
        cw = np.array(cw, copy=True)
        cw[0, 0] = 0.0
    # g[l, k] · cos(π k û) · cos(π l v̂)
    G = np.einsum("lk,khw->lhw", cw, cu)
    return np.einsum("lhw,lhw->hw", G, cv)


def resample_op(src: int, p: int) -> np.ndarray:
    """Bilinear map from one src×src matrix onto a p×p lattice. Shape (p², src²)."""
    if src < 1:
        raise ValueError("src must be >= 1")
    if p == 1:
        return np.full((1, src * src), 1.0 / float(src * src), dtype=np.float64)
    i = np.arange(p, dtype=np.float64)
    yy, xx = np.meshgrid(i, i, indexing="ij")
    x = xx * ((src - 1) / max(p - 1, 1))
    y = yy * ((src - 1) / max(p - 1, 1))
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, src - 1)
    y1 = np.minimum(y0 + 1, src - 1)
    fx = x - x0
    fy = y - y0
    D = np.zeros((p * p, src * src), dtype=np.float64)
    out = np.arange(p * p)
    np.add.at(D, (out, y0.ravel() * src + x0.ravel()), ((1.0 - fx) * (1.0 - fy)).ravel())
    np.add.at(D, (out, y0.ravel() * src + x1.ravel()), (fx * (1.0 - fy)).ravel())
    np.add.at(D, (out, y1.ravel() * src + x0.ravel()), ((1.0 - fx) * fy).ravel())
    np.add.at(D, (out, y1.ravel() * src + x1.ravel()), (fx * fy).ravel())
    return D


def color_dirs(p: int, prime_idx: int) -> np.ndarray:
    """(p, p, 3) RGB directions.

    p=1 is brightness. Every other prime lattice is one of the three color
    phases — the whole p×p shares that hue. Phases cycle with the prime so
    a 2×2 red field never locks to a 3×3 green field. A spatial Bayer inside
    the lattice stripes RGB at cell scale; don't do that.
    """
    if p == 1:
        return np.ones((1, 1, 3), dtype=np.float64)
    vec = COLOR_PHASES[prime_idx % 3]
    out = np.empty((p, p, 3), dtype=np.float64)
    out[...] = vec
    return out


def _solve_c(A: np.ndarray, y: np.ndarray, lam_l2: float = 1e-4) -> np.ndarray:
    """min ||A c - y||^2 / n + lam_l2 ||c||^2."""
    n, p = A.shape
    Ad = np.asarray(A, dtype=np.float64)
    yd = np.asarray(y, dtype=np.float64)
    ATA = Ad.T @ Ad
    ATy = Ad.T @ yd
    ridge = max(float(lam_l2), 1e-12)
    lhs = ATA + (n * ridge) * np.eye(p)
    try:
        return np.linalg.solve(lhs, ATy)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(Ad, yd, rcond=None)[0]


def save_png(path: str, rgb: np.ndarray) -> None:
    from PIL import Image

    a = np.asarray(rgb)
    if a.dtype != np.uint8:
        a = to_float_rgb(a)
        a = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(a, mode="RGB").save(path)


@dataclass
class PrimeField:
    """
    primes: (1, 2, 3, 5, …). c is concatenated p×p scalars, length sum p².
    Color lives in the phase assignment, not a third axis of c.
    """

    primes: tuple[int, ...] = (1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
    src: int = 0
    foveal: float = 1.15
    edge_gain: float = 0.55
    chord_gain: float = 0.12
    PRESETS = {
        "tiny": (1, 2, 3, 5, 7),                          # 88
        "small": (1, 2, 3, 5, 7, 11),                     # 209
        "medium": (1, 2, 3, 5, 7, 11, 13),                # 378
        "full": (1, 2, 3, 5, 7, 11, 13, 17),              # 667
        "max360": (1, 2, 3, 5, 7, 11, 13, 17, 19),  # 1028
        "sharp23": (1, 2, 3, 5, 7, 11, 13, 17, 19, 23),  # 1557
        "sharp": (1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31),  # 3359 — this rig's visual model
        "tied": (1, 2, 3, 5, 7, 11, 13, 17, 19, 23),  # same primes, one src matrix
        "huge": (
            1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53
        ),
    }
    # src>0: one stored matrix; every prime lattice is a resample of it.
    # Looks trippy; not the visual model. Keep for experiments only.
    PRESET_SRC = {
        "tiny": 0,
        "small": 0,
        "medium": 0,
        "full": 0,
        "max360": 0,
        "sharp23": 0,
        "sharp": 0,
        "tied": 23,
        "huge": 0,
    }

    c: np.ndarray = field(init=False, repr=False)
    bank: np.ndarray | None = field(default=None, init=False, repr=False)
    _A_cache: dict = field(default_factory=dict, init=False, repr=False)
    _hues: list = field(default_factory=list, init=False, repr=False)
    _factor: tuple | None = field(default=None, init=False, repr=False)

    @classmethod
    def preset(cls, name: str = "sharp", foveal: float = 1.15, **kw) -> "PrimeField":
        if name not in cls.PRESETS:
            raise KeyError(f"unknown preset {name}; have {list(cls.PRESETS)}")
        src = kw.pop("src", cls.PRESET_SRC.get(name, 0))
        return cls(primes=cls.PRESETS[name], foveal=foveal, src=src, **kw)

    def __post_init__(self):
        self.primes = _ok_primes(tuple(self.primes))
        self.src = int(self.src)
        n_free = nparams_of(self.primes)
        n_store = self.src * self.src if self.src > 0 else n_free
        if n_store > MAX_PARAMS:
            raise ValueError(f"stored params {n_store} exceeds cap {MAX_PARAMS}")
        self.c = np.zeros(n_free, dtype=np.float64)  # expanded prime lattices (derived if src>0)
        self.s = np.zeros((self.src, self.src), dtype=np.float64) if self.src > 0 else None
        self._hues = []
        pidx = 0
        for p in self.primes:
            self._hues.append(color_dirs(p, pidx if p > 1 else 0))
            if p > 1:
                pidx += 1
        self._A_cache = {}
        self._factor = None
        self._D = None
        if self.src > 0:
            self._sync_c()

    def nparams(self) -> int:
        if self.src > 0:
            return int(self.src * self.src)
        return int(self.c.size)

    def _resample_D(self) -> np.ndarray:
        """(sum p², src²) — one source matrix → every prime lattice."""
        if self._D is not None:
            return self._D
        blocks = [resample_op(self.src, p) for p in self.primes]
        self._D = np.concatenate(blocks, axis=0)
        return self._D

    def _sync_c(self) -> None:
        if self.src <= 0 or self.s is None:
            return
        d = self._resample_D()
        self.c = d @ self.s.reshape(-1)
        want = nparams_of(self.primes)
        if self.c.size != want:
            raise RuntimeError(f"derived c has {self.c.size} cells, expected {want}")

    def _apply_params(self, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float64).reshape(-1)
        if self.src > 0:
            self.s = vec.reshape(self.src, self.src)
            self._sync_c()
        else:
            self.c = vec

    def offsets(self) -> list[tuple[int, int, int]]:
        """(p, start, n) for each lattice in c."""
        out = []
        s = 0
        for p in self.primes:
            n = p * p
            out.append((p, s, n))
            s += n
        return out

    def grid(self, p: int) -> np.ndarray:
        for q, s, n in self.offsets():
            if q == p:
                return self.c[s : s + n].reshape(p, p)
        raise KeyError(p)

    def set_grid(self, p: int, g: np.ndarray) -> None:
        g = np.asarray(g, dtype=np.float64).reshape(p, p)
        for q, s, n in self.offsets():
            if q == p:
                self.c[s : s + n] = g.reshape(n)
                return
        raise KeyError(p)

    def formula(self) -> str:
        bits = ",".join(f"{p}×{p}:{kernel_name(p)}" for p in self.primes)
        srcbit = f"S={self.src}×{self.src}→{len(self.primes)} primes  " if self.src > 0 else ""
        sp = self.s if self.src > 0 else self.c
        return (
            f"C(u)=sum_p DCT_p(foveal(u); resample(S,p))·phase + edge + chord  "
            f"| {srcbit}p=[{bits}]  N={self.nparams()}  foveal={self.foveal:g}  "
            f"edge={self.edge_gain:g} chord={self.chord_gain:g}  "
            f"sparse={sparsity(sp):.2f}  phases=3×256"
        )

    def copy(self) -> "PrimeField":
        pf = PrimeField(
            primes=self.primes,
            src=self.src,
            foveal=self.foveal,
            edge_gain=self.edge_gain,
            chord_gain=self.chord_gain,
        )
        pf.c = self.c.copy()
        if self.s is not None:
            pf.s = self.s.copy()
        if self.bank is not None:
            pf.bank = self.bank.copy()
        return pf

    def _uv(self, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
        u, v = pixel_uv(H, W)
        return foveal_map(u, v, self.foveal)

    def rgb_tex(self, p: int) -> np.ndarray:
        g = self.grid(p)
        hues = self._hues[self.primes.index(p)]
        return g[:, :, None] * hues

    def _scalar_maps(self, H: int, W: int) -> list[np.ndarray]:
        u, v = self._uv(H, W)
        return [
            _dct_scalar(self.grid(p), u, v, p, self.edge_gain, skip_dc=False)
            for p in self.primes
        ]

    def field(self, H: int, W: int, c: np.ndarray | None = None) -> np.ndarray:
        """RGB map (H, W, 3), unclipped. DCT grids + edge boost + prime chords."""
        hold_c = hold_s = None
        if c is not None:
            hold_c, hold_s = self.c.copy(), None if self.s is None else self.s.copy()
            self._apply_params(c)
        try:
            scalars = self._scalar_maps(H, W)
            acc = np.zeros((H, W, 3), dtype=np.float64)
            ac = []
            for p, s, hue in zip(self.primes, scalars, self._hues):
                acc += s[:, :, None] * hue[0, 0]
                ac.append(s - float(self.grid(p)[0, 0]))
            if self.chord_gain and len(ac) > 1:
                ch = np.zeros((H, W), dtype=np.float64)
                for a, b in zip(ac, ac[1:]):
                    ch += a * b
                # Beats of consecutive primes: multi-scale edges, luma.
                acc += float(self.chord_gain) * ch[:, :, None]
            return acc
        finally:
            if hold_c is not None:
                self.c = hold_c
                if hold_s is not None:
                    self.s = hold_s

    def spatial_weight(self, H: int, W: int) -> np.ndarray:
        """DC coverage. Cosine grids are full-screen; p=1 is identically 1."""
        u, v = self._uv(H, W)
        return _dct_scalar(np.ones((1, 1)), u, v, 1, 0.0)

    def raster(self, H: int = 360, W: int = 360, stretch: bool = False, backend: str = "cpu") -> np.ndarray:
        if backend == "gpu":
            try:
                from gpu import raster_gpu

                img = raster_gpu(self, H, W)
            except Exception as e:
                print(f"gpu raster failed ({e}); using cpu", flush=True)
                img = self.field(H, W)
        else:
            img = self.field(H, W)
        if stretch:
            lo, hi = np.percentile(img, 1), np.percentile(img, 99)
            if hi <= lo:
                hi = lo + 1e-6
            img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
        else:
            img = np.clip(img, 0.0, 1.0)
        return (img * 255.0 + 0.5).astype(np.uint8)

    def decode(self, H: int, W: int, stretch: bool = False, backend: str = "cpu") -> np.ndarray:
        return self.raster(H, W, stretch=stretch, backend=backend)

    def linear_field(self, H: int, W: int, c: np.ndarray | None = None) -> np.ndarray:
        """DCT + edge boost only (no prime-chord). Linear in c."""
        hold = None
        if c is not None:
            hold, self.c = self.c, np.asarray(c, dtype=np.float64)
        try:
            scalars = self._scalar_maps(H, W)
            acc = np.zeros((H, W, 3), dtype=np.float64)
            for s, hue in zip(scalars, self._hues):
                acc += s[:, :, None] * hue[0, 0]
            return acc
        finally:
            if hold is not None:
                self.c = hold

    def design(self, H: int, W: int, dtype=None) -> np.ndarray:
        """A, shape (H*W*3, N). Linear DCT+edge part (chords are quadratic)."""
        n_pix = H * W
        N_c = nparams_of(self.primes)
        N = self.nparams()
        if dtype is None:
            dtype = np.float32 if n_pix * 3 * N_c * 8 > 400_000_000 else np.float64
        dtype = np.dtype(dtype)
        key = (H, W, self.primes, self.src, float(self.foveal), float(self.edge_gain), dtype.str)
        cached = self._A_cache.get(key)
        if cached is not None:
            return cached
        u, v = self._uv(H, W)
        uh = ((u + 1.0) * 0.5).reshape(n_pix)
        vh = ((v + 1.0) * 0.5).reshape(n_pix)
        A_c = np.zeros((n_pix * 3, N_c), dtype=dtype)
        col = 0
        for p, hues in zip(self.primes, self._hues):
            n_loc = p * p
            hue = hues[0, 0]
            if p == 1:
                A_sp = np.ones((n_pix, 1), dtype=dtype)
            else:
                k = np.arange(p, dtype=np.float64)
                cu = np.cos(np.pi * np.outer(uh, k))
                cv = np.cos(np.pi * np.outer(vh, k))
                w = _mode_weights(p, self.edge_gain)
                A_sp = (cv[:, :, None] * cu[:, None, :] * w[None, :, :]).reshape(n_pix, n_loc).astype(dtype, copy=False)
            block = A_sp[:, None, :] * hue[None, :, None]
            A_c[:, col : col + n_loc] = block.reshape(n_pix * 3, n_loc)
            col += n_loc
        if self.src > 0:
            D = self._resample_D().astype(dtype, copy=False)
            A = A_c @ D
            del A_c
        else:
            A = A_c
        if A.nbytes > 200_000_000:
            self._A_cache.clear()
        elif len(self._A_cache) > 4:
            self._A_cache.clear()
        self._factor = None
        self._A_cache[key] = A
        return A

    def encode(
        self,
        rgb: np.ndarray,
        lam_l2: float = 1e-4,
        reset: bool = True,
        **_ignored,
    ) -> str:
        tgt = to_float_rgb(rgb)
        bank = self.encode_many([tgt], lam_l2=lam_l2, chord_steps=0)
        self.c = bank[0]
        return self.formula()

    def encode_many(
        self,
        images: list[np.ndarray],
        lam_l2: float = 1e-4,
        chord_steps: int = 1,
    ) -> np.ndarray:
        if not images:
            raise ValueError("no images")
        imgs = [to_float_rgb(im) for im in images]
        H, W, _ = imgs[0].shape
        for im in imgs:
            if im.shape != (H, W, 3):
                raise ValueError("all images must share H,W,3")
        if self.nparams() > 500:
            print(
                f"  encode {len(imgs)} imgs @ {H}  N={self.nparams()}  building design…",
                flush=True,
            )
        A = self.design(H, W)
        n, p = A.shape
        lam = max(lam_l2, 1e-12)
        fkey = (H, W, self.primes, float(self.foveal), float(self.edge_gain), A.dtype.str, float(lam))
        cho = ata = None
        use_cho = False
        if self._factor is not None and self._factor[0] == fkey:
            _, cho, ata, use_cho = self._factor
            if self.nparams() > 500:
                print(f"  A {A.shape} {A.dtype} {A.nbytes/1e6:.0f}MB  (cached factor)", flush=True)
        else:
            if self.nparams() > 500:
                print(
                    f"  A {A.shape} {A.dtype} {A.nbytes/1e6:.0f}MB  "
                    f"ATA gemm on 4 cores — a few minutes, not stuck…",
                    flush=True,
                )
            ATA = np.asarray(A.T @ A, dtype=np.float64)
            ATA.flat[:: p + 1] += n * lam
            try:
                cho = np.linalg.cholesky(ATA)
                use_cho = True
                ata = None
                del ATA
            except np.linalg.LinAlgError:
                use_cho = False
                ata = ATA
            self._factor = (fkey, cho, ata, use_cho)
            if self.nparams() > 500:
                print("  factor done  solve bank…", flush=True)
        Y = np.empty((len(imgs), n), dtype=A.dtype)
        for k, im in enumerate(imgs):
            Y[k] = np.asarray(im.reshape(-1), dtype=A.dtype)
        # (K, n) @ (n, p) — no transpose of the giant A.
        BT = np.asarray(Y @ A, dtype=np.float64)
        del Y
        bank = np.zeros((len(imgs), p), dtype=np.float64)
        for k in range(len(imgs)):
            if use_cho:
                z = np.linalg.solve(cho, BT[k])
                bank[k] = np.linalg.solve(cho.T, z)
            else:
                bank[k] = np.linalg.solve(ata, BT[k])
        # Chord GN re-renders the full field per image. Fine for tiny;
        # with p=53 that's minutes of silence. LS on the DCT grids is the fit.
        if self.chord_gain and chord_steps > 0 and self.src == 0 and self.nparams() <= 500:
            hold = self.c.copy()
            for _ in range(int(chord_steps)):
                for k, im in enumerate(imgs):
                    self.c = bank[k]
                    linear = np.asarray(A @ bank[k].astype(A.dtype, copy=False), dtype=np.float64).reshape(H, W, 3)
                    chord = self.field(H, W) - linear
                    y = np.asarray(im.reshape(H * W * 3) - chord.reshape(-1), dtype=A.dtype)
                    ATy = np.asarray(A.T @ y, dtype=np.float64)
                    if use_cho:
                        z = np.linalg.solve(cho, ATy)
                        bank[k] = np.linalg.solve(cho.T, z)
                    else:
                        bank[k] = np.linalg.solve(ata, ATy)
            self.c = hold
        self.bank = bank
        self._apply_params(bank.mean(axis=0))
        return bank

    def reconstruct(self, rgb: np.ndarray, **encode_kw) -> tuple[np.ndarray, dict]:
        tgt = to_float_rgb(rgb)
        self.encode(tgt, **encode_kw)
        pred = self.field(*tgt.shape[:2])
        stats = {
            "mse": mse(pred, tgt),
            "fovea_mse": fovea_mse(pred, tgt),
            "sparsity": sparsity(self.c),
            "nparams": self.nparams(),
        }
        return pred, stats

    def evaluate(self, images: list[np.ndarray], **encode_kw) -> dict:
        imgs = [to_float_rgb(im) for im in images]
        bank = self.encode_many(imgs, **{k: v for k, v in encode_kw.items() if k in ("lam_l2", "chord_steps")})
        H, W = imgs[0].shape[:2]
        A = self.design(H, W)
        C = np.asarray(bank.T, dtype=A.dtype)
        PRED = np.asarray(A @ C, dtype=np.float64)
        mses, foveas, spars = [], [], []
        for k, im in enumerate(imgs):
            pred = PRED[:, k].reshape(H, W, 3)
            mses.append(mse(pred, im))
            foveas.append(fovea_mse(pred, im))
            spars.append(sparsity(bank[k]))
        self.c = bank.mean(axis=0)
        return {
            "mse": float(np.mean(mses)),
            "mse_std": float(np.std(mses)),
            "fovea_mse": float(np.mean(foveas)),
            "sparsity": float(np.mean(spars)),
            "nparams": self.nparams(),
            "n": len(imgs),
            "per_image_mse": mses,
            "bank": bank,
        }

    def generate(self, H: int = 360, W: int = 360, seed: int = 0, amp: float = 1.0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if self.src > 0:
            self.s = rng.normal(0.0, 0.35 * amp, size=(self.src, self.src))
            self.s[self.src // 2, self.src // 2] += 0.4 * amp
            self._sync_c()
        else:
            self.c[:] = 0.0
            for p, off, n in self.offsets():
                scale = amp / float(p)
                self.c[off : off + n] = rng.normal(0.0, scale, size=n)
                if p == 1:
                    self.c[off] = abs(self.c[off]) + 0.35 * amp
        return self.decode(H, W, stretch=True)

    def generate_from_cloud(self, H: int = 360, W: int = 360, seed: int = 0, mix: float = 0.12) -> np.ndarray:
        if self.bank is None or len(self.bank) == 0:
            return self.generate(H, W, seed=seed)
        rng = np.random.default_rng(seed)
        K = len(self.bank)
        i0, i1 = int(rng.integers(0, K)), int(rng.integers(0, K))
        t = float(rng.uniform(0.0, 1.0))
        vec = (1.0 - t) * self.bank[i0] + t * self.bank[i1]
        vec = vec + mix * rng.normal(0.0, np.std(self.bank) + 1e-6, size=vec.shape)
        self._apply_params(vec)
        return self.decode(H, W, stretch=False)

    def layer_field(self, p: int, H: int, W: int) -> np.ndarray:
        """Raster a single prime lattice (others zero)."""
        pf = self.copy()
        pf.c[:] = 0.0
        pf.set_grid(p, self.grid(p))
        return pf.field(H, W)

    def save(self, path: str, **extra) -> None:
        payload = {
            "c": self.c,
            "primes": np.array(self.primes, dtype=np.int32),
            "src": np.int32(self.src),
            "foveal": np.float64(self.foveal),
            "edge_gain": np.float64(self.edge_gain),
            "chord_gain": np.float64(self.chord_gain),
        }
        if self.s is not None:
            payload["s"] = self.s
        if self.bank is not None:
            payload["bank"] = self.bank
        for k, v in extra.items():
            payload[k] = v
        path = str(path)
        if not path.endswith(".npz"):
            path = path + ".npz"
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "PrimeField":
        z = np.load(path, allow_pickle=True)
        primes = tuple(int(p) for p in z["primes"].tolist())
        foveal = float(z["foveal"]) if "foveal" in z.files else 1.15
        edge = float(z["edge_gain"]) if "edge_gain" in z.files else 0.55
        chord = float(z["chord_gain"]) if "chord_gain" in z.files else 0.12
        src = int(z["src"]) if "src" in z.files else 0
        pf = cls(primes=primes, src=src, foveal=foveal, edge_gain=edge, chord_gain=chord)
        pf.c = np.array(z["c"], dtype=np.float64)
        if "s" in z.files:
            pf.s = np.array(z["s"], dtype=np.float64)
            pf._sync_c()
        if "bank" in z.files:
            pf.bank = np.array(z["bank"], dtype=np.float64)
        return pf


def grid_line_overlay(H: int, W: int, primes: tuple[int, ...], foveal: float = 1.15) -> np.ndarray:
    """Debug: draw each prime's interior grid lines. They should not coincide."""
    u, v = pixel_uv(H, W)
    cu, cv = foveal_map(u, v, foveal)
    img = np.zeros((H, W, 3), dtype=np.float64)
    hues = [(1.0, 0.25, 0.2), (0.2, 0.85, 0.35), (0.25, 0.45, 1.0)]
    ux = (cu + 1.0) * 0.5
    uy = (cv + 1.0) * 0.5
    for k, p in enumerate(primes):
        if p <= 1:
            continue
        col = hues[(k - 1) % 3]
        # lines at i/p in chart [0,1], skip the frame
        for i in range(1, p):
            t = i / p
            img += np.exp(-((ux - t) * W) ** 2 / 2.0)[:, :, None] * np.array(col)
            img += np.exp(-((uy - t) * H) ** 2 / 2.0)[:, :, None] * np.array(col)
    img = np.clip(img, 0.0, 1.0)
    return img


if __name__ == "__main__":
    from pathlib import Path

    Path("artifacts").mkdir(exist_ok=True)
    pf = PrimeField.preset("medium")
    img = pf.generate(360, 360, seed=1)
    save_png("artifacts/scalefield_v1_gen360.png", img)
    print(pf.formula())
    print(f"wrote artifacts/scalefield_v1_gen360.png {img.shape} N={pf.nparams()}")
    try:
        from gpu import gpu_info

        print("gpu", gpu_info())
    except Exception as e:
        print("gpu", e)
