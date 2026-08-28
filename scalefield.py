"""
ScaleField — visual models as x*3*i fields, not H*W*3 tensors.

State is an amplitude surface (x neurons) × 3 channels × i scales.
The field is linear in b, so encode is a small least-squares solve.
Rasterize only to paint a panel.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

LN2 = math.log(2.0)
# 32×3×24 = 2304. Fence, not a target.
MAX_PARAMS = 32 * 3 * 24
# Growing x adds more localized blobs. Log-polar tags are opt-in (textured generate).
BASIS_DEFAULTS = ["exp"] * 32
VISUAL_TAGS = ["exp", "const", "lin", "linu", "linv", "gabor", "quad"]
SPIRAL_TAGS = frozenset({"logsin", "logcos", "rip"})
GOLDEN = math.pi * (3.0 - math.sqrt(5.0))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2))


def fovea_mse(pred: np.ndarray, tgt: np.ndarray, frac: float = 0.5) -> float:
    """MSE on a centered crop. frac=0.5 → middle 50% of each side."""
    h, w = tgt.shape[:2]
    dh = max(1, int(h * frac))
    dw = max(1, int(w * frac))
    y0 = (h - dh) // 2
    x0 = (w - dw) // 2
    return mse(pred[y0 : y0 + dh, x0 : x0 + dw], tgt[y0 : y0 + dh, x0 : x0 + dw])


def sparsity(b: np.ndarray, eps: float = 1e-3) -> float:
    return float(np.mean(np.abs(b) < eps))


def to_float_rgb(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb, dtype=float)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("rgb must be H,W,3")
    if a.max() > 1.5:
        a = a / 255.0
    return a


def pack_b(b: np.ndarray) -> np.ndarray:
    """(i, 3, x) -> (i*x, 3)."""
    return np.transpose(b, (0, 2, 1)).reshape(b.shape[0] * b.shape[2], 3)


def unpack_b(B: np.ndarray, i: int, x: int) -> np.ndarray:
    """(i*x, 3) -> (i, 3, x)."""
    return B.reshape(i, x, 3).transpose(0, 2, 1)


def _offset(k: int, psi: dict) -> tuple[float, float]:
    """Blob center in chart coords. Vogel layout is stable when x grows."""
    if f"du{k}" in psi or f"dv{k}" in psi:
        return float(psi.get(f"du{k}", 0.0)), float(psi.get(f"dv{k}", 0.0))
    if k == 0:
        return 0.0, 0.0
    radius = min(0.90, 0.88 * math.sqrt(k / 12.0))
    ang = k * GOLDEN
    return radius * math.cos(ang), radius * math.sin(ang)


def _W_local(tag: str, uu: np.ndarray, vv: np.ndarray, psi: dict, k: int) -> np.ndarray:
    """Localized basis. uu,vv are (chart - center) / sigma."""
    env = np.exp(-0.5 * (uu * uu + vv * vv))
    if tag == "exp":
        return env
    if tag == "gabor":
        ang = (k % 6) * (math.pi / 6.0)
        wave = 2.2 * (uu * math.cos(ang) + vv * math.sin(ang))
        return np.cos(wave) * env
    rad = np.sqrt(uu * uu + vv * vv) + 1e-4
    if tag == "logsin":
        return np.sin(3.0 * np.log(rad) + 0.4 * k) * env
    if tag == "logcos":
        return np.cos(3.0 * np.log(rad) + 0.4 * k) * env
    if tag == "rip":
        ang = (k % 6) * (math.pi / 6.0)
        wave = 2.5 * (uu * math.cos(ang) + vv * math.sin(ang))
        return np.sin(wave) * env
    raise ValueError(tag)


def _basis_stack(
    x: int,
    u: np.ndarray,
    v: np.ndarray,
    tags: list[str],
    psi: dict,
    ell: int = 0,
    r: float = 0.5,
) -> np.ndarray:
    """(x, H, W) basis on a full-frame chart. ell only changes blob size."""
    sigma0 = float(psi.get("sigma0", 0.75))
    sigma_ell = max(0.02, sigma0 * (r ** ell))
    out = []
    for k in range(x):
        tag = tags[k] if k < len(tags) else BASIS_DEFAULTS[k % len(BASIS_DEFAULTS)]
        if tag == "const":
            out.append(np.ones_like(u))
            continue
        if tag == "linu":
            out.append(u)
            continue
        if tag == "linv":
            out.append(v)
            continue
        if tag == "lin":
            a, b, c = psi.get("a", 1.0), psi.get("b", 0.0), psi.get("c", 0.0)
            out.append(a * u + b * v + c)
            continue
        if tag == "quad":
            out.append(u * u + v * v)
            continue
        du, dv = _offset(k, psi)
        # Slight size jitter so neighboring blobs aren't identical.
        sig = sigma_ell * (1.15 if k % 3 == 1 else 0.80 if k % 3 == 2 else 1.0)
        uu = (u - du) / sig
        vv = (v - dv) / sig
        out.append(_W_local(tag, uu, vv, psi, k))
    return np.stack(out, axis=0)


def _solve_B(
    A: np.ndarray,
    Y: np.ndarray,
    lam_l2: float = 1e-4,
    lam_l1: float = 0.0,
    ista_steps: int = 0,
) -> np.ndarray:
    """
    min mean((A B - Y)^2) + lam_l2 * mean(B^2) [+ lam_l1 * mean(|B|) via ISTA].
    A: (n, p), Y: (n, c) → B: (p, c).
    """
    n, p = A.shape
    ATA = A.T @ A
    ATY = A.T @ Y
    # d/dB mean((AB-Y)^2) = (2/(n c)) A^T E, but we solve per-column with
    # (ATA + n lam_l2 I) B = ATY  ↔  min (1/n)||AB-Y||^2 + lam_l2 ||B||^2.
    ridge = max(float(lam_l2), 1e-10)
    lhs = ATA + (n * ridge) * np.eye(p)
    try:
        B = np.linalg.solve(lhs, ATY)
    except np.linalg.LinAlgError:
        B = np.linalg.lstsq(A, Y, rcond=None)[0]

    if lam_l1 <= 0.0 or ista_steps <= 0:
        return B

    # Lipschitz of (2/n) ATA along each column.
    eig_max = float(np.linalg.eigvalsh(ATA)[-1]) if p > 1 else float(ATA[0, 0])
    L = 2.0 * eig_max / max(n, 1) + 2.0 * ridge
    lr = 1.0 / (L + 1e-12)
    # mean(|B|) over p*c entries → prox threshold lr * lam_l1 / (p*c) * (p*c) wait:
    # grad mean(|B|) = sign(B) / B.size, so thresh = lr * lam_l1 / B.size * B.size? No:
    # prox of α * mean(|x|) = prox of (α / m) ||x||_1 with m=x.size → t = α / m.
    # mean(|B|) per column (p entries) → prox t = lr * lam_l1 / p
    thresh = lr * lam_l1 / float(p)
    for _ in range(ista_steps):
        E = A @ B - Y
        B = B - lr * ((2.0 / n) * (A.T @ E) + 2.0 * ridge * B)
        B = np.sign(B) * np.maximum(np.abs(B) - thresh, 0.0)
    return B


@dataclass
class ScaleField:
    """
    x = surface complexity (basis neurons per channel per scale)
    i = number of scales
    3 = RGB
    params ≈ x * 3 * i
    """

    x: int = 3
    i: int = 4
    r: float = 0.5
    tags: list[str] = field(default_factory=lambda: ["exp", "lin", "const"])
    psi: dict = field(default_factory=dict)

    PRESETS = {
        "tiny": dict(x=3, i=4, r=0.5),  # 36
        "small": dict(x=8, i=6, r=0.55),  # 144
        "medium": dict(x=12, i=8, r=0.6),  # 288
        "textured": dict(  # 768, under 1k — log tags are generate-only
            x=16,
            i=16,
            r=0.5,
            tags=["exp", "gabor", "const", "lin", "exp", "gabor", "quad", "exp"],
        ),
    }

    @classmethod
    def preset(cls, name: str = "small") -> "ScaleField":
        if name not in cls.PRESETS:
            raise KeyError(f"unknown preset {name}; have {list(cls.PRESETS)}")
        return cls(**cls.PRESETS[name])

    def __post_init__(self):
        if self.x < 1 or self.i < 1:
            raise ValueError("x and i must be >= 1")
        if self.x * 3 * self.i > MAX_PARAMS:
            raise ValueError(f"x*3*i={self.x * 3 * self.i} exceeds cap {MAX_PARAMS}")
        self.tags = list(self.tags)
        self.psi = dict(self.psi)
        self.b = np.zeros((self.i, 3, self.x), dtype=float)
        self.bank: np.ndarray | None = None
        self._A_cache: dict[tuple, np.ndarray] = {}

    def nparams(self) -> int:
        return int(self.b.size)

    def formula(self) -> str:
        tags = ",".join(
            (self.tags[k] if k < len(self.tags) else BASIS_DEFAULTS[k % len(BASIS_DEFAULTS)])
            for k in range(self.x)
        )
        sparse = sparsity(self.b)
        return (
            f"C(u)=sum_ell W_{{{tags}}}((u-μ)/r^ell) · b[ell]  "
            f"| x={self.x} ch=3 i={self.i} r={self.r:g}  "
            f"N={self.nparams()} sparse={sparse:.2f}"
        )

    def copy(self) -> "ScaleField":
        sf = ScaleField(x=self.x, i=self.i, r=self.r, tags=list(self.tags), psi=dict(self.psi))
        sf.b = self.b.copy()
        if self.bank is not None:
            sf.bank = self.bank.copy()
        return sf

    def _arch_key(self) -> tuple:
        psi_items = tuple(sorted((str(k), float(v)) for k, v in self.psi.items()))
        return (self.x, self.i, float(self.r), tuple(self.tags), psi_items)

    def _chart(self, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
        # Full frame at every scale. Octaves change blob *size*, not the domain.
        ys = np.linspace(-1.0, 1.0, H)
        xs = np.linspace(-1.0, 1.0, W)
        v, u = np.meshgrid(ys, xs, indexing="ij")
        return u, v

    def design(self, H: int, W: int) -> np.ndarray:
        """Design matrix A, shape (H*W, i*x). Column ell*x+k is W_k at scale ell."""
        key = (H, W, self._arch_key())
        cached = self._A_cache.get(key)
        if cached is not None:
            return cached
        u, v = self._chart(H, W)
        cols = []
        for ell in range(self.i):
            basis = _basis_stack(self.x, u, v, self.tags, self.psi, ell=ell, r=self.r)
            cols.append(basis.reshape(self.x, H * W).T)
        A = np.concatenate(cols, axis=1)
        # Keep a handful of architectures around (r/psi search).
        if len(self._A_cache) > 12:
            self._A_cache.clear()
        self._A_cache[key] = A
        return A

    def field(self, H: int, W: int, b: np.ndarray | None = None) -> np.ndarray:
        """Return RGB map (H, W, 3) in float (unclipped)."""
        coeff = self.b if b is None else b
        A = self.design(H, W)
        B = pack_b(coeff)
        return (A @ B).reshape(H, W, 3)

    def raster(self, H: int = 256, W: int = 256, stretch: bool = True) -> np.ndarray:
        """Adapter: field → uint8 RGB. stretch=True is display-only, not the loss."""
        img = self.field(H, W)
        if stretch:
            lo, hi = np.percentile(img, 1), np.percentile(img, 99)
            if hi <= lo:
                hi = lo + 1e-6
            img = np.clip((img - lo) / (hi - lo), 0, 1)
        else:
            img = np.clip(img, 0.0, 1.0)
        return (img * 255.0 + 0.5).astype(np.uint8)

    def decode(self, H: int, W: int, stretch: bool = False) -> np.ndarray:
        return self.raster(H, W, stretch=stretch)

    def encode(
        self,
        rgb: np.ndarray,
        lam_l2: float = 1e-4,
        lam_l1: float = 1e-3,
        ista_steps: int = 12,
        reset: bool = True,
        **_ignored,
    ) -> str:
        """Compress one image into b. Field is linear in b → ridge (+ optional L1)."""
        tgt = to_float_rgb(rgb)
        H, W, _ = tgt.shape
        A = self.design(H, W)
        Y = tgt.reshape(H * W, 3)
        B = _solve_B(A, Y, lam_l2=lam_l2, lam_l1=lam_l1, ista_steps=ista_steps)
        self.b = unpack_b(B, self.i, self.x)
        return self.formula()

    def encode_many(
        self,
        images: list[np.ndarray],
        lam_l2: float = 1e-4,
        lam_l1: float = 1e-3,
        ista_steps: int = 8,
    ) -> np.ndarray:
        """Fit a b for every image. Returns bank shape (K, i, 3, x)."""
        if not images:
            raise ValueError("no images")
        imgs = [to_float_rgb(im) for im in images]
        H, W, _ = imgs[0].shape
        for im in imgs:
            if im.shape != (H, W, 3):
                raise ValueError("all images must share H,W,3")
        # Same A for every image (cached). Loop so L1 is per-image, not 3K-wide.
        bank = np.zeros((len(imgs), self.i, 3, self.x), dtype=float)
        for k, im in enumerate(imgs):
            self.encode(im, lam_l2=lam_l2, lam_l1=lam_l1, ista_steps=ista_steps)
            bank[k] = self.b
        self.bank = bank
        self.b = bank.mean(axis=0)
        return bank

    def reconstruct(self, rgb: np.ndarray, **encode_kw) -> tuple[np.ndarray, dict]:
        """Encode then field() at the source resolution. Returns pred, stats."""
        tgt = to_float_rgb(rgb)
        self.encode(tgt, **encode_kw)
        pred = self.field(*tgt.shape[:2])
        stats = {
            "mse": mse(pred, tgt),
            "fovea_mse": fovea_mse(pred, tgt),
            "sparsity": sparsity(self.b),
            "nparams": self.nparams(),
        }
        return pred, stats

    def evaluate(self, images: list[np.ndarray], **encode_kw) -> dict:
        """Per-image encode (codec test) + mean metrics. Sets bank and mean b."""
        imgs = [to_float_rgb(im) for im in images]
        bank = self.encode_many(imgs, **encode_kw)
        H, W = imgs[0].shape[:2]
        A = self.design(H, W)
        mses, foveas, spars = [], [], []
        for k, im in enumerate(imgs):
            pred = (A @ pack_b(bank[k])).reshape(H, W, 3)
            mses.append(mse(pred, im))
            foveas.append(fovea_mse(pred, im))
            spars.append(sparsity(bank[k]))
        self.b = bank.mean(axis=0)
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

    def train(
        self,
        images,
        steps: int = 8,
        lr: float = 0.12,
        lam: float = 1e-3,
        lam_l2: float = 1e-4,
    ) -> str:
        """
        Fit one b per image, store the bank, leave self.b as the mean field.
        `steps`/`lr` kept for CLI compatibility; the solve does not iterate pixels.
        """
        _ = steps, lr
        imgs = [to_float_rgb(im) for im in images]
        self.encode_many(imgs, lam_l2=lam_l2, lam_l1=lam, ista_steps=12)
        return self.formula()

    def generate(self, H: int = 512, W: int = 512, seed: int = 0, amp: float = 1.0) -> np.ndarray:
        """Sample a structured field (not pixel noise) and raster at any res."""
        rng = np.random.default_rng(seed)
        self.b[:] = 0.0
        for ell in range(self.i):
            scale_amp = amp / (1.0 + LN2 * ell)
            n_on = max(2, self.x // 3)
            for c in range(3):
                ks = rng.choice(self.x, size=n_on, replace=False)
                for k in ks:
                    self.b[ell, c, int(k)] = scale_amp * float(rng.normal(0.0, 0.55))
        return self.decode(H, W, stretch=True)

    def generate_from_cloud(
        self,
        H: int = 256,
        W: int = 256,
        seed: int = 0,
        mix: float = 0.15,
    ) -> np.ndarray:
        """Draw (or interpolate) a trained b-vector and raster it."""
        if self.bank is None or len(self.bank) == 0:
            return self.generate(H, W, seed=seed)
        rng = np.random.default_rng(seed)
        K = len(self.bank)
        i0, i1 = int(rng.integers(0, K)), int(rng.integers(0, K))
        t = float(rng.uniform(0.0, 1.0))
        self.b = (1.0 - t) * self.bank[i0] + t * self.bank[i1]
        self.b = self.b + mix * rng.normal(0.0, np.std(self.bank) + 1e-6, size=self.b.shape)
        return self.decode(H, W, stretch=False)

    def with_capacity(self, x: int | None = None, i: int | None = None) -> "ScaleField":
        x = self.x if x is None else int(x)
        i = self.i if i is None else int(i)
        if x * 3 * i > MAX_PARAMS:
            raise ValueError(f"x*3*i={x * 3 * i} exceeds cap {MAX_PARAMS}")
        sf = ScaleField(x=x, i=i, r=self.r, tags=list(self.tags), psi=dict(self.psi))
        i2, x2 = min(i, self.i), min(x, self.x)
        sf.b[:i2, :, :x2] = self.b[:i2, :, :x2]
        if self.bank is not None:
            K = self.bank.shape[0]
            bank = np.zeros((K, i, 3, x), dtype=float)
            bank[:, :i2, :, :x2] = self.bank[:, :i2, :, :x2]
            sf.bank = bank
        return sf

    def save(self, path: str, **extra) -> None:
        payload = {
            "b": self.b,
            "x": self.x,
            "i": self.i,
            "r": self.r,
            "tags": np.array(self.tags, dtype=object),
            "psi_json": json.dumps({str(k): float(v) for k, v in self.psi.items()}),
        }
        if self.bank is not None:
            payload["bank"] = self.bank
        for k, v in extra.items():
            payload[k] = v
        path = str(path)
        if not path.endswith(".npz"):
            path = path + ".npz"
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "ScaleField":
        z = np.load(path, allow_pickle=True)
        tags = list(z["tags"])
        psi = {}
        if "psi_json" in z.files:
            raw = z["psi_json"]
            s = raw.item() if getattr(raw, "shape", ()) == () else str(raw)
            psi = json.loads(str(s))
        sf = cls(x=int(z["x"]), i=int(z["i"]), r=float(z["r"]), tags=tags, psi=psi)
        sf.b = np.array(z["b"], dtype=float)
        if "bank" in z.files:
            sf.bank = np.array(z["bank"], dtype=float)
        return sf


def save_png(path: str, rgb: np.ndarray) -> None:
    from PIL import Image

    a = np.asarray(rgb)
    if a.dtype != np.uint8:
        a = to_float_rgb(a)
        a = (np.clip(a, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(a, mode="RGB").save(path)


def demo_blob(path: str = "artifacts/scalefield_demo.png") -> str:
    """Known tiny field → high-res raster. No photo needed."""
    sf = ScaleField(x=3, i=4, r=0.5)
    sf.b[0, 0, 0] = 1.4
    sf.b[0, 1, 0] = 0.3
    sf.b[0, 2, 0] = 0.2
    sf.b[0, 0, 1] = 0.2
    sf.b[1, 1, 0] = 0.9
    sf.b[2, 2, 0] = 0.7
    img = sf.raster(256, 256)
    save_png(path, img)
    return f"{sf.formula()}\nwrote {path} shape={img.shape} N={sf.nparams()}"


def demo_roundtrip(
    src: np.ndarray | None = None,
    path: str = "artifacts/scalefield_roundtrip.png",
) -> str:
    """Encode a synthetic target, decode at higher res."""
    H = W = 64
    if src is None:
        yy, xx = np.mgrid[0:H, 0:W]
        src = np.zeros((H, W, 3))
        src[:, :, 0] = np.exp(-((xx - 20) ** 2 + (yy - 24) ** 2) / 180)
        src[:, :, 1] = np.exp(-((xx - 40) ** 2 + (yy - 36) ** 2) / 220)
        src[:, :, 2] = 0.15 + 0.25 * xx / W
    sf = ScaleField(x=3, i=4, r=0.5)
    formula = sf.encode(src, lam_l1=0.0, ista_steps=0)
    out = sf.decode(256, 256, stretch=False)
    save_png(path, out)
    recon = sf.field(H, W)
    return f"{formula}\nmse@{H}={mse(src, recon):.4f} wrote {path} N={sf.nparams()} vs pixels={H * W * 3}"


def demo_generate(path: str = "artifacts/scalefield_small_512.png") -> str:
    sf = ScaleField.preset("small")
    img = sf.generate(512, 512, seed=7)
    save_png(path, img)
    return f"{sf.formula()} wrote {path} {img.shape}"


if __name__ == "__main__":
    from pathlib import Path

    Path("artifacts").mkdir(exist_ok=True)
    print(demo_blob())
    print(demo_roundtrip())
    print(demo_generate())
