"""Frozen 378-d cosine-band field for the wallpaper player. Do not train with this."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v1"))
sys.path.insert(0, str(ROOT / "gen"))
sys.path.insert(0, str(ROOT / "gen2"))

from scalefield import COLOR_PHASES, PrimeField, pixel_uv, foveal_map  # noqa: E402
from scalegen import DiagGMM, LatentPrior  # noqa: E402
from thin import PhaseUnfold  # noqa: E402

EL_PRIMES = (2, 3, 5, 7, 11, 13)
PRIMES_ALL = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
)


def axis_p(p: int) -> np.ndarray:
    small = [q for q in PRIMES_ALL if q < p]
    F = [1.0 / p] + [q / float(p) for q in small]
    need = p - len(F)
    climb = [float(p)]
    for q in PRIMES_ALL:
        x = float(p * q)
        if x <= p * p + 1e-12 and x not in climb:
            climb.append(x)
    climb = sorted(climb)
    if need <= 0:
        return np.asarray(F[:p], dtype=np.float64)
    if need == 1:
        pick = [float(p * p)]
    elif need >= len(climb):
        pick = list(climb)
        for q in PRIMES_ALL:
            if len(pick) >= need:
                break
            if p < q < p * p and float(q) not in pick:
                pick.append(float(q))
        pick = sorted(pick)[:need]
        while len(pick) < need:
            pick.append(float(p * p))
    else:
        idx = np.round(np.linspace(0, len(climb) - 1, need)).astype(int)
        seen: set[int] = set()
        pick = []
        for i in idx:
            i = int(i)
            if i not in seen:
                seen.add(i)
                pick.append(float(climb[i]))
        j = len(climb) - 1
        while len(pick) < need:
            if j not in seen:
                seen.add(j)
                pick.append(float(climb[j]))
            j -= 1
        pick = sorted(pick)
    return np.asarray(F + pick, dtype=np.float64)


def _uhv(pf: PrimeField, H: int, W: int):
    u, v = pixel_uv(H, W)
    u, v = foveal_map(u, v, pf.foveal)
    return (u + 1.0) * 0.5, (v + 1.0) * 0.5


def hue_of(p: int, idx: int) -> np.ndarray:
    if p <= 1:
        return np.ones(3, dtype=np.float64)
    return COLOR_PHASES[idx % 3]


def n_el() -> int:
    return 1 + sum(p * p for p in EL_PRIMES)


def offsets_el() -> list[tuple[int, int, int]]:
    out = [(1, 0, 1)]
    s = 1
    for p in EL_PRIMES:
        n = p * p
        out.append((p, s, n))
        s += n
    return out


class _Bag:
    def __init__(self, bank):
        self.bank = bank


class GeneratorEl:
    def __init__(self, pf, prior, gmm, unf, bank):
        self.pf = pf
        self.field = _Bag(bank)
        self.prior = prior
        self.gmm = gmm
        self.unf = unf

    @classmethod
    def load(cls, stem: str, codec: str | Path) -> "GeneratorEl":
        stem = Path(stem)
        pf = PrimeField.load(str(codec))
        bank = np.array(np.load(str(stem) + ".el.npz")["bank"], dtype=np.float64)
        prior = LatentPrior.load(str(stem) + ".prior.npz")
        gmm_path = Path(str(stem) + ".gmm.npz")
        gmm = DiagGMM.load(str(gmm_path)) if gmm_path.is_file() else None
        z = np.load(str(stem) + ".unf.npz")
        primes = tuple(int(p) for p in z["primes"])
        unf = PhaseUnfold.load_from(z, primes)
        unf.n_prefix = int(z["n_prefix"])
        return cls(pf, prior, gmm, unf, bank)


class LiveField:
    def __init__(self, pf: PrimeField, H: int, W: int):
        self.H, self.W = int(H), int(W)
        uh, vh = _uhv(pf, H, W)
        self.cu: dict[int, np.ndarray] = {}
        self.cv: dict[int, np.ndarray] = {}
        self.hue: dict[int, np.ndarray] = {}
        self.part: dict[int, np.ndarray] = {}
        self.owner = np.zeros(n_el(), dtype=np.int32)
        pidx = 0
        for p, start, n in offsets_el():
            self.owner[start : start + n] = p
            self.hue[p] = hue_of(p, pidx if p > 1 else 0)
            self.part[p] = np.zeros((self.H, self.W), dtype=np.float64)
            if p > 1:
                F = axis_p(p)
                self.cu[p] = np.cos(np.pi * F[:, None, None] * uh[None, :, :])
                self.cv[p] = np.cos(np.pi * F[:, None, None] * vh[None, :, :])
                pidx += 1
        self.amps = np.zeros(n_el(), dtype=np.float64)
        self.acc = np.zeros((self.H, self.W, 3), dtype=np.float64)

    def _rebuild_p(self, p: int) -> None:
        start = n = 0
        for q, s, nn in offsets_el():
            if q == p:
                start, n = s, nn
                break
        if p == 1:
            self.part[p][:, :] = float(self.amps[start])
            return
        coef = self.amps[start : start + n].reshape(p, p)
        inner = np.einsum("ij,jhw->ihw", coef, self.cu[p])
        np.einsum("ihw,ihw->hw", inner, self.cv[p], out=self.part[p])

    def _sum(self) -> None:
        self.acc.fill(0.0)
        for p, part in self.part.items():
            self.acc += part[:, :, None] * self.hue[p]

    def set_all(self, amps: np.ndarray) -> None:
        self.amps = np.asarray(amps, dtype=np.float64).reshape(-1).copy()
        for p, _s, _n in offsets_el():
            self._rebuild_p(p)
        self._sum()

    def set_amp(self, i: int, value: float) -> None:
        i = int(i)
        if self.amps[i] == value:
            return
        self.amps[i] = float(value)
        self._rebuild_p(int(self.owner[i]))
        self._sum()

    def rgb(self) -> np.ndarray:
        img = np.clip(self.acc * self.acc, 0.0, 1.0)
        return (img * 255.0 + 0.5).astype(np.uint8)
