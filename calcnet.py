"""
CalcNet — tiny neural devices that *are* calculus maps.

Each device is a map with an explicit Jacobian.
Parameter count is kept stupid-small; architecture does the work.

Not an LLM. An LLM is the same idea with 10^9 copies of matmul+softmax+norm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

# Optional numpy for the demo calculator
try:
    import numpy as np
except ImportError:
    np = None


# ---------- primitive maps (0–2 params) ----------

def affine_1d(w: float, b: float) -> Callable[[float], float]:
    return lambda x: w * x + b


def sigmoid(x: float) -> float:
    # stable
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def d_sigmoid(x: float) -> float:
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh_f(x: float) -> float:
    return math.tanh(x)


def d_tanh(x: float) -> float:
    t = math.tanh(x)
    return 1.0 - t * t


def relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def d_relu(x: float) -> float:
    return 1.0 if x > 0.0 else 0.0


# ---------- devices ----------

@dataclass
class Neuron1D:
    """Graph of σ(wx+b). A curve. Params: 2."""
    w: float
    b: float
    act: str = "tanh"

    def _sig(self):
        return {"tanh": (tanh_f, d_tanh), "sigmoid": (sigmoid, d_sigmoid), "relu": (relu, d_relu)}[self.act]

    def f(self, x: float) -> float:
        s, _ = self._sig()
        return s(self.w * x + self.b)

    def df(self, x: float) -> float:
        """Chain rule visible: σ'(wx+b) * w."""
        _, ds = self._sig()
        return ds(self.w * x + self.b) * self.w

    def nparams(self) -> int:
        return 2


@dataclass
class Compose:
    """(g ∘ f). Chain rule: Jg(f(x)) * Jf(x)."""
    f: Neuron1D
    g: Neuron1D

    def h(self, x: float) -> float:
        return self.g.f(self.f.f(x))

    def dh(self, x: float) -> float:
        y = self.f.f(x)
        return self.g.df(y) * self.f.df(x)

    def nparams(self) -> int:
        return self.f.nparams() + self.g.nparams()


@dataclass
class Sum2:
    """f+g. Product/sum rule playground."""
    f: Neuron1D
    g: Neuron1D

    def h(self, x: float) -> float:
        return self.f.f(x) + self.g.f(x)

    def dh(self, x: float) -> float:
        return self.f.df(x) + self.g.df(x)

    def nparams(self) -> int:
        return 4


@dataclass
class EulerStep:
    """ResNet as Euler: x + dt * v(x). Discrete ODE."""
    v: Neuron1D
    dt: float = 0.1

    def step(self, x: float) -> float:
        return x + self.dt * self.v.f(x)

    def dstep(self, x: float) -> float:
        return 1.0 + self.dt * self.v.df(x)

    def integrate(self, x0: float, n: int) -> list[float]:
        xs = [x0]
        x = x0
        for _ in range(n):
            x = self.step(x)
            xs.append(x)
        return xs

    def nparams(self) -> int:
        return 1 + self.v.nparams()


@dataclass
class FiniteDiff:
    """Differentiator device. 0 learned params. h is the chart scale."""
    h: float = 1e-3

    def D(self, f: Callable[[float], float], x: float) -> float:
        return (f(x + self.h) - f(x - self.h)) / (2.0 * self.h)


@dataclass
class Trapezoid:
    """Integrator device. 0 learned params."""

    def I(self, f: Callable[[float], float], a: float, b: float, n: int = 64) -> float:
        if n < 1:
            return 0.0
        h = (b - a) / n
        s = 0.5 * f(a) + 0.5 * f(b)
        for i in range(1, n):
            s += f(a + i * h)
        return s * h


@dataclass
class Product:
    """(f*g). Product rule: f'g + fg'."""
    f: Neuron1D
    g: Neuron1D

    def h(self, x: float) -> float:
        return self.f.f(x) * self.g.f(x)

    def dh(self, x: float) -> float:
        return self.f.df(x) * self.g.f(x) + self.f.f(x) * self.g.df(x)

    def nparams(self) -> int:
        return 4


class Expr:
    """
    Calculus graph. Leaves are Neuron1D or the identity x.
    Nodes: add, mul, div, pow, comp, scale.
    eval / deriv walk the same tree. formula() prints the map.
    """

    def __init__(self, kind: str, *kids, c: float = 1.0):
        self.kind = kind
        self.kids = kids
        self.c = c

    @staticmethod
    def leaf(n: Neuron1D) -> "Expr":
        return Expr("leaf", n)

    @staticmethod
    def x() -> "Expr":
        return Expr("id")

    def __add__(self, other: "Expr") -> "Expr":
        return Expr("add", self, other)

    def __sub__(self, other: "Expr") -> "Expr":
        return self + other.scale(-1.0)

    def __mul__(self, other: "Expr") -> "Expr":
        return Expr("mul", self, other)

    def __truediv__(self, other: "Expr") -> "Expr":
        return Expr("div", self, other)

    def __pow__(self, p: float) -> "Expr":
        return Expr("pow", self, c=float(p))

    def compose(self, inner: "Expr") -> "Expr":
        return Expr("comp", self, inner)

    def scale(self, c: float) -> "Expr":
        return Expr("scale", self, c=c)

    def eval(self, x: float) -> float:
        k = self.kind
        if k == "id":
            return x
        if k == "leaf":
            return self.kids[0].f(x)
        if k == "add":
            return self.kids[0].eval(x) + self.kids[1].eval(x)
        if k == "mul":
            return self.kids[0].eval(x) * self.kids[1].eval(x)
        if k == "div":
            return self.kids[0].eval(x) / self.kids[1].eval(x)
        if k == "pow":
            return self.kids[0].eval(x) ** self.c
        if k == "comp":
            return self.kids[0].eval(self.kids[1].eval(x))
        if k == "scale":
            return self.c * self.kids[0].eval(x)
        raise ValueError(k)

    def deriv(self, x: float) -> float:
        k = self.kind
        if k == "id":
            return 1.0
        if k == "leaf":
            return self.kids[0].df(x)
        if k == "add":
            return self.kids[0].deriv(x) + self.kids[1].deriv(x)
        if k == "mul":
            a, b = self.kids
            return a.deriv(x) * b.eval(x) + a.eval(x) * b.deriv(x)
        if k == "div":
            a, b = self.kids
            num = a.deriv(x) * b.eval(x) - a.eval(x) * b.deriv(x)
            den = b.eval(x) ** 2
            return num / den
        if k == "pow":
            u = self.kids[0].eval(x)
            p = self.c
            if u == 0.0 and p <= 1:
                return 0.0
            return p * (u ** (p - 1.0)) * self.kids[0].deriv(x)
        if k == "comp":
            outer, inner = self.kids
            y = inner.eval(x)
            return outer.deriv(y) * inner.deriv(x)
        if k == "scale":
            return self.c * self.kids[0].deriv(x)
        raise ValueError(k)

    def leaves(self) -> list:
        if self.kind == "leaf":
            return [self.kids[0]]
        out = []
        for k in self.kids:
            if isinstance(k, Expr):
                out.extend(k.leaves())
        return out

    def nparams(self) -> int:
        return 2 * len(self.leaves())

    def formula(self) -> str:
        k = self.kind
        if k == "id":
            return "x"
        if k == "leaf":
            n = self.kids[0]
            return f"{n.act}({n.w:.4g}*x{n.b:+.4g})"
        if k == "add":
            return f"({self.kids[0].formula()} + {self.kids[1].formula()})"
        if k == "mul":
            return f"({self.kids[0].formula()} * {self.kids[1].formula()})"
        if k == "div":
            return f"({self.kids[0].formula()} / {self.kids[1].formula()})"
        if k == "pow":
            return f"({self.kids[0].formula()})^{self.c:g}"
        if k == "comp":
            return f"({self.kids[0].formula()} ∘ {self.kids[1].formula()})"
        if k == "scale":
            return f"({self.c:g}*{self.kids[0].formula()})"
        return k


MAX_PARAMS = 1000


class WeirdTrainer:
    """
    Fit only leaf (w,b). Tree shape is the math.
    Score = MSE + lam * nparams. Finite-diff SGD. Cap 1000 params.
    """

    def __init__(self, expr: Expr, lr: float = 0.05, lam: float = 1e-4):
        if expr.nparams() > MAX_PARAMS:
            raise ValueError(f"refusing {expr.nparams()} params; cap is {MAX_PARAMS}")
        self.expr = expr
        self.lr = lr
        self.lam = lam

    def mse(self, xs, ys) -> float:
        e = 0.0
        for x, y in zip(xs, ys):
            d = self.expr.eval(x) - y
            e += d * d
        return e / max(len(xs), 1)

    def step(self, xs, ys) -> float:
        leaves = self.expr.leaves()
        loss0 = self.mse(xs, ys) + self.lam * self.expr.nparams()
        h = 1e-4
        for n in leaves:
            for attr in ("w", "b"):
                old = getattr(n, attr)
                setattr(n, attr, old + h)
                lp = self.mse(xs, ys)
                setattr(n, attr, old - h)
                lm = self.mse(xs, ys)
                setattr(n, attr, old)
                g = (lp - lm) / (2.0 * h)
                setattr(n, attr, old - self.lr * g)
        return loss0

    def fit(self, xs, ys, steps: int = 200) -> str:
        for _ in range(steps):
            self.step(xs, ys)
        return instrument(self.expr)


def layer_map(W, b, act: Callable):
    """One hidden layer: R^n -> R^m. Graph in R^{n+m}."""
    if np is None:
        raise RuntimeError("numpy required for layer_map")
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)

    def f(x):
        x = np.asarray(x, dtype=float)
        return act(W @ x + b)

    def J(x):
        x = np.asarray(x, dtype=float)
        z = W @ x + b
        # diag(σ') W
        if act is np.tanh:
            dp = 1.0 - np.tanh(z) ** 2
        else:
            raise ValueError("demo J only wired for tanh")
        return (dp[:, None]) * W

    return f, J


def instrument(expr: Expr, x: float | None = None) -> str:
    """The tool outputs the calculation. Optional x adds h(x) and h'."""
    line = expr.formula()
    if x is None:
        return line
    return f"{line}  |  h({x:g})={expr.eval(x):.6g}  h'={expr.deriv(x):.6g}"


# ---------- calculator ----------

def demo():
    print("=== CalcNet demo ===")
    n = Neuron1D(w=1.2, b=-0.3, act="tanh")
    print(f"neuron params={n.nparams()}  f(0)={n.f(0):.4f}  f'(0)={n.df(0):.4f}")

    c = Compose(Neuron1D(0.8, 0.1), Neuron1D(-1.1, 0.0))
    print(f"compose params={c.nparams()}  h(0.5)={c.h(0.5):.4f}  h'={c.dh(0.5):.4f}")

    fd = FiniteDiff(h=1e-4)
    err = abs(fd.D(n.f, 0.5) - n.df(0.5))
    print(f"finite-diff vs analytic f' err={err:.2e}")

    tr = Trapezoid()
    # d/dx tanh = 1-tanh^2; integral of n.df should recover n.f up to const
    rec = n.f(0.0) + tr.I(n.df, 0.0, 0.8, n=80)
    print(f"∫ f' ≈ {rec:.4f}  f(0.8)={n.f(0.8):.4f}")

    ode = EulerStep(Neuron1D(w=-0.5, b=0.0, act="tanh"), dt=0.2)
    path = ode.integrate(1.0, 8)
    print("euler path:", [round(v, 4) for v in path], f"params={ode.nparams()}")

    # chained equation: h = 0.5 * f * (g ∘ f)   +   f
    f = Expr.leaf(Neuron1D(1.1, -0.2))
    g = Expr.leaf(Neuron1D(-0.7, 0.15))
    eq = (f * g.compose(f)).scale(0.5) + f
    x = 0.4
    fd = FiniteDiff(h=1e-5)
    print(
        f"expr h({x})={eq.eval(x):.4f}  h'={eq.deriv(x):.4f}  "
        f"fd={fd.D(eq.eval, x):.4f}  params={eq.nparams()}"
    )
    print(instrument(eq, x))

    # weird trainer: recover a 2-param tanh from samples
    target = Neuron1D(w=0.9, b=-0.25, act="tanh")
    xs = [i * 0.2 - 1.0 for i in range(11)]
    ys = [target.f(z) for z in xs]
    model = Expr.leaf(Neuron1D(w=0.1, b=0.0, act="tanh"))
    trn = WeirdTrainer(model, lr=0.4, lam=0.0)
    calc = trn.fit(xs, ys, steps=80)
    print("pulled calculation:", calc)
    print("mse", trn.mse(xs, ys))


if __name__ == "__main__":
    demo()
