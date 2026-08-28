"""iGPU raster for PrimeField. Intel HD 530 via Mesa EGL + OpenGL 4.6.

DCT frequency grids + Laplacian edge boost + consecutive-prime chords.
"""

from __future__ import annotations

import atexit

import numpy as np

_CTX = None
_PROG_CACHE: dict = {}
_ERR: str | None = None
_MAXN = 20


def gpu_available() -> bool:
    try:
        _ctx()
        return True
    except Exception:
        return False


def gpu_info() -> str:
    try:
        ctx = _ctx()
        info = ctx.info
        return f"{info.get('GL_RENDERER')}  GL {ctx.version_code}"
    except Exception as e:
        return f"unavailable: {e}"


def _ctx():
    global _CTX, _ERR
    if _CTX is not None:
        return _CTX
    if _ERR is not None:
        raise RuntimeError(_ERR)
    try:
        import moderngl

        ctx = moderngl.create_context(standalone=True, backend="egl", require=330)
        _CTX = ctx
        atexit.register(_release)
        return ctx
    except Exception as e:
        _ERR = str(e)
        raise


def _release():
    global _CTX, _PROG_CACHE
    _PROG_CACHE.clear()
    if _CTX is not None:
        try:
            _CTX.release()
        except Exception:
            pass
        _CTX = None


_VERT = """
#version 330
out vec2 v_uv;
void main() {
    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
    v_uv = vec2(gl_Position.x, -gl_Position.y);
}
"""


def _frag_src(n: int) -> str:
    samplers = "\n".join(f"uniform sampler2D grid{i};" for i in range(n))
    dct_calls = "\n    ".join(
        f"ss[{i}] = dct_scalar(grid{i}, primes[{i}], uv);" for i in range(n)
    )
    rgb_adds = "\n    ".join(
        f"c += ss[{i}] * hues[{i}];" for i in range(n)
    )
    dc_subs = "\n    ".join(
        f"ac[{i}] = ss[{i}] - texelFetch(grid{i}, ivec2(0, 0), 0).r;" for i in range(n)
    )
    return f"""
#version 330
{samplers}
uniform int primes[{n}];
uniform vec3 hues[{n}];
uniform float foveal_a;
uniform float edge_gain;
uniform float chord_gain;
in vec2 v_uv;
out vec3 f_color;

const float PI = 3.141592653589793;

vec2 foveal(vec2 uv) {{
    if (foveal_a <= 1e-8) return uv;
    float ta = tanh(foveal_a);
    return tanh(foveal_a * uv) / ta;
}}

float dct_scalar(sampler2D tex, int p, vec2 uv) {{
    vec2 t = uv * 0.5 + 0.5;
    // Keep edge_gain live for the p=1 program (compiler would strip the uniform).
    if (p <= 1) return texelFetch(tex, ivec2(0, 0), 0).r * (1.0 + 0.0 * edge_gain);
    float denom = float((p - 1) * (p - 1));
    if (denom < 1.0) denom = 1.0;
    float s = 0.0;
    for (int l = 0; l < p; l++) {{
        float cv = cos(PI * float(l) * t.y);
        for (int k = 0; k < p; k++) {{
            float cu = cos(PI * float(k) * t.x);
            float w = 1.0 + edge_gain * float(k * k + l * l) / denom;
            s += texelFetch(tex, ivec2(k, l), 0).r * w * cu * cv;
        }}
    }}
    return s;
}}

void main() {{
    vec2 uv = foveal(v_uv);
    float ss[{n}];
    float ac[{n}];
    {dct_calls}
    vec3 c = vec3(0.0);
    {rgb_adds}
    {dc_subs}
    float ch = 0.0;
    for (int i = 0; i < {n} - 1; i++) ch += ac[i] * ac[i + 1];
    c += chord_gain * vec3(ch);
    f_color = c;
}}
"""


def _program(n: int):
    ctx = _ctx()
    prog = _PROG_CACHE.get(n)
    if prog is None:
        prog = ctx.program(vertex_shader=_VERT, fragment_shader=_frag_src(n))
        _PROG_CACHE[n] = prog
    return prog


def raster_gpu(field, H: int, W: int) -> np.ndarray:
    """Return float RGB (H, W, 3), unclipped — same contract as field.field()."""
    import moderngl

    ctx = _ctx()
    primes = field.primes
    n = len(primes)
    if n > _MAXN:
        raise ValueError(f"too many primes for GPU shader ({n} > {_MAXN})")
    prog = _program(n)
    vao = ctx.vertex_array(prog, [])
    textures = []
    hue_list = []
    for i, p in enumerate(primes):
        g = np.ascontiguousarray(field.grid(p), dtype=np.float32)
        tex = ctx.texture((p, p), 1, data=g.tobytes(), dtype="f4")
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tex.repeat_x = False
        tex.repeat_y = False
        tex.use(i)
        textures.append(tex)
        prog[f"grid{i}"].value = i
        hue_list.append(tuple(float(x) for x in field._hues[i][0, 0]))
    pvals = tuple(int(p) for p in primes)
    def _set(name, value):
        try:
            prog[name].value = value
        except KeyError:
            pass

    _set("primes", pvals[0] if n == 1 else pvals)
    _set("hues", hue_list[0] if n == 1 else hue_list)
    _set("foveal_a", float(field.foveal))
    _set("edge_gain", float(field.edge_gain))
    _set("chord_gain", float(field.chord_gain))

    fbo = ctx.framebuffer(color_attachments=[ctx.texture((W, H), 3, dtype="f4")])
    fbo.use()
    ctx.viewport = (0, 0, W, H)
    ctx.clear(0.0, 0.0, 0.0)
    vao.render(mode=moderngl.TRIANGLES, vertices=3)
    raw = fbo.read(components=3, dtype="f4", alignment=1)
    img = np.frombuffer(raw, dtype=np.float32).reshape(H, W, 3).astype(np.float64)
    img = img[::-1].copy()

    fbo.release()
    vao.release()
    for t in textures:
        t.release()
    return img
