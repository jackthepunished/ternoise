# Phase 2 (M2) Implementation Plan — Host Renderer & Dataset Generation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A C++/OpenGL 4.3 compute-shader path tracer that renders a Cornell-box scene at 1 spp with G-buffers (world normal, linear depth), accumulates a high-spp reference, quantizes both to INT8 per QUANT_SPEC section 2, and batch-dumps paired training data.

**Architecture:** One GLFW window app (`host/`). A compute shader path-traces into an RGBA32F accumulation image and writes G-buffer images from the deterministic primary hit; a fullscreen-triangle blit displays whichever buffer is selected. Batch mode hides the window, loops over jittered camera views, dumps `noisy` (1 spp, 7ch), `ref` (N spp, 3ch) raw int8 files plus PNG previews. GPU selection: `GALLIUM_DRIVER=d3d12` forced by a launcher script (probed 2026-08-07: default context is llvmpipe; with the override it is GL 4.6 on real hardware).

**Tech Stack:** C++17, GLFW3 (apt), OpenGL 4.3 core (functions loaded via `glfwGetProcAddress`, no GLEW/glad), vendored `stb_image_write.h` for PNG previews. No other deps.

## Global Constraints

- INT8 encodings exactly per QUANT_SPEC section 2: color `clamp(round(c*255)-128)`, normal `clamp(round(n*127))`, depth `clamp(round(d*255)-128)`; channel order `[R,G,B,Nx,Ny,Nz,D]`.
- Radiance is clamped to [0,1] before color quantization (HDR clipping accepted for v1, documented in the dump metadata).
- Depth: `d = clamp(t_hit / 20.0, 0, 1)`; ray miss → `d = 1`, normal = (0,0,0).
- Primary rays are deterministic (pixel center, no AA jitter): G-buffers are noise-free and identical between noisy and reference passes of a view.
- Default frame size 256×256 (`--size` overrides; must be multiple of 8 = the compute local size).
- Reference accumulation default 1024 spp (`--ref-spp` overrides).
- `data/` stays gitignored; only code and the vendored header are committed.
- No emojis in repo content. C++ flags as Phase 1: `-std=c++17 -O2 -Wall -Wextra -Werror` (stb header compiled in its own TU with warnings relaxed).
- RNG: PCG2D hash, seeded from (pixel, frame index, view index) — batch output is deterministic for a given seed set.

## File Structure

```
host/main.cpp             app: modes, GL/GLFW setup, render loop, dump logic
host/gl_loader.h/.cpp     X-macro loader for the GL 4.3 functions we use
host/shader_util.h/.cpp   compile/link + file slurp helpers
host/quantize.h/.cpp      float buffers -> int8 per QUANT_SPEC
host/stb_image_write.h    vendored (public domain)
host/shaders/pathtrace.comp
host/shaders/blit.vert, blit.frag
scripts/run_host.sh       GALLIUM_DRIVER=d3d12 exec ./host_app "$@"
tests/test_dump.py        drives a tiny headless dump, validates files
Makefile                  new host_app / quantize test targets, ctest extension
```

---

### Task 1: GL bootstrap — window, 4.3 core context, loader

**Files:**
- Create: `host/gl_loader.h`, `host/gl_loader.cpp`, `host/main.cpp` (skeleton), `scripts/run_host.sh`
- Modify: `Makefile`

**Interfaces:**
- Produces: `host_app` binary that opens a window (or hidden window with `--hidden`), prints `GL_VERSION` / `GL_RENDERER`, clears the screen, exits cleanly on ESC or `--frames N`. `gl::load(glfwGetProcAddress)` must be called after context creation; all GL 4.3 entry points used later live in namespace `gl`.

- [ ] **Step 1: Write `host/gl_loader.h` — X-macro over every GL function the project uses**

```cpp
#pragma once
#include <GL/glcorearb.h>   // shipped by libgl1-mesa-dev; types + enums only

// Every modern GL function we call, loaded by name at runtime.
#define TERNOISE_GL_FUNCS(X) \
    X(PFNGLCREATESHADERPROC,            CreateShader) \
    X(PFNGLSHADERSOURCEPROC,            ShaderSource) \
    X(PFNGLCOMPILESHADERPROC,           CompileShader) \
    X(PFNGLGETSHADERIVPROC,             GetShaderiv) \
    X(PFNGLGETSHADERINFOLOGPROC,        GetShaderInfoLog) \
    X(PFNGLCREATEPROGRAMPROC,           CreateProgram) \
    X(PFNGLATTACHSHADERPROC,            AttachShader) \
    X(PFNGLLINKPROGRAMPROC,             LinkProgram) \
    X(PFNGLGETPROGRAMIVPROC,            GetProgramiv) \
    X(PFNGLGETPROGRAMINFOLOGPROC,       GetProgramInfoLog) \
    X(PFNGLDELETESHADERPROC,            DeleteShader) \
    X(PFNGLUSEPROGRAMPROC,              UseProgram) \
    X(PFNGLGETUNIFORMLOCATIONPROC,      GetUniformLocation) \
    X(PFNGLUNIFORM1IPROC,               Uniform1i) \
    X(PFNGLUNIFORM1UIPROC,              Uniform1ui) \
    X(PFNGLUNIFORM1FPROC,               Uniform1f) \
    X(PFNGLUNIFORM3FPROC,               Uniform3f) \
    X(PFNGLGENTEXTURESPROC,             GenTextures) \
    X(PFNGLBINDTEXTUREPROC,             BindTexture) \
    X(PFNGLTEXSTORAGE2DPROC,            TexStorage2D) \
    X(PFNGLTEXPARAMETERIPROC,           TexParameteri) \
    X(PFNGLBINDIMAGETEXTUREPROC,        BindImageTexture) \
    X(PFNGLDISPATCHCOMPUTEPROC,         DispatchCompute) \
    X(PFNGLMEMORYBARRIERPROC,           MemoryBarrier) \
    X(PFNGLGENVERTEXARRAYSPROC,         GenVertexArrays) \
    X(PFNGLBINDVERTEXARRAYPROC,         BindVertexArray) \
    X(PFNGLDRAWARRAYSPROC,              DrawArrays) \
    X(PFNGLGETTEXIMAGEPROC,             GetTexImage) \
    X(PFNGLVIEWPORTPROC,                Viewport) \
    X(PFNGLCLEARCOLORPROC,              ClearColor) \
    X(PFNGLCLEARPROC,                   Clear) \
    X(PFNGLGETSTRINGPROC,               GetString) \
    X(PFNGLFINISHPROC,                  Finish)

namespace gl {
#define TERNOISE_DECL(type, name) extern type name;
TERNOISE_GL_FUNCS(TERNOISE_DECL)
#undef TERNOISE_DECL
// loader: pass glfwGetProcAddress; aborts with the symbol name if any lookup fails
void load(void* (*get_proc)(const char*));
}
```

- [ ] **Step 2: Write `host/gl_loader.cpp`**

```cpp
#include "gl_loader.h"
#include <cstdio>
#include <cstdlib>

namespace gl {
#define TERNOISE_DEF(type, name) type name = nullptr;
TERNOISE_GL_FUNCS(TERNOISE_DEF)
#undef TERNOISE_DEF

void load(void* (*get_proc)(const char*)) {
#define TERNOISE_LOAD(type, name) \
    name = (type)get_proc("gl" #name); \
    if (!name) { std::fprintf(stderr, "missing GL symbol gl" #name "\n"); std::exit(2); }
TERNOISE_GL_FUNCS(TERNOISE_LOAD)
#undef TERNOISE_LOAD
}
}  // namespace gl
```

(Note: `glfwGetProcAddress` has signature `GLFWglproc(const char*)`; pass it through a lambda/cast adapter in main.cpp.)

- [ ] **Step 3: Write `host/main.cpp` skeleton**

Behavior contract (args parsed with plain `strcmp` loop, no getopt):
- `--size N` (default 256, must be %8==0 else exit 2 with message), `--frames N` (0 = until ESC), `--hidden` (GLFW_VISIBLE false).
- GLFW window hints: context 4,3, `GLFW_OPENGL_CORE_PROFILE`, no resize.
- After `glfwMakeContextCurrent`: `gl::load`, print `GL_VERSION` and `GL_RENDERER` to stdout.
- Loop: poll events, `gl::ClearColor(0.1f, 0.1f, 0.12f, 1)`, clear, swap; exit after `--frames` iterations or ESC/window close.

- [ ] **Step 4: Write `scripts/run_host.sh`**

```bash
#!/bin/sh
# WSLg's default GL context is llvmpipe (software). Force the D3D12 GPU path.
cd "$(dirname "$0")/.." || exit 1
GALLIUM_DRIVER=d3d12 exec ./host_app "$@"
```

`chmod +x scripts/run_host.sh`

- [ ] **Step 5: Makefile target**

```make
HOSTSRC = host/main.cpp host/gl_loader.cpp
host_app: $(HOSTSRC) host/gl_loader.h
	$(CXX) $(CXXFLAGS) $(HOSTSRC) -o host_app -lglfw -lGL
```

- [ ] **Step 6: Verify**

Run: `make host_app && ./scripts/run_host.sh --hidden --frames 3`
Expected: prints a `GL_VERSION` line containing `4.` and `GL_RENDERER` containing `D3D12`, exits 0. Then run once without `--hidden --frames` on the desktop to see the window.

- [ ] **Step 7: Commit** — `git add host scripts Makefile && git commit -m "feat(host): GLFW + GL 4.3 core bootstrap with minimal loader"`

---

### Task 2: Shader utilities + compute "hello" + blit

**Files:**
- Create: `host/shader_util.h`, `host/shader_util.cpp`, `host/shaders/blit.vert`, `host/shaders/blit.frag`, `host/shaders/hello.comp` (temporary, deleted in Task 3)
- Modify: `host/main.cpp`, `Makefile` (add shader_util.cpp)

**Interfaces:**
- Produces: `GLuint make_program_graphics(const char* vs_path, const char* fs_path)` and `GLuint make_program_compute(const char* cs_path)` — slurp files, compile, link; on any error print the full info log and exit 2. Fullscreen-triangle blit pipeline: empty VAO + `DrawArrays(GL_TRIANGLES, 0, 3)`, sampler uniform `tex`, uniform `int display_mode` (0 = tonemapped color, 1 = raw). Texture unit conventions fixed here and reused for the rest of M2: image unit 0 = accum RGBA32F, 1 = normal RGBA32F, 2 = depth R32F.

- [ ] **Step 1: `host/shaders/blit.vert`** (fullscreen triangle via gl_VertexID)

```glsl
#version 430 core
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
```

- [ ] **Step 2: `host/shaders/blit.frag`**

```glsl
#version 430 core
in vec2 uv;
out vec4 frag;
uniform sampler2D tex;
uniform int display_mode;   // 0 tonemap (pow 1/2.2), 1 raw
void main() {
    vec3 c = texture(tex, uv).rgb;
    if (display_mode == 0) c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
    frag = vec4(c, 1.0);
}
```

- [ ] **Step 3: `hello.comp`** — writes a uv gradient with `imageStore` to image unit 0, local size 8x8. Wire into main.cpp: create the three textures (`TexStorage2D`: RGBA32F, RGBA32F, R32F at size N), dispatch hello over (N/8, N/8), `MemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT)`, blit accum texture.

- [ ] **Step 4: Verify** — `make host_app && ./scripts/run_host.sh` shows a smooth red/green gradient; `--hidden --frames 3` exits 0.

- [ ] **Step 5: Commit** — `"feat(host): shader utils, compute dispatch, fullscreen blit"`

---

### Task 3: The path tracer

**Files:**
- Create: `host/shaders/pathtrace.comp`
- Delete: `host/shaders/hello.comp`
- Modify: `host/main.cpp`

**Interfaces:**
- Produces: compute shader with uniforms `uint frame_index`, `uint view_seed`, `vec3 cam_pos`, `vec3 cam_target`; accumulates linear radiance into image 0 (`accum += sample`, so the display/dump divides by `frame_index+1`), writes normal to image 1 and depth to image 2 on every dispatch (deterministic — same value every frame of a view). Keys in interactive mode: `1`/`2`/`3` select accum/normal/depth view, `R` resets accumulation (clears accum to 0, frame_index to 0). Display divides accum by frame count via a `uniform float inv_spp` on the blit (add it to blit.frag: `c *= inv_spp;` before tonemap when displaying accum).

- [ ] **Step 1: Write `host/shaders/pathtrace.comp` with exactly this scene and logic**

```glsl
#version 430 core
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, binding = 0) uniform image2D accum;
layout(rgba32f, binding = 1) uniform image2D gnormal;
layout(r32f,    binding = 2) uniform image2D gdepth;
uniform uint frame_index;
uniform uint view_seed;
uniform vec3 cam_pos;
uniform vec3 cam_target;

const float T_MAX = 20.0;   // QUANT_SPEC depth normalization bound
const int   BOUNCES = 4;

// ---- RNG: pcg2d ----
uvec2 pcg2d(uvec2 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u; v.y += v.x * 1664525u;
    v ^= v >> 16u;
    v.x += v.y * 1664525u; v.y += v.x * 1664525u;
    v ^= v >> 16u;
    return v;
}
uvec2 rng_state;
float rnd() {   // advance state, return [0,1)
    rng_state = pcg2d(rng_state);
    return float(rng_state.x) * (1.0 / 4294967296.0);
}

// ---- Scene: Cornell box, 2 spheres, ceiling area light ----
// Walls as 5 giant spheres would distort; use analytic slabs instead:
// box interior [-2,2]x[0,4]x[-2,2]; camera looks down -z from z=+6.5 side.
struct Hit { float t; vec3 n; vec3 albedo; vec3 emit; };

bool hit_plane(vec3 ro, vec3 rd, vec3 p0, vec3 n, inout Hit h, vec3 albedo) {
    float d = dot(rd, n);
    if (d > -1e-6) return false;               // one-sided, facing interior
    float t = dot(p0 - ro, n) / d;
    if (t < 1e-4 || t > h.t) return false;
    vec3 hp = ro + t * rd;
    if (abs(hp.x) > 2.001 || hp.y < -0.001 || hp.y > 4.001 || abs(hp.z) > 2.001) return false;
    h.t = t; h.n = n; h.albedo = albedo; h.emit = vec3(0);
    return true;
}
bool hit_sphere(vec3 ro, vec3 rd, vec3 c, float r, inout Hit h, vec3 albedo, vec3 emit) {
    vec3 oc = ro - c;
    float b = dot(oc, rd), cc = dot(oc, oc) - r * r;
    float disc = b * b - cc;
    if (disc < 0.0) return false;
    float t = -b - sqrt(disc);
    if (t < 1e-4 || t > h.t) return false;
    h.t = t; h.n = normalize(ro + t * rd - c); h.albedo = albedo; h.emit = emit;
    return true;
}
bool hit_light(vec3 ro, vec3 rd, inout Hit h) {  // ceiling quad light |x|,|z| <= 0.9 at y=3.999
    float d = rd.y;
    if (d < 1e-6) return false;                  // must travel upward
    float t = (3.999 - ro.y) / d;
    if (t < 1e-4 || t > h.t) return false;
    vec3 hp = ro + t * rd;
    if (abs(hp.x) > 0.9 || abs(hp.z) > 0.9) return false;
    h.t = t; h.n = vec3(0, -1, 0); h.albedo = vec3(0); h.emit = vec3(12.0);
    return true;
}
bool intersect(vec3 ro, vec3 rd, out Hit h) {
    h.t = T_MAX; h.n = vec3(0); h.albedo = vec3(0); h.emit = vec3(0);
    bool any = false;
    any = hit_plane(ro, rd, vec3(0, 0, 0), vec3(0, 1, 0),  h, vec3(0.73)) || any;  // floor
    any = hit_plane(ro, rd, vec3(0, 4, 0), vec3(0, -1, 0), h, vec3(0.73)) || any;  // ceiling
    any = hit_plane(ro, rd, vec3(0, 0, -2), vec3(0, 0, 1), h, vec3(0.73)) || any;  // back
    any = hit_plane(ro, rd, vec3(-2, 0, 0), vec3(1, 0, 0), h, vec3(0.65, 0.05, 0.05)) || any; // left red
    any = hit_plane(ro, rd, vec3(2, 0, 0), vec3(-1, 0, 0), h, vec3(0.12, 0.45, 0.15)) || any; // right green
    any = hit_sphere(ro, rd, vec3(-0.75, 0.8, -0.6), 0.8, h, vec3(0.85), vec3(0)) || any;
    any = hit_sphere(ro, rd, vec3(0.9, 0.5, 0.5),   0.5, h, vec3(0.7, 0.7, 0.95), vec3(0)) || any;
    any = hit_light(ro, rd, h) || any;
    return any;
}

vec3 cosine_hemisphere(vec3 n) {
    float u1 = rnd(), u2 = rnd();
    float r = sqrt(u1), phi = 6.2831853 * u2;
    vec3 t = normalize(abs(n.x) > 0.5 ? cross(n, vec3(0, 1, 0)) : cross(n, vec3(1, 0, 0)));
    vec3 b = cross(n, t);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
}

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = imageSize(accum);
    if (px.x >= res.x || px.y >= res.y) return;
    rng_state = uvec2(uint(px.x + px.y * res.x) ^ view_seed, frame_index * 2654435761u + 1u);

    // deterministic primary ray: pixel center, pinhole, vertical fov 40 deg
    vec3 fwd = normalize(cam_target - cam_pos);
    vec3 right = normalize(cross(fwd, vec3(0, 1, 0)));
    vec3 up = cross(right, fwd);
    vec2 ndc = (vec2(px) + 0.5) / vec2(res) * 2.0 - 1.0;
    float half_h = tan(radians(20.0));
    vec3 rd = normalize(fwd + right * ndc.x * half_h + up * ndc.y * half_h);
    vec3 ro = cam_pos;

    // G-buffer from primary hit (every frame, deterministic)
    Hit h;
    bool hit0 = intersect(ro, rd, h);
    imageStore(gnormal, px, vec4(hit0 ? h.n : vec3(0), 0));
    imageStore(gdepth, px, vec4(clamp((hit0 ? h.t : T_MAX) / T_MAX, 0.0, 1.0)));

    // path trace from the primary hit
    vec3 radiance = vec3(0), throughput = vec3(1);
    for (int bounce = 0; bounce <= BOUNCES; ++bounce) {
        if (bounce > 0 && !intersect(ro, rd, h)) break;
        if (bounce == 0 && !hit0) break;
        radiance += throughput * h.emit;
        if (h.emit != vec3(0)) break;           // light is terminal
        throughput *= h.albedo;
        ro = ro + h.t * rd + h.n * 1e-4;
        rd = cosine_hemisphere(h.n);
    }
    vec4 prev = frame_index == 0u ? vec4(0) : imageLoad(accum, px);
    imageStore(accum, px, prev + vec4(radiance, 1.0));
}
```

- [ ] **Step 2: Wire uniforms + keys in main.cpp**

Base camera `cam_pos = (0, 2.0, 6.5)`, `cam_target = (0, 1.6, 0)`. Per frame: set `frame_index`, dispatch, barrier (`GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT`), blit selected texture with `inv_spp = 1.0f / (frame_index + 1)` when showing accum (1.0 for G-buffers, displayed raw with mode 1). Keys 1/2/3 switch texture, R resets `frame_index = 0` (shader overwrites accum on frame 0 — no clear call needed).

- [ ] **Step 3: Verify** — `./scripts/run_host.sh`: heavy-noise Cornell image that visibly converges over a few seconds; key 2 shows flat-shaded normal colors; key 3 grayscale depth; R restarts the noise. `--hidden --frames 5` exits 0.

- [ ] **Step 4: Commit** — `"feat(host): 1-spp compute path tracer with accumulation and G-buffers"`

---

### Task 4: INT8 quantization (tested) + readback

**Files:**
- Create: `host/quantize.h`, `host/quantize.cpp`, `host/test_quantize.cpp`
- Modify: `Makefile` (test_quantize target, add to ctest)

**Interfaces:**
- Produces:
  - `void quantize_frame(const float* rgb, const float* normal4, const float* depth, int n_px, int8_t* out7)` — planar-izes to 7×H×W int8 per QUANT_SPEC section 2 (rgb is RGBA32F readback stride 4, normal4 stride 4, depth stride 1; out layout channel-major `[R plane, G plane, B plane, Nx, Ny, Nz, D]`).
  - `void quantize_rgb(const float* rgb, int n_px, int8_t* out3)` — same for the 3-channel reference.
  - Readback helper in main.cpp: `gl::GetTexImage` to `std::vector<float>` (RGBA for color/normal, RED for depth), color divided by `frame_count` before quantization.

- [ ] **Step 1: `host/test_quantize.cpp` first** — hardcoded spec examples:

```cpp
// QUANT_SPEC section 2 examples, both endpoints and rounding
#include "quantize.h"
#include <cstdio>

static int fail(const char* what, long got, long want) {
    std::printf("quantize FAIL %s: got %ld want %ld\n", what, got, want);
    return 1;
}
int main() {
    // one pixel: color (0,0.5,1), normal (-1,0,1), depth 0.25
    const float rgb[4]    = {0.0f, 0.5f, 1.0f, 0.0f};
    const float norm4[4]  = {-1.0f, 0.0f, 1.0f, 0.0f};
    const float depth[1]  = {0.25f};
    int8_t out[7];
    quantize_frame(rgb, norm4, depth, 1, out);
    const long want[7] = {-128,  0, 127,   // round(c*255)-128: 0->-128, .5->0 (127.5 rounds away to 128, clamp... see below), 1->127
                          -127, 0, 127,    // round(n*127)
                          -64};            // round(0.25*255)-128 = 64-128
    // NOTE: 0.5*255 = 127.5 -> round half away = 128 -> 128-128 = 0
    for (int i = 0; i < 7; ++i)
        if (out[i] != want[i]) return fail("frame ch", out[i], want[i]);
    std::puts("test_quantize PASS");
    return 0;
}
```

- [ ] **Step 2: Implement with `lround` (round-half-away, matching the contract), clamp, planar layout.** Makefile: build + run in `ctest` before the vector loop.

- [ ] **Step 3: Verify** — `make ctest` green including `test_quantize PASS`.

- [ ] **Step 4: Commit** — `"feat(host): INT8 quantization per QUANT_SPEC with unit test"`

---

### Task 5: Batch dump mode + PNG previews

**Files:**
- Create: `host/stb_image_write.h` (vendor from https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h), `host/stb_impl.cpp` (`#define STB_IMAGE_WRITE_IMPLEMENTATION` + include, compiled with `-Wno-all` via separate rule)
- Modify: `host/main.cpp`, `Makefile`, `tests/test_dump.py` (new)

**Interfaces:**
- Produces: `host_app --dump N --out DIR [--size S] [--ref-spp K] [--seed SEED]` (implies hidden window):
  - For view v in 0..N-1: `view_seed = hash(SEED, v)`; camera jitter from a CPU-side `std::mt19937(view_seed)`: `cam_pos += (U(-0.35,0.35), U(-0.2,0.2), U(-0.5,0.5))`, `cam_target += (U(-0.2,0.2), U(-0.15,0.15), 0)`.
  - Render frame 0 → read back → `DIR/frame_v%04d_noisy.bin` (7·S·S int8) + `DIR/frame_v%04d_noisy.png` (tonemapped RGB preview).
  - Continue accumulating to K spp → `DIR/frame_v%04d_ref.bin` (3·S·S int8) + ref preview PNG. G-buffer previews (`_normal.png`, `_depth.png`) for v==0 only.
  - Write `DIR/meta.json` once: `{"size": S, "ref_spp": K, "count": N, "seed": SEED, "channels": "R,G,B,Nx,Ny,Nz,D", "hdr_clamp": [0,1]}`.
  - Progress line per view to stdout; exit 0 only if all files written.

- [ ] **Step 1: Write `tests/test_dump.py`**

```python
import json, os, subprocess
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs a GL context (WSLg/desktop)")
def test_tiny_dump(tmp_path):
    r = subprocess.run(["scripts/run_host.sh", "--dump", "2", "--out", str(tmp_path),
                        "--size", "64", "--ref-spp", "16", "--seed", "7"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["count"] == 2 and meta["size"] == 64
    for v in range(2):
        noisy = np.fromfile(tmp_path / f"frame_v{v:04d}_noisy.bin", dtype=np.int8)
        ref = np.fromfile(tmp_path / f"frame_v{v:04d}_ref.bin", dtype=np.int8)
        assert noisy.size == 7 * 64 * 64 and ref.size == 3 * 64 * 64
        n = noisy.reshape(7, 64, 64)
        assert n[0:3].std() > 0                      # not a blank frame
        assert np.abs(n[3:6]).max() <= 127           # normals in range
        # noisy vs ref must differ (noise exists) but share the scene (correlated)
        assert not np.array_equal(n[0:3].ravel(), ref)
    assert (tmp_path / "frame_v0000_noisy.png").exists()
```

- [ ] **Step 2: Implement dump mode in main.cpp** per the interface contract above. PNG preview: quantized int8 planes converted back to uint8 (`q + 128`), interleaved, `stbi_write_png`.

- [ ] **Step 3: Verify** — `make host_app && .venv/bin/pytest tests/test_dump.py -q` passes; open the two preview PNGs and eyeball: noisy = speckled Cornell box, ref = smooth-ish.

- [ ] **Step 4: Commit** — `"feat(host): batch dataset dump with INT8 pairs and PNG previews"`

---

### Task 6: Generate the M2 dataset + close out

**Files:**
- Modify: `README.md` (tick M2), `.gitignore` already covers `data/`

- [ ] **Step 1: Full-suite check** — `make test` green (pytest incl. test_dump + ctest incl. test_quantize).
- [ ] **Step 2: Generate** — `./scripts/run_host.sh --dump 300 --out data/v1 --seed 1058` (background; ~300 views × 1024 spp; if wall-clock is unreasonable on the iGPU, retry with `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` in the launcher env to use the RTX 5070, and record which GPU generated the set in the devlog).
- [ ] **Step 3: Spot-check** — `ls data/v1 | wc -l` = 2×300 bins + previews + meta; open 3 random previews.
- [ ] **Step 4: Tick M2 in README, commit, push** — `"docs: tick M2 - renderer + 300-pair dataset"`.

**M2 exit criterion:** live window shows noisy render, normals, depth, and converging accumulation; `data/v1/` holds 300 noisy/ref INT8 pairs generated by the same binary; all tests green.
