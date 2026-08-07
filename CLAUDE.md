# ternoise — agent context

A 1.58-bit (ternary-weight) neural denoiser for 1-spp ray-traced frames, targeting an
FPGA with **zero multipliers**: weights are {-1, 0, +1}, so every MAC is an add, a
subtract, or a skip. No DSP slices, no FP inference. Proof-of-concept that ternary
CNNs can replace FP16/FP32 denoisers in rendering pipelines.

Repo: github.com/jackthepunished/ternoise (remote is SSH). Owner: Bahadir, 3rd-year CS
student, Rust/C++ systems background, building this in public (X/Twitter + devlog/).

## The one rule that governs everything

`docs/QUANT_SPEC.md` is the normative contract. PyTorch eval path, C++ golden model,
and (future) RTL must agree **bit-for-bit** with it and with each other, verified
against the checked-in golden vectors in `vectors/` (regenerable byte-identically:
`python -m vectors.gen_vectors`, seed 1058). Zero tolerance — a one-bit diff is a bug.
Never change quantization behavior without updating QUANT_SPEC, the golden vectors,
and every implementation together.

Two load-bearing consequences (do not "fix" these):
- Requant scales are powers of two only (shift+round+clamp). An arbitrary fixed-point
  scale would smuggle a multiplier back into the datapath.
- Accumulators are sized per layer: ceil(log2(9 * C_in * 128)) + 1 bits. 16-bit
  overflows at C_in = 32. Software uses int64 and asserts int32 range.

## Status (updated 2026-08-07)

- [x] M0 quantization contract + golden vectors
- [x] M1 bit-exact twins: Python ref (`model/ref_ops.py`), C++ golden model (`sim/`),
      PyTorch `BitConv2d`/`DenoiseNet` (STE train path, int eval via ref_ops),
      2-bit weight packer (`tools/pack_weights.py`) + C++ unpacker, CI green
- [x] M2 renderer: GL 4.3 compute path tracer (`host/`), Cornell box, G-buffers,
      accumulation; dataset `data/v1/` = 300 pairs, 256px, 1024-spp refs (gitignored,
      regenerate: `./scripts/run_host.sh --dump 300 --out data/v1 --seed 1058`)
- [ ] M3 training + software denoising — NEXT. Sketch: dataset loader -> QAT training
      (L1 vs ref, PSNR/SSIM val) -> calibration (per-layer shift from accumulator
      stats, bias_int from FP bias; the M1 stubs `shift=5`, `bias_int=0` become real)
      -> export via packer -> C++ golden model denoises a val frame bit-exact vs
      int_forward. GATE: +6 dB PSNR over noisy on val avg, else no RTL gets written.
- [ ] M4 RTL (SystemVerilog + Verilator; line buffers, ternary PE array, bit-exact
      vs C++ on streamed frames; Yosys check: zero DSP/multiplier cells)
- [ ] M5 hardware-in-the-loop (Verilated model linked into host) -> synthesis numbers
      -> only then choose an FPGA board. No board exists yet; buy nothing.

Master plan: `~/.claude/plans/we-must-be-careful-velvety-knuth.md` (outside repo).
Phase plans live in `docs/plans/` (M0/M1 and M2 plans there show the house style).

## Network v1 (fixed until the whole chain works)

3x3 convs, channels 7->16->16->16->16->3, ReLU after layers 1-4, none on 5.
Input 7ch = [R,G,B,Nx,Ny,Nz,D] int8 per QUANT_SPEC section 2. Output = RGB residual,
clamped-added onto input channels 0-2. No U-Net, no 5x5, no per-channel scales, no
temporal anything — v1 simplifications are deliberate; do not widen scope.

## Commands

- `make test` — everything: pytest (`.venv`) + C++ unit tests + all golden cases.
  This must be green before any commit claims completion. CI runs the same.
- `make host_app && ./scripts/run_host.sh` — interactive renderer (keys 1/2/3 views,
  R reset, ESC quit). The launcher forces `GALLIUM_DRIVER=d3d12`; without it WSLg
  gives llvmpipe (software GL). GL function loading is hand-rolled in
  `host/gl_loader.h` — add new GL calls to the X-macro list there.
- Python env: `.venv` created with `uv venv` (system python3-venv also available).
  Torch is CPU-only currently; swap to CUDA wheel for M3 training (RTX 5070 present).

## Conventions

- TDD, strict: failing test first, then implementation, verify, commit. Small
  focused commits (`feat:`, `docs:`, `chore:`, `ci:`, scope `(host)` etc. — see log).
- Phase plans are written to `docs/plans/YYYY-MM-DD-*.md` and discussed with Bahadir
  BEFORE execution. Follow existing plan style: exact file paths, real code in steps,
  verify commands with expected output, one commit per task.
- C++: C++17, `-Wall -Wextra -Werror`, plain loops in reference code (clarity over
  speed, no SIMD). Python: numpy int64 for all integer-path math.
- **No emojis anywhere** — code, docs, commits, posts. Hard rule from Bahadir.
- data/, build artifacts, .venv are gitignored; golden vectors are committed.
- Devlog entries in `devlog/` (markdown, one per work session-ish) feed the
  build-in-public posts; milestone completions are post-worthy moments. Bahadir
  frames the public narrative around ML inference efficiency.

## Agent workflow (since M3)

Three roles, general across phases. The planner is the interactive session with
Bahadir: writes the phase plan (docs/plans/), dispatches the other two, acts on
review verdicts, opens PRs, merges, runs integration. The executor implements
one plan task in an isolated worktree, TDD, and pushes a branch. The reviewer
reviews a pushed branch against its task section, read-only, and returns a
merge verdict. Role definitions live in `.claude/agents/` (plan-executor,
plan-reviewer) — local and untracked on purpose; do not commit `.claude/`.
Loop per task: plan -> execute -> review -> fix if needed -> PR -> merge, in
the plan's merge order. Worktree test invocation:
`make test PYTHON=<main-checkout>/.venv/bin/python`.

## Environment quirks

- WSL2 Ubuntu 24.04 on a Windows laptop; repo lives at `~/ternoise` (native FS —
  never work under /mnt/c, it is 10x slower).
- GPUs: AMD Radeon 610M iGPU (default D3D12 adapter, fine and fast for the renderer)
  and RTX 5070 Laptop 8GB (CUDA 13.2 works; for GL use
  `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`, but it benchmarked SLOWER than the iGPU
  for the dump workload — readback overhead).
- `sudo` needs an interactive terminal (agent shells can't prompt); ask Bahadir to
  run apt installs in a separate terminal.
- GitHub via SSH key (OAuth token lacks workflow scope — don't switch the remote
  back to HTTPS).
