# Fork Tracking - llama.cpp-MI50 (gfx906) vs Mainline

> Tracks divergence of this fork (`FirephoenixX02/llama.cpp-MI50`) from upstream `ggml-org/llama.cpp`.
> Branch: `gfx906/mi50-optimization` - ROCm/HIP optimizations for **MI50 / Vega20 (gfx906)**.

## Snapshot

| Field | Value |
|-------|-------|
| **Fork repo** | `https://github.com/FirephoenixX02/llama.cpp-MI50` |
| **Upstream** | `https://github.com/ggml-org/llama.cpp` |
| **Fork branch (published)** | `origin/gfx906/mi50-optimization` at `aa3c43261` (2026-09-10) |
| **Local branch** | `gfx906/mi50-optimization` at `f51aa0ab2` + 3 dirty turbo ports (2026-09-10) - 1 commit behind published |
| **Base / merge-base** | `311d4211b` - `memory: avoid allocating V cache for indexer` (#28330) |
| **Base date** | 2026-09-10 |
| **Commits ahead of base** | 19 on `origin/gfx906/mi50-optimization`, 18 on local `HEAD` + 3 uncommitted (add-id/rope/vecdotq) |
| **Last doc update** | 2026-09-10 |
| **Upstream `origin/master`** | `311d4211b` (mirrors ggml-org `master` at same date) |

### How to refresh this snapshot

```bash
# fetch upstream if you add it as a remote (recommended)
git remote add upstream https://github.com/ggml-org/llama.cpp.git
git fetch upstream master
git fetch origin

# base and drift
git merge-base --fork-point origin/master HEAD || git merge-base origin/master HEAD
git log --oneline origin/master..origin/gfx906/mi50-optimization --reverse
git diff --stat origin/master..origin/gfx906/mi50-optimization
git diff --shortstat origin/master..origin/gfx906/mi50-optimization

# what upstream did since fork point
git log --oneline 311d4211b..upstream/master | head -n 50

# if local is behind published
git log --oneline HEAD..origin/gfx906/mi50-optimization
```

---

## TL;DR - What this fork does

All changes target **gfx906 (MI50/Radeon VII) on ROCm/HIP**. Goal: make this GCN card performant for modern quantized models (Q4_K, Q5_K, Q6_K, Q8_0, Q3_K) and MoE/hybrid models (Qwen 3.x, Gemma 4, GDN) in both single-stream and concurrent decode.

Key pillars:

1. **GCN weight-repack buffer type** (`GGML_CUDA_REPACK=1`) - transforms weights at upload into a 3-plane layout (nibbles/scales/superblock) the decode matvec streams fully coalesced (~58% -> ~89% HBM bandwidth on matvec).
2. **Repacked MMQ GEMM for prefill** - 2D-tiled dp4a GEMM consuming repacked planes directly.
3. **Multi-quant + broadcast + MoE extensions** - Q5_K/Q6_K/Q8_0/Q3_K in repacked path, 2D weight x 3D activation broadcast (GDN hybrid), 3D MUL_MAT_ID / expert-stack.
4. **Concurrent-decode batching** - batched `ncols` matvecs and MoE per-token loops so 2-8 slots scale.
5. **GCN-specific correctness + micro-opts** - DPP warp reductions, `solve_tri` fix, per-op profiler, ssm-conv / async memcpy / FATTN fixes, Vega20 MMQ tuning.

---

## Commit Inventory (oldest -> newest)

| # | SHA (short) | Date | Title | Category | Upstream? |
|---|-------------|------|-------|----------|-----------|
| 1 | `2797a21d0` | 2026-06-09 | `ggml-hip: support k up to 64 in solve_tri, cap GCN at 64x64` | Fix / GCN cap | fork-original |
| 2 | `07b67fb1b` | 2026-06-09 | `ggml-hip: DPP-based warp reductions on GCN` | Perf / ISA | fork-original |
| 3 | `b947c8f38` | 2026-06-10 | `ggml-hip: GCN Q4_K weight-repacking buffer type (GGML_CUDA_REPACK=1)` | Core infra | fork-original |
| 4 | `b7f720cb6` | 2026-06-10 | `ggml-hip: repacked int8 MMQ tile GEMM for GCN prefill` | Perf / Prefill | fork-original |
| 5 | `59ef37491` | 2026-06-10 | `ggml-hip: Q5_K/Q6_K/Q8_0 repack + 2D-x-3D broadcast for repacked weights` | Feature | fork-original |
| 6 | `d643ec4aa` | 2026-06-10 | `ggml-hip: retune the repacked Q8_0 matvec for small dense shapes` | Tuning | fork-original |
| 7 | `979542125` | 2026-06-10 | `ggml-hip: MUL_MAT_ID / 3D expert-stack support for repacked weights` | Feature / MoE | fork-original |
| 8 | `0338da9e1` | 2026-06-10 | `ggml-hip: half-sub-block work units for the ID-matvec path` | Tuning | fork-original |
| 9 | `01bd06486` | 2026-06-10 | `ggml-hip: direct in-kernel expert routing -> MoE decode parity; repack default-on for GCN` | Perf / MoE | fork-original |
| 10 | `58b0f4235` | 2026-06-16 | `cuda: add Q3_K to the GCN weight-repack kernels` | Feature | fork-original |
| 11 | `8a9313831` | 2026-06-17 | `cuda: env-gated per-op GFXPROF profiler` | Debug / Tooling | fork-original |
| 12 | `4567adbeb` | 2026-07-11 | `cuda(gfx906): batched repacked matvec for concurrent decode` | Perf / Concurrency | cherry-pick |
| 13 | `8a95af4c4` | 2026-07-11 | `cuda(gfx906): batched repacked matvec for dense Q5_K` | Perf / Concurrency | cherry-pick |
| 14 | `a6b6d51ab` | 2026-07-11 | `cuda(gfx906): fold per-sequence ssm_out into the batched matvec` | Perf / GDN | cherry-pick |
| 15 | `1f512b7ea` | 2026-07-11 | `cuda(gfx906): always set device before async memcpy (#21140/#18313)` | Fix | cherry-pick (upstream #21140) |
| 16 | `d08832721` | 2026-07-11 | `cuda: bound ssm_conv_long_token_f32 staging read to valid columns` | Fix | cherry-pick (upstream bug from #20128) |
| 17 | `4b3a86c97` | 2026-09-10 | `build: enable GCN repack for MI50 (gfx906) in ROCm build script` | Build | fork-original |
| 18 | `f51aa0ab2` | 2026-09-10 | `ggml/cuda: add Vega20 (gfx906) MMQ tuning for MI50` | Tuning / MMQ | fork-original |
| 19 | `aa3c43261` | 2026-09-10 | `fattn: clamp Vega20 (gfx906) to nstages=1, occupancy=1 for 64KB LDS` | Fix / FATTN | fork-original - only on `origin/gfx906/mi50-optimization` |

> `git log --reverse --oneline origin/master..origin/gfx906/mi50-optimization` reproduces this order.
> Local `HEAD` (`f51aa0ab2`) is missing #19. Run `git pull origin gfx906/mi50-optimization` or `git merge origin/gfx906/mi50-optimization` to sync.

---

## File-level Impact (vs `311d4211b`)

```
git diff --stat origin/master..origin/gfx906/mi50-optimization

 .gitignore                             |    1 +
 build-llamacpp-rocm.sh                 |   27 +
 ggml/src/ggml-cuda/common.cuh          |  133 +-
 ggml/src/ggml-cuda/fattn-mma-f16.cuh   |   18 +-
 ggml/src/ggml-cuda/ggml-cuda.cu        |  187 ++-
 ggml/src/ggml-cuda/mmq-config-vega.cuh |  270 ++++
 ggml/src/ggml-cuda/mmq.cuh             |   10 +-
 ggml/src/ggml-cuda/repack-gcn.cu       | 2526 ++++++++++++++++++++++++++++++++
 ggml/src/ggml-cuda/repack-gcn.cuh      |   57 +
 ggml/src/ggml-cuda/solve_tri.cu        |  109 +-
 ggml/src/ggml-cuda/ssm-conv.cu         |   10 +-
 src/llama-model.cpp                    |   10 +-
 12 files changed, 3293 insertions(+), 65 deletions(-)  # on published branch (11 files / 3276 on local HEAD)
```

### What each file does

| File | Lines | Role in fork |
|------|-------|--------------|
| `ggml/src/ggml-cuda/repack-gcn.cu` | +2526 (new) | Entire repacked matvec/MMQ + MoE + broadcast machinery. ~90% of fork delta. |
| `ggml/src/ggml-cuda/repack-gcn.cuh` | +57 (new) | Public API for repack buffer type and `ggml_cuda_mul_mat_*_repacked`. |
| `ggml/src/ggml-cuda/mmq-config-vega.cuh` | +270 (new) | Vega20 MMQ config table ported from RDNA2 with occupancy 1, `I=128`, `launch_bounds 256,1` for 64KB LDS/W64 dp4a. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | +178/-9 | Hook repack buft via `get_extra_bufts`, dispatch `mul_mat`/`mul_mat_id` to repacked kernels, extra buft precedence fix, `GGML_CUDA_REPACK`/`*_Q8_0`/`*_MOE` env gates, `GGML_CUDA_REPACK_NO_MMQ` fallback, `REPACK_TRACE`/`NOFOLD` diagnostics, `GFXPROF` HIP-events profiler, fixed `ggml_cuda_set_device` device-id bug. |
| `ggml/src/ggml-cuda/common.cuh` | +132/-1 | DPP warp reductions (`GCN` only), `V_DOT2_F32_F16` gating, `MATRIX_ROW_PADDING` usage, CC defines `GGML_CUDA_CC_GCN`/`VEGA20`, `W64` handling. |
| `ggml/src/ggml-cuda/mmq.cuh` | +8/-2 | Dispatch to Vega MMQ via `GGML_CUDA_CC_IS_GCN` / `__gfx906__` (`mmq-config-vega.cuh`), occupancy 1 path. |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | +17/-1 | **Only on published branch** - clamps FATTN `nstages_target=1`, `occupancy=1` for GCN (64KB LDS overflow fix for `num_stages=2`). |
| `ggml/src/ggml-cuda/solve_tri.cu` | +62/-47 | Extend `MAX_K_FAST` to 64 with per-thread column loop; cap GCN to `n<=64,k<=64` (Qwen3-Next delta-net prefill). |
| `ggml/src/ggml-cuda/ssm-conv.cu` | +9/-1 | Bound `ssm_conv_long_token_f32` staging read to `d_conv-1+local_n_t` (fix OOB when `n_t % 32 != 0`, GDN). |
| `build-llamacpp-rocm.sh` | +27 (new) | Canonical ROCm 7.2.4 build (`/opt/rocm-7.2.4`, `HIPCXX`, `GFX906`, `-DGGML_CUDA_REPACK=1 -DGGML_CUDA_REPACK_Q8_0=1`, `-isystem`/`-L` isolation from Ubuntu HIP 5.x). |
| `.gitignore` | +1 | Track `build-llamacpp-rocm.sh` (exception to ignore). |
| `src/llama-model.cpp` | +6/-4 | Scheduler `make_gpu_buft_list` inserts extra bufts before device default so repack type can be selected; supports 2D->3D broadcast shape. |

---

## Detailed Change Log

### 1. `2797a21d0` - solve_tri k=64, GCN cap
- **Why:** `solve_tri` fast path capped at `k<=32` but Qwen3-Next delta-net prefill needs 64.
- **Change:** `MAX_K_FAST=64` with per-thread column loop above `MAX_K_BLOCK` to keep block shape `(WARP_SIZE,32)`; GCN rejects `n>64||k>64` to other backends.
- **Files:** `ggml/src/ggml-cuda/solve_tri.cu:62`

### 2. `07b67fb1b` - DPP-based warp reductions on GCN
- **Why:** `__shfl_xor` on GCN lowers to `ds_bpermute_b32` (LDS + `s_waitcnt` per step).
- **Change:** For GCN, specialize `warp_reduce_sum/max` for `float`/`float2`/`int` via DPP `quad_perm` / `row_ror` / `ds_swizzle` (5 DPP + 1 shfl for wave64 sum vs 6 LDS). `s_nop` hazard waits. `half2` stays generic, CDNA untouched.
- **Files:** `ggml/src/ggml-cuda/common.cuh:452`
- **Perf:** gemma4 31B Q4_K tg128 `21.93 -> 22.97 t/s (+4.7%)` on MI50.

### 3. `b947c8f38` - GCN Q4_K weight-repacking buffer type
- **Why:** On-disk Q4_K superblock interleaves nibbles/scales every 144B -> wave64 caps at ~58% HBM BW.
- **Change:** New extra buffer type (`GGML_CUDA_REPACK=1`), host-side repack at `set_tensor` into 3 planes (nibble 16B/chunk, `sc|m`, `d|dmin`), `MATRIX_ROW_PADDING` padding when `K/32` is power-of-two (HBM channel aliasing fix), `supports_op` limits to 2D Q4_K `MUL_MAT` f32, fused `mul_mat` refuses repacked. Repacked matvec hits ~89% BW.
- **Files:** `ggml/src/ggml-cuda/repack-gcn.cu:1`, `ggml/src/ggml-cuda/repack-gcn.cuh:1`, `ggml/src/ggml-cuda/ggml-cuda.cu:420`, `src/llama-model.cpp:6`

### 4. `b7f720cb6` - Repacked int8 MMQ tile GEMM for prefill
- **Why:** Repacked `ne11>1` fell back to dequant+fp16 `GemmEx`.
- **Change:** 256-thread WG, `64x64` tile, `BK=4, TM=TN=4`, `launch_bounds(256,2)`, LDS-staged `4`-sub-block chunks, `4x4` register micro-tile (bank-conflict-free due to `block_q8_1` stride 36B, `gcd(9,32)=1`). Fallback behind `GGML_CUDA_REPACK_NO_MMQ=1`.
- **Perf:** MI50 gemma4 31B UD-Q4_K_XL pp512 `186.7 -> 213.3 t/s (+14.2%)` vs `GGML_CUDA_REPACK=0`; `tg128 22.29 -> 26.57`.

### 5. `59ef37491` - Q5_K/Q6_K/Q8_0 repack + 2D-x-3D broadcast
- **Q5_K:** nibble + `qh` u32 plane (`bit 4g+b` -> byte bit 4) + `sc|m` + `d|dmin`; `dsc*idot - deff*sx` fold.
- **Q6_K:** nibble + 8B `h2` plane (bits 4-5) + signed per-16 scale pairs + `d`-only plane; `-32` folded via activation half-sums.
- **Q8_0:** qs bytes + fp16 `d` stream; `ROWS=1`/`2` split at `ne01>=4096`; gated `GGML_CUDA_REPACK_Q8_0=1` (prefill +43% on 0.8B but decode -6%, needs retune).
- **Broadcast fix:** `supports_op` now accepts 2D weight x 3D activation (weight broadcast over `ne2`) for Qwen3.5 GDN hybrid; batch PPL within noise.
- **Perf:** Pure-Q8_0 0.8B pp512 `4763 -> 6822 t/s (+43%)`.

### 6. `d643ec4aa` - Retune Q8_0 matvec for small dense shapes
- **Change:** Small-`ne01` path to 4-wave blocks with 16-weight half-sub-block units (`ne0=1024` -> 32 sub-blocks for 64 lanes was starving). Rejected: full-block units, 8-weight quarter, `ROWS=4`.
- **Perf:** 0.8B Q8_0 tg128 `222.7 -> 231.0 t/s` (canonical `238.0`, so still gated).

### 7. `979542125` - MUL_MAT_ID / 3D expert-stack for repacked weights
- **Why:** MoE expert tensors are 3D (`[K, N, n_experts]`); repack was 2D-only.
- **Change:** Per-expert stacked slabs, `HAS_IDS` templated kernels (4 matvec + 4 MMQ), `blockIdx.y` as assignment, `expert_bounds`/`tile_off` binary search, 16-token tiles for MoE (64-wide would be mostly empty).
- **Perf:** qwen3.6 35B-A3B UD-Q4_K_XL pp512 `864 -> 967 t/s (+12.0%)`, `tg128 89.6 -> 83.1 (-7.3%)` at this stage (decode still gated).

### 8. `0338da9e1` - Half-sub-block units for ID matvec
- **Change:** MoE `HAS_IDS` matvecs use 16-weight half units (Q4_K/Q5_K split `deff*sx` on even half; Q6_K splits clean with folded `-32`); dense keeps full units.
- **Perf:** `tg128 83.11 -> 83.75 (+0.8%)`, PPL `5.07` vs `5.12` canon. Still gated (`GGML_CUDA_REPACK_MOE`) - bottleneck is fused `ffn_gate+ffn_up` (2 launches vs 1).

### 9. `01bd06486` - Direct in-kernel routing -> MoE parity; repack default-on for GCN
- **Why:** At 1 token, `mm_ids_helper` compaction is pure overhead vs canonical 2-launch `mmvq-id`. Broadcast `src1` (`ne[1]==1`) needs `x_stride=0` (NaN if wrong).
- **Change:** MoE decode reads raw `ids` directly; compaction remains for grouped GEMM batch (`n_tokens>8`). Add fused `gate+up` GLU matvec (hook dormant for tested models). Repack on by default for GCN dense after this.
- **Perf:** qwen3.6 35B-A3B tg128 `89.94 vs 89.86 canon` (parity, was `83.11` gated).

### 10. `58b0f4235` - Q3_K in repack kernels
- **Change:** `lo2` (2-bit) + 1-bit `hi` plane (`3.56 bpw`), `q3 = lo2 | (hbit<<2)`, Q6_K symmetric scale+bias `4`.
- **Effect:** Lets gfx906 run Q3_K MoE/dense via dp4a repack instead of rocBLAS.

### 11. `8a9313831` - Env-gated `GFXPROF` profiler
- **Change:** `GFXPROF=1` wraps each graph op in HIP events -> per `(op, src0 type, KxN)` GPU time + `TMAC/s` at exit. No-op when unset. For gfx906 where distro ROCm ships no `rocprof`.
- **Files:** `ggml/src/ggml-cuda/ggml-cuda.cu:8`

### 12. `4567adbeb` - Batched repacked matvec for concurrent decode
- **Why:** Only `ne11==1` used tuned matvec; `2..8` hit GEMM -> concurrent decode < single-stream.
- **Change:** Dense Q6_K `mul_mat_vec_q6k_repacked_ncols<NCOLS>` (`ROWS=1`, inline unpack, no per-column cache); `ne11<=8` Q6_K routed there. MoE: `n_tokens<=8` loops single-token matvec per token instead of 16-wide grouped GEMM (1 token/expert avg -> 15/16 empty); compaction only for `>8`.
- **Perf:** `batched-bench` concurrent decode `+34-72%` (`npl=2 32->56 t/s`), single-stream byte-identical.

### 13. `8a95af4c4` - Batched repacked matvec for dense Q5_K
- **Why:** Fused `attn_qkv` (36 GDN layers) is Q5_K and fell to per-column loop (-12% at 6 slots).
- **Change:** `mul_mat_vec_q5k_repacked_ncols<NCOLS>` (nibble + `repack_spread4` high bit, `dsc*idot - deff*sx`).
- **Perf:** `batched-bench` concurrent decode `+7-9%` (2-8 slots).

### 14. `a6b6d51ab` - Fold per-sequence `ssm_out` into batched matvec
- **Why:** GDN `ssm_out` sees `[d,1,n_seqs]` -> `ne11=1, ne12=n_seqs` -> `n_seqs` separate matvecs reloading weight each time (~8% at 6 slots).
- **Change:** In `ggml_cuda_mul_mat_repacked`, when `ne11==1 && 2<=ne12<=8` with contiguous `dst`, fold `ne12` into column batch as one `ne11=ne12` call (`ncols` kernel). Adds `REPACK_TRACE` (shape logging) and `REPACK_NOFOLD` env gates.
- **Perf:** `ssm_out 648x ne11=1 -> 108x ne11=6`; `batched-bench` concurrent decode `+3-6%` (`npl=6 70.6 -> 74.3 t/s`).

### 15. `1f512b7ea` - Always set device before async memcpy
- **Why:** `ggml_cuda_set_device` skipped `cudaSetDevice` when `cudaGetDevice` already matched, but ROCm multi-GPU thread may report stale `-1` -> async memcpy -> `illegal memory access`.
- **Change:** Drop skip; `cudaSetDevice` on current device is near-no-op; explicitly set before every async H2D/D2H.
- **Upstream:** Port of `llama.cpp #21140 / #18313`.
- **Files:** `ggml/src/ggml-cuda/ggml-cuda.cu:121`

### 16. `d08832721` - Bound `ssm_conv_long_token_f32` staging read
- **Why:** Kernel stages `load_cols = d_conv-1+split_n_t (32)` columns per row but when `n_t % 32 != 0` trailing block has `local_n_t < 32` -> reads past `conv_input` (exact `ggml_nbytes`, no padding) -> `HSA_STATUS_ERROR_MEMORY_FAULT` on multi-slot spec-decode / dynamic prompts.
- **Change:** Bound to `d_conv-1+local_n_t`, zero tail. Output-preserving (compute loop already bounded). Verified on Qwen3.5-122B GDN fan-out (4 slots, 40k ctx, n-gram spec) that crashed at ~11min now runs 14min clean.
- **Upstream bug:** From `PR #20128`; see `#20024, #21383`.
- **Files:** `ggml/src/ggml-cuda/ssm-conv.cu:9`

### 17. `4b3a86c97` - Build: enable GCN repack for MI50
- **Change:** `build-llamacpp-rocm.sh` sets `-DGGML_CUDA_REPACK=1 -DGGML_CUDA_REPACK_Q8_0=1`, `GPU_TARGETS=gfx906`, `ROCm 7.2.4` `/opt` isolation. `.gitignore` exception.
- **Files:** `build-llamacpp-rocm.sh:27`, `.gitignore:1`

### 18. `f51aa0ab2` - Vega20 MMQ tuning for MI50
- **Why:** No dedicated Vega20 config; RDNA2 occupancy 2 overflows 64KB LDS on gfx906 W64.
- **Change:** New `ggml/src/ggml-cuda/mmq-config-vega.cuh` with occupancy 1, `I=128` saturates LDS, `launch_bounds(256,1)` avoids VGPR spill; dispatch via `GGML_CUDA_CC_IS_GCN` host + `GCN`/`__gfx906__` device.
- **Files:** `ggml/src/ggml-cuda/mmq-config-vega.cuh:270`, `ggml/src/ggml-cuda/mmq.cuh:10`

### 19. `aa3c43261` - FATTN clamp for gfx906 (published only)
- **Why:** FATTN MMA on gfx906 uses W64 dp4a, no `cp_async`, 64KB LDS/CU; `nstages=2` doubles K+V tiles -> spill.
- **Change:** `ggml/src/ggml-cuda/fattn-mma-f16.cuh:230` clamp `nstages_target=1`, `occupancy=1` for `GGML_CUDA_CC_IS_GCN` host and `GCN`/`__gfx906__` device before RDNA fallback. `num_warps=4`, adaptive `J` via Vega MMQ.
- **Note:** This commit is on `origin/gfx906/mi50-optimization` but not on local `HEAD`. Cherry-pick or pull to include.

---

## Correctness Notes

- All repacked matvec/MMQ variants validated via `wikitext PPL` vs GEMM/canonical within noise (tokens in commit logs).
- `ub=1 PPL` checks: Q5_K/Q6_K, Q8_0 (5.07 vs 5.12), MoE ID paths, FATTN clamp (no PPL regression cited; LDS correctness).
- `n_tokens/ne11==1` paths collapse to original launch -> single-stream byte-unchanged for concurrent-decode batching commits.
- Env gates for debugging without rebuild: `GGML_CUDA_REPACK`, `GGML_CUDA_REPACK_Q8_0`, `GGML_CUDA_REPACK_MOE`, `GGML_CUDA_REPACK_NO_MMQ`, `REPACK_TRACE`, `REPACK_NOFOLD`, `GFXPROF`.

---

## Build & Runtime Flags

| Flag / Env | Default | Effect |
|------------|---------|--------|
| `GGML_CUDA_REPACK=1` (cmake) | ON for GCN after `01bd06486`, gated before | Enable repacked Q4_K/Q5_K/Q6_K dense + MoE path |
| `GGML_CUDA_REPACK_Q8_0=1` | needs explicit flag | Enable Q8_0 repacked (decode still -3% vs canonical, keep for prefill +43%) |
| `GGML_CUDA_REPACK_MOE=0/1` | OFF until `01bd06486` | Q4_K/Q5_K half-block MoE path |
| `GGML_CUDA_REPACK_NO_MMQ=1` | unset | Force dequant+GemmEx fallback for `ne11>1` repacked |
| `REPACK_TRACE=1` | unset | Log `mul_mat` dispatch shapes |
| `REPACK_NOFOLD=1` | unset | Disable `ssm_out` `ne12` fold |
| `GFXPROF=1` | unset | HIP-events per-op profiler |
| `GGML_CUDA_ALLREDUCE=internal\|nccl` | auto | Multi-GPU all-reduce backend choice |
| `GPU_TARGETS=gfx906` | - | HIP compile target |

Build via: `./build-llamacpp-rocm.sh` (requires `/opt/rocm-7.2.4`).

---

## What Needs Upstream Attention

| Area | Status | Next step |
|------|--------|-----------|
| `ssm_conv_long_token_f32` OOB fix | Upstream bug (#20128), cherry-picked here; should be PR'd if not already | Check `upstream/master` - if missing, open PR with `d08832721` diff |
| `ggml_cuda_set_device` fix | Upstream #21140/#18313, already merged upstream via other PRs - verify not needed after rebase | Drop on next merge-base bump if upstream contains it |
| Repack buffer type | Fork-only; invasive (~2.5k LOC). Upstreaming would need feature-flag review | Keep fork-local, rebase-friendly, avoid touching hot paths when gated off |
| FATTN GCN clamp | Fork-only; correctness on 64KB LDS | Validate against upstream FATTN changes at each rebase |
| Vega20 MMQ table | Derived from RDNA2 occupancy 1; not upstream | Keep; upstream MMQ tables live in `mmq.cuh` and change often - rebase conflict expected |
| DPP warp reductions | GCN-only `#ifdef`; no CDNA/RDNA impact | Keep; re-check if upstream refactors `common.cuh` reductions |

---

## Rebase / Merge Checklist

1. Update `origin/master` (or `upstream/master`):
   ```bash
   git fetch origin
   git log --oneline 311d4211b..origin/master | wc -l   # how many upstream commits since base
   ```
2. Rebase fork branch:
   ```bash
   git checkout gfx906/mi50-optimization
   git rebase origin/master   # or `git rebase upstream/master` then `git push --force-with-lease origin gfx906/mi50-optimization`
   ```
   Expected conflicts: `ggml/src/ggml-cuda/ggml-cuda.cu`, `common.cuh`, `mmq.cuh`, `fattn-mma-f16.cuh`, `src/llama-model.cpp`.
3. Verify no silent behavior change when repack is off (`GGML_CUDA_REPACK=0` PPL vs upstream).
4. Bump **Base / merge-base** and **Commits ahead** in this doc, and move commit `aa3c43261` into the main table after it lands locally.
5. Update **File-level Impact** (`git diff --shortstat origin/master..HEAD`) and this checklist if upstream adds new files (e.g., `llama-model.cpp` graph changes).

---

## Branch Topology

```
upstream/master (ggml-org)  ──────────────────────►  311d4211b ──► ... (new upstream commits)
origin/master (this repo)      ──────────────────────►  311d4211b  (synced, no fork commits)
origin/gfx906/mi50-optimization  ────────── 311d4211b ──► 18 commits ──► f51aa0ab2 ──► aa3c43261 (published HEAD)
local gfx906/mi50-optimization   ────────── 311d4211b ──► 18 commits ──► f51aa0ab2 (1 behind published)
```

---

## Links & References

- Upstream issues/PRs cited: `#20024`, `#20128`, `#21140`, `#21383`, `#28330`, `#28667`, s390x repack.
- Related files: `ggml/src/ggml-cuda/repack-gcn.cu:1`, `ggml/src/ggml-cuda/repack-gcn.cuh:1`, `ggml/src/ggml-cuda/mmq-config-vega.cuh:1`, `ggml/src/ggml-cuda/common.cuh:452`, `ggml/src/ggml-cuda/ggml-cuda.cu:121`, `ggml/src/ggml-cuda/solve_tri.cu:1`, `ggml/src/ggml-cuda/ssm-conv.cu:1`, `ggml/src/ggml-cuda/fattn-mma-f16.cuh:230`, `build-llamacpp-rocm.sh:1`.

---

## Turbo High-Value Ports (2026-09-10, uncommitted)

Ported from `arte-fact/llamacpp-gfx-906-turbo` (`gfx906/` 29-file W64 kernels). Build verified via `./build-llamacpp-rocm.sh` (ROCm 7.2.4, `gfx906`, `GGML_CUDA_REPACK`).

| # | Port | File | Change | Status |
|---|------|------|--------|--------|
| 20 | `add-id` vectorized | `ggml/src/ggml-cuda/add-id.cu:3` | `float4` coalesced path + contiguous fast path; `ne0%4==0` aligned -> 4x fewer global loads | Built, pending commit |
| 21 | `rope` GCN `__sincosf` | `ggml/src/ggml-cuda/rope.cu:20` | `rope_yarn` uses `__sincosf` + `__logf` on `GCN` (vs `cosf+sinf+logf`); saves 1 transcendental per `n_dims/2` lanes | Built, pending commit |
| 22 | `vecdotq` fast `b1/b2` + MXFP4 LUT | `ggml/src/ggml-cuda/vecdotq.cuh:7` | `memcpy` `flat_load` for `b1/b2`, `__builtin_amdgcn_perm` 8-entry MXFP4 `v_perm` (0-12), `GGML_GCN_VEC_DOT_MXFP4_Q8_1` macro in `vec_dot_mxfp4_q8_1` + `gfx906_get_int_b2_fast` in `vec_dot_q8_0_q8_1` | Built, pending commit |
| 23 | `quantize` DPP | — | **Deferred**: `quantize_q8_1` already hits `common.cuh:452` DPP `warp_reduce_max/sum`; turbo's `quantize_mmq_q8_1` ultra-fused 4-val/thread DPP asm (`quantize.cu:216`) is high-risk asm, low benefit vs existing path. Keep for future if `quantize_mmq` profiling shows hot. |

`git diff --stat HEAD`: `add-id.cu 117 +-`, `rope.cu 10 +`, `vecdotq.cuh 59 +` (175 ins). Isolated to GCN guards (`defined(GCN)` / `__gfx906__`), no CDNA/RDNA/NV impact when `GGML_CUDA_REPACK=0`; gated FATTN/MMQ unchanged.

Remaining turbo candidates (deferred, see `NEXT_OPTIMIZATIONS.md`):
- `mmvq warp-coop` (`gfx906/matmul/mmvq-q4_0/q4_1/q8_0.cuh:11` half-warp 32t/row) — benefits canonical `Q4_0/Q4_1` decode when `REPACK=0`; conflicts with repacked path, needs `!repack` gate.
- `mmq vectorized + prefetch` (`mmq.cuh:11`, `mmq-prefetch.cuh:11` `global_load_dword` Y/X L2 prefetch) — only for upstream MMQ, not repacked MMQ.
- `sgemm/mmf` custom `F32 32x32x64` / `F16 32x64x64` (`gfx906/matmul/sgemm.cuh:16`, `mmf.cuh:17`) — standalone, 2x small GEMM, no deps; candidate for next port.
- `fattn-q8` tile (`gfx906/attention/fattn-q8.cuh:1` 8 `DKQ/DV` instances) — heavy, only for `Q8_0 KV`.

---

## Maintaining This Document

- Edit this file on every `git push` to `gfx906/mi50-optimization` or `origin/master` sync.
- Keep commit table in reverse-chronological log order (`git log --reverse`).
- Copy commit body details from `git show <sha>` rather than paraphrasing from memory.
- Performance deltas are from commit messages - include `llama-bench` / `batched-bench` / `wikitext PPL` numbers when present.
- Remove or collapse older tuning experiments (e.g., rejected `ROWS=4`, quarter-units) only after they are stable for >2 rebases.

