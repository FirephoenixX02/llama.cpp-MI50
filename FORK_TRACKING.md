# Fork Tracking - llama.cpp-MI50 (gfx906) vs Mainline

> Tracks divergence of this fork (`FirephoenixX02/llama.cpp-MI50`) from upstream `ggml-org/llama.cpp`.
> Branch: `gfx906/mi50-optimization` - ROCm/HIP optimizations for **MI50 / Vega20 (gfx906)**.

## Snapshot

| Field | Value |
|-------|-------|
| **Fork repo** | `https://github.com/FirephoenixX02/llama.cpp-MI50` |
| **Upstream** | `https://github.com/ggml-org/llama.cpp` |
| **Fork branch (published)** | `origin/gfx906/mi50-optimization` at `d2685bff2` (2026-11-05, bf16 repack); `5a2255fcc` / `2aa788490` / `ffc7e39c1` + docs commit not yet pushed |
| **Local branch** | `gfx906/mi50-optimization` rebased onto `master` = `7951f601e` (2026-10-19), 99 commits ahead, clean |
| **Base / merge-base** | `7951f601e` - `vendor : update cpp-httplib to 0.62.0 (#30785)` |
| **Base date** | 2026-10-19 |
| **Commits ahead of base** | 99 on `HEAD` (`git rev-list --count origin/master..HEAD`) |
| **Last doc update** | 2026-11-06 (lds_nc column-group restructure + rms_norm_mul_rope SGPR slim committed `ffc7e39c1` and validated same day, see "lds_nc column-group + rope SGPR slim" below; fork-tracking trimmed - early rounds compressed to tables, per-port tables deduplicated against the Commit Inventory) |
| **Upstream `origin/master`** | `7951f601e` (2026-10-19) - rebase done, see "Rebase 2026-10-19" below |
| **Backup** | `backup/mi50-pre-rebase-20261018` at `87f850290` (pre-rebase HEAD) |
| **Donor** | `furnace/gfx906-perf` at `fc353a5ac` (2026-10-04, sixvolts) - 11 single-GPU generic opts ported, GDN/qwen4exp/server/tests-only excluded; round 2 at `4f30c1bda` (2026-11-01) - LDS rows, deterministic top-k, per-type switches, M-RoPE fusion (see below) |

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
git log --oneline 4364bf723..upstream/master | head -n 50

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
5. **GCN-specific correctness + micro-opts** - DPP warp reductions, `solve_tri` fix, GCN `supports_op` gate fix (FILL/TRI/DIAG), per-op profiler, ssm-conv / async memcpy / FATTN mask / HIP graph kernel guard / `J_max` padding fixes, Vega20 MMQ tuning (`Q6_K` `I=64 occ2`), native `q8_0` FATTN tile (shadow skip, `GGML_CUDA_FATTN_PATH`).

---

## Commit Inventory (oldest -> newest)

| # | SHA (short) | Date | Title | Category | Upstream? |
|---|-------------|------|-------|----------|-----------|
| 1 | `0beb3a854` | 2026-06-09 | `ggml-hip: support k up to 64 in solve_tri, cap GCN at 64x64` | Fix / GCN cap | fork-original |
| 2 | `562a6764a` | 2026-06-09 | `ggml-hip: DPP-based warp reductions on GCN` | Perf / ISA | fork-original |
| 3 | `d7fe6c82c` | 2026-06-10 | `ggml-hip: GCN Q4_K weight-repacking buffer type (GGML_CUDA_REPACK=1)` | Core infra | fork-original |
| 4 | `00cca237f` | 2026-06-10 | `ggml-hip: repacked int8 MMQ tile GEMM for GCN prefill` | Perf / Prefill | fork-original |
| 5 | `d6b882696` | 2026-06-10 | `ggml-hip: Q5_K/Q6_K/Q8_0 repack + 2D-x-3D broadcast for repacked weights` | Feature | fork-original |
| 6 | `12cf36601` | 2026-06-10 | `ggml-hip: retune the repacked Q8_0 matvec for small dense shapes` | Tuning | fork-original |
| 7 | `c2f7fb648` | 2026-06-10 | `ggml-hip: MUL_MAT_ID / 3D expert-stack support for repacked weights` | Feature / MoE | fork-original |
| 8 | `f80e84a60` | 2026-06-10 | `ggml-hip: half-sub-block work units for the ID-matvec path` | Tuning | fork-original |
| 9 | `349128ba3` | 2026-06-10 | `ggml-hip: direct in-kernel expert routing -> MoE decode parity; repack default-on for GCN` | Perf / MoE | fork-original |
| 10 | `5bd62a7c1` | 2026-06-16 | `cuda: add Q3_K to the GCN weight-repack kernels` | Feature | fork-original |
| 11 | `470fb288c` | 2026-06-17 | `cuda: env-gated per-op GFXPROF profiler` | Debug / Tooling | fork-original |
| 12 | `b739ac2f7` | 2026-07-11 | `cuda(gfx906): batched repacked matvec for concurrent decode` | Perf / Concurrency | cherry-pick |
| 13 | `aceb9bdac` | 2026-07-11 | `cuda(gfx906): batched repacked matvec for dense Q5_K` | Perf / Concurrency | cherry-pick |
| 14 | `7e274ffeb` | 2026-07-11 | `cuda(gfx906): fold per-sequence ssm_out into the batched matvec` | Perf / GDN | cherry-pick |
| 15 | `a62a93461` | 2026-07-11 | `cuda(gfx906): always set device before async memcpy (#21140/#18313)` | Fix | cherry-pick (upstream #21140) |
| 16 | `e95f70b60` | 2026-07-11 | `cuda: bound ssm_conv_long_token_f32 staging read to valid columns` | Fix | cherry-pick (upstream bug from #20128) |
| 17 | `a06580527` | 2026-09-10 | `build: enable GCN repack for MI50 (gfx906) in ROCm build script` | Build | fork-original |
| 18 | `8ed3854ef` | 2026-09-10 | `ggml/cuda: add Vega20 (gfx906) MMQ tuning for MI50` | Tuning / MMQ | fork-original |
| 19 | `b143fd169` | 2026-09-10 | `fattn: clamp Vega20 (gfx906) to nstages=1, occupancy=1 for 64KB LDS` | Fix / FATTN | fork-original |
| 20 | `a1bc8adc6` | 2026-09-10 | `ggml/cuda: port turbo GCN opts - add-id vec4, rope __sincosf, vecdotq fast` | Perf | fork-original turbo port |
| 21 | `d80593382` | 2026-09-10 | `ggml/cuda: add mi50grad single-GPU GCN opts - RMSNorm vec4, dual 4x, GEMV v8, FA v3` | Perf | fork-original mi50 port |
| 22 | `05526fce6` | 2026-09-10 | `ggml/cuda: fix mi50 ops DCE and update fork tracking with MoE bench` | Fix / Docs | fork-original |
| 23 | `671a00b66` (dropped) | 2026-09-10 | `Merge remote-tracking branch 'origin/gfx906/mi50-optimization' into gfx906/mi50-optimization` | Merge | — |
| 24 | `4cf619f7a` | 2026-09-11 | `docs: update FORK_TRACKING snapshot and file impact for 671a00b66` | Docs | fork-original |
| 25 | `2817c3c0d` | 2026-09-11 | `ggml/cuda: add GCN vectorized fused RMSNorm+MUL for gfx906` | Perf / Fusion | fork-original |
| 26 | `4691e0306` | 2026-09-11 | `docs: update FORK_TRACKING for fused RMSNorm+MUL ca6deda1a` | Docs | fork-original |
| 27 | `0b8b50219` | 2026-09-11 | `ggml/cuda: fix Vega20 Q2_0 MMQ config, graph Global capture, test filter` | Fix / Testing | fork-original |
| 28 | `88071e8d8` | 2026-09-11 | `docs: sync FORK_TRACKING SHA for Vega fix` | Docs | fork-original |
| 29 | `4f600e164` | 2026-09-11 | `docs: finalize FORK_TRACKING for Vega fix` | Docs | fork-original |
| 30 | `2b620d7b9` | 2026-09-13 | `ggml/cuda: GCN native q8_0 FATTN tile, FATTN mask fix, HIP graph kernel guard, MMQ padding` | Perf / Fix / FATTN+MMQ+Graph | port from `milpster/gfx906-llama-cpp` |
| 31 | `f3d2967a5` (dropped) | 2026-09-28 | `Merge branch 'gfx906/mi50-rebase-test' into gfx906/mi50-optimization` (rebase onto `4364bf723`) | Merge / Rebase | fork-original |
| 32 | `73d46a563` | 2026-06-16 | `cuda: plain-FMA F32 GEMM for GCN without rocBLAS` | Perf / F32 | port from furnace `d1adf86fb`, sixvolts + Claude Opus 4.8 |
| 33 | `0a86efc3c` | 2026-09-14 | `cuda: skinny-M F32 GEMM for GCN (M <= 16)` | Perf / F32 | port from furnace `8c8ac0ee6`, sixvolts + Claude Fable 5.1 |
| 34 | `01228e730` | 2026-09-23 | `cuda: register-blocked F32 GEMM for GCN, replacing the 16x16 tile` | Perf / F32 | port from furnace `674d0d666`, sixvolts + Claude Opus 5.5 |
| 35 | `daf518ebd` | 2026-09-23 | `cuda: tiled transpose path for dim-0 concat with a transposed src1` | Perf / Concat | port from furnace `19822e03a`, sixvolts + Claude Opus 5.5 |
| 36 | `ee30334e4` | 2026-09-24 | `cuda: F32 matvec for few long rows on GCN (one workgroup per row)` | Perf / F32 | port from furnace `8d74e9e3c`, sixvolts + Claude Opus 5.5, tests dropped |
| 37 | `6e5d7565d` | 2026-09-24 | `cuda: re-enable the repacked gate/up/GLU fusion` | Perf / Fusion | port from furnace `bd43d856a`, sixvolts + Claude Opus 5.5 |
| 38 | `16f036303` | 2026-09-24 | `cuda: fuse the shared-expert gate tail (SIGMOID -> MUL -> ADD)` | Perf / Fusion | port from furnace `efdceeee4`, sixvolts + Claude Opus 5.5, qwen4exp SCALE/SILU parts excluded, tests dropped |
| 39 | `4498864a3` | 2026-09-24 | `cuda: topk-moe - parallel rank selection for 256/512 experts on HIP` | Perf / MoE (reverted next) | port from furnace `4a8dee414`, sixvolts + Claude Opus 5.5 |
| 40 | `51d1cf038` | 2026-10-01 | `ggml/cuda: add overlap check to shared-expert gate fusion` | Fix / Fusion guard | manual, subset of furnace `22a1849de` (sixvolts + Claude Opus 5.5), qwen4exp parts excluded |
| 41 | `5821810bb` | 2026-09-24 | `cuda: topk-moe - drop the rank-selection kernel` | Revert / MoE | port from furnace `a9356d3ed`, sixvolts + Claude Opus 5.5 (rank kernel slower, tests kept) |
| 42 | `4e78727bc` | 2026-10-01 | `ggml/cuda: allow in-place aliasing in shared-expert fusion` | Fix / Fusion guard | manual, subset of furnace `d8e22b45e` (sixvolts + Claude Opus 5.5), qwen4exp parts excluded |
| 43 | `9ce1a327f` | 2026-09-24 | `cuda: GCN F32 matvec with one wave per row (router, GDN beta/alpha)` | Perf / F32 | port from furnace `2658517c7`, sixvolts + Claude Opus 5.5, tests dropped |
| 44 | `a240ea530` | 2026-09-26 | `cuda: F32 row matvec - issue all loads up front for K=2560` | Perf / F32 | port from furnace `c60c156f1`, sixvolts + Claude Opus 5.5 |
| 45 | `b07b7c25c` | 2026-09-27 | `cuda: get_rows - pack short rows into one block` | Perf / GET_ROWS | port from furnace `596c95b54`, sixvolts + Claude Opus 5.5 |
| 46 | `b7aaeb655` | 2026-09-27 | `cuda: HIP top-k radix select - parallel bin search` | Perf / TOPK | port from furnace `47e140b4d`, sixvolts + Claude Opus 5.5 |
| 47 | `fdd1a263b` | 2026-09-27 | `cuda: topk-moe - DPP lane moves in the argmax butterfly on GCN` | Perf / MoE | port from furnace `0a2860296`, sixvolts + Claude Opus 5.5 |
| 48 | `8994ed0cd` | 2026-10-01 | `ggml/hip: Q5_1 repack for GCN` | Perf / Repack | manual port of furnace `4998a31c5`, sixvolts + Claude Opus 5.5, adapted to LEVER dispatch |
| 49 | `280425e22` | 2026-10-01 | `ggml/hip: Q5_1 short-row seg matvec` | Perf / Repack | manual port of furnace `91c65edd7`, sixvolts + Claude Opus 5.5 |
| 50 | `6b9805627` | 2026-10-01 | `ggml/hip: Q8_0 short-row seg and splitk matvec` | Perf / Repack | manual port of furnace `5770fe67b`, sixvolts + Claude Opus 5.5 |
| 51 | `815b69876` | 2026-10-01 | `ggml/hip: Q8_0 flat K320 and rowu K6144 plus GLU one-row wave` | Perf / Repack | manual port of furnace `09115fed0` + `c0070db50` + `a765d59d8`, sixvolts + Claude Opus 5.5 |
| 52 | `68b49f038` | 2026-10-01 | `ggml/hip: Q4_K MoE GLU 16-lane decode` | Perf / Repack | manual port of furnace `112246080`, sixvolts + Claude Opus 5.5 |
| 53 | `769b807b9` | 2026-09-23 | `ggml-hip: fold the Q5_K high bit into int8 at LDS staging in the repacked MMQ` | Perf / MMQ | port from furnace `783fee0db`, sixvolts + Claude Opus 5.5 |
| 54 | `c59ffc780` | 2026-09-23 | `cuda: repack MMQ MoE - hoist ids gather, parallel tile->expert map, occupancy 4` | Perf / MMQ | port from furnace `6bed0852d`, sixvolts + Claude Opus 5.5 |
| 55 | `607b018ac` | 2026-09-23 | `cuda: repack MMQ Q8_0 - b128 X/W LDS planes, K-tail skip, coalesced epilogue` | Perf / MMQ | port from furnace `efa372001`, sixvolts + Claude Opus 5.5 |
| 56 | `7ab57b9a9` | 2026-09-23 | `cuda: repack MMQ Q4_K MoE - 1-wave 64x16 tile, 4x4 per-lane micro-tile` | Perf / MMQ | port from furnace `2d06ccf8e`, sixvolts + Claude Opus 5.5 |
| 57 | `715b4f06b` | 2026-09-23 | `cuda: repack MMQ Q4_K MoE - drop the GGML_CUDA_REPACK_Q4K_OLD A/B path` | Cleanup / MMQ | port from furnace `243742bf5`, sixvolts + Claude Opus 5.5 |
| 58 | `599884f37` | 2026-09-23 | `cuda: repack MMQ Q8_0 dense - W register prefetch, occupancy 3` | Perf / MMQ | port from furnace `4d7f65477`, sixvolts + Claude Opus 5.5 |
| 59 | `7f4503e19` | 2026-09-26 | `cuda: merge dense repacked Q8_0 matvecs that share their activation` | Perf / Batching | port from furnace `782a256af`, sixvolts + Claude Opus 5.5, quantize via local pattern |
| 60 | `017defa1a` | 2026-09-26 | `cuda: group sibling matvecs independent of the token count` | Perf / Batching | port from furnace `063188a8f`, sixvolts + Claude Opus 5.5 |
| 61 | `6125bf33a` | 2026-09-26 | `cuda: merge F32 single-column matvecs that share their input` | Perf / Batching | port from furnace `8a1434bd1`, sixvolts + Claude Opus 5.5, launch block only (grouping already in 60) |
| 62 | `3ac07a4a8` | 2026-10-01 | `ggml/hip: Q8_0 matvec for 2-8 columns` | Perf / Repack | manual port of furnace `5029cad4e`, sixvolts + Claude Opus 5.5, dispatched ahead of LEVERs |
| 63 | `3a966036f` | 2026-10-01 | `ggml/cuda: repack MUL_MAT_ID route and activation cache` | Perf / Cache | manual port of furnace `45f7f7eba`, sixvolts + Claude Opus 5.5, LEVER-2 loop kept |
| 64 | `61359c17d` | 2026-09-27 | `cuda: repacked MoE for 2-8 tokens through the per-slot decode kernels` | Perf / Batching | port from furnace `9d82e1845`, sixvolts + Claude Opus 5.5, LEVER-2 loop replaced by single-grid + expansion |
| 65 | `d3a3b7a52` | 2026-09-27 | `cuda: faster 2-8 token forwards (split-K / short-row Q8_0, MoE GLU, F32)` | Perf / Batching | port from furnace `eadd8281a`, sixvolts + Claude Opus 5.5, GLU adapted to local quantize |
| 66 | `f8f51d0d0` | 2026-09-27 | `cuda: small-batch paths up to 16 tokens` | Perf / Batching | port from furnace `acf92306c`, sixvolts + Claude Opus 5.5 |
| 67 | `10d1de8ba` | 2026-10-01 | `ggml/cuda: Q8_0 fused GLU and per-graph activation cache` | Perf / Fusion+Cache | manual port of furnace `8ac1bc756`, sixvolts + Claude Opus 5.5 |
| 68 | `496ce80d4` | 2026-09-27 | `cuda: fold one-column broadcast slices into the Q8_0 nc matvec; nc load prefetch` | Perf / Repack | port from furnace `9791f13e8`, sixvolts + Claude Opus 5.5, broadcast-fold deferred (LEVER-4 kept) |
| 69 | `2e9662724` | 2026-09-28 | `cuda: flatten contiguous src1 slices in repacked mul_mat` | Perf / Batching | port from furnace `8bb172726`, sixvolts + Claude Opus 5.5, supersedes LEVER-4 |
| 70 | `875d429a1` | 2026-09-28 | `cuda: single-token fusions for a few columns` | Perf / Fusion | port from furnace `533d05dca`, sixvolts + Claude Fable 5.1, hc_up kernel excluded (qwen4exp) |
| 71 | `2b56a7e65` | 2026-10-01 | `docs: update FORK_TRACKING for repack tiling ports with attribution` | Docs | fork-original |
| 72 | `bf3d8b4aa` | 2026-10-01 | `tests: add SIGMOID_MUL_ADD whole-graph cases for shared-expert fusion` | Testing | fork-original |
| 73 | `5e601af03` | 2026-10-05 | `cuda: MoE expert weight-once few-token gate/up (dedup)` | Perf / MoE | manual port of furnace `99e97eafc`, sixvolts + Claude Opus 5.5 |
| 74 | `5e601af03` | 2026-10-05 | `cuda: weight-once 2..16-col dense KQ matvec` | Perf / Repack | manual port of furnace `0934af928`, sixvolts + Claude Opus 5.5, Q5_K/Q6_K <=8 keep LEVER ncols |
| 75 | `5e601af03` | 2026-10-05 | `cuda: Q6_K int8 expansion at MMQ LDS staging` | Perf / MMQ | manual port of furnace `15d76b19d`, sixvolts + Claude Opus 5.5 |
| 76 | `5e601af03` | 2026-10-05 | `cuda: residual ADD into RMS_NORM -> MUL (+q8_1) fusion` | Perf / Fusion | manual port of furnace `8a8f44363`, sixvolts + Claude Opus 5.5 |
| 77 | `5e601af03` | 2026-10-05 | `cuda: split-K for skinny F32 GEMMs` | Perf / F32 | manual port of furnace `c5bc7f986`, sixvolts + Claude Opus 5.5 |
| 78 | `5e601af03` | 2026-10-05 | `cuda: rank-select top-k for 256/512 experts` | Perf / TOPK | manual port of furnace `bafaf7d73`, sixvolts + Claude Fable 5.1 |
| 79 | `5e601af03` | 2026-10-05 | `cuda: MoE decode chain (down+reduce, q8 emit, LDS nc, loads-first)` | Perf / MoE | manual port of furnace `69db7bf5d`, sixvolts + Claude Fable 5.1, cur_cgraph added for emit |
| 80 | `5e601af03` | 2026-10-05 | `cuda: Q4_0/IQ repack, Q3_K relabel, narrow MMQ tiles` | Perf / Repack | manual port of furnace `7f70fff53`, sixvolts + Claude Fable 5.1, tests + test entry points skipped |
| 81 | `5e601af03` | 2026-10-05 | `cuda: loads-first dense KQ/Q8_0 matvecs + Q4_K GLU` | Perf / Repack | manual port of furnace `b8eb8050f`, sixvolts + Claude Fable 5.1 |
| 82 | `5e601af03` | 2026-10-05 | `cuda: MMQ 8B-aligned staging + padded LDS rows, expert guards` | Perf / MMQ | manual port of furnace `ae7fdb2d7`, sixvolts + Claude Fable 5.1 |
| 83 | `5e601af03` | 2026-10-05 | `cuda: KQ thread-packed experts, Q8 dispatch by K, Q8 hoist opt-in` | Perf / Repack | manual port of furnace `7bac8169e`, sixvolts + Claude Fable 5.1, tests skipped |
| 84 | `d2685bff2` | 2026-11-05 | `ggml/hip: BF16 -> F16 upload conversion repack for GCN` | Perf / Repack | fork-original (resolves the BF16 WIP section below) |
| 85 | `5a2255fcc` | 2026-11-06 | `cuda: bounds-check the MoE weighted-reduction fused-node walk` | Fix / Graph | port from `mx-llama.cpp ef7246cd9` |
| 86 | `2aa788490` | 2026-11-06 | `docs: gfx906 ISA optimization notes + build-object register audit tool` | Docs / Tooling | fork-original |
| 87 | `ffc7e39c1` | 2026-11-06 | `cuda: lds_nc column groups of 4 for NC = 6..8; rms_norm_mul_rope SGPR slim` | Perf / Repack + Rope | fork-original (restructure of the `69db7bf5d` port + notes sec. 12 SGPR slim); validated same day, see below |
| 88 | this commit | 2026-11-06 | `docs: lds_nc column-group + rope WIP validation results` | Docs | fork-original |

> `git log --reverse --oneline origin/master..HEAD` reproduces this order. Furnace ports keep original `Author: sixvolts <rigel@sixvolts.org>` and `Co-Authored-By: Claude` trailers via cherry-pick; manual guards (`51d1cf038`, `4e78727bc`) credit the furnace source commit in the body.
>
> Filter applied (donor `furnace/gfx906-perf` at `fc353a5ac`, range `905021dba..7bac8169e`): A: multigpu excluded (none new in range). B: non-opt excluded (`300fdfe3b` server `/props` config, `746887a5c` tests-only bench shapes, `47f6e5b8a` server `_exit`). C: GDN/qwen4exp arch excluded (`058d97848` GDN gather elision fix, `50798f69b` GDN gate/softplus/beta fusion, `a2207dd89` GDN q/k norm pairs, `3a30ac2c2` recurrent conv-step fusion - local has no GDN fusion base). D: only `sixvolts` + `Claude` commits ported. Earlier filter (donor at `905021dba`): multigpu (`TurboPrefill`, `NO_PEER_COPY`, layer-split), README/tests-only/server/MTP/`llama` scheduler staging, `QSA`/sparse/`hc`/`dsv4` arch, `cohere2` handler, build switches, `solve_tri` scope, bounce fix; redundant skipped (`b3c18e502`, `2ffa92b3c`). Prior repack-tiling deferrals resolved via manual LEVER-adapted ports. qwen4exp arch still deferred per request.

---

## File-level Impact (vs `7951f601e`)

Raw: 34 files, 11804 insertions(+), 200 deletions(-) vs `origin/master` (2026-11-06; full
per-file listing via `git diff --stat origin/master..HEAD`). The 11 furnace ports that were
uncommitted on 2026-10-05 are a single commit (`5e601af03`).

Since 2026-11-05: the BF16 repack WIP landed as `d2685bff2` (`repack-gcn.cu` 6383 -> 6596,
`ggml-cuda.cu` relabel + guard), the MoE weighted-reduction guard is `5a2255fcc`
(`ggml-cuda.cu` -> 1177), and `2aa788490` adds the ISA notes and the audit tool. `ffc7e39c1`
then restructures the `lds_nc` Q8_0 kernel (column groups of 4 for NC 6..8, repack-gcn.cu
6596 -> 6610) and slims the fused `rms_norm_mul_rope` kernel args (rope.cu 75 -> 129 stat
lines, i32 strides + plain mul-modulo).

### What each file does

| File | Lines | Role in fork |
|------|-------|--------------|
| `ggml/src/ggml-cuda/repack-gcn.cu` | +6610 (new) | Entire repacked matvec/MMQ + MoE + broadcast machinery + Q5_1/Q4_0/IQ repack + BF16-as-f16 upload; `lds_nc` column groups of 4 for NC 6..8 (`ffc7e39c1`). ~90% of fork delta. |
| `ggml/src/ggml-cuda/repack-gcn.cuh` | +85 (new) | Public API for repack buffer type and `ggml_cuda_mul_mat_*_repacked`. |
| `ggml/src/ggml-cuda/mmq-config-vega.cuh` | +282 (new) | Vega20 MMQ config table ported from RDNA2 with occupancy 1, `I=128` (`Q6_K I=64 occ2` after `2b620d7b9`), `launch_bounds 256,1` for 64KB LDS/W64 dp4a. +12 for Q2_0 fix (J_best=0). |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | +1177 | Repack buft hook + `mi50_force_link` + `FATTN`/`MMQ`/`REPACK` dispatch; `GFXPROF`, `set_device` fix (#21140), `Global` graph capture, HIP `kernel_funcs` graph guard (`2b620d7b9`); furnace ports: plain-FMA/skinny/register-blocked F32 GEMM, F32 row matvecs + multi-nc, transpose dispatch, shared-expert fusion + guards, GLU re-enable; MoE weighted-reduction walk bounds + zero-token alloc deps (`5a2255fcc`, ported from mx-llama.cpp); bf16 -> f16 relabel (`d2685bff2`). |
| `ggml/src/ggml-cuda/common.cuh` | +143/-1 | DPP warp reductions (`GCN` only), `V_DOT2` gating, `MATRIX_ROW_PADDING`, `CC_GCN`/`VEGA20`, `W64`, `cuda_graph` `kernel_funcs` + `unstable_disabled` (`2b620d7b9`). |
| `ggml/src/ggml-cuda/norm.cu` | +164 | `GCN` `RMSNorm` `float4` 4× `vec` (`tid*4`), `block_reduce` DPP, `rsqrtf` — `elementwise_v2.hip:62` port + `rms_norm_f32_gfx906_fused<256/1024>` `fused RMSNorm+MUL(+ADD)` `1` launch vs `2` (`2817c3c0d`). |
| `ggml/src/ggml-cuda/add-id.cu` | +106/-11 | Turbo `float4` `vec4` + `contiguous` fast paths for `MoE` `ADD_ID` (`ne0%4` aligned). |
| `ggml/src/ggml-cuda/rope.cu` | +90/-39 | `GCN` `__sincosf`+`__logf` in `rope_yarn`; round-2 `RMS_NORM_MUL_ROPE` M-RoPE fusion (`963dc1514` port, `ed4f655a6`); kernel SGPR slim - i32 strides + plain mul-modulo (`ffc7e39c1`), tg128 +0.9% on 0.8B BF16, validated 2026-11-06 |
| `ggml/src/ggml-cuda/vecdotq.cuh` | +59 | `GCN` `memcpy` `b1/b2_fast`, `v_perm` `MXFP4` `8-entry`, `GGML_GCN_VEC_DOT_MXFP4` `Q8_0` fast. |
| `ggml/src/ggml-cuda/mmq.cuh` | +19/-2 | Vega `MMQ` `GCN`/`__gfx906__` dispatch; `J_max` fallback padding (`2b620d7b9`). |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | +17/-1 | Clamps `FATTN` `nstages=1` `occupancy=1` for `GCN` `64KB LDS`. |
| `ggml/src/ggml-cuda/fattn-common.cuh` | +93 | `GGML_USE_HIP` `FATTN` auto native decision (`GGML_CUDA_FATTN_PATH`) + per-device `free_mem` probe (`2b620d7b9`). |
| `ggml/src/ggml-cuda/fattn-tile.cuh` | +330/-2 | GCN native `q8_0` `half2/float` loaders, SW-pipelined `KQ` prefetch, 16-wide narrow tile + occupancy-tuned 256 config (`2b620d7b9`). |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | +6/-1 | Mask stride `nb31/2` fix for sequence-split `ne11` slice vs full mask rows (`2b620d7b9`). |
| `ggml/src/ggml-cuda/fattn.cu` | +63 | GCN native `q8_0`/`f16+q8_0` predicates + alloc skip + `VEC` vs `TILE` routing for quantized KV (`2b620d7b9`). |
| `ggml/src/ggml-cuda/solve_tri.cu` | +62/-47 | `MAX_K_FAST` 64 + `GCN` cap `64×64`. |
| `ggml/src/ggml-cuda/ssm-conv.cu` | +9/-1 | Bound `ssm_conv_long_token_f32` OOB fix. |
| `ggml/src/ggml-cuda/concat.cu` | +78 | Tiled `dim-0` transpose path from furnace `19822e03a` (sixvolts + Claude Opus 5.5). |
| `ggml/src/ggml-cuda/getrows.cu` | +49 | Short-row packing from furnace `596c95b54` (sixvolts + Claude Opus 5.5). |
| `ggml/src/ggml-cuda/top-k.cu` | +171 net | Radix-select bin search from furnace `47e140b4d` (sixvolts + Claude Opus 5.5); round-2 deterministic column-order radix top-k (`01a6f0ce6`). |
| `ggml/src/ggml-cuda/topk-moe.cu` | +44/-12 net | DPP argmax moves from furnace `0a2860296` (sixvolts + Claude Opus 5.5); rank kernel added then dropped (`4a8dee414`/`a9356d3ed`). |
| `ggml/src/ggml-cuda/unary.cu` / `unary.cuh` | +21/+4 | `sigmoid_mul_add` kernel for shared-expert fusion from furnace `efdceeee4` (sixvolts + Claude Opus 5.5). |
| `ggml/src/ggml-cuda/mmq.cu` | +13 | NVFP4 `y_scale` + `ids_dst` `+J_max` / `+128` padding for `J`-wide overread (`2b620d7b9`). |
| `build-llamacpp-rocm.sh` | +25 (new) | ROCm 7.2.4 `HIPCXX` `GFX906` isolation (the inert `-D REPACK` lines dropped in `961362c5c`). |
| `MI50_ISA_OPTIMIZATION_NOTES.md` | +438 (new) | gfx906 ISA digest: verified hardware config, hazard tables, dot/packed ops, applied register/LDS fixes and audit backlog (sections 0-12). |
| `amdgcn-audit.py` | +191 (new) | Per-kernel VGPR/SGPR/LDS audit from HIP code-object metadata in the build objects (no ROCm tooling or GPU needed). |
| `FORK_TRACKING.md` | +774 (new) | This doc. |
| `.gitignore` | +10 | `results.*` `rocprof` traces + `build-llamacpp-rocm.sh`. |
| `src/llama-model.cpp` | +6/-4 | `make_gpu_buft_list` `extra_bufts` precedence + 2D→3D broadcast. |
| `tests/test-backend-ops.cpp` | +59 | Substring `matches_filter` plus furnace `512`-expert topk cases (`4a8dee414`); per-commit kernel tests dropped to keep diff small. |

---

## Detailed Change Log

The 2026-06..09 rounds compressed to one row each; the commit bodies (`git show <sha>`) keep the
full why/change/files text. Rounds from 2026-09-13 on have their own sections below (FATTN, furnace
rounds 1-2, bf16 repack, ISA notes).

| # | SHA | Change | Result |
|---|-----|--------|--------|
| 1 | `0beb3a854` | solve_tri `MAX_K_FAST=64` + GCN cap 64x64 (Qwen3-Next delta-net prefill needs k=64) | fixes delta-net prefill |
| 2 | `562a6764a` | DPP warp reductions (`quad_perm`/`row_ror`/`ds_swizzle`, float/float2/int) instead of shfl -> `ds_bpermute` LDS roundtrips | gemma4 31B tg128 +4.7% |
| 3 | `d7fe6c82c` | GCN Q4_K repack buffer type: 3-plane host repack, pow2 `MATRIX_ROW_PADDING` HBM fix, 2D f32 MUL_MAT only | matvec ~58% -> ~89% HBM BW |
| 4 | `00cca237f` | Repacked int8 MMQ tile GEMM (64x64, BK=4, 4x4 micro-tile, LDS-staged sub-blocks), `GGML_CUDA_REPACK_NO_MMQ` fallback | gemma4 pp512 +14.2%, tg128 22.29 -> 26.57 |
| 5 | `d6b882696` | Q5_K/Q6_K/Q8_0 repack planes; 2D weight x 3D activation broadcast (`supports_op`) for GDN hybrid | pure-Q8_0 0.8B pp512 +43% |
| 6 | `12cf36601` | Q8_0 matvec retune: 4-wave blocks, 16-weight half-sub-block units | 0.8B tg128 222.7 -> 231.0 |
| 7 | `c2f7fb648` | MUL_MAT_ID 3D expert-stack: per-expert slabs, HAS_IDS matvec/MMQ, 16-token MoE tiles | qwen3.6 pp512 +12.0% (tg still gated) |
| 8 | `f80e84a60` | Half-sub-block units for ID matvec (Q4_K/Q5_K split `deff*sx`, Q6_K folded -32) | tg128 +0.8% |
| 9 | `349128ba3` | In-kernel expert routing for n_tokens<=8 (compaction only >8), fused gate+up GLU; repack default-on for GCN | qwen3.6 tg128 parity 89.94 vs 89.86 |
| 10 | `5bd62a7c1` | Q3_K repack: 2-bit lo + 1-bit hi plane, Q6_K-style scale+bias | Q3_K MoE/dense via dp4a |
| 11 | `470fb288c` | `GFXPROF=1` per-op HIP-event profiler (distro ROCm ships no rocprof) | tooling |
| 12 | `b739ac2f7` | Batched Q6_K repacked matvec (`ncols` kernels) + MoE per-token loop for n_tokens<=8 | batched decode +34-72% |
| 13 | `aceb9bdac` | Batched Q5_K repacked matvec (for fused `attn_qkv`) | batched decode +7-9% |
| 14 | `7e274ffeb` | Fold per-sequence GDN `ssm_out` [d,1,n_seqs] into one ne11=n_seqs matvec; adds `REPACK_TRACE`/`REPACK_NOFOLD` | batched decode +3-6% |
| 15 | `a62a93461` | Always set device before async memcpy (upstream #21140/#18313) | stale-device fix |
| 16 | `e95f70b60` | Bound `ssm_conv_long_token_f32` staging read (upstream bug #20128) | spec-decode memfault fix |
| 17 | `a06580527` | Build script repack defaults (dropped in `961362c5c`: runtime env took over) | - |
| 18 | `8ed3854ef` | Vega20 MMQ config: new `mmq-config-vega.cuh`, occupancy 1, `I=128`, `launch_bounds(256,1)` | MMQ works on 64KB LDS |
| 19 | `b143fd169` | FATTN MMA clamp nstages=1 / occupancy=1 for GCN (64KB LDS W64 dp4a) | kills FATTN spill |
| 20 | `0b8b50219` | Q2_0 MMQ config rows (fixes `J_best=0` abort), `Global` graph capture, substring `matches_filter` | MUL_MAT 1288/1288 OK |
| 21 | `2b620d7b9` | Milpster port batch: native q8_0 FATTN tile + SW prefetch, FATTN mask `s31` fix, HIP graph kernel-func guard, MMQ `Q6_K I=64 occ2` + `J_max`/NVFP4 padding | see FATTN section below |

---

## Benchmark vs upstream master (2026-10-19)

`master` (`7951f601e`) was built in a separate worktree with the same ROCm flags
(`-DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGPU_TARGETS=gfx906`, no repack) and compared on the
same MI50. `llama-bench -ngl 99`, r=3-5, two rounds:

| Model | Test | master | fork | delta |
|-------|------|--------|------|-------|
| Qwen3.5-9B UD-Q4_K_XL | pp512 | 757.3 | 963.0 | **+27 %** |
| Qwen3.5-9B UD-Q4_K_XL | tg128 | 65.1 | 83.2 | **+28 %** |
| Qwen3.5-9B UD-Q4_K_XL | pp32 | 354.1 | 415.9 | +17 % |
| Qwen3.5-9B UD-Q4_K_XL | tg16 | 65.2 | 80.1 | +23 % |
| Qwen3.5-0.8B BF16 | pp512 | 4356.2 | 4767.3 | +9 % |
| Qwen3.5-0.8B BF16 | tg128 | 183.8 | 190.4 | +4 % |

The quantized win comes from the repacked MMQ kernels (`mul_mat_q` Q4_K 500.5 ms ->
`mmq_gemm_q4k_repacked` 411.3 ms, Q5_K 351.5 -> 245.4 ms, Q6_K 132.2 -> 109.9 ms for a 512
token prefill).

### F32 GEMM: hipblas is the default (2026-10-19)

ROCm 7.1 shipped no gfx906 Tensile kernels, so the fork first replaced hipblas with its own GCN
F32 GEMM; on ROCm 7.2.4 hipblas has them and wins. `rocprofv3` on the 0.8B BF16 prefill (same 366
GEMM dispatches): hipblas/Tensile 138.1 ms vs fork `gcn_f32_gemm_tn_rb`/`_splitk`/`_skinny` 234.4 ms
GPU time. 0.8B BF16 pp512: master 4356.2, fork hipblas default 4767.3, fork custom path 3347.3.
For the quantized model the flip is neutral (9B pp512 963.0 vs 951.0, tg128 83.2 vs 80.3) and PPL
is unchanged (`6.3304`, repack on). Custom path stays opt-in: `GGML_CUDA_GCN_F32_GEMM=1`.

---

## Rebase 2026-10-19 (onto upstream master `7951f601e`)

The fork history carried two copies of the first 29 commits (pre-`rebase-test` set on `ec91ab5ad`,
rebased set on `4364bf723`, joined by merge `f3d2967a5`); the tree was identical, so the redundant
set (32 commits), both merges and the now-empty `d87a4cd04` were dropped and the branch rebuilt as
`master` + 32 + 50 post-merge commits. Conflict resolutions (upstream side kept unless noted): GCN
`rms_norm` kernels moved into the upstream `rms_norm_f32_dispatch`; fork `J_max` MMQ padding dropped
(upstream `src1_q8_1_padding`/`J_best` supersedes, NVFP4 `y_scale +128` kept); `__vsub4` hunks
dropped (upstream PR #29478); repack buft extended with the new `alloc_buffer_n`/`get_alloc_size_n`
interface slots; upstream `should_use_mmvf` warp arg and `ggml_cuda_fusion_disabled()` wrap the
re-added GCN blocks. The full conflict table is in the git history of this file (pre-trim revision);
expected conflict files for the next rebase: see Rebase / Merge Checklist.

---

## Correctness Notes

- All repacked matvec/MMQ variants validated via `wikitext PPL` vs GEMM/canonical within noise (tokens in commit logs).
- `ub=1 PPL` checks: Q5_K/Q6_K, Q8_0 (5.07 vs 5.12), MoE ID paths, FATTN clamp (no PPL regression cited; LDS correctness).
- `n_tokens/ne11==1` paths collapse to original launch -> single-stream byte-unchanged for concurrent-decode batching commits.
- Env gates for debugging without rebuild: `GGML_CUDA_REPACK`, `GGML_CUDA_REPACK_Q8_0`, `GGML_CUDA_REPACK_MOE`, `GGML_CUDA_REPACK_NO_MMQ`, `REPACK_TRACE`, `REPACK_NOFOLD`, `GFXPROF`, `GGML_CUDA_FATTN_PATH` (`auto`/`force_convert`/`force_native` for `2b620d7b9` native `q8_0` tile).
- `test-backend-ops` on `gfx906` (`HIP` `ROCm 7.2.4`): after `Vega Q2_0` fix, `MUL_MAT` `1288/1288 OK` (`ROCm0`), `q2_0` `47/47`, `q4_K` `64/64`, `q5_K` `29/29`, `q6_K` `13/13`, `RMS_NORM` `51/51 OK` (fused + vectorized). Prior `J_best=0` abort blocked repack `Q4_K/Q5_K/Q6_K` coverage. `2b620d7b9` adds `mask s31` fix, `graph kernel guard` (MUL_MAT_ID quantize swap), `MMQ` `J_max`/`NVFP4` padding — no PPL delta, graph stable on spec-decode fan-out.
- 2026-10-05 ports: full `build-llamacpp-rocm.sh` clean; `test-backend-ops -o MUL_MAT` / `-o RMS_NORM`
  / `-o TOPK` / `-o ADD` all `OK`; `llama-bench` Qwen3.5-9B UD-Q4_K_XL `-ngl 99 -p32 -n16` runs
  (`pp32 ~385`, `tg16 ~77`). No new `tests/` files (repo policy; furnace `test-repack-host`/
  `test-repack-bench` hunks skipped). The env gates these ports added are all in the Build &
  Runtime Flags table.
- 2026-10-19 F32 GEMM default flip: `GGML_CUDA_GCN_F32_GEMM=1` re-enables the hand-written GCN F32 GEMM; by default hipblas handles F32 GEMM (faster on ROCm 7.2.4). PPL unchanged (`6.3304` on Qwen3.5-9B UD-Q4_K_XL, repack on); see "Benchmark vs upstream master".
- 2026-10-19 rebase validation (1xMI50 gfx906, ROCm 7.2.4): full `build-llamacpp-rocm.sh` clean (no errors); `test-backend-ops -o MUL_MAT` `2450/2450 OK` on `ROCm0` (upstream grew the matrix since the last rebase), `-o RMS_NORM` / `-o TOPK` / `-o ADD` / `-o MUL_MAT_ID` / `-o FLASH_ATTN_EXT` / `-o GET_ROWS` / `-o UNARY` / `-o SOFT_MAX` all `OK`. `llama-bench` Qwen3.5-9B UD-Q4_K_XL `-ngl 99 -p 32 -n 16 -r 2`: `pp32 392.59 ± 63.02`, `tg16 78.82 ± 2.25` (pre-rebase reference: `pp32 ~385`, `tg16 ~77`). PPL (wikitext `wiki.test.raw`, 2 chunks, ctx 1024, Qwen3.5-9B UD-Q4_K_XL): repack on `4.3965 / 6.3304` vs `GGML_CUDA_REPACK=0` `4.4011 / 6.3036` (within noise). `GGML_CUDA_FATTN_PATH=force_convert` and `force_native` with `-ctk q8_0 -ctv q8_0`: identical `PPL 6.3199`.

---

## Build & Runtime Flags

| Flag / Env | Default | Effect |
|------------|---------|--------|
| `GGML_CUDA_REPACK` (runtime env) | ON for GCN after `349128ba3` (`off`/`false`/`no`/`0` opts out) | Master switch for the repack buffer type. All `GGML_CUDA_REPACK*` switches are runtime env vars - no cmake option consumes them (`961362c5c` dropped the inert `-D` lines from `build-llamacpp-rocm.sh`). |
| `GGML_CUDA_REPACK_Q3_K` / `_Q4_K` / `_Q5_K` / `_Q6_K` | ON (`0`/`off`/`false`/`no` opts out) | Per-type K-quant repack switches (furnace `9a548180e` + `0c63b6968`). Q3_K is stored/run as Q6_K planes but the switch follows the tensor type: `_Q6_K=0` leaves Q3_K on the Q6_K kernels, `_Q3_K=0` does not touch Q6_K. `_Q4_K=0` also gives up the fused gate/up GLU decode. A non-repacked tensor runs the stock MMQ / mmvq path. |
| `GGML_CUDA_REPACK_Q8_0` | ON for GCN since `961362c5c` (2026-11-05, `off` opts out); opt-in before | Spark-X2.6-4B-Q8_0 repacked pp512 `+57%` / tg128 `+6%` vs stock - the retuned decode kernels won, no retune pending anymore |
| `GGML_CUDA_REPACK_Q5_1` | ON for GCN since `961362c5c` (`off` opts out); opt-in before | Q5_1 repack, K multiple of 32 (`qwen4exp ffn_down_exps` has K=640) |
| `GGML_CUDA_REPACK_BF16` | ON, committed `d2685bff2` (`off` opts out); `GGML_CUDA_CUBLAS_COMPUTE_TYPE=bf16` also opts out | 2D MUL_MAT BF16 weights stored as f16 in the repack buffer (`set_tensor` converts bf16 -> f16 row-major); `ggml_cuda_mul_mat` relabels BF16 -> F16 (thread_local tensor) so the stock f16 mmvf/Hgemm path runs. `token_embd` unaffected (loader probe maps it to GET_ROWS, which the repack buft gate rejects, so it stays in the default buffer). |
| `GGML_CUDA_REPACK_MOE=0/1` | OFF until `349128ba3` | Q4_K/Q5_K half-block MoE path |
| `GGML_CUDA_REPACK_NO_MMQ=1` | unset | Force dequant+GemmEx fallback for `ne11>1` repacked |
| `REPACK_TRACE=1` | unset | Log `mul_mat` dispatch shapes |
| `REPACK_NOFOLD=1` | unset | Disable `ssm_out` `ne12` fold |
| `GFXPROF=1` | unset | HIP-events per-op profiler |
| `GGML_CUDA_FATTN_PATH=auto\|force_convert\|force_native` | `auto` | FATTN quantized-KV path selector (`2b620d7b9`): `auto` probes free mem vs `f16` shadow (20% headroom) per device, `force_*` overrides for bench. |
| `GGML_CUDA_REPACK_Q4_0=1` / `GGML_CUDA_REPACK_IQ=1` | ON (0 opts out) | Q4_0 / IQ4_NL / IQ4_XS / IQ3_S repack, 2D weights only (`7f70fff53` port) |
| `GGML_CUDA_Q3K_RELABEL=1` | ON (0 keeps 3-bit kernels) | Q3_K stored/computed as Q6_K planes (exact) |
| `GGML_CUDA_NO_MMQ_NARROW=1` | unset | Restore 64-wide MMQ token tile for <=48 columns |
| `GGML_CUDA_NO_KQ_NC=1` | unset | KQ/nibble 2..16-col matvec back to MMQ/LEVER path |
| `GGML_CUDA_NO_MOE_DEDUP=1` | unset | Few-token MoE expert-dedup GLU off |
| `GGML_CUDA_GCN_F32_GEMM=1` | unset (hipblas is the default) | Use the hand-written GCN F32 GEMM instead of hipblas |
| `GGML_CUDA_NO_F32_SPLITK=1` | unset | Skinny F32 GEMM split-K off (only with `GGML_CUDA_GCN_F32_GEMM=1`) |
| `GGML_CUDA_NO_TOPK_RANK=1` | unset | 256/512-expert rank top-k off |
| `GGML_CUDA_NO_ADD_NORM_FUSION=1` | unset | Residual ADD into RMSNORM+MUL off |
| `GGML_CUDA_NO_NORM_MROPE_FUSION=1` | unset (fusion on) | RMSNORM+MUL+ROPE M-RoPE / IMRoPE modes back to separate rope + set_rows (furnace `963dc1514`) |
| `GGML_CUDA_DOWN_REDUCE_R=N` / `GGML_CUDA_Q5_1_DOWN_R=N` / `GGML_CUDA_KQ_DOWN_R=N` | 4 (0 off) | Fused down+reduce / pack kernel row groups |
| `GGML_CUDA_NO_Q8_LDS_NC` / `NO_Q8_NC_CHUNK` / `NO_Q8_MULTI_UNROLL` / `NO_Q8_GLU_UNROLL` | unset | Q8_0 staged-nc / chunk / unroll variants off |
| `GGML_CUDA_NO_Q8_LDS_ROWS=1` | unset (path on) | Q8_0 2..4-column activation-in-LDS matvec back to `_nc` (furnace `01f700b37` + `013ad9e02`) |
| `GGML_CUDA_Q8_LDS_ROWS_ROUNDS=N` | auto (1 round, 2 from 64K rows) | Force the resident-block round count of `q8_lds_rows` |
| `GGML_CUDA_NO_KQ_HOIST=1` / `GGML_CUDA_NO_Q4K_FENCE=1` | unset | KQ hoisted loads / Q4_K sched fence off |
| `GGML_CUDA_Q8_HOIST=1` (opt-in) / `GGML_CUDA_Q8_ROWU=all\|r2\|old` | unset / default rowu@6144 | Q8_0 hoist A/B; row-unrolled dispatch select |
| `GGML_CUDA_ALLREDUCE=internal\|nccl` | auto | Multi-GPU all-reduce backend choice |
| `GPU_TARGETS=gfx906` | - | HIP compile target |

Build via: `./build-llamacpp-rocm.sh` (requires `/opt/rocm-7.2.4`).

---

## What Needs Upstream Attention

| Area | Status | Next step |
|------|--------|-----------|
| `ssm_conv_long_token_f32` OOB fix | Upstream bug (#20128), cherry-picked here; should be PR'd if not already | Check `upstream/master` - if missing, open PR with `e95f70b60` diff |
| `ggml_cuda_set_device` fix | Upstream #21140/#18313, already merged upstream via other PRs - verify not needed after rebase | Drop on next merge-base bump if upstream contains it |
| `FATTN` mask `s31` fix | Upstream bug (mask stride vs sequence-split slice), fixed here `2b620d7b9` | PR upstream if still missing - see `fattn-vec.cuh:111` |
| gfx906 packed-byte subtraction | Upstream merged PR #29478 (`__vsub4` + VMEM sched barrier) - the fork port `d87a4cd04` was dropped in the 2026-10-19 rebase | Nothing left to port |
| `MUL_MAT` `J_max`/`NVFP4` padding | Upstream bug (stream-k tail overread). Upstream reworked the padding (`src1_q8_1_padding` + `J_best`) and kept the `NVFP4` `y_scale` `+128` guard, so the fork hunk was dropped in the 2026-10-19 rebase | Nothing left to port - re-check at the next rebase |
| `HIP` graph `hipGraphExecUpdate` kernel guard | HIP bug (launch-bounds not validated on func change) | Keep; consider upstreaming `common.cuh:1411` + `ggml-cuda.cu:2676` |
| Repack buffer type | Fork-only; invasive (~2.5k LOC). Upstreaming would need feature-flag review | Keep fork-local, rebase-friendly, avoid touching hot paths when gated off |
| FATTN GCN clamp | Fork-only; correctness on 64KB LDS | Validate against upstream FATTN changes at each rebase |
| FATTN GCN native `q8_0` tile | Fork-only; GCN 64KB LDS compact vs `f16` shadow | Keep; `GGML_USE_HIP` gated, validated via `ggml_cuda_fattn_use_native_tile` probe |
| Vega20 MMQ table | Derived from RDNA2 occupancy 1; `Q6_K I=64 occ2` after `2b620d7b9` (`+4.2%` PP) | Keep; upstream MMQ tables live in `mmq.cuh` and change often - rebase conflict expected |
| DPP warp reductions | GCN-only `#ifdef`; no CDNA/RDNA impact | Keep; re-check if upstream refactors `common.cuh` reductions |
| `MoE` weighted-reduction empty-tail | Fixed here `5a2255fcc` (node-walk bounds + zero-token alloc deps, ported from mx-llama.cpp `ef7246cd9`) | Check if upstream still walks past `n_nodes` on empty last-layer chains; PR if missing - see `ggml-cuda.cu:3968` |

---

## Rebase / Merge Checklist

1. Update `origin/master` (or `upstream/master`):
   ```bash
   git fetch origin
   git log --oneline 7951f601e..origin/master | wc -l   # how many upstream commits since base
   ```
2. Rebase fork branch:
   ```bash
   git branch backup/mi50-pre-rebase-$(date +%Y%m%d)   # keep the old HEAD reachable
   git checkout gfx906/mi50-optimization
   git rebase origin/master   # or `git rebase upstream/master` then `git push --force-with-lease origin gfx906/mi50-optimization`
   ```
    Expected conflicts: `ggml/src/ggml-cuda/ggml-cuda.cu`, `common.cuh`, `norm.cu`, `mmq.cu`, `mmq.cuh`, `topk-moe.cu`, `vecdotq.cuh`, `repack-gcn.cu` (buft iface), `src/llama-model.cpp`.
3. Verify no silent behavior change when repack is off (`GGML_CUDA_REPACK=0` PPL vs upstream) and when `GGML_CUDA_FATTN_PATH=force_convert` matches non-native path.
4. Bump **Base / merge-base** and **Commits ahead** in this doc.
5. Update **File-level Impact** (`git diff --shortstat origin/master..HEAD`) and this checklist if upstream adds new files (e.g., `fattn.cu` tile changes).
6. Re-run `./build-llamacpp-rocm.sh` and `test-backend-ops` (`-o MUL_MAT`, `-o RMS_NORM`, `-o TOPK`) on `gfx906` before pushing.

---

## Branch Topology

```
upstream/master (ggml-org)  ──────────────────────►  7951f601e  (2026-10-19, current base)
origin/master (this repo)      ──────────────────────►  7951f601e  (synced, no fork commits)
origin/gfx906/mi50-optimization  ─── 7951f601e ──► d2685bff2  (published, 2026-11-05; 5a2255fcc..ffc7e39c1 unpushed)
backup/mi50-pre-rebase-20261018  ─── 4364bf723 ──► 87f850290  (same as pre-rebase backup)
local gfx906/mi50-optimization   ─── 7951f601e ──► 99 commits ──► ffc7e39c1 + docs
```

---

## Links & References

- Upstream issues/PRs cited: `#20024`, `#20128`, `#21140`, `#21383`, `#28330`, `#28667`, s390x repack.
- Ported from: `https://github.com/milpster/gfx906-llama-cpp` (`d14628d04` `q8_0-native`, `eaca6d43` `mixed`, `7cfc00904` dispatch, `923a9661d` wip, `42fc61855` `fattn-vec`+graph, `cd97aeae0` `Q6_K`, `469de3c3d` `MMQ OOB`, `91259ac30` probe) -> local `2b620d7b9`.
- Related files: see the File-level Impact table above.

---

## Turbo High-Value Ports (2026-09-10, `a1bc8adc6`)

From `arte-fact/llamacpp-gfx-906-turbo` (`gfx906/` 29-file W64 kernels). Landed: `add-id` `float4`
vec4 + contiguous fast path (`add-id.cu`), `rope_yarn` `__sincosf`+`__logf` on GCN (`rope.cu`),
`vecdotq` fast `b1/b2` memcpy + MXFP4 `v_perm` LUT. Deferred: turbo asm `quantize_mmq_q8_1`
(high-risk, low gain), `mmvq` warp-coop, `mmq` vectorize/prefetch, `sgemm/mmf`, `fattn-q8` tile
(the native FATTN tile later landed a better version). Bench Qwen3.5-9B UD-Q4_K_XL pp512/tg128
neutral, and Ling-3.0-tiny Q8_0 MoE shows the add-id/vecdotq gains (pp 2076.98 -> 2082.00,
tg 109.93 -> 110.83). Dense Q4_K shows no gain (dual/v8/FA v3 target GPTQ+head_dim 256, not
Q4_K).

## mi50grad Single-GPU Ports (2026-09-10, staged)

From the `mi50grad/` standalone HIP stack; multi-GPU (kernel P2P AR, deferred AR, fused GEMV+AR)
excluded per request. Landed: `rms_norm_f32_gfx906<256/1024>` - `float4` 4x per thread + DPP
`block_reduce` + `rsqrtf` (`norm.cu`, gate `GCN && ncols%4==0`). Compiled but gated-inactive in
`gfx906-mi50-opts.cuh` (GCN-only, kept by `ggml_cuda_mi50_force_link()` in `ggml_cuda_init()`;
an earlier gate-only build had the kernels DCE'd - fix `05526fce6`): `mi50_gemv_dual_fused_4x`,
`mi50_gemv_v8_coop`, `mi50_flash_attn_256_v3_prefill` target GPTQ dequant-dot + head_dim 256
prefill and need Q4_K repack gate+up wiring. `FUTURE_RESEARCH.md`: `v_dot8`, fused QKV deferred.

## Fused RMSNorm+MUL GCN (2026-09-11, `2817c3c0d`)

`rms_norm_f32_gfx906_fused<256/1024,do_add>` (`norm.cu:338`): 4x `float4` `x` loads, broadcast
`mul/add` fastdiv, DPP `block_reduce`, `rsqrtf` - one launch instead of `rms_norm` +
`k_bin_bcast`/`scale_f32`. Gate `GCN && ncols%4==0 && ncols<8192` in `rms_norm_mul_f32_cuda`;
`test-backend-ops -o RMS_NORM` 51/51 OK. Bench: Merged 1.9B BF16 pp 1174 -> 1179, tg 93.2 -> 93.6
(neutral). A F16/BF16 repack prototype (`GGML_CUDA_REPACK_F16/BF16`, q8-quantized matvec) was much
slower on 0.8B BF16 (pp 2995 -> ~900; rocBLAS Hgemm already optimal, quant overhead) and was
reverted - bf16 later landed correctly as `d2685bff2` (stored as f16, no quant step). `.gitignore`
carries `results.*` / rocprof traces.

## GCN Native FATTN q8_0 Tile + HIP Graph + MMQ (2026-09-13, `2b620d7b9`, ported from `milpster/gfx906-llama-cpp`)

Ported from `https://github.com/milpster/gfx906-llama-cpp` (`d14628d04`, `eaca6d43`, `7cfc00904`,
`923a9661d`, `42fc61855`, `cd97aeae0`, `469de3c3d`, `91259ac30`), adapted to `311d4211b`,
HIP/GCN-only. Four areas (detail: change-log row 21 + the commit body):

- `FATTN` native q8_0 KV tile: `fattn-common.cuh:6` auto decision (`GGML_CUDA_FATTN_PATH=auto |
  force_convert | force_native`, 20% free-mem headroom per device, re-checked when KV grows),
  `fattn-tile.cuh:6` q8_0 K/V loaders (once-per-block dequant, no `f16` shadow materialization),
  16-wide narrow tile for Q<=16 verify batches, `256/256/16` occupancy 3. Default `auto` keeps the
  `f16` convert while the shadow fits and switches to the native tile at deep KV.
- `fattn-tile.cuh:293` SW double-buffer prefetch hides ~31% LDS wait; `fattn-vec.cuh:111` mask
  stride `nb31/2` fix for sequence-split slices (upstream still open - see Upstream Attention).
- HIP graph kernel-func guard `common.cuh:1411` + `ggml-cuda.cu:2676`: `hipGraphExecUpdate` does not
  validate `launch_bounds` on a function change, so a MUL_MAT_ID quantize-kernel swap (128 vs 32)
  replayed UB; the guard compares kernel-func sets and re-instantiates on change.
- `mmq-config-vega.cuh` `Q6_K` retune `I=64 occ2` (+4.2% PP) and `mmq.cu`/`mmq.cuh` `J_max`/NVFP4
  `+128` tail padding (padding since superseded upstream).

## Furnace gfx906-perf Generic-Opt Ports (2026-10-01, donor `905021dba`; 2026-10-05, donor `fc353a5ac`)

Source: `/home/otis/Dokumente/Build/llamacpp-gfx906-furnace` branch `gfx906-perf`; author sixvolts
<rigel@sixvolts.org> with `Co-Authored-By: Claude` (Opus 4.8 / Fable 5.1 / Opus 5.5, see commit
trailers). Cherry-picked with original authorship preserved; every Commit Inventory row from
`73d46a563` through `5e601af03` carries its donor SHA and adaptation notes, so no per-port table
is repeated here.

Coverage by area: repack tiling and kernels (Q3_K relabel, Q4_K GLU/fence/hoist, Q5_K/Q5_1/Q8_0/
Q4_0/IQ repack and matvec/mmq variants), dispatch (2-16-column matvecs incl. the LDS-staged
`lds_rows`/`lds_nc` kernels, sibling-matvec grouping, small-batch paths to 16 tokens, per-graph
activation cache), MoE (route cache, expert dedup, decode chain down+reduce), F32 (GEMM variants,
row matvecs), fusion (residual add/norm, GLU, shared-expert SIGMOID tail), and single kernels
(concat dim-0 transpose, get_rows short-row packing, radix top-k, topk-moe DPP moves).

Port filter (excluded): multigpu (`TurboPrefill`, `NO_PEER_COPY`, layer-split), server bits
(`300fdfe3b`, `47f6e5b8a`), tests-only shapes (`746887a5c`), `QSA`/sparse/`hc`/`dsv4`/GDN/qwen4exp/
`MTP` arch (no local base; qwen4exp deferred per request), build switches, bounce fix, redundant
(`b3c18e502`, `2ffa92b3c`).

Validation (2026-10-05, 1x MI50): full clean build; `test-backend-ops -o MUL_MAT` (all repack types
incl. Q4_0/IQ + narrow tiles) / `-o RMS_NORM` / `-o TOPK` / `-o ADD` all `OK`; `llama-bench` 9B
`-p32 -n16` runs (`pp32 ~385`, `tg16 ~77`); the 2026-10-19 rebase validation re-ran the full
2450/2450 `MUL_MAT` matrix.

## Furnace gfx906-perf Ports, Round 2 (2026-11-05, donor `4f30c1bda`)

Source: `/home/otis/Dokumente/Build/llamacpp-gfx906-furnace` branch `gfx906-perf` at `4f30c1bda` (2026-11-01). Author `sixvolts <rigel@sixvolts.org>` (`01f700b37`, `013ad9e02`, `5ab3eaa4a`, `963dc1514`, `0c63b6968`) with `Co-Authored-By: Claude Opus 5.5`; `Nik <69503704+nik343@users.noreply.github.com>` (`9a548180e`) with `Co-authored-by: Codex`. Manual ports credit the source commit in code comments and commit body.

| Local | Furnace | Change | Status |
|-------|---------|--------|--------|
| `2ba96ded4` | - | `ggml_cuda_mul_mat_repacked_multi` quantized only one activation column (`ne1 = 1`), so repack Q8_0 at ubatch 2..16 produced garbage. Use `repack_quantize_x` as the donor does. | Fork fix |
| `3b56169f8` | `01f700b37` + `013ad9e02` | Q8_0 2..4-column matvec with the activation in LDS; grid in whole rounds of resident blocks (`GGML_CUDA_NO_Q8_LDS_ROWS`, `GGML_CUDA_Q8_LDS_ROWS_ROUNDS`). `unary.cu` gate hunk not applicable (no `add_unary_mul`). | Ported |
| `01a6f0ce6` | `5ab3eaa4a` | Deterministic radix top-k: per-chunk count/gather, column-order ties. | Ported |
| `ed4f655a6` | `9a548180e` + `0c63b6968` | Per-type GCN K-quant repack switches (`GGML_CUDA_REPACK_Q3_K` / `_Q4_K` / `_Q5_K` / `_Q6_K`) and `repack_env_enabled()` (`off` / `false` / `no`). | Ported |
| `1e2bfa436` | `963dc1514` | RMS_NORM + MUL + ROPE fusion for M-RoPE / IMRoPE (`GGML_CUDA_NO_NORM_MROPE_FUSION`) plus the f16 store rounding fix; new `RMS_NORM_MUL_ROPE` IMRoPE/M-RoPE cases. | Ported, adapted to the sample/channel grid clamping |

Excluded: `0eb61b009` (batch embd view; targets MTP-specific copy overhead, fork has newer `llama_batch_ext` / mixed-batch code), `77eda3529` (GDN/MTP verify fusion, model-specific), `ed263812e` (needs `rms_norm_scale2`, absent). Model-specific Qwen/QSA/GDN/MTP work excluded per the donor criteria.

Validation (2026-11-05, 1x MI50 gfx906, ROCm 7.2.4, `Spark-X2.6-4B-Q8_0`):

- `test-backend-ops -b ROCm0`: `TOP_K` 533/533 (incl. tied keys), `ROPE` 450/450, `RMS_NORM_MUL_ROPE` 240/240 (incl. IMRoPE/M-RoPE, full and 64-dim rotary, with and without SET_ROWS).
- `llama-perplexity` wikitext-2 ch512, ub4: LDS rows, `q8_lds_nc` and per-column fallback give identical chunk PPL (38.9503, 24.4469, 19.7742, 20.9448). `GGML_CUDA_REPACK=off` and `GGML_CUDA_REPACK_Q8_0=off` match the stock path exactly.
- `llama-bench -p512 -n128 -r3`: repack off pp512 `1347.67`, tg128 `112.83`; Q8_0 repack on pp512 `2113.85`, tg128 `119.60`.
- `llama-batched-bench -npp 128 -ntg 128 -ub 512`, TG t/s with `GGML_CUDA_NO_Q8_LDS_ROWS=1` -> on: B=2 `169.95` -> `199.24` (+17.2%), B=3 `231.53` -> `262.74` (+13.5%), B=4 `254.71` -> `295.09` (+15.9%). B=1 unchanged within run noise.
- `llama-bench` defaults, commit `c68bd0d45`, `-ngl 99 -p512 -n128 -r5`: `Qwen3.5-0.8B BF16` pp512 `4785.28 ± 330.75`, tg128 `194.60 ± 0.88` (2026-10-19 ref `4767.3` / `190.4`, +0.4% / +2.2%); `Qwen3.5-9B UD-Q4_K_XL` pp512 `974.42 ± 14.91`, tg128 `83.75 ± 0.43` (2026-10-19 ref `963.0` / `83.2`, +1.2% / +0.7%). M-RoPE fusion A/B on the 9B (tg128, `-p16 -r3`): `GGML_CUDA_NO_NORM_MROPE_FUSION=1` `83.16 ± 0.85` vs default `84.16 ± 0.30` (~+1%). No regression on either Qwen model.
- Q8_0 / Q5_1 repack default-on (`961362c5c`): `Spark-X2.6-4B-Q8_0` `llama-bench` with no env pp512 `2127.86 ± 50.99`, tg128 `118.96 ± 0.51` = the repack-on reference above; `llama-perplexity` wikitext-2 ch512 ub4 `--chunks 2` PPL `24.5763` matches the repack-on chunk values; `test-backend-ops -o MUL_MAT` all `OK`. `Qwen3.5-9B` (its Q8_0 tensors now repack): pp512 `997.70 ± 23.03`, tg128 `83.02 ± 0.37` vs `974.42` / `83.75` with Q8_0 off (prefill +2.4%, tg within noise).

---

## BF16 -> F16 Upload Conversion Repack (2026-11-05, committed `d2685bff2`)

Design: gfx906 has no bf16 compute, so the stock path dequantizes bf16 weights to f32 and runs Sgemm at half the
packed-f16 rate. 2D MUL_MAT BF16 weights are stored in the repack buffer as f16 instead: `set_tensor` converts
bf16 -> f16 row-major (`repack_bf16_as_f16_host`, same 2 bytes, no interleave) and `ggml_cuda_mul_mat` relabels
the tensor BF16 -> F16 so the stock f16 mmvf/Hgemm path runs. Opt-outs: `GGML_CUDA_REPACK_BF16=off` and
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=bf16` (that env would read the f16 bits as bf16). Committed as
`d2685bff2`: `repack-gcn.cu` +21 (`nbytes`/`supported`/`set_tensor` cases), `ggml-cuda.cu` +31/-3
(relabel + `REPACK_BF16_DEBUG` dispatch prints).

Bench (1xMI50, ROCm 7.2.4, `llama-bench -p512 -n128 -r3`, Qwen3.5-0.8B BF16):
| run | pp512 | tg128 |
|---|---|---|
| repack bf16 ON | 5808 ± 1453 | 173.1 ± 6.4 |
| repack bf16 OFF (stock) | 4384 ± 766 | 171.7 ± 1.5 |

pp +30-35% (matches the +33% cited in the commit comment); tg neutral. tg drifts 173 -> 197 t/s between sessions
(clock noise ~±12%), so treat single-run tg deltas as noise. PPL wikitext ch1 c1024: ON 7.8480 vs OFF 7.8239
(bf16 -> f16 mantissa truncation; within the ±1.03 error bar, re-check on a target eval before trusting exactness).

### Crash: frame-resident relabelled tensor (fixed 2026-11-05 - pattern to remember)

First WIP build segfaulted on every run: `ggml_backend_buffer_get_usage(buffer=0x4)` from the
`bad_padding_clear` line in `ggml_cuda_mul_mat`. Forensics (gdb + hardware watchpoint): the
frame-resident `ggml_tensor f16_weights = *src0;` stack copy never materialized - HIP clang 20
(-O2 host compile) does not reliably store a large frame struct whose address escapes, so `src0`
kept pointing at stale junk a shallower call had left at the same stack depth
(`get_rows_switch_src0_type` had written `0x4`-looking garbage there). A debug printf flipping the
outcome 5/5 crash -> 5/5 pass marked the whole thing as codegen-sensitive == looked
nondeterministic.

Fix: the relabelled tensor is now `static thread_local` inside `ggml_cuda_mul_mat` (stable address,
per backend thread). Validated: 5/5 runs + 4-chunk PPL clean, `llama-batched-bench` clean,
2450/2450 `MUL_MAT`.

Rule for future ports: never put a `ggml_tensor` copy on the call frame when a pointer to it is
passed to another function - use `static thread_local`, heap, or mutate/restore in place. The same
pattern still exists for `dst_vec` (transposed F32 vector path) and `src0_chunk`/`dst_chunk`
(cublas chunked convert) in `ggml_cuda_mul_mat`; if a crash shows up with `buffer=<small int>`
garbage or an empty tensor name from one of these copies, treat it as this bug first. Repro:

    gdb -ex "break ggml_backend_buffer_get_usage" -ex "cond 1 buffer < 0x10000"
    # then dump the stack around src0; a hardware watchpoint armed at ggml_cuda_op_get_rows
    # (first graph op at the same depth) shows whether the copy ever stored

---

## ISA Notes + Kernel Register Audit (2026-11-06, `2aa788490`)

`MI50_ISA_OPTIMIZATION_NOTES.md` (committed) distills the "Vega" 7nm ISA doc and the verified
MI50 hardware config; `amdgcn-audit.py` (committed) reads per-kernel VGPR/SGPR/LDS and spill
counts from the HIP code-object metadata inside the built `.o` files - no ROCm tooling or GPU
needed, occupancy math `min(10, 1024/VGPR, 800/SGPR)` + `floor(64 KiB / LDS_per_wg)`.

Audit findings (notes sec. 11): `flash_attn_tile` dkq128/J>=8 spills ~900 dword (J<=4: 5-12);
`gcn_f32_matvec_rows_multi_nc<7..16>` V256 + spill up to 394; `rms_norm_mul_rope` SGPR 100 +
spill 15-20; `lds_nc<6>/<8>` occupancy co-bound by LDS (6912/9212 B), so VGPR-only shaving was
a dead end.

### lds_nc column-group + rope SGPR slim: committed `ffc7e39c1`, validated 2026-11-06

Both changes are output-preserving restructures. The in-tree `libggml-hip` was built after the
edits (library 17:44 vs sources 17:38), so every run below includes them; no separate rebuild or
`amdgcn-audit.py` re-audit was done - run the two audit commands in notes sec. 12 to confirm
the register/LDS expectations.

| File | Change | Result |
|---|---|---|
| `repack-gcn.cu` | `lds_nc` column groups of 4 for NC = 6..8 (LDS 6912/9216 -> 4608 B, live acc 16; NC <= 5 single-pass unchanged). Per (row, column) K order unchanged - bit-identical; pass 2 re-reads weight halves as L2 hits | validated: batched-bench B=8 TG `326.3-328.0` vs per-column fallback `254-268` (+22%); B=2..4 match the recorded refs within noise; B=6 dips (`~271-275`: group 4+2 leaves 25% dead slots, below B=4's `295`) - pack concurrent slots to multiples of 4, best 8 |
| `rope.cu` | `rms_norm_mul_rope`: x/mul/dst strides -> i32 (2^31-element asserts in the fused entry), per-channel mul dims -> plain `%` (per-element fastdiv kept). Bit-identical | validated: 0.8B BF16 tg128 `197.65 +/- 0.91` on vs `195.95 +/- 0.93` fusion-off (+0.9%); 48-token greedy text identical; SGPR audit still pending |

Validation (2026-11-06, 1x MI50 gfx906, ROCm 7.2.4, build `2aa788490` + WIP):

- `test-backend-ops -o MUL_MAT` 240/240 OK; `-o RMS_NORM_MUL_ROPE` OK.
- `Spark-X2.6-4B-Q8_0` wikitext `-c 512 --chunks 4`: chunk PPL bit-identical across rope fusion
  on/off (`GGML_CUDA_NO_NORM_MROPE_FUSION`), staged vs per-column fallback (`GGML_CUDA_NO_Q8_LDS_NC`
  / `GGML_CUDA_NO_Q8_LDS_ROWS`), and `-b 512` (n_seq=1) vs `-b 2048` (n_seq=4): `39.2040, 24.5763,
  19.8034, 20.9564`. The pooled `--chunks 2` ref `24.5763` from `e80a06d39` reproduces bit-exactly.
- The old 4-chunk list (`38.9503, 24.4469, 19.7742, 20.9448`) is not digit-exact anymore: the
  perplexity tool's Final/pooling semantics changed since the 2026-11-05 validation ("Final
  estimate" now prints the last chunk's value) - tooling artifact, not a WIP signal (all
  invariance toggles on this build are bit-identical).
- `llama-bench -ngl 99 -p512 -n128 -r5`: `Spark-X2.6-4B-Q8_0` pp512 `2165.48 +/- 39.63` / tg128
  `120.93 +/- 1.46` (recorded `2127.86 +/- 50.99` / `118.96 +/- 0.51`); `Qwen3.5-0.8B-BF16` pp512
  `6516.33 +/- 501.34` (vs `4785.28 +/- 330.75`; the prefill jump is the committed bf16 repack, not
  this WIP), tg128 `197.65 +/- 0.91` (vs `194.60`).
- `llama-batched-bench -npp 128 -ntg 128 -ub 512 -c 4096` TG t/s: B2 `200.37` (ref `199.24`), B3
  `266.64` (ref `262.74`), B4 `295.65` (ref `295.09`); new: B6 `271-275`, B7 `280-305`, B8
  `326.3-328.0`. `GGML_CUDA_NO_Q8_LDS_NC=1` (per-column fallback): B6 `265-266`, B7 `262-266`,
  B8 `254-268`.
- Open: B=6/7 vs the pre-WIP single-pass `lds_nc<6>/<7>` needs a pre-WIP `.so` A/B (rebuild
  declined); the env-fallback numbers above are the available bound. Also seen while benching:
  `llama-batched-bench` OOMs unless `-c` is pinned (default context allocates ~36 GB at N_KV for
  8 sequences), and `GFXPROF=1` aborts in `llama-cli` (pre-existing tooling issue, untouched).

---

## Milpster gfx906-llama-cpp Ports (2026-10-05, donor `HEAD` at `1373e889a`)

From `https://github.com/milpster/gfx906-llama-cpp` (`/home/otis/Dokumente/Build/gfx906-llama-cpp`,
branch `master`), ported with `Original author: milp` credit. Excluded: multi-GPU/offload
(`a2bf395db`, `916727ecf`, `2eb48f94c`), cross-backend MTP/spec (`87110171f`), GDN/SSM arch
(`80d985f4a`, `f81efa909`, `4d41a52d6`), server/loader/journal/bench, VEGA_TUNE experiments
(superseded by `GGML_CUDA_FATTN_PATH`), already-equivalent commits, upstream-PR MMVQ merge
(arrives via rebase; repack bypasses canonical mmvq anyway).

| Local | Donor | Change | Status |
|---|---|---|---|
| `d87a4cd04` | `1373e889a` | `__vsubss4`->`__vsub4` + VMEM sched barrier (donor Q6_K tg ~+8%) | dropped 2026-10-19: upstream merged PR #29478 |
| `ec5e7a7a7` | `2da432a72` | Skip recurrent pos scan when batch pos is given (`llama-batch.cpp`) | landed |
| `137d79114` | `2da432a72` | Native q8_0-K / q4_0-V FATTN tile (engages with quantized KV only; wins are memory - no f16 shadow for q4_0 V) | landed |
| `bc87c64d0` | - | Fix GCN `supports_op` gate that forced FILL/TRI/DIAG onto CPU (KDA/delta-net PPL blowup on Ling-3.0-tiny: b1 c32 29.6 -> 1838.7; fix restores 28.03) | fork-original fix |

## Maintaining This Document

- Edit this file on every `git push` to `gfx906/mi50-optimization` or `origin/master` sync.
- Keep commit table in reverse-chronological log order (`git log --reverse`).
- The full commit bodies stay in `git show <sha>` - do not paraphrase from memory; this doc only
  mirrors them for recent rounds and compresses older ones into the tables above.
- Performance deltas are from commit messages - include `llama-bench` / `batched-bench` / `wikitext PPL` numbers when present.
- Remove or collapse older tuning experiments (e.g., rejected `ROWS=4`, quarter-units) only after they are stable for >2 rebases.
