# Fork Tracking - llama.cpp-MI50 (gfx906) vs Mainline

> Tracks divergence of this fork (`FirephoenixX02/llama.cpp-MI50`) from upstream `ggml-org/llama.cpp`.
> Branch: `gfx906/mi50-optimization` - ROCm/HIP optimizations for **MI50 / Vega20 (gfx906)**.

## Snapshot

| Field | Value |
|-------|-------|
| **Fork repo** | `https://github.com/FirephoenixX02/llama.cpp-MI50` |
| **Upstream** | `https://github.com/ggml-org/llama.cpp` |
| **Fork branch (published)** | `origin/gfx906/mi50-optimization` at `d2685bff2` (2026-11-05, bf16 repack); `5a2255fcc` / `2aa788490` / `ffc7e39c1` + docs commit not yet pushed |
| **Local branch** | `gfx906/mi50-optimization` rebased onto `master` = `a691b2345` (2026-11-08), 99 commits ahead, clean |
| **Base / merge-base** | `a691b2345` - `webgpu : add COUNT_EQUAL support (#31578)` |
| **Base date** | 2026-11-08 |
| **Commits ahead of base** | 99 on `HEAD` (`git rev-list --count origin/master..HEAD`) |
| **Last doc update** | 2026-11-09 (HIP fast-math flags: compile/IR/ISA + op-tests 288/288 + same-host A/B, neutral on bench models; GCN F32 GEMM chunk-`ldc` fix; fattn-tile GQA-limit experiment (default reverted to grouped, env variant); chunked GDN prefill (mx iacopPBK PR #2) measured +67% op-level at qwen35 pp4096 and rejected, kept as a doc finding; attention audit refreshed; GDN gate fusion + fused conv step ported and validated (see "GDN Gate Fusion", "Fused Recurrent Conv Step"), uncommitted), 2026-11-08 (rebase onto `a691b2345`; MMID tile-spec adaptation + per-chunk HIP graph guard, see "Rebase 2026-11-08" below; all branch SHAs rewritten; + FATTN quantized-KV decode GQA-gap measured on the Pro VII, see "FATTN Decode Attention" below; grouped-VEC candidate built + measured (2-3x worse at every depth) + reverted same day, tree matches `f518082a3`) |
| **Upstream `origin/master`** | `a691b2345` (2026-11-08) - rebase done, see "Rebase 2026-11-08" below |
| **Backup** | `backup/mi50-pre-rebase-20261108` at `270fc1275` (pre-rebase HEAD), plus the older `backup/mi50-pre-rebase-20261018` at `87f850290` |
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
> Filter applied (donor `furnace/gfx906-perf` at `fc353a5ac`, range `905021dba..7bac8169e`): A: multigpu excluded (none new in range). B: non-opt excluded (`300fdfe3b` server `/props` config, `746887a5c` tests-only bench shapes, `47f6e5b8a` server `_exit`). C: GDN/qwen4exp arch excluded (`058d97848` GDN gather elision fix, `50798f69b` GDN gate/softplus/beta fusion, `a2207dd89` GDN q/k norm pairs, `3a30ac2c2` recurrent conv-step fusion; 2026-11-08 update: the "no GDN fusion base" reason is wrong at base `a691b2345` - the fused `GATED_DELTA_NET` op is default-on there and profiles at 3.3-4.4% GPU time on qwen35, see "GDN and attention shares" below; these micro-fusions remain excluded on share grounds, not on base grounds). D: only `sixvolts` + `Claude` commits ported. Earlier filter (donor at `905021dba`): multigpu (`TurboPrefill`, `NO_PEER_COPY`, layer-split), README/tests-only/server/MTP/`llama` scheduler staging, `QSA`/sparse/`hc`/`dsv4` arch, `cohere2` handler, build switches, `solve_tri` scope, bounce fix; redundant skipped (`b3c18e502`, `2ffa92b3c`). Prior repack-tiling deferrals resolved via manual LEVER-adapted ports. qwen4exp arch still deferred per request.

---

## File-level Impact (vs `a691b2345`)

Raw: 34 files, 11704 insertions(+), 200 deletions(-) vs `origin/master` (2026-11-08; full
per-file listing via `git diff --stat origin/master..HEAD`). The 11 furnace ports that were
uncommitted on 2026-10-05 are a single commit (`5e601af03` in the pre-rebase history).

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
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | dropped 2026-11-08 | The `GCN` MMA clamp (+17/-1, commit 19) removed: no MMA dispatch site was ever reachable on gfx906 at any fork base (see FATTN Decode Attention); provenance was a defensive import of vLLM-gfx906 Triton tuning, never an upstream or measured llama.cpp behavior. File is byte-identical to `origin/master` now |
| `ggml/src/ggml-cuda/fattn-common.cuh` | +93 | `GGML_USE_HIP` `FATTN` auto native decision (`GGML_CUDA_FATTN_PATH`) + per-device `free_mem` probe (`2b620d7b9`). |
| `ggml/src/ggml-cuda/fattn-tile.cuh` | +330/-2 | GCN native `q8_0` `half2/float` loaders, SW-pipelined `KQ` prefetch, 16-wide narrow tile + occupancy-tuned 256 config (`2b620d7b9`). |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | +6/-1 | Mask stride `nb31/2` fix for sequence-split `ne11` slice vs full mask rows (`2b620d7b9`). |
| `ggml/src/ggml-cuda/fattn.cu` | +63 | GCN native `q8_0`/`f16+q8_0` predicates + alloc skip + `VEC` vs `TILE` routing for quantized KV (`2b620d7b9`). |
| `ggml/src/ggml-cuda/solve_tri.cu` | +62/-47 | `MAX_K_FAST` 64 + `GCN` cap `64x64`. Cold path since base `a691b2345` - the fused `GATED_DELTA_NET` op covers qwen35 prefill and decode (`delta-net-base.cpp` fused defaults + `llama-context.cpp:348-350`) |
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
| 1 | `0beb3a854` | solve_tri `MAX_K_FAST=64` + GCN cap 64x64 (Qwen3-Next delta-net prefill needs k=64) | fixes delta-net prefill (cold path since base `a691b2345`: qwen35 prefill takes the fused GDN op; solve_tri reached only if fused ops are disabled) |
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
| 19 | `b143fd169` | FATTN MMA clamp nstages=1 / occupancy=1 for GCN (64KB LDS W64 dp4a) | killed (never-launched) mma-kernel spills; dropped 2026-11-08, see FATTN Decode Attention |
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

## Rebase 2026-11-08 (onto upstream master `a691b2345`)

Rebased the 99-commit stack from `7951f601e` onto `a691b2345` (468 upstream commits). All branch
SHAs were rewritten; `git log --reverse --oneline a691b2345..HEAD` is authoritative. Backup of the
pre-rebase tip: `backup/mi50-pre-rebase-20261108` at `270fc1275`.

Conflicts resolved:

- `ggml-cuda.cu` include block: keep upstream `top-k-mask.cuh` and the guarded
  `gfx906-mi50-opts.cuh` include.
- `ggml-cuda.cu` HIP graph guard: upstream split graph capture into chunks
  (`ggml_cuda_graph::instances`, `ggml_cuda_graph_launch_chunk`). The guard now operates per chunk
  (`ggml_cuda_graph_update_executable(graph, idx, cuda_graph)`), and `kernel_funcs` became a vector
  of per-chunk vectors.
- `unary.cuh`: keep upstream `ggml_cuda_op_glu_single` switch and add the fork
  `ggml_cuda_op_sigmoid_mul_add` declaration.
- `tests/test-backend-ops.cpp`: keep both upstream 384-expert top-k cases and the fork 512-expert
  cases; keep upstream `yarn`/`truncate` RMS_NORM_MUL_ROPE fields plus the fork `rope_dims`.
- `ggml-cuda.cu` RMS_NORM+MUL+ROPE: upstream already accepts M-RoPE/IMRoPE in
  `ggml_cuda_should_fuse_rope_set_rows`; keep that mode check (the fork env gate remains in
  `ggml_cuda_should_fuse_rms_norm_mul_rope`).

Code adaptations required by upstream API changes (not conflicts):

- `mmid.cu` now emits `ggml_cuda_mm_id_tile_spec` instead of `expert_bounds`.
  `ggml_cuda_mul_mat_id_repacked` passes a `tile_spec` buffer and the 16-wide tile width, then
  rebuilds `expert_bounds` from the compact routing (`repack_expert_histogram` +
  `repack_expert_bounds_scan`) before running the existing `repack_tile_map`; the routed MMQ kernels
  are unchanged. The tail specs cannot be used for bounds: the helper mutates `col_low` while
  emitting seam tiles.

Validation (1x MI50 gfx906, ROCm 7.2.4, `build-llamacpp-rocm.sh` clean):

- `test-backend-ops -b ROCm0 -o MUL_MAT,MUL_MAT_ID,RMS_NORM,RMS_NORM_MUL_ROPE,ROPE,TOPK,ADD,FLASH_ATTN_EXT,GET_ROWS,UNARY,SOFT_MAX`:
  9193/9193 tests passed, 2/2 backends passed.
- `llama-bench -ngl 99 -p 512 -n 128 -r 3`, upstream `a691b2345` vs this branch, same session,
  both built with `build-llamacpp-rocm.sh`:

  | Model | master pp512 | fork pp512 | pp delta | master tg128 | fork tg128 | tg delta |
  |-------|-------------:|-----------:|---------:|-------------:|-----------:|---------:|
  | Qwen3.5-9B UD-Q4_K_XL | 793.34 +/- 3.18 | 987.92 +/- 31.66 | +24.5% | 67.60 +/- 0.22 | 83.06 +/- 0.14 | +22.9% |
  | Qwen3.5-0.8B BF16 | 4386.02 +/- 59.30 | 6635.71 +/- 115.05 | +51.3% | 184.87 +/- 0.35 | 198.32 +/- 0.28 | +7.3% |
  | Spark-X2.6-4B-Q8_0 | 1370.77 +/- 13.52 | 2149.44 +/- 12.02 | +56.8% | 110.43 +/- 0.08 | 121.52 +/- 0.27 | +10.0% |

  These are the same gains as before the rebase (pre-rebase fork 963.0/83.2, 4767.3/190.4),
  so the 2026-11-08 base did not regress the fork optimizations.
- `llama-perplexity -c 512 --chunks 2` (wikitext-2, chunk PPL): Ling-3.0-tiny Q8_0 MoE repack on
  `8.1083, 12.4490` vs `GGML_CUDA_REPACK=0` `8.0729, 12.5455`; Spark-X2.6-4B Q8_0 repack on
  `39.2040, 24.5763` vs off `39.2735, 24.6055`. This MoE check is what caught the first
  `expert_bounds` reconstruction attempt (tail-spec `col_low` is mutated by the seam loop; PPL
  blew up to 255).

---

## Correctness Notes

- All repacked matvec/MMQ variants validated via `wikitext PPL` vs GEMM/canonical within noise (tokens in commit logs).
- `ub=1 PPL` checks: Q5_K/Q6_K, Q8_0 (5.07 vs 5.12), MoE ID paths, FATTN clamp (no PPL regression cited; LDS correctness - clamp kernels are unreachable on gfx906 at base `a691b2345`, see FATTN Decode Attention).
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
| `GGML_CUDA_NO_CONV_STEP_FUSION=1` / `NO_CONV_STEP_CONV=1` | unset (fusion on) | GDN conv-step fusion (gather + concat + tail CPYs + SSM_CONV-SILU) off; the second env disables only the SILU absorption. See Fused Recurrent Conv Step. |
| `GGML_CUDA_NO_ADD_UNARY_FUSION=1` | unset (fusion on) | GDN gate ADD -> softplus -> MUL (+ folded SIGMOID) fusion off (ported furnace `50798f69b`). Bit-exact, +1.3% tg on the 9B. |
| `GGML_CUDA_FATTN_TILE_GQA=1` | unset (always-grouped, pre-experiment behavior) | 2026-11-09 experiment, see the Tile GQA-Limit section: any set value enables the `gqa_limit = 16` shape (grouped up to 16 query tokens, ungrouped `ncols2 = 1` beyond, only where an ungrouped site exists: DV <= 256, DKQ not 192/320). Wins -12.5% FA while the KV window fits L2, loses +6% once it crosses; default stays grouped. |
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
| FATTN GCN clamp | Fork-only; dropped 2026-11-08 (was dead code on every fork base, see FATTN Decode Attention) | Done - file matches `origin/master`; nothing left to port or validate |
| FATTN GCN native `q8_0` tile | Fork-only; GCN 64KB LDS compact vs `f16` shadow | Keep; `GGML_USE_HIP` gated, validated via `ggml_cuda_fattn_use_native_tile` probe |
| Vega20 MMQ table | Derived from RDNA2 occupancy 1; `Q6_K I=64 occ2` after `2b620d7b9` (`+4.2%` PP) | Keep; upstream MMQ tables live in `mmq.cuh` and change often - rebase conflict expected |
| DPP warp reductions | GCN-only `#ifdef`; no CDNA/RDNA impact | Keep; re-check if upstream refactors `common.cuh` reductions |
| `MoE` weighted-reduction empty-tail | Fixed here `5a2255fcc` (node-walk bounds + zero-token alloc deps, ported from mx-llama.cpp `ef7246cd9`) | Check if upstream still walks past `n_nodes` on empty last-layer chains; PR if missing - see `ggml-cuda.cu:3968` |
| FATTN quantized-KV decode at batch=1 | On GCN (and by the same dispatch shape on CUDA), nc=1 with quantized K/V always runs `fattn-vec` with `cols_per_block=1`: the per-(q-head, chunk) kernel re-does V dequant + P.V FMA per q head. Loads are L2-absorbed; the VALU work is not. Measured 2026-11-08, see "FATTN Decode Attention" below | Both grouped routes falsified 2026-11-08 (head-grouped VEC 2-3x worse; grouped native tile 1409/2288/2852). The gap is wave-count-bound, not op-count-bound; leave it, or revisit with a new kernel geometry |

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
upstream/master (ggml-org)  ──────────────────────►  a691b2345  (2026-11-08, current base)
origin/master (this repo)      ──────────────────────►  a691b2345  (synced, no fork commits)
origin/gfx906/mi50-optimization  ─── 7951f601e ──► d2685bff2  (published, 2026-11-05; rest unpushed)
backup/mi50-pre-rebase-20261018  ─── 4364bf723 ──► 87f850290  (older pre-rebase backup)
backup/mi50-pre-rebase-20261108  ─── 7951f601e ──► 270fc1275  (tip before the 2026-11-08 rebase)
local gfx906/mi50-optimization   ─── a691b2345 ──► 99 commits + rebase fixups
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

## GDN and attention shares (2026-11-08, same Pro VII pass, qwen35 + one dense model)

Method note: the GDN op was invisible to `GFXPROF=1` tables because the fused-node walker
(`ggml-cuda.cu` gated_delta_net + state-cpy fused into one launch) evaluates outside the per-node
loop the profiler hooks. A temporary event-timed hook around `ggml_cuda_op_gated_delta_net_fused_cache`
(measured, then reverted) closed the blind spot. Fused-op defaults confirmed at
`llama-context.cpp:348-350` (`fused_gdn_ar/ch = true`), in-graph via a 3-line trace (measured,
reverted): prefill nt=512 and decode nt=1 both take `build_delta_net_fused`.

Share table (serialized GPU ms, nograph; GDN op includes its fused state-cpy):

| op | qwen35 pp4096 | qwen35 decode d4K | qwen35 decode d8K |
|---|---:|---:|---:|
| GATED_DELTA_NET+cpy (fused) | 188.7 ms = 4.4% | 418 us/tok = 3.6% | 358 us/tok = 3.3% |
| FLASH_ATTN_EXT (8 full-attn layers) | 195.8 ms = 4.6% | 556 us/tok = 4.8% | 688 us/tok = 6.4% |
| dense-model FLASH_ATTN_EXT (`Spark-X2.6-4B` pp4096) | 509.3 ms = 23.8% | - | - |
| rest (repaked matvecs, MoE, glue) | ~91% | ~91.6% | ~90% |

Consequences:

- GDN recurrence tuning is a dead end on this base: the upstream fused op plus the fused-cache
  path already run the recurrence at 3.3-4.4% of GPU time on the GDN-hybrid flagships. A 2x
  better GDN kernel would move the wall by ~2%. GDN decode is depth-invariant (linear attention),
  as expected. `solve_tri` is cold: only the non-fused chunking branch uses it, which nothing
  takes at these defaults.
- Attention-side work is dense-model-weighted: the tile prefill inefficiency (5.8% of `v_dot2`
  peak, FATTN section above) costs ~24% of pp GPU time on dense models but only ~4.6% on the
  GDN-hybrid; quantized-KV decode GQA grouping matters 21-25% on dense models (FATTN Decode
  Attention) vs <7% on the hybrid. Both candidates pay only where full-attention layers
  dominate, i.e. for dense-architecture targets.

---

## HIP Fast Math + F32 GEMM ldc Fix (2026-11-09, applied and validated)

### HIP fast math: off -> on

Upstream history: `-ffast-math` HIP (#23862) + `-fno-finite-math-only` (#25373), disabled (#25495); then `-funsafe-math-optimizations` (#24668), removed in #26696 because its implied `-fassociative-math` reorders FP reductions and flipped greedy argmax on RDNA3.5. Base `a691b2345` ships HIP fast math off (the CUDA branch of `ggml/src/ggml-cuda/CMakeLists.txt` keeps its own `-use_fast_math`).

Applied to `ggml/src/ggml-hip/CMakeLists.txt` (same hunk as the recent PR, comment included):

```
-funsafe-math-optimizations -fno-associative-math -fno-math-errno -fno-rounding-math -fgpu-flush-denormals-to-zero -fsigned-zeros -fgpu-approx-transcendentals
```

Verified offline with the ROCm 7.2.4 clang 22 cross-compile for gfx906 (no GPU execution):

| Check | Result |
|---|---|
| expf lowering | ~20-inst OCML polynomial + clamps -> 2 insts: `v_mul_f32` (x*log2e) + `v_exp_f32` |
| tanhf / silu / rsqrtf / 1/x | brev/bfi 2-inst form / bare `v_rsq_f32` / `v_rcp_f32` (was `v_div_scale` Newton chain) |
| mixed test kernel body | 121 -> 77 instrs (`-funsafe-math` alone: 94; the 94->77 step is `-fgpu-flush-denormals-to-zero`) |
| `-fgpu-approx-transcendentals` alone | inert (implied by `-funsafe-math-optimizations`, kept for symmetry with the PR) |
| IR `reassoc` | `unsafe` alone: 2 -> with `-fno-associative-math`: 0 (the #26696 regression class is off) |
| IR `nsz` | `unsafe` alone: 2 -> with `-fsigned-zeros`: 0 |
| FMA contraction | still fast (8/8 `v_fma` in a serial MAD loop preserved) |
| finite-math | stays off (`-ffinite-math-only` not enabled) -> INFINITY masking in SOFT_MAX/FATTN safe |
| FP mode metadata | `float_denorm_mode_32` 3 -> 0 (FTZ f32 only), `round_mode` stays RN, f16/f64 denorm mode unchanged |

Ops affected: silu/GLU MoE gate (expf), SOFT_MAX + FATTN softmax (expf per KQ element), logit-softcap tanhf, rms-style rsqrtf (`norm.cu` already called it explicitly). dp4a/integer repack+MMQ paths and hipblas are untouched. `ggml-hip` rebuilt clean with the flags on 2026-11-09 (fattn-tile q8 spill counts unchanged, see audit below).

Runtime validation (this host, Radeon Pro VII 16 GB, ROCm 7.2.4, r=3, same host+session A/B vs a second build dir `build-nofm` with the flag set removed; keep that dir around until the round is closed):

- `test-backend-ops -b ROCm0 -o SOFT_MAX -o UNARY -o FLASH_ATTN_EXT -o RMS_NORM -o RMS_NORM_MUL_ROPE`: 288/288 OK.
- `Qwen3.5-9B UD-Q4_K_XL`: no-fast-math pp512 `1016.25 +/- 9.47` / tg128 `84.26 +/- 0.02` vs fast-math pp512 `1013.23 +/- 11.16` / tg128 `84.31 +/- 0.01` - neutral (the pre-fast-math recorded refs of 987.92/83.06 are from the MI50 box, cross-machine; on this host the delta is noise).
- `Qwen3.5-0.8B BF16`: pp512 `6566.97 +/- 575.47` / tg128 `199.00 +/- 0.56` vs `6565.81 +/- 592.93` / `198.89 +/- 0.35` - identical within noise.
- PPL (9B, wikitext, c 1024, 2 chunks): fast-math `6.2945` vs recorded repack-on `6.3304` / repack-off `6.3036` (different host + approx-transcendental kernels; within the +/- 0.49 error bar, no regression signal).

Interpretation: flags behave exactly as the PR claims on gfx906, numerics gate passes, but both bench models under-represent the affected kernels (repacked MMQ/hipblas prefill, small expf share) so end-to-end is neutral. Kept for CUDA parity and for the expf/silu-heavy paths the bench models do not cover (Ling-type MoE GLU decode, softcap models, FATTN q8 tile softmax). Worth a GFXPROF per-op A/B on a GLU-heavy MoE model before judging it done.

### GCN GEMM ldc fix for chunked dst (build-validated 2026-11-09; PPL check pending)

Fork-latent bug, env-gated (`GGML_CUDA_GCN_F32_GEMM=1` + `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` + F16/BF16 weights whose F32 copy exceeds 512 MiB). Upstream `#29442` (`ab09ea4c1`, 2026-10-06) added F16/BF16 -> F32 conversion chunking that calls `ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>` with a dst view (data offset one chunk, `ne[0]` = chunk rows, `nb[1]` = full row stride kept). The fork's own GCN GEMM (`41dee49df`, 2026-09-23, predates the chunking) passed `(int) ne0` as the output-stride arg to `gcn_f32_gemm_skinny` / `gcn_f32_gemm_tn_rb`: on the first chunk that equals the true stride, on later chunks every output column past the first row-block is written at the chunk stride instead of the full stride — same bug class as mx-llama.cpp `2f5cf9337` (their `chunk_src0` branch and the outer-chunk loop do not exist in this base; the fix does not map 1:1).

Fix applied in `ggml-cuda.cu`: skinny and tn_rb call sites pass `(int) ldc` (`ldc = nb1/sizeof(float)`, equals `ne0` for full tensors so unchanged there); split-K part slices stay contiguous with `ne0` (kernel calls unchanged) and `gcn_f32_splitk_reduce` gained `ne0`/`ldc` params and scatters `y[(i % ne0) + (i / ne0) * ldc]`, an identity when `ldc == ne0`. The mx commit (`2f5cf9337`) itself needs no port. Still to run: PPL on a BF16 model with a >512 MiB F32 copy and `GGML_CUDA_GCN_F32_GEMM=1` (e.g. 0.8B BF16, 1.5 GiB F32).

### Attention candidate refresh (audit on the 2026-11-09 fast-math rebuild)

Method: `amdgcn-audit.py` on the in-tree objects; cross-checked against the fattn-tile dispatch (`fattn-tile.cuh:1592` `cols_per_block = DKQ <= 128 ? 64 : 32`, `Q->ne[1] > 4` -> `flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2>`).

| Item | Measured (fast-math rebuild, audit re-run 2026-11-09) | Candidate |
|---|---|---|
| `flash_attn_tile` dkq128/dv128 f16 | spill-free at V128, LDS 29696 (2 WG/CU) | ok, no action |
| `flash_attn_tile` dkq128/dv128 quantized (k_q8_0 / v_q8_0 / v_q4_0) | spill 867-920 dwords at every ncols1 >= 8 geometry, unchanged by fast math: exactly the geometries the native q8 prefill/batch-decode tile dispatches for GQA r=8 (8x8) and r=4 (16x4) with cols_per_block 64; driver is `Q_k[2][cpw][cpy_ne]` + cpw=16 unroll | flag-aware config table: nthreads 512/occ1 for quantized instances at ncols >= 32 (halves cpw 16 -> 8 and the double-buffer register state); ~40-line diff, measurable via GFXPROF spill -> 0 |
| `flash_attn_tile` dkq256/dv256 quantized | spill 120-130 (cols_per_block 32 -> smaller ncols); f16 spill-free | same fix, smaller payoff (Spark dense prefill) |
| `flash_attn_tile` VKQ (P*V) phase | `__hmul2`+`__hadd2` with f16 accumulators: 1 inst per scalar madd, while the KQ phase already runs `v_dot2_f32_f16` asm (0.5 inst + f32 acc) via `ggml_cuda_mad` | transposed KQ'/V' LDS staging (k-adjacent pairs contiguous per jc and per v) would convert the VKQ loop to dot2: halves the P*V instruction count and lifts the f16 accumulator. LDS traffic equal (each element read once either way). NOT bit-identical (pairwise f32 accumulation order differs; f16 accumulator today), so needs the PPL chain. Bigger restructure, the 5.8%-of-`v_dot2`-peak item from the ISA notes; do config table first |
| `flash_attn_mask_to_KV_max<32>/<64>` | SGPR spill 108/331 (decode helper) | low priority |
| `rms_norm_mul_rope_f32` | S100 + SGPR spill 3-14 per instance (i32 slim from `ffc7e39c1` did not fully clear it, fast math does not move SGPR), occupancy 8; grid y/z clamped to 65535 so the channel/sample loops never fold away | port mx-llama.cpp `f6da2e696`: template the loop-carrying norm kernels, use a no-loop instance when the grid fits (norm family kernels in `norm.cu:17/124/332` share the pattern). Target S <= 80 (10 waves) + spill 0. Runs per decode token on M-RoPE models |

Falsified routes from 2026-11-08 remain falsified (quantized-KV decode GQA grouping, native tile at nc=1); the decode attention gap stays open per FORK_TRACKING.

## Tile GQA-Limit Experiment (2026-11-09, `GGML_CUDA_FATTN_TILE_GQA`, default reverted to grouped)

Hypothesis: upstream skips the fattn-tile GQA grouping when `Q->ne[1] > 16` on NVIDIA because "the GQA optimizations seem detrimental in some cases"; gfx906 always groups (`gqa_limit = INT_MAX`), so prefill tiles always launch grouped (`ncols2 = gqa_ratio`). The tile-GQA grouping fixes `cols_per_block` (64 for DKQ <= 128) as the total ncols regardless of ncols2, so grouping only trades K/V tile re-reads (L2-absorbed) against work-group count.

Experiment: first as an env toggle (`GGML_CUDA_FATTN_TILE_GQA=0` -> ungrouped everywhere), then folded into the dispatch. Protocol: `GGML_CUDA_DISABLE_GRAPHS=1 GFXPROF=1 llama-bench`, FA `tot_ms` on 16 pp512 FA calls, 2x interleaved runs per variant (the 2026-11-08 measurement-hygiene rule); 9B (dkq128, GQA r8, f16 KV), Pro VII.

| Phase | grouped (old default) | ungrouped | Delta |
|---|---:|---:|---|
| pp512 fill FA tot | 40.8-41.1 ms | 35.7-36.8 ms | **ungrouped -12.5%** (4/4 interleaved runs consistent) |
| end-to-end pp512 (graphs) | 999.5 +/- noise | 991.7-1009 | ~+0.5% (FA is 3.9% of the pp wall on this hybrid model) |
| decode FA d0..129 | n/a (identical dispatch) | n/a | no dispatch change: the `K->ne[1] % FATTN_KQ_STRIDE` condition blocks grouping for depths 1..255 in both the old and new defaults; the 3 interleaved pairs showing G < U (239.5-245.4 vs 252.9-260.8 ms) are session drift, not dispatch |
| 0.8B BF16 pp512 FA | 27.6-33.6 | 29.0-35.0 | unresolvable in noise (drift +/-25% swamps the ~0.3 ms/call delta at this size) |
| PPL (9B, c 1024, 2 chunks) | 6.3304 / 6.3036 refs | 6.3005 | within band, no quality shift |

pp4096 A/B (with graphs, GFXPROF per-op FA, 2x interleaved pairs per model; pp4096 also tightened the e2e sigma to +/-0.3-0.5%):

| Phase | grouped (old default) | ungrouped (beyond 16 tokens) | Verdict |
|---|---:|---:|---|
| pp4096 FA per-op, 9B (96 calls) | 317.5/318.7 ms, 2.480-2.490 ms/call | 338.2/340.0 ms, 2.642-2.656 ms/call | **ungrouped +6%** (2/2 tight pairs) |
| pp4096 FA per-op, 0.8B (96 calls) | 135.6/136.8 ms, 1.412-1.425 ms/call | 144.1/144.5 ms, 1.501-1.505 ms/call | **ungrouped +6%** (2/2 tight pairs) |
| pp4096 e2e, 9B | 991.3 +/- 0.3% mean | 989.3 +/- 0.3% | grouped +0.2% (2/3 pairs ordered, borderline) |
| pp4096 e2e, 0.8B | 6430 +/- 0.4% mean | 6374.6 +/- 0.6% | grouped +0.9% (2/3 pairs ordered) |
| pp512 fill FA (nograph) | 40.8-41.1 ms | 35.7-36.8 ms | ungrouped -12.5% (4/4; the first round's win) |

Depth explains the flip: each FA call's KV working set grows with depth; at pp512 (<=1 MB per call) the 8x K/V re-reads of the ungrouped geometry stay inside the 8 MiB L2 and the extra work-groups win (-12.5%); once the KV window crosses L2 (8.4 MB at pp4096) the re-reads pay HBM (+6%). Deep prompts are the common case, so the net is negative.

Landed decision (revert): the gfx906 dispatch is back to always-grouped (pre-experiment behavior). `GGML_CUDA_FATTN_TILE_GQA` (any set value) re-enables the `gqa_limit = 16` shape (grouped up to 16 query tokens, ungrouped `ncols2 = 1` beyond), valid only for `DV <= 256 && DKQ != 192 && DKQ != 320` (`576/320/512-DV512`/192/320 have no ungrouped site - the first landing GGML_ABORT'ed there, caught by `test-backend-ops -o FLASH_ATTN_EXT`). Validation after the revert: 9B pp4096 per-op FA 318.5 ms / 2.4887 ms/call = the grouped number; FLASH_ATTN_EXT suite all OK; `build-llamacpp-rocm.sh` clean.

Follow-up idea if ever revisited: a depth-conditional gqa_limit (ungrouped only while `K->ne[1] * DKQ * 4B * gqa_ratio` fits L2) would recover the shallow-fill win without the deep-prompt cost, for <1% pp in exchange for a dynamic dispatch rule - not taken under the simplicity bias. A dense-model (Spark dkq256) check is likewise pending with the reverted default.

## Chunked GDN Prefill: measured and rejected (2026-11-09, no code landed)

Source: mx-llama.cpp iacopPBK PR #2 (`1248e5e37`, 2026-08-08, merged into mx master) - a chunked `GATED_DELTA_NET` prefill kernel filling the TODO at `gated_delta_net.cu` (`launch_gated_delta_net`). The kernel lives at mx `ggml/src/ggml-cuda/gated_delta_net_chunk.cu/.cuh` (+331/16 lines, self-contained) plus a 15-line dispatch hook; re-applying is a drop-in (provenance header + hook at the TODO site). Not landed here.

Mechanism: one (head, sequence) block loops over CS-token chunks (CS 64, 16 for the per-feature KDA gate) with the `S_v x S_v` recurrent state in registers across the whole prefill; one launch replaces the per-token step kernel. Engages when `!keep_rs` (K == 1, standard prefill) and `n_tokens >= 2*CS` and `S_v <= 128`; covers both the non-KDA branch (Qwen GDN, glm5-next, clef, bailingmoe3) and KDA (kimi-linear/k3/k3.1). Ships a dedicated `__gfx906__` branch with DPP/`ds_swizzle` warp reduces; the only wave64-sensitive line is the final `__shfl_xor_sync(0xffffffff, r, 32, 64)` stage - validated OK on wave64 by the S_v 64/128 test shapes.

Measurement (9B qwen35 pp4096, 3x interleaved pairs, Pro VII, build f518082a3 + port):

| variant | pp4096 t/s |
|---|---:|
| chunked (as shipped, 2 warps) | 961.8-968.6 |
| chunked with 4-warp launcher | 971.02 +/- 2.23 |
| per-token step kernel (base) | 989.1-999.2 |

The chunked op runs ~315 ms vs the step kernel's 188.7 ms (+67%; e2e -3.1% consistent 3/3). The step kernel wins on CTA efficiency (4 column segments of 16 lanes each with cheap 16-lane reduces vs the chunk kernel's 2 warps x full-64-lane reduces) and its per-token costs (state round trips ~16 ms + launches ~10 ms at pp4096) are far below the gap at this shape. A 4-warps-per-CTA launcher variant recovers about a third of it but stays behind.

Quality (measured while the port was in tree, for the record if ever revived): `test-backend-ops -o GATED_DELTA_NET -o GATED_DELTA_NET_CACHE_FUSION` 43/43 OK with the chunked route engaged (chunked shapes 64/127/256/65/100/200 tokens, KDA 64/33/100, K>1 snapshots, cache fusion). PPL 9B c1024x2 chunked `4.3786 / 6.2862` vs step `4.3684 / 6.2945` (within the +/- 0.49 bar; not bit-identical by construction). Build clean via the script.

Revisit triggers: a Kimi-class model in the bench set (KDA chunk = 16 tokens, smaller S_v, possibly different CTA economics), or a future upstream/mx performance fix to the chunk kernel's column layout. Not worth keeping as an env on this card: every default-ON route regresses pp on the only models benched.

## GDN Gate Fusion: ADD -> UNARY -> MUL + folded SIGMOID (2026-11-09, committed)

Ported from furnace gfx906-perf `50798f69b` (sixvolts + Claude Opus 5.5, adapted). New `add_unary_mul_kernel<op>` + `ggml_cuda_op_add_unary_mul` in `unary.cu/.cuh` (+72) and a pattern hook in `ggml_cuda_try_fuse` (ggml-cuda.cu, +22): `softplus(alpha + dt_bias) * ssm_a` and the beta `sigmoid(beta_proj)` (when it is the next non-view node and independent) run in one launch per recurrent layer per token instead of three. Gate: `GGML_CUDA_NO_ADD_UNARY_FUSION=1`; the base `ggml_cuda_fusion_disabled()` global guard covers it too.

Adapted to the base's newer fuse checker: furnace used the legacy 4-arg `ggml_can_fuse`; this tree calls `ggml_can_fuse_subgraph(cgraph, i, 3, {ADD, UNARY, MUL}, {i+2}, 1)` (the new API asserts a non-empty `outputs` list, so the MUL is declared the keep-node; ADD/UNARY uses must confine to the subgraph).

Measurement (9B qwen35, Pro VII, 3x interleaved pairs, tg1024 + pp4096, GFXPROF nograph at n=1):

| | fusion off | fusion on |
|---|---:|---:|
| launches per decode token (GFXPROF rows) | 50 ADD + 48 UNARY | 2 ADD + 48 fused |
| serialized per-token GPU total | 131.0 | 120.9 |
| tg1024 | 82.99 mean | 84.11 mean (**+1.3%**, 3/3) |
| pp4096 | ~980.6/989.5 | ~986.5/991.5 (+0.3-0.7%, 2/2) |
| PPL 9B c1024x2 | 4.3684 / 6.2945 | **4.3684 / 6.2945 (bit-identical)** |

The beta SIGMOID fold-in is opportunistic (checks the same size, contiguity, no-data-overlap against the gate output and independence); it did not engage on qwen35-9B in this run - the win is the 3-node gate folding alone. Same-arithmetic per element, so PPL is bit-exact. `test-backend-ops -o UNARY -o ADD -o RMS_NORM -o GLU` all OK; `build-llamacpp-rocm.sh` clean.

Applies to every GATED_DELTA_NET model (qwen3.5 family, Kimi KDA, glm5-next, clef, bailingmoe3). Donor's own reference on Qwen3.5-4B: launches 745 -> 634, tg256 +6.6%.

## Fused Recurrent Conv Step (2026-11-09, committed)

Ported from furnace gfx906-perf `3a30ac2c2` + `075dbfe61` (sixvolts + Claude Opus 5.5). For the GDN conv state on decode / verify batches: GET_ROWS(conv cache) -> RESHAPE -> CONCAT(state, x^T) and the state-tail CPYs run as one launch (`conv_step_concat_f32` in `ssm-conv.cu`, +334 lines; the gather is elided - the kernel reads the cache through the row indices - when nothing between the gather and the concat writes the cache rows or reuses the index memory; one thread per channel reads every sequence before writing any, so the multi-sequence write-under-read hazard is excluded). The SSM_CONV -> SILU that reads the concat is then absorbed into the same step kernel: `graph_optimize` moves it right behind the CONCAT's CPYs (`ggml_cuda_conv_step_graph_optimize`, pre-allocation; the moved nodes need only the concat and the conv weights), and the step computes silu(conv) per channel in `ssm_conv_f32<true>` order. Gates: `GGML_CUDA_NO_CONV_STEP_FUSION=1` (whole thing) and `GGML_CUDA_NO_CONV_STEP_CONV=1` (SILU absorption only; parity on this card - the base's SSM_CONV+ADD+SILU fusion already covers it - kept as a launch-count notch for when general fusion is off).

Changes: `ssm-conv.cu` +251, `ssm-conv.cuh` +11, `ggml-cuda.cu` +26 (CONCAT hook first in `try_fuse`, the conv-state `GET_ROWS` elision continue in the capture loop, the graph_optimize call beside the q8-multi-group moves, `ggml_cuda_is_view_or_noop_public`).

Validation (0.8B BF16 + 9B qwen35, Pro VII):

- Launches at 1 decode token (GFXPROF nograph): CONCAT 36 -> 0, GET_ROWS 76 -> 40 (the 36 conv-state gathers elided on the 0.8B).
- PPL, both models (c 1024, 2 chunks): bit-identical on/off (`0.8B 7.8077 / 13.0645`; `9B 4.3684 / 6.2945`).
- tg1024 A/B (2x interleaved each): 0.8B 200.08-200.12 vs **202.85-205.98** (+1.4-3%); 9B 83.44-83.55 vs **84.82-85.21** (+1.6-2%).
- Batched decode `-npp 128 -ntg 128 -npl 1,2,4,8` (S_TG, on vs off): B=1 205.9 vs 197.2 (+4.4%), B=2 312.9 vs 298.0 (+5.0%), B=4 435.7 vs 424.3 (+2.7%), B=8 422.5 vs 419.7 (+0.7%).
- `test-backend-ops -o SSM_CONV -o GATED_DELTA_NET` all OK; `build-llamacpp-rocm.sh` clean.

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

---

## FATTN Decode Attention: quantized-KV GQA gap measured (2026-11-08, no code landed)

Problem: at single-token decode (`Q->ne[1] == 1`) on GCN with quantized K/V, `fattn.cu` always
returns `BEST_FATTN_KERNEL_VEC`: the three native-tile predicates (`fattn.cu:556/572/588`) require
`Q->ne[1] > 2`, and the GCN guard at `fattn.cu:732` falls through to VEC below that. VEC then runs
with `cols_per_block = 1` (`fattn-vec.cuh:560-569`), i.e. one block per (q-head, KV chunk), mapping
`K += nb12*(head / gqa_ratio)` (`fattn-vec.cuh:112-113`): per q-head K/V loads, and per q-head
re-executed V dequant + P.V FMA. f16/f32 KV with `gqa_opt_applies` already routes to the TILE kernel
with `ncols2 = gqa_ratio_eff` (heads grouped per KV head), so only quantized-KV decode pays the
linear-in-heads loop. Same dispatch shape exists on CUDA (`fattn.cu` Turing/MMA branch returns VEC at
`Q->ne[1] == 1` for quantized KV): not GCN-specific.

Measurement (Radeon Pro VII 16 GB, ROCm 7.2.4, build `f518082a3`, `GGML_CUDA_DISABLE_GRAPHS=1`
(see side finding below), `GFXPROF=1` per-op HIP events): model `Spark-X2.6-4B-Q8_0` (36 layers,
16 heads, 4 KV heads, GQA r=4, head_dim 256, dense attention), `llama-bench -fa on -p 0 -d <L>
-n {1,129} -r 1 --no-warmup`, decode attention per token = `(FLASH_ATTN_EXT tot_ms at n=129 -
tot_ms at n=1) / 128` (depth fill cancels out; call counts in the GFXPROF rows confirm the
ubatch bookkeeping).

| KV | kernel at nc=1 | depth 512 | 4K | 8K |
|---|---|---:|---:|---:|
| f16 | TILE, grouped (ncols2=4) | 966 | 1373 | 1609 |
| q8_0 | VEC, cols=1 (current dispatch) | 1504 | 2138 | 2669 |
| q8_0 | native TILE at ncols1=1 (4-line patch + `GGML_CUDA_FATTN_PATH=force_native`) | 1409 | 2288 | 2852 |

(all values `us/token` over 36 layers)

Findings:

- Loads are not the cost: the x4 K/V re-read is mostly L2-absorbed (641 MB HBM per step once-read,
  ~241 GB/s effective). The kernel is VALU bound: V dequant + P.V FMA run once per q head (x4
  here); the duplicated K-side dp4a is cheap.
- Depth scaling confirms compute/latency bound, not bandwidth: x16 depth grows time only 1.77x;
  steady-state segment ~258 GB/s once-read equivalent, vs 751 GB/s the f16 TILE sustains at the
  same nc1=1 x ncols2=4 geometry - the hardware headroom is real.
- Rerouting decode to the existing native q8_0 tile is falsified: a 4-line patch (relax
  `Q->ne[1] > 2` to `>= 1` in `fattn.cu:556/572/588` and the `fattn-tile.cuh:1598` native-cascade
  gate, A/B via `GGML_CUDA_FATTN_PATH`, validated with `test-backend-ops -o FLASH_ATTN_EXT` first)
  measured 1409/2288/2852 - the tile's int8/V LDS staging does not amortize at ncols1=1. Patch
  measured and reverted; tree and build match `f518082a3`.
- Grouping inside the VEC kernel itself is falsified as well (2026-11-08, same protocol, env-gated
  WIP on `f518082a3`, reverted after measurement): template `flash_attn_ext_vec` with
  `head_grouped` - one block covers all `gqa_ratio` q heads of one KV head, KQ dot lanes 4 -> 8 to
  halve the per-head q8 VGPR (D=256/hg=4 247 VGPR, D=128/hg=8 255 VGPR, both spill-free per
  `amdgcn-audit.py`; hg 8 gated to D <= 128, D=256/hg=8 = 256 VGPR + 32 spill). hg must equal
  `gqa_ratio`. `launch_fattn<D, 1, hg>` and `flash_attn_combine_results` reused unchanged. Dispatch
  gate `GGML_CUDA_FATTN_GQA_VEC=1`, GCN-only. Correctness: `test-backend-ops -o FLASH_ATTN_EXT`
  4304/4304 with the env set. Measured decode attention per token (FLASH tot n129 - n1, r=1
  no-warmup, `force_native`, Spark 4B q8_0 KV, interleaved OFF/ON per depth): stock
  1484/2144/2797 us at d512/4K/8K (matches the 1504/2138/2669 recorded above) vs grouped hg=4
  4794/4909/5475 - 2.0-3.2x WORSE at every depth. End-to-end tg128 d8K: 88.59 stock vs 70.42
  grouped. Reverted; tree and build match `f518082a3`.
- Why grouping loses (corrected bottleneck model): decode attention concurrency is bounded by work
  units = KV heads x KV chunks (4 x ~4 = 16 units at d512). Stock VEC manufactures 4x more units
  (q heads x chunks) out of the redundant per-head K/V reads - reads that are mostly L2 absorbed,
  i.e. nearly free parallelism. Grouping amortizes the V dequant but cannot create units: at d512
  the grouped grid holds 16 blocks x 2 warps = 32 wave64s on 60 CUs (over half the machine idle)
  while each wave does up to 4x the work. Deep KV raises parallel_blocks (~30 at 8K fills the
  machine), but the 247 VGPR 4-head Q state cuts occupancy to 4 waves/CU (vs 7 at 129 VGPR) and
  the per-wave ILP still carries the x4 work - still ~2x. The ceiling priced the fix at the f16
  tile's 751 GB/s streaming rate; the q8_0 decode path is compute/latency bound, not bandwidth
  bound, so amortizing reads cannot pay for the lost wave count.
- Measurement hygiene: GFXPROF FA deltas from back-to-back r=1 runs drift with clock/cache state.
  Always interleave OFF/ON per depth in one session; the first (sequential) pass of the grouped-VEC
  A/B produced flat-off stock numbers that made the regression look like nothing - the interleaved
  pass above reproduced the doc's stock rows exactly.
- End-to-end cost of the gap (`llama-bench -n 128 -r 3`, nograph): q8_0 KV tg128 d4K `98.27 +/- 0.76`
  / d8K `92.99 +/- 0.73` vs f16 KV `110.84 +/- 1.30` / `109.74 +/- 1.34`. Attention share of decode
  wall: q8_0 21% (d4K) / 25% (d8K); f16 15% / 18%. Quantized KV currently costs 8-16% tg for the
  VRAM it saves (~1 GB at 8K on this card).

Candidate fix (built and falsified 2026-11-08): the head-grouped narrow decode variant inside
`fattn-vec.cuh` - block = (KV chunk, all r q-heads of one KV head), K/V loads and V dequant once,
per-head KQ dp4a + online max, reusing the VEC KV-split + `k_combine` merge and the GCN fast
vecdotq, spill-free at 247-255 VGPR via 8-lane KQ dots - measured 2.0-3.2x WORSE than stock at
every depth; see the grouped-VEC bullet above. Also not worth pursuing (checked):
`parallel_blocks` tuning (does not remove the xgqa VALU work; and the unit count, not the split,
is the binding constraint), `force_convert` at decode (re-copies KV to f16 per step, >=3x
traffic), mixed K/V type tricks (no native tile path exists for them at nc=1), and sub-groups
(hg < gqa_ratio would need a partial-group grid; the unit-count argument covers it). With
grouping falsified in both the VEC and the TILE geometry, the quantized-KV decode gap on GCN is
structural at this kernel generation: closing it needs a different kernel shape (per-wave work
must not scale with gqa_ratio while unit count stays KV heads x chunks - a hard ask on wave64
dp4a), otherwise accept the 8-16% tg cost for the VRAM quantized KV saves.

Side findings:

- `llama-bench` decode aborts under HIP graphs on this build: `operation not permitted when stream
  is capturing` in the buffer pool alloc (`ggml-cuda.cu:565`), also reproducing on the documented
  `-p 512 -n 128 -r 1` invocation without GFXPROF. Prefill completes; the abort hits the first
  decode capture. Probably VRAM/allocator-timing dependent - the 2026-11-08 validation runs pass
  on the 32 GB MI50 per the table above, this is the 16 GB Pro VII. Open item, unrelated to FATTN;
  all numbers here are nograph.
- `ggml_cuda_fattn_auto_native` keeps f16-convert while the f16 shadow fits (20% free-mem headroom),
  so for small caches the native tile machinery is dormant by design; it only engages via
  `GGML_CUDA_FATTN_PATH=force_native`, and then only where the `fattn.cu` predicates pass
  (`Q->ne[1] > 2` today). Relevant whenever touching the tile dispatch.
- fattn-mma is unreachable on gfx906 at base `a691b2345`: every `BEST_FATTN_KERNEL_MMA_F16`
  return site is gated on Turing/Volta (NVIDIA), `amd_mfma_available` (CDNA only,
  `common.cuh:359-361`) or `amd_wmma_available` (RDNA3/4, `common.cuh:367-368`). Per the
  Vega 7nm ISA doc, gfx906 has no MFMA/matrix-core instructions at all: packed math only
  (`V_PK_FMA_F16`, `V_DOT2_F32_F16`, `V_DOT4_I32_I8`, `V_DOT8_I32_I4`). What runs is fattn-tile,
  which does emit real `v_dot2_f32_f16` (`V_DOT2_F32_F16_AVAILABLE`, `common.cuh:902-904`).
- Fork commit 19 (`b143fd169`, 2026-09-10) clamped the fattn-mma config for GCN
  (nstages=1, occupancy=1), justified in the body by vLLM-gfx906 Triton flash-attn tuning
  ("2 stages doubles K+V LDS > 64KB, spills"). It never had a live window: at every fork base
  (`ec91ab5ad`, `311d4211b`, `a691b2345`) the dispatch already excluded gfx906, upstream has no
  analogous behavior (no GCN references in upstream `fattn-mma-f16.cuh`), and no fork commit ever
  touched the MMA dispatch sites. No llama.cpp measurement ever backed the "kills FATTN spill"
  effect. Dropped 2026-11-08: `fattn-mma-f16.cuh` is byte-identical to `origin/master` now,
  build clean, `llama-bench` smoke pass.
- fattn-tile efficiency on gfx906 (same GFXPROF data): prefill attention at the depth-512 fill,
  f16 KV = 51.1 ms for 38.7 GMAC whole-model = 0.76 TMAC/s = 5.8% of the `v_dot2` peak
  (13.1 TMAC/s at 1.82 GHz, same quarter-wave-rate accounting as the ~28 TMAC/s dp4a figure),
  ~21% of the pp512 wall on `Spark-X2.6-4B-Q8_0`. A same-instruction-class tiled kernel holds
  ~13% of peak - roughly 2.3x headroom. Open item: better tiled prefill attention geometry, or
  leave if prefill share benchmarks smaller on the main bench models.

## Maintaining This Document

- Edit this file on every `git push` to `gfx906/mi50-optimization` or `origin/master` sync.
- Keep commit table in reverse-chronological log order (`git log --reverse`).
- The full commit bodies stay in `git show <sha>` - do not paraphrase from memory; this doc only
  mirrors them for recent rounds and compresses older ones into the tables above.
- Performance deltas are from commit messages - include `llama-bench` / `batched-bench` / `wikitext PPL` numbers when present.
- Remove or collapse older tuning experiments (e.g., rejected `ROWS=4`, quarter-units) only after they are stable for >2 rebases.
