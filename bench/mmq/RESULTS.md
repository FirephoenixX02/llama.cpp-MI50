# MMQ config static sweep, gfx906 (2026-11-27)

Static sweep only: no GPU runs, no timing, no correctness tests. Every number below
comes from the amdgpu code-object metadata of freshly compiled MMQ instance objects
(`amdgcn-audit.py`) plus the shared-memory formula the launcher itself uses.

## Scope

`ggml/src/ggml-cuda/mmq-config-vega.cuh`, the table that `ggml_cuda_mmq_get_config_vega`
returns for gfx906. Types the local GGUFs actually dispatch (Qwen3.5-0.8B/9B, Spark 4B):
`Q4_0, Q8_0, Q4_K, Q5_K, Q6_K`.

Current table is almost uniform: `nthreads 256, occupancy 1, I 128, K_vram MMQ_ITER_K (256)`
for every type, except Q6_K which came out of an earlier probe as `I 64, occupancy 2`.

Swept knobs per (type, J, fallback) row:

| knob | values |
|------|--------|
| `I` (tile rows) | 64, 128, 192, 256 |
| `nthreads` | 128, 256, 512 |
| `occupancy` (`__launch_bounds__` 2nd arg) | 1, 2, 3 |
| `K_vram` | 256, 512 |

`sram_layout`, `stream_k` and `J` stay as the table has them: they change data layout or
work decomposition, not just resource use, so they are not a safe static sweep.

## Method

1. Patch every `CASE(...)` row for the 5 types in `mmq-config-vega.cuh`.
2. Compile the single instance TU (`template-instances/mmq-instance-<type>.cu`) with the
   exact flags from `build/compile_commands.json`, 5 TUs in parallel. ~20 s per TU, whole
   sweep (21 cells) ~8 min on 48 cores. Nothing else in the tree is rebuilt.
3. Read `.vgpr_count`, `.sgpr_count`, `.vgpr_spill_count`, `.sgpr_spill_count` per kernel
   from the code object with `amdgcn-audit.py`, decoding the template arguments out of the
   mangled `mul_mat_q<type, J, fallback, prec, has_fusion>` symbol.
4. Compute LDS per (type, I, J, nthreads) with `mmq_get_nbytes_shared`:
   `J*4 + txs(I)*4 + pad(J*144, nthreads*4)`, `txs` from `mmq_get_dp4a_tile_x_sizes`
   (gfx906 has no MFMA/WMMA, so the dp4a layout is the one compiled).
5. Residency per CU: `min(occupancy, 65536/LDS, (1024/VGPR)/waves_per_block,
   4*(800/SGPR)/waves_per_block)`, `waves_per_block = nthreads/64`.

Why step 4 is needed: MMQ uses `extern __shared__`, so the metadata reports `LDS 0` for
every instance and the audit alone cannot answer "does a second block fit". The formula is
the launcher's own, and it reproduces the one known point: Q6_K, I=64, J=64, nthreads=256
gives 27432 B, that is 2 blocks per 64 KiB CU, which matches the comment in the table.

## Results

### A. Current Vega table: residency is capped by `__launch_bounds__`, not by LDS

| type | I | nthreads | occ | J | VGPR | spill | SGPR | LDS KiB | blocks by LDS | resident blocks |
|------|---|----------|-----|---|------|-------|------|---------|---------------|-----------------|
| Q4_0 | 128 | 256 | 1 | 8 | 80 | 0 | 34 | 22.7 | 2 | 1 |
| Q4_0 | 128 | 256 | 1 | 16 | 116 | 0 | 34 | 23.7 | 2 | 1 |
| Q4_0 | 128 | 256 | 1 | 32 | 115 | 0 | 34 | 25.8 | 2 | 1 |
| Q4_0 | 128 | 256 | 1 | 64 | 108 | 0 | 34 | 29.9 | 2 | 1 |
| Q4_K | 128 | 256 | 1 | 8 | 125 | 0 | 34 | 21.1 | 3 | 1 |
| Q4_K | 128 | 256 | 1 | 16 | 114 | 0 | 34 | 22.1 | 2 | 1 |
| Q4_K | 128 | 256 | 1 | 32 | 121 | 0 | 34 | 24.2 | 2 | 1 |
| Q4_K | 128 | 256 | 1 | 64 | 137 | 0 | 34 | 28.3 | 2 | 1 |
| Q5_K | 128 | 256 | 1 | 8 | 118 | 0 | 34 | 37.1 | 1 | 1 |
| Q5_K | 128 | 256 | 1 | 16 | 119 | 0 | 34 | 38.1 | 1 | 1 |
| Q5_K | 128 | 256 | 1 | 32 | 129 | 0 | 34 | 40.2 | 1 | 1 |
| Q5_K | 128 | 256 | 1 | 64 | 141 | 0 | 34 | 44.3 | 1 | 1 |
| Q6_K | 64 | 256 | 2 | 8 | 76 | 0 | 32 | 19.6 | 3 | 2 |
| Q6_K | 64 | 256 | 2 | 16 | 83 | 0 | 34 | 20.6 | 3 | 2 |
| Q6_K | 64 | 256 | 2 | 32 | 84 | 0 | 34 | 22.7 | 2 | 2 |
| Q6_K | 64 | 256 | 2 | 64 | 80 | 0 | 34 | 26.8 | 2 | 2 |
| Q8_0 | 128 | 256 | 1 | 8 | 126 | 0 | 34 | 38.7 | 1 | 1 |
| Q8_0 | 128 | 256 | 1 | 16 | 110 | 0 | 34 | 39.7 | 1 | 1 |
| Q8_0 | 128 | 256 | 1 | 32 | 119 | 0 | 34 | 41.8 | 1 | 1 |
| Q8_0 | 128 | 256 | 1 | 64 | 125 | 0 | 34 | 45.9 | 1 | 1 |

### B. J = 64, fallback = false: static matrix (type x I x nthreads x occ)

| type | I | nthreads | occ | VGPR | spill | SGPR | LDS KiB | blocks by LDS | resident blocks |
|------|---|----------|-----|------|-------|------|---------|---------------|-----------------|
| Q4_0 | 64 | 128 | 1 | 90 | 0 | 34 | 19.6 | 3 | 1 |
| Q4_0 | 64 | 128 | 2 | 108 | 0 | 34 | 19.6 | 3 | 2 |
| Q4_0 | 64 | 256 | 1 | 73 | 0 | 34 | 19.6 | 3 | 1 |
| Q4_0 | 64 | 256 | 2 | 66 | 0 | 34 | 19.6 | 3 | 2 |
| Q4_0 | 64 | 256 | 3 | 66 | 0 | 34 | 19.6 | 3 | 3 |
| Q4_0 | 64 | 512 | 1 | 77 | 0 | 34 | 20.6 | 3 | 1 |
| Q4_0 | 64 | 512 | 2 | 77 | 0 | 34 | 20.6 | 3 | 1 |
| Q4_0 | 128 | 128 | 1 | 184 | 0 | 34 | 29.9 | 2 | 1 |
| Q4_0 | 128 | 128 | 2 | 128 | 29 | 44 | 29.9 | 2 | 2 |
| Q4_0 | 128 | 256 | 1 | 108 | 0 | 34 | 29.9 | 2 | 1 |
| Q4_0 | 128 | 256 | 2 | 108 | 0 | 34 | 29.9 | 2 | 2 |
| Q4_0 | 128 | 512 | 1 | 79 | 0 | 34 | 30.9 | 2 | 1 |
| Q4_0 | 128 | 512 | 2 | 79 | 0 | 34 | 30.9 | 2 | 1 |
| Q4_0 | 192 | 256 | 1 | 108 | 0 | 34 | 40.2 | 1 | 1 |
| Q4_0 | 256 | 128 | 1 | 256 | 39 | 44 | 50.5 | 1 | 1 |
| Q4_0 | 256 | 128 | 2 | 128 | 277 | 44 | 50.5 | 1 | 1 |
| Q4_0 | 256 | 256 | 1 | 190 | 0 | 34 | 50.5 | 1 | 1 |
| Q4_0 | 256 | 256 | 2 | 128 | 63 | 44 | 50.5 | 1 | 1 |
| Q4_0 | 256 | 512 | 1 | 128 | 6 | 44 | 51.5 | 1 | 1 |
| Q4_0 | 256 | 512 | 2 | 128 | 6 | 44 | 51.5 | 1 | 1 |
| Q4_K | 64 | 128 | 1 | 99 | 0 | 34 | 18.8 | 3 | 1 |
| Q4_K | 64 | 128 | 2 | 137 | 0 | 34 | 18.8 | 3 | 2 |
| Q4_K | 64 | 256 | 1 | 73 | 0 | 34 | 18.8 | 3 | 1 |
| Q4_K | 64 | 256 | 2 | 73 | 0 | 34 | 18.8 | 3 | 2 |
| Q4_K | 64 | 256 | 3 | 73 | 0 | 34 | 18.8 | 3 | 3 |
| Q4_K | 64 | 512 | 1 | 64 | 0 | 34 | 19.8 | 3 | 1 |
| Q4_K | 64 | 512 | 2 | 64 | 0 | 34 | 19.8 | 3 | 2 |
| Q4_K | 128 | 128 | 1 | 189 | 0 | 34 | 28.3 | 2 | 1 |
| Q4_K | 128 | 128 | 2 | 128 | 65 | 44 | 28.3 | 2 | 2 |
| Q4_K | 128 | 256 | 1 | 137 | 0 | 34 | 28.3 | 2 | 1 |
| Q4_K | 128 | 256 | 2 | 128 | 10 | 44 | 28.3 | 2 | 2 |
| Q4_K | 128 | 512 | 1 | 111 | 0 | 34 | 29.3 | 2 | 1 |
| Q4_K | 128 | 512 | 2 | 111 | 0 | 34 | 29.3 | 2 | 1 |
| Q4_K | 192 | 256 | 1 | 137 | 0 | 34 | 37.8 | 1 | 1 |
| Q4_K | 256 | 128 | 1 | 256 | 92 | 44 | 47.4 | 1 | 1 |
| Q4_K | 256 | 128 | 2 | 128 | 444 | 44 | 47.4 | 1 | 1 |
| Q4_K | 256 | 256 | 1 | 235 | 0 | 34 | 47.4 | 1 | 1 |
| Q4_K | 256 | 256 | 2 | 128 | 193 | 44 | 47.4 | 1 | 1 |
| Q4_K | 256 | 512 | 1 | 128 | 67 | 44 | 48.4 | 1 | 1 |
| Q4_K | 256 | 512 | 2 | 128 | 67 | 44 | 48.4 | 1 | 1 |
| Q5_K | 64 | 128 | 1 | 102 | 0 | 34 | 26.8 | 2 | 1 |
| Q5_K | 64 | 128 | 2 | 141 | 0 | 34 | 26.8 | 2 | 2 |
| Q5_K | 64 | 256 | 1 | 77 | 0 | 34 | 26.8 | 2 | 1 |
| Q5_K | 64 | 256 | 2 | 76 | 0 | 34 | 26.8 | 2 | 2 |
| Q5_K | 64 | 256 | 3 | 76 | 0 | 34 | 26.8 | 2 | 2 |
| Q5_K | 64 | 512 | 1 | 85 | 0 | 34 | 27.8 | 2 | 1 |
| Q5_K | 64 | 512 | 2 | 85 | 0 | 34 | 27.8 | 2 | 1 |
| Q5_K | 128 | 128 | 1 | 196 | 0 | 34 | 44.3 | 1 | 1 |
| Q5_K | 128 | 128 | 2 | 128 | 72 | 44 | 44.3 | 1 | 1 |
| Q5_K | 128 | 256 | 1 | 141 | 0 | 34 | 44.3 | 1 | 1 |
| Q5_K | 128 | 256 | 2 | 128 | 14 | 44 | 44.3 | 1 | 1 |
| Q5_K | 128 | 512 | 1 | 115 | 0 | 34 | 45.3 | 1 | 1 |
| Q5_K | 128 | 512 | 2 | 115 | 0 | 34 | 45.3 | 1 | 1 |
| Q5_K | 192 | 256 | 1 | 141 | 0 | 34 | 61.9 | 1 | 1 |
| Q5_K | 256 | 128 | 1 | 256 | 102 | 44 | 79.4 | 0 | 0 |
| Q5_K | 256 | 128 | 2 | 128 | 456 | 44 | 79.4 | 0 | 0 |
| Q5_K | 256 | 256 | 1 | 245 | 0 | 34 | 79.4 | 0 | 0 |
| Q5_K | 256 | 256 | 2 | 128 | 204 | 44 | 79.4 | 0 | 0 |
| Q5_K | 256 | 512 | 1 | 128 | 70 | 44 | 80.4 | 0 | 0 |
| Q5_K | 256 | 512 | 2 | 128 | 70 | 44 | 80.4 | 0 | 0 |
| Q6_K | 64 | 128 | 1 | 106 | 0 | 34 | 26.8 | 2 | 1 |
| Q6_K | 64 | 128 | 2 | 139 | 0 | 34 | 26.8 | 2 | 2 |
| Q6_K | 64 | 256 | 1 | 80 | 0 | 34 | 26.8 | 2 | 1 |
| Q6_K | 64 | 256 | 2 | 80 | 0 | 34 | 26.8 | 2 | 2 |
| Q6_K | 64 | 256 | 3 | 80 | 0 | 34 | 26.8 | 2 | 2 |
| Q6_K | 64 | 512 | 1 | 68 | 0 | 34 | 27.8 | 2 | 1 |
| Q6_K | 64 | 512 | 2 | 68 | 0 | 34 | 27.8 | 2 | 1 |
| Q6_K | 128 | 128 | 1 | 194 | 0 | 32 | 44.3 | 1 | 1 |
| Q6_K | 128 | 128 | 2 | 128 | 69 | 44 | 44.3 | 1 | 1 |
| Q6_K | 128 | 256 | 1 | 139 | 0 | 34 | 44.3 | 1 | 1 |
| Q6_K | 128 | 256 | 2 | 128 | 11 | 44 | 44.3 | 1 | 1 |
| Q6_K | 128 | 512 | 1 | 113 | 0 | 34 | 45.3 | 1 | 1 |
| Q6_K | 128 | 512 | 2 | 113 | 0 | 34 | 45.3 | 1 | 1 |
| Q6_K | 192 | 256 | 1 | 139 | 0 | 34 | 61.9 | 1 | 1 |
| Q6_K | 256 | 128 | 1 | 256 | 96 | 44 | 79.4 | 0 | 0 |
| Q6_K | 256 | 128 | 2 | 128 | 446 | 44 | 79.4 | 0 | 0 |
| Q6_K | 256 | 256 | 1 | 240 | 0 | 34 | 79.4 | 0 | 0 |
| Q6_K | 256 | 256 | 2 | 128 | 193 | 44 | 79.4 | 0 | 0 |
| Q6_K | 256 | 512 | 1 | 128 | 92 | 44 | 80.4 | 0 | 0 |
| Q6_K | 256 | 512 | 2 | 128 | 92 | 44 | 80.4 | 0 | 0 |
| Q8_0 | 64 | 128 | 1 | 105 | 0 | 34 | 27.6 | 2 | 1 |
| Q8_0 | 64 | 128 | 2 | 125 | 0 | 34 | 27.6 | 2 | 2 |
| Q8_0 | 64 | 256 | 1 | 81 | 0 | 34 | 27.6 | 2 | 1 |
| Q8_0 | 64 | 256 | 2 | 80 | 0 | 34 | 27.6 | 2 | 2 |
| Q8_0 | 64 | 256 | 3 | 80 | 0 | 34 | 27.6 | 2 | 2 |
| Q8_0 | 64 | 512 | 1 | 56 | 0 | 34 | 28.6 | 2 | 1 |
| Q8_0 | 64 | 512 | 2 | 56 | 0 | 34 | 28.6 | 2 | 2 |
| Q8_0 | 128 | 128 | 1 | 208 | 0 | 34 | 45.9 | 1 | 1 |
| Q8_0 | 128 | 128 | 2 | 128 | 74 | 44 | 45.9 | 1 | 1 |
| Q8_0 | 128 | 256 | 1 | 125 | 0 | 34 | 45.9 | 1 | 1 |
| Q8_0 | 128 | 256 | 2 | 125 | 0 | 34 | 45.9 | 1 | 1 |
| Q8_0 | 128 | 512 | 1 | 87 | 0 | 34 | 46.9 | 1 | 1 |
| Q8_0 | 128 | 512 | 2 | 87 | 0 | 34 | 46.9 | 1 | 1 |
| Q8_0 | 192 | 256 | 1 | 125 | 0 | 34 | 64.2 | 0 | 0 |
| Q8_0 | 256 | 128 | 1 | 256 | 141 | 44 | 82.5 | 0 | 0 |
| Q8_0 | 256 | 128 | 2 | 128 | 374 | 44 | 82.5 | 0 | 0 |
| Q8_0 | 256 | 256 | 1 | 239 | 0 | 34 | 82.5 | 0 | 0 |
| Q8_0 | 256 | 256 | 2 | 128 | 93 | 44 | 82.5 | 0 | 0 |
| Q8_0 | 256 | 512 | 1 | 128 | 4 | 44 | 83.5 | 0 | 0 |
| Q8_0 | 256 | 512 | 2 | 128 | 4 | 44 | 83.5 | 0 | 0 |

### C. Candidate row `I=64, nthreads=256, occ=2` against the current row, all J

| type | J | current: VGPR / spill / LDS KiB / blocks | I64 nt256 occ2: VGPR / spill / LDS KiB / blocks |
|------|---|------------------------------------------|-----------------------------------------------|
| Q4_0 | 8 | 80 / 0 / 22.7 / 1 | 58 / 0 / 12.3 / 2 |
| Q4_0 | 16 | 116 / 0 / 23.7 / 1 | 57 / 0 / 13.4 / 2 |
| Q4_0 | 24 | 114 / 0 / 24.7 / 1 | 63 / 0 / 14.4 / 2 |
| Q4_0 | 32 | 115 / 0 / 25.8 / 1 | 64 / 0 / 15.4 / 2 |
| Q4_0 | 40 | 126 / 0 / 26.8 / 1 | 64 / 0 / 16.5 / 2 |
| Q4_0 | 48 | 101 / 0 / 27.8 / 1 | 64 / 0 / 17.5 / 2 |
| Q4_0 | 64 | 108 / 0 / 29.9 / 1 | 66 / 0 / 19.6 / 2 |
| Q4_K | 8 | 125 / 0 / 21.1 / 1 | 64 / 0 / 11.6 / 2 |
| Q4_K | 16 | 114 / 0 / 22.1 / 1 | 64 / 0 / 12.6 / 2 |
| Q4_K | 24 | 118 / 0 / 23.2 / 1 | 85 / 0 / 13.6 / 2 |
| Q4_K | 32 | 121 / 0 / 24.2 / 1 | 89 / 0 / 14.7 / 2 |
| Q4_K | 40 | 137 / 0 / 25.2 / 1 | 88 / 0 / 15.7 / 2 |
| Q4_K | 48 | 179 / 0 / 26.2 / 1 | 90 / 0 / 16.7 / 2 |
| Q4_K | 64 | 137 / 0 / 28.3 / 1 | 73 / 0 / 18.8 / 2 |
| Q5_K | 8 | 118 / 0 / 37.1 / 1 | 66 / 0 / 19.6 / 2 |
| Q5_K | 16 | 119 / 0 / 38.1 / 1 | 87 / 0 / 20.6 / 2 |
| Q5_K | 24 | 121 / 0 / 39.2 / 1 | 86 / 0 / 21.6 / 2 |
| Q5_K | 32 | 129 / 0 / 40.2 / 1 | 88 / 0 / 22.7 / 2 |
| Q5_K | 40 | 174 / 0 / 41.2 / 1 | 71 / 0 / 23.7 / 2 |
| Q5_K | 48 | 180 / 0 / 42.3 / 1 | 78 / 0 / 24.7 / 2 |
| Q5_K | 64 | 141 / 0 / 44.3 / 1 | 76 / 0 / 26.8 / 2 |
| Q6_K | 8 | 76 / 0 / 19.6 / 2 | 76 / 0 / 19.6 / 2 |
| Q6_K | 16 | 83 / 0 / 20.6 / 2 | 83 / 0 / 20.6 / 2 |
| Q6_K | 24 | 88 / 0 / 21.6 / 2 | 88 / 0 / 21.6 / 2 |
| Q6_K | 32 | 84 / 0 / 22.7 / 2 | 84 / 0 / 22.7 / 2 |
| Q6_K | 40 | 75 / 0 / 23.7 / 2 | 75 / 0 / 23.7 / 2 |
| Q6_K | 48 | 77 / 0 / 24.7 / 2 | 77 / 0 / 24.7 / 2 |
| Q6_K | 64 | 80 / 0 / 26.8 / 2 | 80 / 0 / 26.8 / 2 |
| Q8_0 | 8 | 126 / 0 / 38.7 / 1 | 63 / 0 / 20.3 / 2 |
| Q8_0 | 16 | 110 / 0 / 39.7 / 1 | 121 / 0 / 21.4 / 2 |
| Q8_0 | 24 | 114 / 0 / 40.7 / 1 | 60 / 0 / 22.4 / 2 |
| Q8_0 | 32 | 119 / 0 / 41.8 / 1 | 62 / 0 / 23.4 / 2 |
| Q8_0 | 40 | 124 / 0 / 42.8 / 1 | 70 / 0 / 24.5 / 2 |
| Q8_0 | 48 | 130 / 0 / 43.8 / 1 | 73 / 0 / 25.5 / 2 |
| Q8_0 | 64 | 125 / 0 / 45.9 / 1 | 80 / 0 / 27.6 / 2 |

## Findings

1. **On this card the Vega MMQ table runs at 1 block per CU for every type except Q6_K.**
   That is what `occupancy 1` in `__launch_bounds__` asks for, and for Q5_K/Q8_0 it is also
   what the 37-46 KiB of LDS allow. For Q4_0/Q4_K it is not: LDS would fit 2 blocks at
   I=128 for every J up to 64 (21-30 KiB), and 3 blocks at J<=16.

2. **`I=64` with `occupancy 2` fits 2 blocks per CU for all five types, all J <= 64,
   with zero spill.** VGPR is 57-121 (down from 80-180), LDS is 11.6-28.3 KiB (down from
   21.1-45.9 KiB). Q6_K is unchanged by this, it already sits there. Q5_K and Q8_0 are the
   big movers: 37-46 KiB -> 20-28 KiB, 1 block -> 2 blocks.

3. **`I=64, nthreads=256, occupancy 3` reaches 3 blocks for Q4_0 and Q4_K only**
   (VGPR 66/73, zero spill, LDS 19.6/18.8 KiB). Q5_K/Q6_K/Q8_0 stay at 2 blocks because
   26.8-27.6 KiB still only fits twice in 64 KiB.

4. **Occupancy and spill trade off sharply as `I` grows.**
   - `nthreads=128, occupancy 2` clamps the block to 128 VGPRs and spills in 109 cells.
     Worst: `I=256, J=64` gives Q5_K 522, Q6_K 513, Q4_K 500 spill dwords.
   - `nthreads=256, occupancy 2 or 3` is spill-free at `I=64` for every type and J.
   - `nthreads=256, occupancy 2` at `I=128` already spills 3-14 dwords on Q4_K/Q5_K/Q6_K
     (J >= 24, fallback=false) and on the Q8_0 fallback rows.
   - `nthreads=256, occupancy 2` at `I=256` spills 51-233 dwords for every type.
   - `nthreads=256, occupancy 1` is the only spill-clean setting at `I=128..256`, apart
     from 4 cells (`I=256`, J=48/64, Q4_K/Q5_K/Q6_K).

5. **`I=192` and `I=256` are mostly dead ends here.** I=192 gives 1 block for Q5_K/Q6_K/Q8_0
   (61.9-64.2 KiB); I=256 with Q5_K/Q6_K/Q8_0 needs 79-83 KiB, that is over the 64 KiB CU
   budget, so those rows would be skipped by `ggml_cuda_mmq_get_J_best` and must not be
   landed. I=256 is only sane for Q4_0/Q4_K, and then at 1 block.

6. **`I` must be a multiple of the physical wave size, 64 on gfx906.** `I=96` does not
   compile: `static_assert(I % warp_size == 0, "DP4A MMQ tile height must be a multiple of
   the physical warp size")` in `mmq.cuh`. So the legal set is {64, 128, 192, 256}.

7. **`K_vram` (256 vs 512) changes nothing static.** Same VGPR, same SGPR, same LDS for all
   types: it only sets the VRAM-side K chunk of the loop, not the tile footprint.

8. **SGPR is never the limit** (32-44, no SGPR spill anywhere), and the `has_fusion`
   instantiations are empty on gfx906: `fusion_available` is false without
   `TURING_MMA_AVAILABLE`, so those kernels compile to the `NO_DEVICE_CODE` stub (VGPR 41,
   SGPR 38, identical for every J). Same stub shows up for the J=56..128 rows that the Vega
   table does not define. Neither is a sweep result, both are binary dead weight.

## What this does not say

- No performance claim. Static residency is an upper bound: 2 resident blocks only help if
  there are enough tiles to fill them, and the tile split at `I=64` doubles the tile count
  in the row direction (which is also what the launcher uses for its stream-k decision).
- Small `I` costs more loop iterations per tile and more `x` tile reloads, small `J` costs
  more `y` quantization passes; none of that shows up in VGPR/LDS numbers.
- `stream_k` was left at `false` everywhere, and `sram_layout` per type.

Next step if any of this is taken further: A/B `I=64, nthreads=256, occupancy 2` for
Q4_0/Q8_0/Q4_K/Q5_K against the current rows, and `occupancy 3` for Q4_0/Q4_K, on real
prefill shapes. The audit says the second (and third) block fits and does not cost a spill;
only a benchmark can say whether it pays.

## Reproduction

```sh
# sweep (patches the table, compiles, restores the table at the end)
python3 bench/mmq/static-sweep.py
# raw rows: one line per (cell, type, J, fallback, prec)
cut -d, -f1-14 bench/mmq/static-sweep.csv | head
```

## A/B bench of the candidate row (2026-11-27)

Candidate: `nthreads 256, occupancy 2, I 64` for Q4_0/Q8_0/Q4_K/Q5_K (Q6_K untouched).
Reverted afterwards, tree clean.

Isolated `test-backend-ops perf -o MUL_MAT`, n=512, m=4096, k=14336, two runs each:

| type_a | baseline us/run | candidate us/run | delta |
|--------|-----------------|------------------|-------|
| q4_K | 7445, 7431 | 5873, 5801 | +21..22% |
| q6_K | 5374 (5305 with FORCE_MMQ) | 5352 | noise, row unchanged |
| q8_0 | 5583 | 5582 | 0%, row never launched (see below) |

Model prefill, `llama-bench -p 512,4096 -r 5`:

| model | test | baseline t/s | candidate t/s |
|-------|------|--------------|---------------|
| Qwen3.5-9B Q4_K | pp512 | 1013.19 +/- 5.34 | 1007.18 +/- 6.07 |
| Qwen3.5-9B Q4_K | pp4096 | 968.83 +/- 3.83 | 972.47 +/- 3.67 |
| Spark-X2.6-4B Q8_0 | pp512 | 2091.39 +/- 23.11 | 2077.97 +/- 24.23 |
| Spark-X2.6-4B Q8_0 | pp4096 | 1799.61 +/- 2.68 | 1793.56 +/- 7.81 |

Per-op inside the 9B model (`GFXPROF=1 GFXPROF_NAMES=1 GGML_CUDA_DISABLE_GRAPHS=1`, -p 512, r=1,
ms/call, baseline -> candidate): ffn_gate Q4_K 3.308 -> 3.297, ffn_up Q4_K 3.261 -> 3.238,
ffn_out 12288x4096 Q4_K 3.561 -> 3.534, ffn_out 12288x4096 Q6_K 3.496 -> 3.525. Flat.

Two reasons the static win does not show up in the models:

1. The local models do not run upstream MMQ at prefill. Their weights are repacked and the
   launch goes to `mmq_gemm_q4k_repacked` / `mmq_gemm_q8_0_repacked` / `mmq_gemm_q6k_repacked`
   in `repack-gcn.cu`, which carry their own `__launch_bounds__` (occ 2/3). Nothing in
   `mmq-config-vega.cuh` is consulted on that path. The isolated q4_K test uses plain
   (non-repacked) tensors, so it does reach MMQ - and there the new row is 22% faster.
2. Q8_0 at n=512 never uses the patched rows anyway: `ggml_cuda_mmq_get_J_best` prefers
   J=128, which the Vega table does not define, so it falls through to the generic GCN row
   (`512, 2, 128, 128`). The Vega rows only reach J<=64.

Conclusion: the occupancy finding is real but not reachable from the models measured here.
Landing `I 64 / occupancy 2` in the Vega table would only change the non-repacked MMQ path
(other models, or repacking disabled), where it looks like a 22% win on one shape and is
unmeasured elsewhere. Not landed.

## Landed: Vega rows now match the GCN rows plus the swept I=64 (2026-11-27)

Answer to the obvious question first: no, at `J <= 64` the GCN rows were not in use.
`ggml_cuda_mmq_get_config_vega` is consulted first for every (type, J, fallback), and it
defined `J <= 64` rows for all of these types, so only `J = 80..128` ever fell through to
`ggml_cuda_mmq_get_config_gcn`. The GCN table was the better reference all along: it asks
for `occupancy 2` everywhere and already uses `I 64` for many J, while the Vega table
shadowed all of it with `256, 1, 128`.

Change in `ggml/src/ggml-cuda/mmq-config-vega.cuh`, 54 rows for Q4_0/Q8_0/Q4_K/Q5_K/Q6_K:
`nthreads 256, occupancy 1, I 128` -> `nthreads 256, occupancy 2, I 64`. J, fallback,
sram_layout, K_vram and stream_k unchanged. Types outside these five keep the old rows.

Static result (from the sweep): 2 resident blocks per CU for all five types, all J <= 64,
VGPR 57-121, zero spill, LDS 11.6-28.3 KiB (was 21.1-45.9 KiB, 1 block).

Verification on the built objects: q4_K J=8/16/24/32/64 now reports VGPR 64/64/89/73 with
LDS 11.6/12.6/13.6/14.7/18.8 KiB; q6_K unchanged (its rows were already `256, 2, 64`).
`test-backend-ops test -o MUL_MAT` for the five types, m 4096/6272, n 1/8/32/512: 4/4 pass.
Isolated `MUL_MAT n=512 m=4096 k=14336`: q4_K 7431..7445 -> 5874 us, q6_K and q8_0 flat.

Model level is unaffected, for the reason in the A/B section: the local models prefill
through `mmq_gemm_*_repacked` in `repack-gcn.cu`, not through upstream MMQ. 9B pp512
1005-1007 vs 1013 baseline, 9B pp4096 971 vs 969, Spark pp512 2077 vs 2091, Spark pp4096
1746-1790 vs 1800 (run to run spread on this card is +-2.5% at pp4096).

Two things this does not fix:

- Q8_0 at n >= 128 selects J=128, which the Vega table still does not define, so the GCN
  row `512, 2, 128, 128` applies. That is not a win that was left on the table: at J=128
  the y tile alone is 18.4 KiB and the x tile is 18.8 KiB, so no legal I (>=64) can fit two
  blocks in 64 KiB. The Q8_0 rows edited here only run for n <= 64.
- `occupancy 3` for Q4_0/Q4_K (3 blocks, VGPR 66/73, zero spill, LDS 19.6/18.8 KiB) is
  supported by the static numbers but has no benchmark behind it. Not landed.

## Unified table (2026-11-27)

`mmq-config-vega.cuh` is gone. Its rows were an unbenchmarked RDNA2 port (`256, 1, 128`)
except for `Q6_K I=64 occ2` (ported A/B, +4.2% pp), and that value is also what the sweep
picks and what the upstream GCN rows already had. So the file kept nothing worth a second
table:

- `ggml/src/ggml-cuda/mmq-config-gcn.cuh` now holds the upstream GCN rows plus a gfx906
  block. Q4_0/Q8_0/Q4_K/Q5_K/Q6_K at `J <= 64` use the bench-swept values (per type, see
  the bench section below); `J > 64` and the other quant types keep the upstream GCN rows.
- Callers pass a `gfx906` flag: host uses `cc == GGML_CUDA_CC_VEGA20`, device uses
  `#if defined(__gfx906__)`. That is the only way to keep host and device agreeing, since
  the device side is a `constexpr` function baked into `__launch_bounds__`.

Two things found while merging:

- **Do not widen the gfx906 rows to `J >= 80`.** A first attempt mirrored the generic Q8_0
  rows (which go to J=128) with `I=64`: `MUL_MAT n=512 m=4096 k=14336` went from 5.6 ms to
  16.6 ms, 3x slower. With `I=64, J=128` the block computes a 64x128 tile over 8 warps of
  32 rows each, so most of the tile work is masked out. Comment added at the site.
- Unswept types now take the generic GCN rows instead of the old Vega rows. Static check on
  two of them (J=8..64, fallback=false/true): Q2_K goes from VGPR 102-256 with up to 31 spill
  dwords to VGPR 65-128 with 11-13 at J=64; Q1_0 goes from VGPR 180-216 clean to VGPR 95-128
  with 11-15 spill dwords at J=16. Fewer registers, 2 blocks instead of 1, a few small spills
  where there were none - the same trade the swept types make, without a measurement.

Verified after the merge: `test-backend-ops test -o MUL_MAT` for the five types passes,
isolated `MUL_MAT n=512 m=4096 k=14336` gives q4_K 5758 us (+22% vs the pre-merge 7431),
q8_0 5578 and q6_K 5304 (both unchanged), 9B pp4096 968.8 (was 968.8), Spark pp4096 1794
(was 1800).

## Bench sweep of the gfx906 rows (2026-11-27)

Static residency alone was not enough: the second block is only worth having if the kernel
is latency-bound. So the sweep was repeated with wall-clock timing on real quantized weights.

Harness: `bench/mmq/mmq-bench.cpp` builds one `MUL_MAT` with a plain (not repacked) weight
tensor, so the launch goes through the upstream MMQ kernel, times it over repeated graph
computes, and prints a checksum of the result. Shapes: the two ffn GEMMs of a 4096-wide model
at ubatch 512 (`k=4096`, `m=12288` and `m=4096`), a small-batch case (`m=12288, n=64`), the
synthetic `k=14336` shape test-backend-ops uses, and one `k=4096, m=4096, n=512` shape per
type. Grid: `I` in {64,128,192,256} x `nthreads` in {128,256,512} x `occupancy` in {1,2,3},
38 cells, one rebuild per cell (`bench/mmq/bench-sweep.csv`).

### Winners per type (K_vram 256, us/run)

| type | Q4_K ffn 4096x12288 n512 | k=14336 4096 n512 | n=64 | best cell |
|------|--------------------------|-------------------|------|-----------|
| Q4_K | 3558 (was 5444) | 5066 (was 7389) | 637 (was 902) | `256, 2, 128` |
| Q4_0 | - | - | - | `256, 2, 64` |
| Q5_K | - | - | - | `512, 3, 64` |
| Q6_K | - | - | - | `256, 3, 64` |
| Q8_0 | - | - | - | insensitive (`256, 2, 64` kept) |

"was" = the pre-merge Vega rows (`256, 1, 128`). Per-type, `k=4096, m=4096, n=512`:
Q4_0 1050, Q4_K 1483 (was 2167), Q5_K 1594 (was 2483), Q6_K 1482 (was 1510),
Q8_0 1634 (was 1655).

Cross-check in test-backend-ops (`MUL_MAT n=512 m=4096 k=14336`): q4_K 7431 -> 5137 us,
q6_K 5374 -> 5216, q4_0 3739, q5_K 5478, q8_0 5570 (unchanged: it selects J=128, generic row).

### Traps this sweep ran into

1. **`K_vram` above `MMQ_ITER_K` (256) silently skips K work.** `K_vram` 512, 1024 and 2048
   reported exactly 2x, 4x and 8x speedups (down to 574 us, 89 TFLOPS on a 53 TFLOPS part),
   because the kernel advances `kb0` by `K_vram/qk` while `load_tiles` stages one block.
   Proven by result checksum: q4_K ffn `3.57e8` at `K_vram=2048` against `2.86e9` at 256, a
   8x drop, i.e. 7/8 of the K range never contributes. All rows keep `MMQ_ITER_K`.
   test-backend-ops did not catch this: its only MMQ-sized q4_K case is `n=8`, which takes the
   matvec path, so the "correctness passes" seen earlier were vacuous for MMQ. Use the
   checksum in `mmq-bench` instead.
2. **`I=192` produces NaN** on the Q5_K path (checksum `-nan`), even though it compiles and
   the static numbers look fine. Q5_K therefore runs `512, 3, 64` (+7% over the best legal
   `I`) instead of `512, 1, 192`. No generic GCN row uses `I=192`.
3. **`nthreads=128` is a large loss** (Q4_K ffn 8989 us, 2.5x worse) and `occupancy 3` at
   `nthreads=256, I>=128` is worse than `occupancy 1` (5444 -> 5457, VGPR clamped to 85).
   `I=256` fails outright: LDS exceeds 64 KiB for Q5_K/Q6_K/Q8_0 and `J_best` aborts.

### Result

`MUL_MAT` on plain q4_K weights at `n=512, m=4096, k=14336` went 7431 -> 5137 us (+45%), and
the ffn shape 5444 -> 3558 us (+53%). Model-level prefill does not move (9B pp512 999 vs
1013, pp4096 963 vs 969; Spark pp4096 1737 vs 1800, high variance while the card is hot),
because the local models prefill through `mmq_gemm_*_repacked` in `repack-gcn.cu`, not MMQ.
