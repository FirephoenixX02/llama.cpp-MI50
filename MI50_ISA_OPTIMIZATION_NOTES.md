# MI50 (gfx906 / Vega 7nm) ISA notes for kernel optimization

Source: `Vega_7nm_Shader_ISA_26November2019.pdf` ("Vega" 7nm Instruction Set Architecture
Reference Guide, 26-November-2019). Text dump: `Vega_7nm_Shader_ISA_26November2019.txt`
(each page prefixed with `===== PDF PAGE n =====`).

This document covers the GCN 5.1.1 "Vega" 7nm generation, which includes Vega 20 = gfx906
(MI50 / MI60 / Radeon VII). All page references below use the document's internal page
numbering (`NN of 290` in the page footer). Mapping: internal page `n` = PDF page `n + 8`
(e.g. internal p.82 LDS overview = PDF p.90).

Scope warning: this is an ISA/programmer reference, not a datasheet. It states the
instruction set, program state and (some) structural limits. It does NOT contain clock
speeds, per-instruction VALU throughput/latency tables, occupancy tables, or L1/L2/HBM
sizes (only LDS/GDS sizes are given). Do not quote such numbers from this doc. Hardware
configuration for the MI50 target is collected in section 0 below, with per-value
provenance.

## 0. MI50 (gfx906) hardware configuration

Reconciled from the provided spec table plus the ISA doc. Provenance per row:

- `doc`: stated in this ISA reference (see section/refs below).
- `tooling`: published gfx906 `rocminfo` / `hipDeviceProp` output. This sandbox has no GPU,
  so these are NOT verified against a live device - confirm with the commands at the end of
  this section on the MI50 box.
- `datasheet`: AMD marketing numbers, approximate, no ISA-doc backing.
- `unverified`: could not be confirmed; ask the table source or re-measure.

| Field | Value | Provenance | Note |
|---|---|---|---|
| Model | MI50, 16 GiB variant | datasheet | Vega 20 full die, HBM2 |
| Architecture | GCN 5.1 | doc | "Vega" 7nm generation covers gfx906 |
| LLVM target | gfx906 | doc | `HIPCXX ... GPU_TARGETS=gfx906` in build-llamacpp-rocm.sh |
| VRAM | 16 GiB HBM2, ~1.02 TB/s peak | datasheet | ISA doc carries no bandwidth numbers |
| Compute units | 60 | tooling (verified live) | full gfx906 die (3840 SP) |
| Wavefront size | 64 | doc (verified live) | p.8; `rocminfo`: `Wavefront Size: 64` |
| LDS | 64 KiB per CU, 32 banks | doc | p.9, section 2/3 below |
| GDS | 64 KiB, 128 B/cycle | doc | p.10 |
| L3 cache | none on gfx906 | tooling | table row ambiguous - see note below |
| L2 cache | 8 MiB | tooling (verified live) | `rocminfo`: `L2: 8192 KB`; table showed 4 MiB - source table is wrong or mislabeled |
| L1 vector/data cache | 16 KiB | tooling (verified live) | `rocminfo`: `L1: 16 KB` (aggregate) |
| L1 scalar cache | 16 KiB ("per 3 CUs") | unverified | see note below |
| L1 instruction cache | 32 KiB ("per 3 CUs") | unverified | see note below |
| Cacheline | 64 B | tooling (verified live) | `rocminfo`: `Cacheline Size: 64`. Matches the 64 B line granularity in the GLC write-combine rules (p.64) |
| VGPR file | 256 KiB per SIMD (1 MiB per CU) | tooling | consistent with doc: 256 VGPR/thread cap (p.39), alloc in 4-Dword groups (p.17) |
| SGPR file | 12.5 KiB per CU (800 regs per SIMD) | tooling | consistent with doc: alloc 16..102 in 16-Dword units, VCC at SGPR 106/107 (p.17) |
| Waves per SIMD cap | 10 | tooling | not in the ISA doc |

Table notes:

- Live `rocminfo` output from the MI50 has since confirmed: L2 = 8192 KB, L1 = 16 KB,
  cacheline 64 B, 60 CUs, wavefront 64. The source table's L3/L2 pairing was therefore
  wrong: gfx906 has no L3, and L2 is 8 MiB (not 4 MiB - the source table's value matches
  Vega 10, not Vega 20).
- The "per 3 CUs" grouping for L1I/L1S does not match the usual GCN pattern (4 CUs per
  shader engine), is not shown by `rocminfo`, and is not covered by the ISA doc. Re-check
  the source table before quoting.
- Occupancy formulas derived from the above (tag the external values when repeating them):
  - VGPR-limited waves/SIMD = min(10, floor(1024 / VGPRs_per_thread)): 10 waves up to ~100
    VGPRs, 8 at 128, 4 at 256. Matches the fork's `launch_bounds(256,1)` reasoning: at 256
    VGPRs/thread, occupancy 2 is impossible without the compiler spilling.
  - SGPR-limited waves/SIMD = floor(800 / allocated_SGPRs_per_wave incl. VCC overhead):
    10 waves up to 80, 8 at 96, 7 at 112+. This is the second reason deep-KV MMQ configs
    dropped to `Q6_K I=64 occ2` territory.

Verification commands (run on the MI50 host):

```sh
rocminfo | grep -E "Marketing|Cache|Compute Unit|Wavefront|Cacheline"
```

Run 2026-11: Cache Info L1 16 KB / L2 8192 KB, Cacheline 64 B, 60 CUs, wavefront 64.
Still open: L1I/L1S "per 3 CUs" rows, VGPR/SGPR file size as reported by the runtime.

## 1. Execution model basics

- A wavefront is 64 work-items executing in parallel (p.8). Vector instructions operate on
  all EXEC-enabled work-items at once; dormant lanes treat the instruction as a NOP.
- All control flow is scalar (SALU); vector compares write VCC (SGPR alias of the wavefront's
  SGPRs 106/107) and VCCZ/EXECZ summary bits are available for early-exit branches (p.18).
- Up to 16 wavefronts (1024 work-items) form a workgroup on one compute unit, sharing LDS;
  `S_BARRIER` synchronizes; waves that die early via `S_ENDPGM` count as satisfied once they
  reach the barrier (p.23).
- `EXEC = 0` is not optimized by hardware: every instruction still issues and wastes
  bandwidth. `S_CBRANCH` or `VSKIP` are the recommended fast skips (p.13).
  - `S_SETVSKIP` sets `MODE.VSKIP`: all VALU/VMEM/LDS/GDS/exports of the wave are skipped at
    high speed - 10 wavefronts per clock can skip a single instruction, "much faster than
    issuing and discarding" (p.15). Useful for wave-uniform divergent padding (e.g. MoE
    expert tails) where whole waves hold EXEC=0.
- Wavefront priority: `S_SETPRIO` sets `WAVE_PRIO` 0-3 (SPI sets SPI_PRIO at creation)
  (p.13, p.12-49 referenced). Only 2 bits; usable for decode-vs-prefill biasing at the HIP
  level only if the runtime allows it (not exposed by HIP; listed for completeness).

## 2. Per-CU resources and allocation limits

- LDS: 64 kB per compute unit, 32 banks x 512 entries x 4 B each (32-bit), plus 32 integer
  atomic units (p.9). One workgroup can claim up to the whole 64 kB (p.82). This 64 kB/CU
  budget is the hard cap behind FATTN tile sizing on gfx906 (fork's nstages=1 / occ=1 clamp).
- LDS is approximately one order of magnitude faster than uncached global, assuming no bank
  conflicts (p.82).
- GDS: 64 kB, shared across all CUs of a kernel, 128 B/cycle access (p.10).
- LDS allocation: contiguous blocks of 128 Dwords on 128-Dword alignment; allocations do not
  wrap; all accesses are clamped to the allocated wave/workgroup range (p.17).
- SGPRs: allocated 16..102 per wavefront in units of 16. VCC physically occupies the two
  SGPRs above the allocated range (SGPR 106/107 alias) (p.17). Trap setup reserves 16 more.
  Alignment: even SGPR for 64-bit data, quad-aligned SGPR for 4+ Dword scalar-memory results
  (p.17).
- VGPRs: allocated in groups of 4 Dwords; allocations can wrap the VGPR pool (p.17). The
  encoding universe is VGPR 0..255 per thread (9-bit source field, codes 256-511 map to
  VGPR 0-255) (p.37-39). Note: the per-SIMD physical register file size / occupancy
  tradeoff is NOT in this document.
- GPR_ALLOC / LDS_ALLOC hardware registers (readable via `S_GETREG`) encode the wave's
  allocation bases/sizes: VGPR in [7:2] (4-Dword units), SGPR in [7:3] (8-Dword units),
  LDS in 64-Dword units (p.35).
- M0 (one per wavefront) is the LDS/GDS addressing register: for LDS ops `M0[15:0]` is the
  byte offset/size; every LDS operation requires M0 initialized before use, and the effective
  LDS range is `MIN(M0[15:0], allocated LDS size)`. Set `M0 = 0xFFFFFFFF` to disable
  clamping (p.18, p.87).
- Out-of-range behavior (p.16): SGPR/VGPR source out of range reads register 0; destination
  out of range drops the write silently. For VMEM loads/atomics, ANY destination VGPR out of
  range (including VDST+3 for a DWORDx4 and the PRT GPR) nullifies the whole instruction
  (issued as if EXEC=0). Kernels that index VGPRs indirectly must keep offsets in range.

## 3. LDS structure, conflicts, and instructions

- 32 banks; dwords are placed in banks serially (consecutive dword = consecutive bank).
  Dword address bit 2..6 selects the bank. All banks execute a store or load simultaneously;
  reads across a wavefront are dispatched over 4 cycles in waterfall (p.82).
- A wavefront can concurrently run 32 dword accesses (or 64-bit READ2/WRITE2 pieces) per
  cycle; more than one access to the same bank at the same time serializes the colliding
  accesses (p.82-83). Indexed and atomic LDS ops complete in 2 to 64 cycles depending on
  bank conflicts (p.85).
- Practical: pad per-lane rows so that a lane stride of 32 dwords lands on a spread of
  banks; LDS-staged GEMM tiles with 33/36/64+1 dword strides avoid serialization (the
  fork's `gcd(9,32)=1` / padded-LDS-row tiling is the same idea).
- LDS instructions (p.85-88):
  - `DS_READ_{B32,B64,B96,B128,U8,I8,U16,I16}`, `DS_WRITE_{B32,B64,B96,B128,B8,B16}`.
  - `DS_READ2_{B32,B64}` / `DS_WRITE2_{B32,B64}`: two unique addresses per lane (separate
    8-bit byte offsets, or one address only if both offsets are equal. The ST64 forms
    scale BOTH offsets by 64 (`OFFSET*4*64`), i.e. addresses in 64-dword units.
  - Address math: `LDS_Addr = LDS_BASE + VGPR[ADDR] + InstrOffset` (byte addresses; for
    READ2 the adjustment `ADJ` is 4 for 8/16/32-bit and 8 for 64-bit data).
  - 8/16-bit LDS loads sign/zero-extend to dword; 64-bit data uses double-GPRs.
  - `DS_PERMUTE_B32` (forward): `ret = LDS[lane dst]` where the wavefront computes
    `LDS[tid] = src0` conceptually but writes NO LDS memory - a pure cross-lane register
    shuffle for all 64 lanes. `DS_BPERMUTE_B32` (backward): `LDS[tid] = src0`,
    `ret = LDS[dst lane]`. These are the wave64-capable shuffles (`__shfl_xor` on gfx906
    lowers to `ds_bpermute_b32`).
  - `ds_*_src2_*` atomics use a signed dword offset mode selected by bit 7 of OFFSET1.
- LDS direct read (VALU source): set SRC0 = LDS_DIRECT; a single dword is read per wave and
  broadcast to all 64 lanes: `M0[15:0]` = byte address, `M0[18:16]` = data type
  (0=ubyte,1=ushort,2=dword,4=sbyte,5=sshort) (p.84). Requires the `SALU writes M0 -> LDS
  direct read` wait state (see section 8). This is a cheap wave-broadcast of one word.
- Buffer load to LDS (MUBUF, `LDS=1` bit): a subset (`BUFFER_LOAD_{ubyte,sbyte,ushort,
  sshort,dword,format_x}`) can spill read data directly into LDS, bypassing VGPRs. The LDS
  write offset is `M0[15:0]` bytes (per wave), the memory offset comes from an SGPR
  (SOFFSET), TFE must be 0, and the write is clamped to the wave's LDS allocation - set a
  matching active mask to keep out-of-range lanes from reading memory for nothing (p.62-63).

## 4. gfx906 arithmetic for LLM kernels

New instructions in Vega 7nm (p.3) + packed math (p.45, p.155-156):

| Opcode# (VOP3P) | Instruction | Semantics |
|---|---|---|
| 35 | `V_DOT2_F32_F16` | `D.f32 = S0.f16[0]*S1.f16[0] + S0.f16[1]*S1.f16[1] + S2.f32` |
| 38 | `V_DOT2_I32_I16` | packed signed i16 pair dot + i32 accumulator |
| 39 | `V_DOT2_U32_U16` | packed unsigned u16 pair dot + u32 accumulator |
| 40 | `V_DOT4_I32_I8` | 4x signed i8 dot per dword + i32 accumulator (dp4a) |
| 41 | `V_DOT4_U32_U8` | 4x unsigned u8 dot per dword + u32 accumulator (dp4a) |
| 42 | `V_DOT8_I32_I4` | 8x signed 4-bit dot per dword + i32 acc |
| 43 | `V_DOT8_U32_U4` | 8x unsigned 4-bit dot + u32 acc |
| 14 | `V_PK_FMA_F16` | two independent f16 FMAs per instruction |
| 15-18 | `V_PK_{ADD,MUL,MIN,MAX}_F16` | two independent f16 ops |
| 32-34 | `V_MAD_MIX_{F32,MIXLO/MIXHI_F16}` | mixed 16/32-bit MAD; OPSEL picks which 16-bit half of each source (0=[31:0],1=..,2=[15:0],3=[31:16]); NEG_HI field acts as absolute-value |
| 1-13 | `V_PK_*_{I16,U16}` | packed 16-bit integer ALU (mad/add/sub/max/min/shifts) |

Key facts for the fork's kernels:

- There is no 8x-int8 dot instruction on gfx906. `V_DOT8_*` consumes 4-bit values (2 per
  nibble). The widest 8-bit primitive is `V_DOT4_*` (dp4a), matching the fork's dp4a MMQ.
- `V_DOT2_F32_F16` is fused: both f16 products accumulate into the f32 accumulator of the
  same instruction (S2), i.e. a 2-term K=2 GEMV primitive with f32 accumulation - the
  "fdot2" used in flash-attention 256-v3. Feeding S2 with a running sum avoids extra adds.
- `V_PK_FMA_F16` gives 2 lanes of f16*f16+f16 per instruction with f16 output (16-bit
  precision ceiling), while `V_DOT2_F32_F16` keeps f32 accumulation at the same packed rate;
  for dot products prefer dot2 when the accumulator must stay f32.
- FMA/MAC coverage in the classic formats: `V_FMA_{F16,F32,F64}` (VOP3A), `V_MAC_F32/F16`
  (VOP2 read-modify-write), plus the new `V_FMAC_F32`. Note `V_MAC_F32` and `V_FMAC_F32`
  cannot use SDWA (p.222).
- Packing/convert primitives adjacent to the kernels: `V_CVT_PKRTZ_F16_F32` (single->packed
  pair of f16), `V_PACK_B32_F16`, `V_CVT_PK_U8_F32`, `V_CVT_PKACCUM_U8_F32`,
  `V_CVT_PK_U16_U32`, `V_CVT_PK_I16_I32`, `V_CVT_PKNORM_{I16,U16}_{F32,F16}`,
  `V_LERP_U8`, `V_SAD_{U8,HI_U8,U16,U32}`, `V_MIN3/MED3`, `V_ALIGNBYTE_B32`,
  `V_ALIGNBIT_B32`, `V_BFI_B32`, `V_BFE_{U32,I32}` (bitfield extract; matches the fork's
  GPTQ `ubfe` dequant).
- `V_PERM_B32` exists (VOP3A list, p.~167 region): per-byte select from 3 dwords - the
  8-entry LUT / `v_perm` MXFP4 trick in `vecdotq.cuh` is expressible on this ISA.
- Scalar side: new `S_PACK_LL/LH/HH_B32_B16` (SOPK opcodes 50-52) pack two 16-bit halves of
  two SGPRs into one dword (p.97, p.228) - useful for building packed f16/i16 constants.
- Also new in Vega 7nm but less used: `V_XNOR_B32`.

VGPR/operand encodings matter for dot-heavy kernels (p.37-39):

- At most ONE SGPR per VALU instruction, but the value can feed more than one operand.
- A 32-bit literal constant is only usable on the 32-bit encodings and only when no SGPR or
  M0 source is used; VOP3 (64-bit) encodings cannot take a literal at all - plan register
  budget accordingly when hoisting scale/zero constants per wave.
- Only SRC0 can be LDS_DIRECT; DPP and SDWA are also SRC0-only encodings.
- Inline constants cover ints 1..64 and -1..-16 and floats 0.5/-0.5/1.0/-1.0/2.0/-2.0/
  4.0/-4.0/1/(2*pi) (as half 0x3118, single 0x3e22f983) and can be used with DPP/SDWA -
  cheap fused f32 scale constants for dequant (p.38-39).
- VOP3 ABS/NEG per operand, OMOD (*0.5/1/2/4) and clamp on output (p.38).
- Vega changed the CLAMP bit meaning: FP ops clamp to [0,1], integer ops clamp to the
  representable min/max, and for V_CMP it signals floating exceptions (p.43). Ported CDNA-era
  asm relying on clamp should be re-checked.

FP mode knobs (p.14, p.22, p.42-43): MODE has separate FP_ROUND [1:0] for f32 and [3:2] for
f16/f64, DENORM modes (flush/allow), DX10_CLAMP NaN handling, IEEE-mode NaN propagation, and
`FP16_OVFL` (clamp overflowed f16 results to +/-MAX_FP16 while preserving INF - "control over
arithmetic overflow for FP16 VALU operations" is a Vega 7nm feature, p.2). If HIP codegen
differs from expectations on f16 rounding/denormals, these MODE bits are the underlying
control (set via S_SETREG; the usual wait states apply).

## 5. Cross-lane moves: DPP, DS_SWIZZLE, readlane/writelane

DPP (Data Parallel Primitive) is a second instruction dword for VOP1/VOP2/VOPC that reads
SRC0 from other lanes - no LDS traffic, and it composes with any SRC0-legal VALU op, so
FMA+shuffle fuse (this is what makes DPP-based warp reductions faster than `ds_bpermute`
round-trips on gfx906).

- Controls (p.267-269):
  - `DPP_QUAD_PERM` (0x00-0xFF): permute inside each group of 4 lanes:
    `pix[n] = pix[(n & 0x3c) + ctl[sel bits]]`.
  - `DPP_ROW_SL/SR n` (1-15): shift within 16-lane row; `DPP_ROW_RR n`: rotate within row.
  - `DPP_WF_SL1/WF_RL1/WF_SR1/WF_RR1`: whole-wavefront shift/rotate by one lane (wave64).
  - `DPP_ROW_MIRROR`, `DPP_ROW_HALF_MIRROR` (8 lanes), `DPP_ROW_BCAST15` (lane 15 of each
    row to the next row), `DPP_ROW_BCAST31` (lane 31 of rows 0-1 to rows 2-3).
  - `ROW_MASK` and `BANK_MASK` gate destination-write groups (rows of 16, wave64 shows rows
    [63:48]/[47:32] too; banks are 4-lane groups). `BC` bound control: 0 = do not write the
    destination when the source lane is out of range (this is how butterfly trees avoid
    garbage at the edges).
  - NEG/ABS modifiers ride along per source.
  - Operand-table note: DPP is "only valid as Source-0" and is described as "DPP over 16
    lanes" (p.39); the row ops above are the 64-lane (wave64) forms.
- DPP is NOT available on (p.221-222): `V_MADMK/MADAK_{F32,F16}`, `V_READFIRSTLANE_B32`,
  all F64 ops and the 64-bit compares, `V_CLREXCP`, `V_SWAP_B32`, plus `V_CMP*_CLASS_F64`.
  Also not listed for any VOP3P/packed op - packed ops take the standard VOP3P modifiers, not
  DPP.
- `V_READLANE_B32` / `V_WRITELANE_B32`: explicit scalar<->lane transfers (p.41-42). Hazards
  apply when VCC/SGPR written by VALU feed the lane select (see section 8).
- `DS_SWIZZLE_B32` (no LDS memory access at all, p.197-198):
  - offset 0xC000-0xDFFF: rotate the 32-lane group left/right by offset[9:5], optional mask
    that swaps odd/even lanes;
  - offset >= 0xE000: FFT swizzle of a 32-lane group by offset[4:0];
  - else `offset[15]==1`: groups of 4 consecutive threads fully share data - each lane holds
    its group-offset bits ([1:0] first lane, [3:2] second, etc.), any-to-any within the quad;
  - else `offset[15]==0`: a 32-lane group transformed by {xor,or,and} 5-bit masks: given
    patterns include SWAPX1..X16, REVERSEX2..X32, BCASTX2..X32 (broadcast one lane into the
    group, e.g. BCASTX32 = xor 0x00, or = thread, and 0x00).
  - This is the in-register alternative to LDS round-trips for broadcast/butterfly patterns
    at wave32 (per-32) granularity; combine with DPP quad-perm for wave64.

## 6. Scalar memory (SMEM)

- Reads 1-16 dwords into SGPRs, stores 1-4 dwords; through the scalar data cache (p.46-47).
- `GLC` for SMEM loads selects L1 policy: 0 = hit_lru, 1 = miss_evict; for stores 0 =
  write-combine, 1 = write-thru (p.46).
- Alignment: address LSBs are forced dword-aligned; `SDATA` must be even for 2-dword reads
  and a multiple of 4 for 4+ dword reads; `SBASE` even (p.49-50). Unaligned SDATA gives
  invalid data without a fault - easy silent bug.
- Ordering: SMEM reads can return OUT OF ORDER and can return partial results crossing cache
  lines; the only safe wait is `S_WAITCNT 0` (p.49). Never do dependent SMEM back-to-back
  with a narrower wait.
- A scalar-memory clause is a run of same-type memory instructions; a clause's instructions
  must not overwrite each other's sources (XNACK replay can re-execute a clause) (p.47-48).
- `S_MEMTIME` / `S_MEMREALTIME`: 64-bit counters into SGPR pairs (const-clock for REALTIME)
  (p.49) - usable for in-kernel timing (the GFXPROF-style timing relies on HIP events, but
  in-kernel `s_memrealtime` is available).

## 7. Vector memory (VMEM) and FLAT/GLOBAL/SCRATCH

- MUBUF/MTBUF/MIMG move per-lane data between VGPRs and memory through the texture cache
  (L1 + L2) (p.51-52). Maximum transfer is 4 dwords per work-item per instruction
  (`*_DWORDX4` = 16 B/lane = 1 kB per wavefront per instruction) (p.52).
- Range checking (p.58): OOB reads return zero (V# dst_sel=1 components return 1), OOB
  writes/atomics are dropped, no fault. Raw buffers (AddTID=0, SwizzleEn=0, IdxEn=0) are OOB
  when `(InstOffset + vgpr_offset) >= NumRecords`; structured buffers when `Index >=
  NumRecords` (units of stride). DWORDx2/x3/x4 are range-checked per component; format
  loads/stores and atomics are all-or-nothing per instruction.
- Address = V# base + SGPR offset + per-lane (index*stride + offset); linear AoS addressing
  (p.56-57). For dword+ ops the two LSBs of the final address are ignored (dword alignment
  forced) (p.61). D16 variants load/store 16 bits into the low (`D16`) or high (`D16_HI`)
  half of a VGPR (p.61) - the f16/KV-cache stream primitive on gfx906.
- V# descriptor: 48-bit base, 14/18-bit stride (stride extends to 18 bits only with
  ADD_TID_EN scratch), 32-bit num_records feeding the per-component range check above
  (p.57, p.61). A descriptor of all zeros is an UNBOUND resource:
  reads return 0 and writes are dropped (p.62) - can be used to cheaply nullify a stream
  instead of branching.
- GLC bit (p.62-64):
  - Load GLC=0: may hit L1 and the line persists across wavefronts (normal streaming loads).
  - Load GLC=1: forces L1 miss (invalidates a matching line) and refetch from L2; no L1
    persistence across waves. BUT: within a single load instruction, L2 is NOT re-read for
    every lane of the same wavefront - the first lane in the wave brings the (64 B) line in
    and the other 63 lanes read that line from L1. Do not use GLC=1 as a way to kill
    intra-wave locality for aligned reads.
  - Store GLC=0: write-combine; a fully dirtied 64 B line stays valid/clean in L1; partial
    lines do not persist. GLC=1 same, but lines never persist. SLC=1 forces L2 miss +
    system coherence (use only for host-shared data, e.g. ping-pong buffers read by HIP
    code) (p.54, p.64).
  - Atomic GLC=1 returns the pre-op value.
- Vector memory returns IN ORDER (reads and writes, including mixed), while LDS/GDS and
  SMEM may complete out of order (p.31-32, p.49).
- FLAT/GLOBAL/SCRATCH (new GLOBAL/SCRATCH formats in Vega, p.76-81): per-lane flat address;
  GLOBAL uses an SGPR base (`SADDR`) + per-lane 32-bit offset and a 13-bit signed instruction
  offset; FLAT has a 12-bit unsigned offset; SCRATCH can take the address from an SGPR.
  GLOBAL/SCRATCH `LDS=1` forms move data between LDS and memory with M0 as the LDS offset
  (`LDS=1` must be zero for FLAT). FLAT
  instructions internally execute as BOTH a Buffer and an LDS instruction: they bump VM_CNT
  and LGKM_CNT, are done only when both complete, and two FLAT instructions can complete out
  of order (LDS-hit vs cache-hit) - results to the same VGPR from different FLAT instructions
  are unordered (p.77-78). Prefer buffer/`global` instructions with known space for hot
  loads; use FLAT only where the space is genuinely unknown.
- FLAT cannot return data to LDS; only MUBUF can (p.77).

## 8. Data dependencies: S_WAITCNT and mandatory wait states

Three counters (p.23-24, p.49):

- `VM_CNT`: MIMG/MUBUF/MTBUF reads/writes. In-order return including mixed read/write.
- `LGKM_CNT`: LDS/GDS (+1 per instruction), SMEM (+1 for 1 dword, +2 for 2+ dwords),
  S_SENDMSG. Different types complete out of order; GDS and LDS specifically can return
  out of order relative to each other.
- `EXP_CNT`: exports reading their VGPRs out.

Hardware does NOT check these (subset of Table 8, p.24-25; the ones that matter for
hand-written GCN/DPP code):

| First | Second | Waits |
|---|---|---|
| VALU writes VGPR | VALU DPP reads that VGPR | 2 (NOPs or independent instrs) |
| VALU writes EXEC | any VALU DPP op | 5 (no EXEC forwarding to DPP) |
| VALU sets VCC/EXEC | VALU reading EXECZ or VCCZ | 5 |
| VALU writes SGPR/VCC | v_readlane/v_writelane lane select | 4 |
| VALU writes VCC (incl. v_div_scale) | V_DIV_FMAS | 4 |
| VALU writes SGPR | VMEM instruction reading that SGPR | 5 ("hardware assumes there is no dependency here") |
| wide stores (FLAT/BUFFER/IMAGE x3/x4, 64-bit atomics w/ return) | writes to the same VGPRs | 1 |
| SALU writes M0 | LDS add-TID / buffer_store_LDS_dword / LDS=1 scratch-global / VINTERP / LDS direct | 1 |
| SALU writes M0 | GDS, S_SENDMSG, S_TTRACE_DATA | 1 |
| SALU writes M0 | S_MOVEREL | 1 |
| S_SETREG | S_GETREG/S_SETREG same reg | 2 |

- VCC alias trap: the dependency checker does not understand that "VCC by name" and "the
  logical SGPR pair holding VCC" are the same register; mixing the two styles in producer/
  consumer pairs needs 1 wait state (p.25).
- These facts document the fork's DPP reduction `s_nop` waits (2 per VALU->DPP step) and the
  upstream `#29478` VMEM scheduling barrier pattern: a VALU-writing SGPR consumed by VMEM
  needs explicit spacing (5) - HIP may or may not insert these; hand-scheduled asm must.

## 9. Wavefront/flow tools

- `S_CBRANCH_FORK/JOIN`, `S_SWAPPC`, `S_CALL_B64` subroutines, `S_GETPC` (p.22-23) are
  available; `S_CBRANCH_*_FORK` names here are literal instruction names, unrelated to the
  fork repo.
- `EXEC_SAVEEXEC` family (`S_{AND,OR,XOR,ANDN2,ORN2,NAND,NOR,XNOR}_SAVEEXEC_B64` and
  `S_ANDN{1,2}_WREXEC_B64`): save + transform EXEC in one instruction - the standard
  wave-uniform early-out idiom for per-expert loops, pairs with `S_CBRANCH_EXECNZ` (p.34).

## 10. Mapping to the fork's active work

| Fork item | ISA backing in this doc |
|---|---|
| DPP warp reductions (`common.cuh`) | DPP quad_perm/row_ror + wave64 row ops (p.267-269); VALU->DPP 2-NOP hazard (p.25); `__shfl_xor` -> `ds_bpermute_b32` (p.86); DPP-unsupported list (p.221-222); VCC-alias 1-wait (p.25) |
| dp4a MMQ / vecdotq | `V_DOT4_U32_U8` / `V_DOT4_I32_I8` (p.156); no wider i8 dot exists (V_DOT8 is 4-bit only); `V_BFE_*` for GPTQ-style extraction; `V_PERM_B32` for LUTs |
| FA v3 fdot2 / q8 FATTN dequant | `V_DOT2_F32_F16` fused f32-acc semantics (p.156); `V_PK_FMA_F16` for packed f16 paths; D16 loaders for KV f16 (p.61) |
| BF16 relabel to F16 | no bf16 ALU ops appear anywhere in this ISA reference - consistent with gfx906 having no native bf16 compute |
| Repack streaming loads | `*_DWORDX4` 16 B/lane (p.52); GLC=0 L1 persistence vs GLC=1 semantics (p.62-64); forced dword alignment (p.61); all-zero V# = unbound trick (p.62); out-of-range dest nullification (p.16) |
| FATTN / MMQ 64KB LDS budget | 64 kB LDS/CU, 32-bank structure, 2-64 cycle serialization on conflicts (p.82-85); LDS allocation 128-Dword blocks + M0 clamp (p.17, p.87); `DS_READ2/WRITE2` (p.86); buffer-load-to-LDS with M0 offset (p.62-63) |
| Divergent MoE / padding skips | EXEC=0 still issues everything; VSKIP skips at 10 WF/clock (p.13-15); SAVEEXEC idioms (p.34) |
| SMEM constant/descriptor loads | SMEM in-order rules out (p.49) -> WAITCNT 0 only; SDATA alignment (p.49); clause source-overwrite prohibition (p.47) |
| Graph/replay robustness | SMEM clause source rules exist only for XNACK replay paths; note the doc covers XNACK_MASK registers (p.37) - gfx906:xnack- has the mechanism disabled but the asm-level restrictions remain the safe assumption for hand asm |

## 11. Kernel register/LDS audit (built objects, gfx906)

Method: `amdgcn-audit.py` (workspace root) parses the per-kernel msgpack metadata
(`.vgpr_count`, `.sgpr_count`, `.vgpr_spill_count`, `.sgpr_spill_count`,
`.group_segment_fixed_size`) embedded in the HIP-compiled ELF objects - no ROCm tooling or
GPU needed, works on the synced build tree. Occupancy math from section 0: waves/SIMD =
min(10, floor(1024/VGPRs)) and floor(800/SGPRs); LDS additionally caps at
floor(64 KiB / LDS_per_workgroup) workgroups per CU (for 256-thread wave64 kernels that is
waves/SIMD directly).

```sh
python3 amdgcn-audit.py 'build/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/*.o' [filter]
python3 amdgcn-audit.py 'build/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/template-instances/*.o' [filter]
```

Key findings (full dumps in `/tmp/audit_main.txt`, `/tmp/audit_ti.txt` at audit time):

| Kernel family | State | Implication / fix target |
|---|---|---|
| `flash_attn_tile` dkq128-dv128, J>=8 | V128 + ~890-920 dword spill (5-12 at J<=4); LDS 29696 (2 WG/CU) | The J>=8 instances are the prefill/batched FATTN path for head-128 models. Reduce live tile state (subtile DV, fewer per-lane K accums) or cap J; spill traffic per thread at this scale is a big LDS round-trip tax. |
| `flash_attn_tile` dkq96 / dkq256, J>=8 | spill 140-265 dwords | Same direction, smaller. |
| `gcn_f32_matvec_rows_multi_nc<16>` (furnace port, ggml-cuda.cu) | V256 + spill up to 394; nc=7..16 spill 38-394; nc<=6 clean, 10 waves at V<=64 | Concurrent-decode F32 matvec spills. Shave per-thread accumulators or split work; nc<=6 variant is fine. |
| `rms_norm_mul_rope_f32` (round-2 M-RoPE fusion) | SGPR 100 + SGPR spill 15-20, W8 | SGPR spill = scratch scratchpad traffic on a per-token hot kernel. Pack the nine 64-bit args / offsets; target <= 80 SGPR (10 waves, no spill). |
| `mul_mat_vec_q8_0_repacked_lds_nc<6>` / `<8>` | V105 / V122 -> 9 / 8 waves | Shave 5 / 22 VGPRs -> back over the 100 cliff to 10 waves, on the 6-8 slot concurrent-decode Q8_0 path. |
| `mmq_gemm_*_repacked` (prefill GEMM) | V115-128, spill 2-3 dwords on several; LDS 12.3-25.6 KB (2 WG/CU at >= 16 KB) | Spills are trivial to eliminate (few regs). VGPR cliffs don't matter here - LDS keeps them at 2 WG/CU; the knob is LDS per WG, not VGPR, on these. |
| canonical MMQ instances (q1_0/q2_0/q2_k J=64) | V211-256, some spill (e.g. q2_k J64 B1: 394) | Check which J values the Vega table actually dispatches before optimizing; the spilling ones are dead weight if never launched. |
| `mul_mat_vec_q` MXFP4 ncols 5-8 | V132-201, 5-7 waves | Only matters for MXFP4 models; would need per-thread acc reduction. |
| decode matvec core (q5k/q6k repacked 2..16 col, glu16) | V56-64, no spill, 10 waves | Already at the cliff-free optimum; leave alone. |

Caveats: dynamic-LDS kernels (canonical MMQ) report LDS 0 in metadata - the real LDS comes
from the config table at launch, so occupancy there is config-bound first. Spill units come
from the runtime metadata (`vgpr_spill_count` semantics vary by code-object version; treat
>0 as bad, magnitude relative between variants).



## Appendix: where to look in the dump

| Topic | Internal page (PDF page) |
|---|---|
| New features / new instructions | 2-3 (10-11) |
| Terminology, wavefront, organization | 6-10 (14-18) |
| State: EXEC, priority, FP16_OVFL, VSKIP | 13-15 (21-23) |
| SGPR/VGPR/LDS allocation, M0 | 15-18 (23-26) |
| S_WAITCNT counters, NOP table | 23-25 (31-33) |
| VALU operand rules, inline constants, packed math | 37-45 (45-53) |
| SMEM | 46-50 (54-58) |
| VMEM buffer, V# fields, GLC, load-to-LDS | 51-64 (59-72) |
| FLAT/GLOBAL/SCRATCH + timing | 76-81 (84-89) |
| LDS banks, conflicts, instruction set | 82-88 (90-96) |
| VOP3P opcode list incl. dots | 155-156 (163-164) |
| DS_SWIZZLE_B32 details | 197-198 (205-206) |
| DPP / SDWA limitations | 221-222 (229-230) |
| VOP3P format fields (OPSEL/NEG_HI) | 263 (271) |
| DPP format + control enumeration | 267-269 (275-277) |
