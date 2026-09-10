#pragma once

// Single-GPU MI50 (gfx906) optimizations ported from mi50grad
// - RMSNorm vectorized is in norm.cu: rms_norm_f32_gfx906
// - Dual FFN 4x, GEMV v8 4x, FlashAttn v3 16x16 below are gated GCN kernels
//   They are compiled but dispatched only when eligible (single-GPU, no P2P)

#if defined(GGML_USE_HIP) && defined(GCN)
#include <hip/hip_fp16.h>

// ---------------------------------------------------------------------------
// Dual FFN 4x register blocking: fused gate+up GEMV + SiLU
// 4 words (32 INT4) per iteration, dual acc gate0/1 up0/1 to break dependency
// Grid: (ceil(N/256), k_splits), Block: 256, each col = one output
// Weights: GPTQ INT4 [K/8,N] + scales/zeros [num_groups,N] FP16
// For GGUF Q4_K repacked, the same 4x pattern applies to repack-gcn.cu's
// mul_mat_vec_q4k_repacked_glu — this kernel is the standalone GPTQ variant
// kept for reference and for Q4_0/Q4_K_M fallback when REPACK=0.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float mi50_dequant_dot8_fp32(
    unsigned int packed, float scale, float zero, const __half* __restrict__ A, unsigned int a_base) {
    float a0 = __half2float(A[a_base]); float a1 = __half2float(A[a_base+1]);
    float a2 = __half2float(A[a_base+2]); float a3 = __half2float(A[a_base+3]);
    float a4 = __half2float(A[a_base+4]); float a5 = __half2float(A[a_base+5]);
    float a6 = __half2float(A[a_base+6]); float a7 = __half2float(A[a_base+7]);
    float w0 = ((float)__builtin_amdgcn_ubfe(packed,0,4)-zero)*scale;
    float w1 = ((float)__builtin_amdgcn_ubfe(packed,4,4)-zero)*scale;
    float w2 = ((float)__builtin_amdgcn_ubfe(packed,8,4)-zero)*scale;
    float w3 = ((float)__builtin_amdgcn_ubfe(packed,12,4)-zero)*scale;
    float w4 = ((float)__builtin_amdgcn_ubfe(packed,16,4)-zero)*scale;
    float w5 = ((float)__builtin_amdgcn_ubfe(packed,20,4)-zero)*scale;
    float w6 = ((float)__builtin_amdgcn_ubfe(packed,24,4)-zero)*scale;
    float w7 = ((float)__builtin_amdgcn_ubfe(packed,28,4)-zero)*scale;
    return w0*a0+w1*a1+w2*a2+w3*a3+w4*a4+w5*a5+w6*a6+w7*a7;
}

__global__ void mi50_gemv_dual_fused_4x(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_gate, const __half* __restrict__ gate_scales, const __half* __restrict__ gate_zeros,
    const unsigned int* __restrict__ B_up, const __half* __restrict__ up_scales, const __half* __restrict__ up_zeros,
    float* __restrict__ C_gate, float* __restrict__ C_up, unsigned int* __restrict__ done,
    __half* __restrict__ out, unsigned int K, unsigned int N, unsigned int group_size, unsigned int k_splits) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    unsigned int num_k_groups = K >> 3;
    unsigned int groups_per_scale = group_size >> 3;
    unsigned int k_split_id = blockIdx.y;
    unsigned int kg_start = (num_k_groups * k_split_id) / k_splits;
    unsigned int kg_end = (num_k_groups * (k_split_id+1)) / k_splits;
    float acc_gate0=0, acc_gate1=0, acc_up0=0, acc_up1=0;
    float gate_s=0, gate_z=0, up_s=0, up_z=0;
    unsigned int last_sg = 0xFFFFFFFF;
    unsigned int kg = kg_start;
    unsigned int kg_end_4 = kg_start + ((kg_end - kg_start) & ~3u);
    for (; kg < kg_end_4; kg+=4) {
        unsigned int gp0=B_gate[kg*N+col], up0=B_up[kg*N+col];
        unsigned int gp1=B_gate[(kg+1)*N+col], up1=B_up[(kg+1)*N+col];
        unsigned int gp2=B_gate[(kg+2)*N+col], up2=B_up[(kg+2)*N+col];
        unsigned int gp3=B_gate[(kg+3)*N+col], up3=B_up[(kg+3)*N+col];
        unsigned int sg0=kg/groups_per_scale;
        if(sg0!=last_sg){gate_s=__half2float(gate_scales[sg0*N+col]); gate_z=__half2float(gate_zeros[sg0*N+col]); up_s=__half2float(up_scales[sg0*N+col]); up_z=__half2float(up_zeros[sg0*N+col]); last_sg=sg0;}
        acc_gate0+=mi50_dequant_dot8_fp32(gp0,gate_s,gate_z,A,kg<<3); acc_up0+=mi50_dequant_dot8_fp32(up0,up_s,up_z,A,kg<<3);
        unsigned int sg1=(kg+1)/groups_per_scale;
        if(sg1!=last_sg){gate_s=__half2float(gate_scales[sg1*N+col]); gate_z=__half2float(gate_zeros[sg1*N+col]); up_s=__half2float(up_scales[sg1*N+col]); up_z=__half2float(up_zeros[sg1*N+col]); last_sg=sg1;}
        acc_gate1+=mi50_dequant_dot8_fp32(gp1,gate_s,gate_z,A,(kg+1)<<3); acc_up1+=mi50_dequant_dot8_fp32(up1,up_s,up_z,A,(kg+1)<<3);
        unsigned int sg2=(kg+2)/groups_per_scale;
        if(sg2!=last_sg){gate_s=__half2float(gate_scales[sg2*N+col]); gate_z=__half2float(gate_zeros[sg2*N+col]); up_s=__half2float(up_scales[sg2*N+col]); up_z=__half2float(up_zeros[sg2*N+col]); last_sg=sg2;}
        acc_gate0+=mi50_dequant_dot8_fp32(gp2,gate_s,gate_z,A,(kg+2)<<3); acc_up0+=mi50_dequant_dot8_fp32(up2,up_s,up_z,A,(kg+2)<<3);
        unsigned int sg3=(kg+3)/groups_per_scale;
        if(sg3!=last_sg){gate_s=__half2float(gate_scales[sg3*N+col]); gate_z=__half2float(gate_zeros[sg3*N+col]); up_s=__half2float(up_scales[sg3*N+col]); up_z=__half2float(up_zeros[sg3*N+col]); last_sg=sg3;}
        acc_gate1+=mi50_dequant_dot8_fp32(gp3,gate_s,gate_z,A,(kg+3)<<3); acc_up1+=mi50_dequant_dot8_fp32(up3,up_s,up_z,A,(kg+3)<<3);
    }
    for(; kg<kg_end; ++kg){unsigned int sg=kg/groups_per_scale; if(sg!=last_sg){gate_s=__half2float(gate_scales[sg*N+col]); gate_z=__half2float(gate_zeros[sg*N+col]); up_s=__half2float(up_scales[sg*N+col]); up_z=__half2float(up_zeros[sg*N+col]); last_sg=sg;} acc_gate0+=mi50_dequant_dot8_fp32(B_gate[kg*N+col],gate_s,gate_z,A,kg<<3); acc_up0+=mi50_dequant_dot8_fp32(B_up[kg*N+col],up_s,up_z,A,kg<<3);}
    float acc_gate=acc_gate0+acc_gate1, acc_up=acc_up0+acc_up1;
    atomicAdd(&C_gate[col], acc_gate); atomicAdd(&C_up[col], acc_up);
    __threadfence();
    unsigned int old=atomicAdd(&done[col],1u);
    if(old==k_splits-1u){float g=C_gate[col], u=C_up[col]; float sig=1.0f/(1.0f+__expf(-g)); out[col]=__float2half(g*sig*u); C_gate[col]=0; C_up[col]=0; done[col]=0;}
}

// ---------------------------------------------------------------------------
// GEMV INT4 v8 4x: standalone GEMV with 4x register blocking, dual acc
// Grid: (ceil(N/THREADS_PER_COL),1), Block:256, COLS_PER_WG=256/TPC
// ---------------------------------------------------------------------------
template<int THREADS_PER_COL>
__device__ void mi50_gemv_v8_coop(
    const __half* __restrict__ A, const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales, const __half* __restrict__ zeros,
    __half* __restrict__ C, unsigned int K, unsigned int N, unsigned int group_size) {
    constexpr unsigned int COLS_PER_WG=256/THREADS_PER_COL;
    constexpr unsigned int NUM_WF=4;
    constexpr unsigned int TPC_PER_WF=THREADS_PER_COL/NUM_WF;
    unsigned int col_in_wg=threadIdx.x%COLS_PER_WG;
    unsigned int k_split_id=threadIdx.x/COLS_PER_WG;
    unsigned int col=blockIdx.x*COLS_PER_WG+col_in_wg;
    if(col>=N) return;
    unsigned int num_k_groups=K>>3;
    unsigned int gps=group_size>>3;
    unsigned int gps_log2=31u-__builtin_clz(gps);
    unsigned int kg_start=(num_k_groups*k_split_id)/THREADS_PER_COL;
    unsigned int kg_end=(num_k_groups*(k_split_id+1))/THREADS_PER_COL;
    if(kg_start>=kg_end) return;
    float acc0=0, acc1=0;
    float cur_scale=0, cur_zero=0;
    unsigned int last_sg=0xFFFFFFFF;
    unsigned int kg=kg_start;
    unsigned int kg_end_4=kg_start+((kg_end-kg_start)&~3u);
    for(;kg<kg_end_4;kg+=4){
        unsigned int p0=B_q4[kg*N+col], p1=B_q4[(kg+1)*N+col], p2=B_q4[(kg+2)*N+col], p3=B_q4[(kg+3)*N+col];
        unsigned int g0=kg>>gps_log2; if(g0!=last_sg){cur_scale=__half2float(scales[g0*N+col]); cur_zero=__half2float(zeros[g0*N+col]); last_sg=g0;}
        acc0+=mi50_dequant_dot8_fp32(p0,cur_scale,cur_zero,A,kg<<3);
        unsigned int g1=(kg+1)>>gps_log2; if(g1!=last_sg){cur_scale=__half2float(scales[g1*N+col]); cur_zero=__half2float(zeros[g1*N+col]); last_sg=g1;}
        acc1+=mi50_dequant_dot8_fp32(p1,cur_scale,cur_zero,A,(kg+1)<<3);
        unsigned int g2=(kg+2)>>gps_log2; if(g2!=last_sg){cur_scale=__half2float(scales[g2*N+col]); cur_zero=__half2float(zeros[g2*N+col]); last_sg=g2;}
        acc0+=mi50_dequant_dot8_fp32(p2,cur_scale,cur_zero,A,(kg+2)<<3);
        unsigned int g3=(kg+3)>>gps_log2; if(g3!=last_sg){cur_scale=__half2float(scales[g3*N+col]); cur_zero=__half2float(zeros[g3*N+col]); last_sg=g3;}
        acc1+=mi50_dequant_dot8_fp32(p3,cur_scale,cur_zero,A,(kg+3)<<3);
    }
    float acc=acc0+acc1;
    for(;kg<kg_end;++kg){unsigned int p=B_q4[kg*N+col]; unsigned int g=kg>>gps_log2; if(g!=last_sg){cur_scale=__half2float(scales[g*N+col]); cur_zero=__half2float(zeros[g*N+col]); last_sg=g;} acc+=mi50_dequant_dot8_fp32(p,cur_scale,cur_zero,A,kg<<3);}
    if(TPC_PER_WF>=2) acc+=__shfl_down(acc,COLS_PER_WG);
    if(TPC_PER_WF>=4) acc+=__shfl_down(acc,COLS_PER_WG*2);
    __shared__ float s_red[4*64]; // max COLS_PER_WG 64
    if((k_split_id & (TPC_PER_WF-1))==0){unsigned int wf=k_split_id/TPC_PER_WF; s_red[wf*COLS_PER_WG+col_in_wg]=acc;}
    __syncthreads();
    if(k_split_id==0){float sum=s_red[col_in_wg]+s_red[COLS_PER_WG+col_in_wg]+s_red[2*COLS_PER_WG+col_in_wg]+s_red[3*COLS_PER_WG+col_in_wg]; C[col]=__float2half(sum);}
}
__global__ void mi50_gemv_v8_t16(const __half* A, const unsigned int* B, const __half* sc, const __half* ze, __half* C, unsigned int K, unsigned int N, unsigned int gs){mi50_gemv_v8_coop<16>(A,B,sc,ze,C,K,N,gs);}
__global__ void mi50_gemv_v8_t8(const __half* A, const unsigned int* B, const __half* sc, const __half* ze, __half* C, unsigned int K, unsigned int N, unsigned int gs){mi50_gemv_v8_coop<8>(A,B,sc,ze,C,K,N,gs);}
__global__ void mi50_gemv_v8_t4(const __half* A, const unsigned int* B, const __half* sc, const __half* ze, __half* C, unsigned int K, unsigned int N, unsigned int gs){mi50_gemv_v8_coop<4>(A,B,sc,ze,C,K,N,gs);}

// ---------------------------------------------------------------------------
// FlashAttention 256 v3 prefill: BLOCK_M=16, BLOCK_N=16, v_dot2_f32_f16
// 256 threads = 4 WF x 64, 4 Q rows/WF, 16 dims/thread, 8 fdot2 per score
// LDS 16KB (k_lds 8K + v_lds 8K), 2 WG/CU on 64KB LDS
// Single-GPU prefill only (decode uses 4-WF split already in v3 decode)
// For llama.cpp this is wired when head_dim==256 && ncols==16 && GCN
// ---------------------------------------------------------------------------
#define MI50_FA_HEAD_DIM 256
#define MI50_FA_BLOCK_M 16
#define MI50_FA_BLOCK_N 16
__global__ void mi50_flash_attn_256_v3_prefill(
    const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, __half* __restrict__ Out,
    unsigned int kv_seq_len, unsigned int num_q_rows, unsigned int num_heads, unsigned int num_kv_heads, unsigned int causal) {
    __shared__ __half k_lds[MI50_FA_BLOCK_N*MI50_FA_HEAD_DIM];
    __shared__ __half v_lds[MI50_FA_BLOCK_N*MI50_FA_HEAD_DIM];
    const unsigned int tid=threadIdx.x, wf=tid/64, lane=tid%64;
    const unsigned int q_in_wf=lane/16, part=lane%16, dim_base=part*16;
    const unsigned int head=blockIdx.x, q_blk=blockIdx.y;
    const unsigned int q_row=wf*4+q_in_wf;
    const unsigned int q_global=q_blk*MI50_FA_BLOCK_M+q_row;
    const unsigned int kv_head=(unsigned int)((unsigned long long)head*num_kv_heads/num_heads);
    const float scale=rsqrtf((float)MI50_FA_HEAD_DIM);
    float qreg[16];
    if(q_global < num_q_rows){
        size_t q_off=((size_t)q_global*num_heads+head)*MI50_FA_HEAD_DIM+dim_base;
        #pragma unroll
        for(int d=0;d<16;++d) qreg[d]=__half2float(Q[q_off+d])*scale;
    } else {
        #pragma unroll
        for(int d=0;d<16;++d) qreg[d]=0;
    }
    float acc[16]={0}; float rmax=-1e30f, rsum=0;
    const size_t kv_stride=(size_t)num_kv_heads*MI50_FA_HEAD_DIM;
    const size_t kv_off_base=(size_t)kv_head*MI50_FA_HEAD_DIM;
    unsigned int kv_limit=causal? min(q_global+1,kv_seq_len):kv_seq_len;
    unsigned int nblk=(kv_limit+MI50_FA_BLOCK_N-1)/MI50_FA_BLOCK_N;
    unsigned int load_row=tid/16, load_dim=(tid%16)*16;
    for(unsigned int blk=0;blk<nblk;++blk){
        unsigned int kv0=blk*MI50_FA_BLOCK_N;
        unsigned int kv1=min(kv0+MI50_FA_BLOCK_N,kv_limit);
        unsigned int bc=kv1-kv0;
        {
            __half* dst=&k_lds[load_row*MI50_FA_HEAD_DIM+load_dim];
            if(load_row<bc){
                size_t k_off=(size_t)(kv0+load_row)*kv_stride+kv_off_base+load_dim;
                *(float4*)dst=*(const float4*)(K+k_off);
                *(float4*)(dst+8)=*(const float4*)(K+k_off+8);
            } else {
                *(float4*)dst=make_float4(0,0,0,0); *(float4*)(dst+8)=make_float4(0,0,0,0);
            }
        }
        __syncthreads();
        float scores[16];
        #pragma unroll
        for(int r=0;r<MI50_FA_BLOCK_N;++r){
            const __half* krow=&k_lds[r*MI50_FA_HEAD_DIM+dim_base];
            float partial=0;
            #pragma unroll
            for(int d=0;d<16;d+=2){
                __half2 qv=__half2(__float2half(qreg[d]),__float2half(qreg[d+1]));
                __half2 kv=*(const __half2*)(krow+d);
                partial=__builtin_amdgcn_fdot2(*(const _Float16_2*)&qv, *(const _Float16_2*)&kv, partial, false);
            }
            partial+=__shfl_down(partial,8); partial+=__shfl_down(partial,4); partial+=__shfl_down(partial,2); partial+=__shfl_down(partial,1);
            scores[r]=__shfl(partial,(int)(q_in_wf*16));
            if(causal && kv0+(unsigned int)r>q_global) scores[r]=-1e30f;
        }
        #pragma unroll
        for(int r=0;r<MI50_FA_BLOCK_N;++r){
            float s=scores[r], nm=fmaxf(rmax,s), corr=__expf(rmax-nm), p=__expf(s-nm);
            #pragma unroll
            for(int d=0;d<16;++d) acc[d]*=corr;
            rsum=rsum*corr+p; rmax=nm; scores[r]=p;
        }
        {
            __half* dst=&v_lds[load_row*MI50_FA_HEAD_DIM+load_dim];
            if(load_row<bc){
                size_t v_off=(size_t)(kv0+load_row)*kv_stride+kv_off_base+load_dim;
                *(float4*)dst=*(const float4*)(V+v_off);
                *(float4*)(dst+8)=*(const float4*)(V+v_off+8);
            } else {
                *(float4*)dst=make_float4(0,0,0,0); *(float4*)(dst+8)=make_float4(0,0,0,0);
            }
        }
        __syncthreads();
        #pragma unroll
        for(int r=0;r<MI50_FA_BLOCK_N;++r){
            float p=scores[r]; const __half* vrow=&v_lds[r*MI50_FA_HEAD_DIM+dim_base];
            #pragma unroll
            for(int d=0;d<16;++d) acc[d]+=p*__half2float(vrow[d]);
        }
        __syncthreads();
    }
    if(q_global < num_q_rows && rsum>0){
        float inv=1.0f/rsum;
        size_t out_off=((size_t)q_global*num_heads+head)*MI50_FA_HEAD_DIM+dim_base;
        #pragma unroll
        for(int d=0;d<16;++d) Out[out_off+d]=__float2half(acc[d]*inv);
    }
}

#endif // GGML_USE_HIP && GCN
