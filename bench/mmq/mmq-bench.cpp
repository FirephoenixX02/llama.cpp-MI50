// MMQ tile-config bench: times one MUL_MAT of a given quant type on plain (not repacked)
// weights, so the launch goes through the upstream MMQ kernel and the gfx906 config rows
// in ggml-cuda/mmq-config-gcn.cuh. Used to sweep those rows; see RESULTS.md.
//
// build: clang++ -O2 -std=c++17 -I ggml/include mmq-bench.cpp -L build/bin -lggml -lggml-base -Wl,-rpath,$PWD/build/bin
// run:   ./a.out [repeat]

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct shape {
    ggml_type type;
    int64_t   k;  // src0->ne[0]
    int64_t   m;  // src0->ne[1]
    int64_t   n;  // src1->ne[1]
};

// Shapes worth measuring: the two ffn GEMMs of a 4096-wide model at ubatch 512, a
// 4096x4096 attention projection, a small-batch (verify / MTP) case, and the synthetic
// shape test-backend-ops uses (k=14336) so numbers can be cross-checked against it.
static const shape SHAPES[] = {
    { GGML_TYPE_Q4_K,  4096, 12288, 512 },
    { GGML_TYPE_Q4_K,  4096,  4096, 512 },
    { GGML_TYPE_Q4_K,  4096, 12288,  64 },
    { GGML_TYPE_Q4_K, 14336,  4096, 512 },
    { GGML_TYPE_Q6_K,  4096,  4096, 512 },
    { GGML_TYPE_Q8_0,  4096,  4096, 512 },
    { GGML_TYPE_Q4_0,  4096,  4096, 512 },
    { GGML_TYPE_Q5_K,  4096,  4096, 512 },
};

static const char * short_name(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_0: return "q4_0";
        case GGML_TYPE_Q5_K: return "q5_K";
        case GGML_TYPE_Q6_K: return "q6_K";
        case GGML_TYPE_Q8_0: return "q8_0";
        case GGML_TYPE_Q4_K: return "q4_K";
        default:             return "?";
    }
}

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 20;

    ggml_backend_load_all();

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            dev = d;
            break;
        }
    }
    if (!dev) {
        fprintf(stderr, "no GPU backend\n");
        return 1;
    }
    std::string dev_name = ggml_backend_dev_name(dev);
    if (dev_name.rfind("ROCm", 0) != 0) {
        fprintf(stderr, "expected a ROCm device, got %s\n", dev_name.c_str());
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    printf("# %s, reps %d\n", dev_name.c_str(), reps);
    printf("%-6s %6s %6s %4s %10s %8s\n", "type", "k", "m", "n", "us/run", "TFLOPS");

    for (const shape & s : SHAPES) {
        ggml_init_params ip = { 512u*1024u*1024u, nullptr, true };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * src0 = ggml_new_tensor_2d(ctx, s.type, s.k, s.m);
        ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, s.n);
        ggml_tensor * dst  = ggml_mul_mat(ctx, src0, src1);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

        std::vector<float> x(s.k);
        std::vector<float> act((size_t) s.k * s.n);
        for (auto & v : act) {
            v = dist(rng);
        }
        // real quantized weights, so a config that skips K work shows up in the checksum
        std::vector<float> w((size_t) s.k * s.m);
        for (auto & v : w) {
            v = dist(rng);
        }
        std::vector<char> qw(ggml_nbytes(src0));
        ggml_quantize_chunk(s.type, w.data(), qw.data(), 0, s.m, s.k, nullptr);
        ggml_backend_tensor_set(src0, qw.data(), 0, qw.size());
        ggml_backend_tensor_set(src1, act.data(), 0, ggml_nbytes(src1));

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dst);

        // warm up, then time reps graph computes (one node each) after a sync
        ggml_backend_synchronize(backend);
        for (int i = 0; i < 3; i++) {
            ggml_backend_graph_compute(backend, gf);
        }
        ggml_backend_synchronize(backend);

        const double t0 = ggml_time_us();
        for (int i = 0; i < reps; i++) {
            ggml_backend_graph_compute(backend, gf);
        }
        ggml_backend_synchronize(backend);
        const double us = (ggml_time_us() - t0) / (double) reps;
        const double flop = 2.0 * (double) s.k * (double) s.m * (double) s.n;
        std::vector<float> out((size_t) s.m * s.n);
        ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
        double ck = 0.0;
        for (float v : out) {
            ck += (double) v * (double) v;
        }
        printf("%-6s %6lld %6lld %4lld %10.1f %8.2f  ck %.6e\n", short_name(s.type),
               (long long) s.k, (long long) s.m, (long long) s.n, us, flop / us * 1e-6, ck);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    ggml_backend_free(backend);
    return 0;
}
