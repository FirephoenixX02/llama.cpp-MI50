#!/usr/bin/env bash
set -euo pipefail

# Build against the full ROCm 7.2.4 stack in /opt exclusively.
# Ubuntu's HIP 5.x headers/libs in /usr must not leak into the build.
# -isystem includes /opt headers before Ubuntu's /usr/include/hip.
# -L /opt libs before Ubuntu's /usr libs, or the driver's -lamdhip64 resolves
#   to Ubuntu's libamdhip64.so.5 and both HIP runtimes get loaded (comgr crash).

HIPCXX=/opt/rocm-7.2.4/lib/llvm/bin/clang
HIP_PATH=/opt/rocm-7.2.4

HIPCXX="$HIPCXX" HIP_PATH="$HIP_PATH" \
    cmake -S . -B build \
        -DCMAKE_PREFIX_PATH=/opt/rocm-7.2.4 \
        -DCMAKE_IGNORE_PATH=/usr/lib/x86_64-linux-gnu/cmake \
        -DCMAKE_HIP_FLAGS="-isystem /opt/rocm-7.2.4/include" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/opt/rocm-7.2.4/lib" \
        -DCMAKE_EXE_LINKER_FLAGS="-L/opt/rocm-7.2.4/lib" \
        -DGGML_HIP=ON \
        -DGGML_HIP_GRAPHS=ON \
        -DGGML_CUDA_REPACK=1 \
        -DGGML_CUDA_REPACK_Q8_0=1 \
        -DGPU_TARGETS=gfx906 \
        -DCMAKE_BUILD_TYPE=Release

cmake --build build -j"$(nproc)"
