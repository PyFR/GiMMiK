#!/usr/bin/env bash
# Build + run the OpenCL vs SYCL GiMMiK benchmark on the Intel GPU.
# Run from the GiMMiK repository root.
set -e

REPS=${1:-50}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

source /opt/intel/oneapi/compiler/latest/env/vars.sh >/dev/null 2>&1
ONEAPI=/opt/intel/oneapi/compiler/latest

echo "== Generating kernels =="
python3 bench/bench_gen.py

echo "== Building OpenCL host =="
icpx -O3 -std=c++17 -Ibench -I"$ONEAPI/include" \
     bench/ocl_bench.cpp -o bench/build/ocl_bench \
     -L"$ONEAPI/lib" -lOpenCL

echo "== Building SYCL host =="
icpx -fsycl -O3 -std=c++17 -Ibench \
     bench/sycl_bench.cpp bench/build/sycl_registry.cpp bench/build/sycl/*.cpp \
     -o bench/build/sycl_bench

echo
echo "########## OpenCL ##########"
bench/build/ocl_bench "$REPS"
echo
echo "########## SYCL ##########"
ONEAPI_DEVICE_SELECTOR='level_zero:gpu' bench/build/sycl_bench "$REPS"
