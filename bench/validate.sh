#!/usr/bin/env bash
# Build + run the correctness-validation suite (beta, fp32/float2, static+dynamic).
set -e
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source /opt/intel/oneapi/compiler/latest/env/vars.sh >/dev/null 2>&1
ONEAPI=/opt/intel/oneapi/compiler/latest

echo "== Generating validation kernels =="
python3 bench/validate_gen.py

echo "== Building OpenCL validator =="
icpx -O2 -std=c++17 -Ibench -I"$ONEAPI/include" \
     bench/validate_ocl.cpp -o bench/vbuild/validate_ocl \
     -L"$ONEAPI/lib" -lOpenCL

echo "== Building SYCL validator =="
icpx -fsycl -O2 -std=c++17 -Ibench \
     bench/validate_sycl.cpp bench/vbuild/vsycl_registry.cpp bench/vbuild/sycl/*.cpp \
     -o bench/vbuild/validate_sycl

echo; echo "########## OpenCL validation ##########"
bench/vbuild/validate_ocl
echo; echo "########## SYCL validation ##########"
ONEAPI_DEVICE_SELECTOR='level_zero:gpu' bench/vbuild/validate_sycl
