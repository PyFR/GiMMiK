#!/usr/bin/env python3
"""Generate a correctness-validation suite for the OpenCL and SYCL backends.

Covers the axes that the perf benchmark skips:
  * beta == 0 and beta != 0
  * fp64 and fp32 (including the fp32 float2 vectorized variants)
  * static-n and dynamic-n kernel signatures

For every (dtype, beta, mode, aligne) case it emits *all* kernel variants from
both backends, shared input/reference data, and C/C++ registries so the host
programs can run+verify each kernel against a NumPy fp64 reference.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gimmik import OpenCLMatMul, SYCLMatMul

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, 'vbuild')

# (dtype, beta, mode, aligne)
CASES = [
    ('float64', 0.0, 'static',  None),
    ('float64', 0.5, 'static',  None),
    ('float64', 0.0, 'dynamic', None),
    ('float64', 0.5, 'dynamic', None),
    ('float64', 0.0, 'static',  2),     # fp64 double2 vector kernels
    ('float64', 0.5, 'static',  2),     # + preload-C candidates (beta != 0)
    ('float64', 0.0, 'dynamic', 2),
    ('float32', 0.0, 'static',  None),
    ('float32', 0.5, 'static',  None),
    ('float32', 0.0, 'static',  2),     # enables float2 vector kernels
    ('float32', 0.0, 'dynamic', None),
]

M, K, N = 32, 48, 8192   # N even & divisible by 64 for the split/vector kernels


def elem_type(dtype, width):
    base = 'float' if dtype == 'float32' else 'double'
    if width > 1:
        return f'sycl::{base}{width}'
    return base


def main():
    os.makedirs(os.path.join(BUILD, 'ocl'), exist_ok=True)
    os.makedirs(os.path.join(BUILD, 'sycl'), exist_ok=True)

    rng = np.random.default_rng(42)
    A = rng.standard_normal((M, K))
    A[rng.random((M, K)) < 0.5] = 0.0
    for j in range(M):
        if not A[j].any():
            A[j, rng.integers(K)] = rng.standard_normal()

    ldb = ldc = N
    cases_meta = []
    ocl_kernels = []
    sycl_kernels = []

    for cid, (dtype, beta, mode, aligne) in enumerate(CASES):
        npdt = np.dtype(dtype)
        code = 1 if dtype == 'float32' else 0
        dsize = npdt.itemsize

        # shared data in the target dtype (exactly what the GPU reads)
        rB = rng.standard_normal((K, N)).astype(npdt)
        rCi = rng.standard_normal((M, N)).astype(npdt)
        Cref = A @ rB.astype(np.float64) + beta * rCi.astype(np.float64)

        rB.tofile(os.path.join(BUILD, f'case{cid}_B.bin'))
        rCi.tofile(os.path.join(BUILD, f'case{cid}_Ci.bin'))
        Cref.astype('<f8').tofile(os.path.join(BUILD, f'case{cid}_Cref.bin'))

        cases_meta.append({'id': cid, 'dtype': dtype, 'code': code,
                           'dsize': dsize, 'beta': beta, 'mode': mode,
                           'm': M, 'k': K, 'n': N, 'ldb': ldb, 'ldc': ldc})

        if mode == 'static':
            kw = dict(n=N, ldb=ldb, ldc=ldc)
        else:
            kw = dict()

        for plat, cls, sub, ext in [('opencl', OpenCLMatMul, 'ocl', 'cl'),
                                    ('sycl', SYCLMatMul, 'sycl', 'cpp')]:
            mm = cls(A, beta=beta, aligne=aligne, **kw)
            for idx, (src, meta) in enumerate(
                    mm.kernels(npdt.type, kname='gimmik_mm')):
                tpl, width = meta['tplname'], meta['width']
                entry = f'gmk_{plat}_c{cid}_{idx}_{tpl.replace("-", "_")}_w{width}'
                src = src.replace('gimmik_mm', entry)
                with open(os.path.join(BUILD, sub, f'{entry}.{ext}'), 'w') as f:
                    f.write(src)

                lws = meta.get('local_work_size')
                rec = {'entry': entry, 'file': f'{sub}/{entry}.{ext}',
                       'tpl': tpl, 'width': width, 'case': cid, 'mode': mode,
                       'lws': list(lws) if lws else None}
                if plat == 'opencl':
                    ocl_kernels.append(rec)
                else:
                    rec['etype'] = elem_type(dtype, width)
                    sycl_kernels.append(rec)

    # ---- cases header ----
    with open(os.path.join(BUILD, 'vcases.h'), 'w') as f:
        f.write('#pragma once\n')
        f.write('typedef struct { int id; int code; int dsize; double beta; '
                'int is_dynamic; int m,k,n,ldb,ldc; } VCase;\n')
        f.write('static const VCase g_cases[] = {\n')
        for c in cases_meta:
            f.write(f'  {{{c["id"]}, {c["code"]}, {c["dsize"]}, {c["beta"]}, '
                    f'{1 if c["mode"]=="dynamic" else 0}, {c["m"]}, {c["k"]}, '
                    f'{c["n"]}, {c["ldb"]}, {c["ldc"]}}},\n')
        f.write('};\n')
        f.write(f'static const int g_cases_n = {len(cases_meta)};\n')

    # ---- OpenCL registry ----
    with open(os.path.join(BUILD, 'vocl_registry.h'), 'w') as f:
        f.write('#pragma once\n')
        f.write('typedef struct { const char* entry; const char* file; '
                'const char* tpl; int cas; int width; int is_dynamic; '
                'int has_lws; size_t l0,l1; } VOcl;\n')
        f.write('static const VOcl g_vocl[] = {\n')
        for r in ocl_kernels:
            l0 = r['lws'][0] if r['lws'] else 0
            l1 = r['lws'][1] if r['lws'] else 0
            f.write(f'  {{"{r["entry"]}", "{r["file"]}", "{r["tpl"]}", '
                    f'{r["case"]}, {r["width"]}, '
                    f'{1 if r["mode"]=="dynamic" else 0}, '
                    f'{1 if r["lws"] else 0}, {l0}, {l1}}},\n')
        f.write('};\n')
        f.write(f'static const int g_vocl_n = {len(ocl_kernels)};\n')

    # ---- SYCL registry (typed wrappers -> uniform thunk) ----
    with open(os.path.join(BUILD, 'vsycl_registry.cpp'), 'w') as f:
        f.write('#include "vsycl_common.hpp"\n')
        for r in sycl_kernels:
            et = r['etype']
            if r['mode'] == 'static':
                f.write(f'sycl::event {r["entry"]}(sycl::queue&, '
                        f'const {et}*, {et}*);\n')
            else:
                f.write(f'sycl::event {r["entry"]}(sycl::queue&, int, '
                        f'const {et}*, int, {et}*, int);\n')
        for r in sycl_kernels:
            et = r['entry']
            pt = r['etype']
            f.write(f'static sycl::event w_{et}(sycl::queue& q, void* b, '
                    f'void* c, int n, int ldb, int ldc) {{ return {et}(q, ')
            if r['mode'] == 'static':
                f.write(f'(const {pt}*)b, ({pt}*)c); }}\n')
            else:
                f.write(f'n, (const {pt}*)b, ldb, ({pt}*)c, ldc); }}\n')
        f.write('const VSycl g_vsycl[] = {\n')
        for r in sycl_kernels:
            f.write(f'  {{"{r["entry"]}", "{r["tpl"]}", {r["case"]}, '
                    f'&w_{r["entry"]}}},\n')
        f.write('};\n')
        f.write(f'const int g_vsycl_n = {len(sycl_kernels)};\n')

    print(f'A: {M}x{K}, nnz={int((A!=0).sum())}')
    print(f'cases={len(cases_meta)}  opencl_kernels={len(ocl_kernels)}  '
          f'sycl_kernels={len(sycl_kernels)}')


if __name__ == '__main__':
    main()
