#!/usr/bin/env python3
"""Generate OpenCL and SYCL GiMMiK kernels + shared input data for benchmarking.

Produces, under bench/build/:
  ocl/<name>.cl      - one OpenCL kernel per variant (unique entry name)
  sycl/<name>.cpp    - one SYCL launcher per variant (unique entry name)
  B.bin, Cref.bin    - shared fp64 input / reference output (row-major, k x n / m x n)
  manifest.json      - description of every kernel + problem dims
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gimmik import OpenCLMatMul, SYCLMatMul

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, 'build')


def make_operator(m, k, sparsity, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, k))
    A[rng.random((m, k)) < sparsity] = 0.0
    # guarantee no fully-zero row so every output column is exercised
    for j in range(m):
        if not A[j].any():
            A[j, rng.integers(k)] = rng.standard_normal()
    return A


def main():
    m = int(os.environ.get('GMK_M', 32))
    k = int(os.environ.get('GMK_K', 48))
    n = int(os.environ.get('GMK_N', 200000))
    sparsity = float(os.environ.get('GMK_SPARSITY', 0.5))

    ldb = ldc = n
    A = make_operator(m, k, sparsity)
    nnz = int((A != 0).sum())

    rng = np.random.default_rng(7)
    B = rng.standard_normal((k, n))
    Cref = A @ B

    os.makedirs(os.path.join(BUILD, 'ocl'), exist_ok=True)
    os.makedirs(os.path.join(BUILD, 'sycl'), exist_ok=True)

    B.astype('<f8').tofile(os.path.join(BUILD, 'B.bin'))
    Cref.astype('<f8').tofile(os.path.join(BUILD, 'Cref.bin'))

    manifest = {
        'm': m, 'k': k, 'n': n, 'ldb': ldb, 'ldc': ldc,
        'nnz': nnz, 'sparsity': sparsity, 'dtype': 'double', 'dsize': 8,
        'nbix': int(np.count_nonzero(np.any(A != 0, axis=0))),
        'kernels': [],
    }

    backends = [('opencl', OpenCLMatMul, 'ocl', 'cl'),
                ('sycl', SYCLMatMul, 'sycl', 'cpp')]

    for plat, cls, subdir, ext in backends:
        mm = cls(A, beta=0.0, n=n, ldb=ldb, ldc=ldc)
        for idx, (src, meta) in enumerate(mm.kernels(np.float64, kname='gimmik_mm')):
            tpl = meta['tplname']
            width = meta['width']
            entry = f'gmk_{plat}_{idx}_{tpl.replace("-", "_")}_w{width}'
            src = src.replace('gimmik_mm', entry)
            fname = f'{entry}.{ext}'
            with open(os.path.join(BUILD, subdir, fname), 'w') as f:
                f.write(src)

            gws = meta.get('global_work_size')
            lws = meta.get('local_work_size')
            manifest['kernels'].append({
                'platform': plat, 'entry': entry, 'file': f'{subdir}/{fname}',
                'tpl': tpl, 'width': width,
                'gws': list(gws) if gws else None,
                'lws': list(lws) if lws else None,
                'local_mem_size': meta.get('local_mem_size', 0),
            })

    with open(os.path.join(BUILD, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    # Shared problem dimensions
    with open(os.path.join(BUILD, 'dims.h'), 'w') as f:
        f.write('#pragma once\n')
        f.write(f'#define GMK_M {m}\n#define GMK_K {k}\n#define GMK_N {n}\n')
        f.write(f'#define GMK_LDB {ldb}\n#define GMK_LDC {ldc}\n')
        f.write(f'#define GMK_NNZ {nnz}\n#define GMK_NBIX {manifest["nbix"]}\n')

    ocl = [x for x in manifest['kernels'] if x['platform'] == 'opencl']
    scl = [x for x in manifest['kernels'] if x['platform'] == 'sycl']

    # OpenCL registry (entry name, source file, work sizes)
    with open(os.path.join(BUILD, 'ocl_registry.h'), 'w') as f:
        f.write('#pragma once\n')
        f.write('typedef struct { const char* entry; const char* file; '
                'const char* tpl; int gdim; size_t g0,g1; size_t l0,l1; } '
                'OclKernel;\n')
        f.write('static const OclKernel g_ocl[] = {\n')
        for x in ocl:
            g = x['gws']
            l = x['lws']
            gdim = len(g)
            g0 = g[0]
            g1 = g[1] if gdim > 1 else 1
            l0 = l[0] if l else 0
            l1 = l[1] if l else 0
            f.write(f'  {{"{x["entry"]}", "{x["file"]}", "{x["tpl"]}", '
                    f'{gdim}, {g0}, {g1}, {l0}, {l1}}},\n')
        f.write('};\n')
        f.write(f'static const int g_ocl_n = {len(ocl)};\n')

    # SYCL registry (declarations + function pointer table)
    with open(os.path.join(BUILD, 'sycl_registry.cpp'), 'w') as f:
        f.write('#include "sycl_common.hpp"\n')
        for x in scl:
            f.write(f'sycl::event {x["entry"]}'
                    f'(sycl::queue&, const double*, double*);\n')
        f.write('const SyclKernel g_sycl[] = {\n')
        for x in scl:
            f.write(f'  {{"{x["entry"]}", "{x["tpl"]}", &{x["entry"]}}},\n')
        f.write('};\n')
        f.write(f'const int g_sycl_n = {len(scl)};\n')

    print(f'A: {m}x{k}, nnz={nnz} ({100*nnz/(m*k):.1f}% dense), '
          f'used B rows={manifest["nbix"]}')
    print(f'n={n}  B={B.nbytes/1e6:.1f} MB  C={Cref.nbytes/1e6:.1f} MB')
    print(f'Generated {len(manifest["kernels"])} kernels into {BUILD}')


if __name__ == '__main__':
    main()
