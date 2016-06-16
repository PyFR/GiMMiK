# -*- coding: utf-8 -*-

% if dtype == 'double':
#if __OPENCL_VERSION__ < 120
# pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif
% endif

% for q, (start, end) in enumerate(tiling):
void
${funcn}_${q}(int i,
              __global const ${dtype}* restrict b, int ldb,
              __global ${dtype}* restrict c, int ldc)
{
    ${dtype} dotp;

% for j, jx in enumerate(mat[start:end], start=start):
    dotp = ${' + '.join('{kx}*b[i + {k}*ldb]'.format(k=k, kx=kx)
                        for k, kx in enumerate(jx) if kx != 0)};
% if beta == 0:
    c[i + ${j}*ldc] = dotp;
% elif beta == 1:
    c[i + ${j}*ldc] += dotp;
% else:
    c[i + ${j}*ldc] = dotp + ${beta}*c[i + ${j}*ldc];
% endif
% endfor
}
% endfor

__kernel void
${funcn}(int n,
         __global const ${dtype}* restrict b, int ldb,
         __global ${dtype}* restrict c, int ldc)
{
    int i = get_global_id(0);

    if (i < n)
    {
    % for q in range(len(tiling)):
        ${funcn}_${q}(i, b, ldb, c, ldc);
    % endfor
    }
}
