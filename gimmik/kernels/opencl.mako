# -*- coding: utf-8 -*-

% if dtype == 'double':
#if __OPENCL_VERSION__ < 120
# pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif
% endif

__kernel void
% if n is None:
${funcn}(int n,
         __global const ${dtype}* restrict b, int ldb,
         __global ${dtype}* restrict c, int ldc)
{
% else:
${funcn}(__global const ${dtype}* restrict b, __global ${dtype}* restrict c)
{
    const int n = ${n};
    const int ldb = ${ldb};
    const int ldc = ${ldc};
% endif
    int i = get_global_id(0);
    ${dtype} dotp;

    if (i < n)
    {
    % for j, jx in enumerate(mat):
        dotp = ${' + '.join(f'{kx}*b[i + {k}*ldb]'
                            for k, kx in enumerate(jx) if kx != 0) or 0};
    % if beta == 0:
        c[i + ${j}*ldc] = dotp;
    % elif beta == 1:
        c[i + ${j}*ldc] += dotp;
    % else:
        c[i + ${j}*ldc] = dotp + ${beta}*c[i + ${j}*ldc];
    % endif
    % endfor
    }
}
