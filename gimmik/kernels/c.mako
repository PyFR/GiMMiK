# -*- coding: utf-8 -*-

void
${funcn}(int n,
         const ${dtype}* restrict b, int ldb,
         ${dtype}* restrict c, int ldc)
{
    ${dtype} dotp;

    #pragma omp simd
    for (int i = 0; i < n; i++)
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
