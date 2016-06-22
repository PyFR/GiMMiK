# -*- coding: utf-8 -*-

#include <omp.h>

#define COLALIGN (32 / sizeof(${dtype}))

% for p, tiling in tilings.items():
% for q, (start, end) in enumerate(tiling):
void
${funcn}_${p}_${q}(int n,
                   const ${dtype}* restrict b, int ldb,
                   ${dtype}* restrict c, int ldc)
{
    ${dtype} dotp;

    #pragma omp simd
    for (int i = 0; i < n; i++)
    {
    % for j, jx in enumerate(mat[start:end], start=start):
        dotp = ${' + '.join('{kx}*b[i + {k}*ldb]'.format(k=k, kx=kx)
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
% endfor
% endfor

void
${funcn}(int ncol,
         const ${dtype}* restrict b, int ldb,
         ${dtype}* restrict c, int ldc)
{
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();

    % for p in sorted(tilings, reverse=True):
        ${'if' if loop.first else 'else if'} (nth % ${p} == 0)
        {
            // Distribute threads
            int ncolth = nth / ${p};

            // Row and column indices for our thread
            int rowix = tid / ncolth;
            int colix = tid % ncolth;

            // Round up ncol to be a multiple of ncolth
            int rncol = ncol + ncolth - 1 - (ncol - 1) % ncolth;

            // Nominal column tile size
            int ntilecol = rncol / ncolth;

            // Handle column alignment
            ntilecol += COLALIGN - 1 - (ntilecol - 1) % COLALIGN;

            // Assign the starting and ending column to each thread
            int colb = ntilecol * colix;
            int cole = (colb + ntilecol < ncol) ? colb + ntilecol : ncol;

        % for q in range(p):
            if (rowix == ${q} && colb < ncol)
                ${funcn}_${p}_${q}(cole - colb, b + colb, ldb, c + colb, ldc);
        % endfor
        }
    % endfor
    }
}
