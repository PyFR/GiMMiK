void
% if n is None:
${kname}(int n,
         const ${dtype}* restrict b, int ldb,
         ${dtype}* restrict c, int ldc)
{
% else:
${kname}(const ${dtype}* restrict b, ${dtype}* restrict c)
{
    const int n = ${n};
    const ${'long long' if k*ldb >= 2**31 else 'int'} ldb = ${ldb};
    const ${'long long' if m*ldc >= 2**31 else 'int'} ldc = ${ldc};
% endif

    #pragma omp simd
    for (int i = 0; i < n; i++)
    {
% for j, jx in enumerate(A):
  % if beta == 0:
        c[i + ${j}*ldc] = ${dot(lambda kx: f'b[i + {kx}*ldb]', jx)};
  % elif beta == 1:
        c[i + ${j}*ldc] += ${dot(lambda kx: f'b[i + {kx}*ldb]', jx)};
  % else:
        c[i + ${j}*ldc] = ${dot(lambda kx: f'b[i + {kx}*ldb]', jx)}
                        + ${beta}*c[i + ${j}*ldc];
  % endif
% endfor
    }
}
