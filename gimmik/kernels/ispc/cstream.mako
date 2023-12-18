export void
% if n is None:
${kname}(uniform int n,
         const uniform ${dtype} b[], uniform int ldb,
         ${dtype} uniform c[], uniform int ldc)
{
% else:
${kname}(const uniform ${dtype} b[], ${dtype} uniform c[])
{
    const uniform int n = ${n};
    const uniform ${'long long' if k*ldb >= 2**31 else 'int'} ldb = ${ldb};
    const uniform ${'long long' if m*ldc >= 2**31 else 'int'} ldc = ${ldc};
% endif

    foreach (i = 0 ... n)
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
