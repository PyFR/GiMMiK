export void
% if n is None:
${kname}(uniform int n,
         const uniform ${dtype} b[], uniform int ldb_,
         ${dtype} uniform c[], uniform int ldc_)
{
    const uniform int64 ldb = ldb_;
    const uniform int64 ldc = ldc_;
% else:
${kname}(const uniform ${dtype} b[], ${dtype} uniform c[])
{
    const uniform int n = ${n};
    const uniform ${'int64' if k*ldb >= 2**31 else 'int'} ldb = ${ldb};
    const uniform ${'int64' if m*ldc >= 2**31 else 'int'} ldc = ${ldc};
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
