# -*- coding: utf-8 -*-

export void
% if n is None:
${funcn}(uniform int n,
         const uniform ${dtype} b[], uniform int ldb,
         ${dtype} uniform c[], uniform int ldc)
{
% else:
${funcn}(const uniform ${dtype} b[], ${dtype} uniform c[])
{
    const uniform int n = ${n};
    const uniform int ldb = ${ldb};
    const uniform int ldc = ${ldc};
% endif
    ${dtype} dotp;

    foreach (i = 0 ... n)
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
