# -*- coding: utf-8 -*-

export void
${funcn}(uniform int n,
         const uniform ${dtype} b[], uniform int ldb,
         ${dtype} uniform c[], uniform int ldc)
{
    ${dtype} dotp;

    foreach (i = 0 ... n)
    {
    % for j, jx in enumerate(mat):
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
