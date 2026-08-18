<%inherit file='base'/>

${parent.prologue(['uint i [[thread_position_in_grid]]'])}\

    if (i < n)
    {
        ${dtype} bv, csub[${m}];

## Iterare through the used rows of B
% for kx in bix:
        bv = b[i + ${kx}*ldb];
  % for j, jx in enumerate(A[:, kx]):
    % if jx != 0 and kx == afix[j]:
        csub[${j}] = ${jx}*bv;
    % elif jx != 0:
        csub[${j}] += ${jx}*bv;
    % endif
    ##
    % if kx == alix[j] and beta == 0:
        c[i + ${j}*ldc] = csub[${j}];
    % elif kx == alix[j] and beta == 1:
        c[i + ${j}*ldc] += csub[${j}];
    % elif kx == alix[j]:
        c[i + ${j}*ldc] = csub[${j}] + ${beta}*c[i + ${j}*ldc];
    % endif
  % endfor
% endfor

## Handle rows of A which are all zero
% for j, jx in enumerate(afix):
  % if jx == -1 and beta == 0:
        c[i + ${j}*ldc] = make_zero();
  % elif jx == -1 and beta != 1:
        c[i + ${j}*ldc] *= ${beta};
  % endif
% endfor
    }
}
