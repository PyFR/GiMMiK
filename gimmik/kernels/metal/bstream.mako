<%inherit file='base'/>

kernel void
% if n is None:
${kname}(constant int& n_,
         device ${dtype}* b, constant int& ldb_,
         device ${dtype}* c, constant int& ldc_,
         uint i [[thread_position_in_grid]])
{
    const int n = ((n_ + ${width} - 1) / ${width}) * ${width};
    const int ldb = ldb_ / ${width};
    const int ldc = ldc_ / ${width};
% else:
${kname}(device const ${dtype}* b, device ${dtype}* c,
         uint i [[thread_position_in_grid]])
{
    const int n = ${-(-n // width)};
    const ${'long' if k*ldb >= width*2**31 else 'int'} ldb = ${ldb // width};
    const ${'long' if m*ldc >= width*2**31 else 'int'} ldc = ${ldc // width};
% endif

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
