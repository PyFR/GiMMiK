<%inherit file='base'/>

<% ksplit = 2 if m < 36 else 1 %>

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
    ${dtype} dotp;

    if (i < n)
    {
% for j, jx in enumerate(A):
  % if (dotex := dot(lambda kx: f'b[i + {kx}*ldb]', jx, maxsplit=ksplit)) != '0.0':
        dotp = ${dotex};
  % else:
        dotp = make_zero();
  % endif
  % if beta == 0:
        c[i + ${j}*ldc] = dotp;
  % elif beta == 1 and dotex != '0.0':
        c[i + ${j}*ldc] += dotp;
  % else:
        c[i + ${j}*ldc] = dotp + ${beta}*c[i + ${j}*ldc];
  % endif
% endfor
    }
}
