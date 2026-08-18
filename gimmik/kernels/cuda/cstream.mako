<%inherit file='base'/>

<% ksplit = 2 if m < 36 else 1 %>

${parent.prologue()}\
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
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
