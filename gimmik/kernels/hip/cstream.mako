<%inherit file='base'/>

<% ksplit = 2 if m < 36 else 1 %>

__global__ __launch_bounds__(128) void
% if n is None:
${kname}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
  % if width > 1:
    n = ((n + ${width} - 1) / ${width}) * ${width};
    ldb /= ${width};
    ldc /= ${width};
  % endif
% else:
${kname}(const ${dtype}* __restrict__ b, ${dtype}* __restrict__ c)
{
    const int n = ${-(-n // width)};
    const ${'long long' if k*ldb >= width*2**31 else 'int'} ldb = ${ldb // width};
    const ${'long long' if m*ldc >= width*2**31 else 'int'} ldc = ${ldc // width};
% endif
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
