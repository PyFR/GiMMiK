<%inherit file='base'/>

<% ksplit = 2 if m < 36 else 1 %>

__global__ __launch_bounds__(${blockx}) void
% if n is None:
${kname}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
  % if width > 1:
    n = (n + ${width} - 1) / ${width};
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
  <%
  nzixs = [kx for kx, val in enumerate(jx) if val != 0]
  if nzixs:
      first_kx = nzixs[0]
      dotex = f"{jx[first_kx]}*b[i + {first_kx}*ldb]"
      for kx in nzixs[1:]:
          dotex = f"{dotex} + {jx[kx]}*b[i + {kx}*ldb]"
  else:
      dotex = 'make_zero()'
  %>
        dotp = ${dotex};
  % if not nzixs:
        dotp = make_zero();
  % endif
  % if beta == 0:
        store_c(&c[i + ${j}*ldc], dotp);
  % elif beta == 1 and nzixs:
        store_c(&c[i + ${j}*ldc], load_c(&c[i + ${j}*ldc]) + dotp);
  % else:
        store_c(&c[i + ${j}*ldc], dotp + ${beta}*load_c(&c[i + ${j}*ldc]));
  % endif
% endfor
    }
}
