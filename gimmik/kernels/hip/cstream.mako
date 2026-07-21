<%inherit file='base'/>

<%
preload = context.get('preload', False)
%>

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
  terms = (f"{jx[kx]}*b[i + {kx}*ldb]" for kx in nzixs)
  dotex = ' + '.join(terms) or 'make_zero()'
  %>
        dotp = ${dotex};
  % if preload and nzixs:
    % if beta == 0:
        nt_store(&c[i + ${j}*ldc], dotp);
    % elif beta == 1:
        dotp = nt_load(&c[i + ${j}*ldc]) + dotp;
        nt_store(&c[i + ${j}*ldc], dotp);
    % else:
        dotp = ${beta}*nt_load(&c[i + ${j}*ldc]) + dotp;
        nt_store(&c[i + ${j}*ldc], dotp);
    % endif
  % elif preload:
    % if beta == 0:
        nt_store(&c[i + ${j}*ldc], make_zero());
    % elif beta != 1:
        nt_store(&c[i + ${j}*ldc], ${beta}*nt_load(&c[i + ${j}*ldc]));
    % endif
  % elif beta == 0:
        nt_store(&c[i + ${j}*ldc], dotp);
  % elif beta == 1 and nzixs:
        nt_store(&c[i + ${j}*ldc], nt_load(&c[i + ${j}*ldc]) + dotp);
  % else:
        nt_store(&c[i + ${j}*ldc], dotp + ${beta}*nt_load(&c[i + ${j}*ldc]));
  % endif
% endfor
    }
}
