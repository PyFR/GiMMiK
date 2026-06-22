<%inherit file='base'/>

<%include file='vector'/>

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
    ${dtype} bv, dotp;

    if (i < n)
    {
% for j, row in enumerate(A):
  <%
  nzixs = [kx for kx, val in enumerate(row) if val != 0]
  %>
  % if nzixs:
    % if beta == 0:
      <% first_kx = nzixs[0] %>
        bv = b[i + ${first_kx}*ldb];
        dotp = gimmik_vmul(${row[first_kx]}, bv);
      % for kx in nzixs[1:]:
        bv = b[i + ${kx}*ldb];
        dotp = gimmik_vmadd(dotp, ${row[kx]}, bv);
      % endfor
        nt_store_c(&c[i + ${j}*ldc], dotp);
    % elif beta == 1:
        dotp = nt_load_c(&c[i + ${j}*ldc]);
      % for kx in nzixs:
        bv = b[i + ${kx}*ldb];
        dotp = gimmik_vmadd(dotp, ${row[kx]}, bv);
      % endfor
        nt_store_c(&c[i + ${j}*ldc], dotp);
    % else:
        dotp = gimmik_vmul(${beta}, nt_load_c(&c[i + ${j}*ldc]));
      % for kx in nzixs:
        bv = b[i + ${kx}*ldb];
        dotp = gimmik_vmadd(dotp, ${row[kx]}, bv);
      % endfor
        nt_store_c(&c[i + ${j}*ldc], dotp);
    % endif
  % else:
    % if beta == 0:
        nt_store_c(&c[i + ${j}*ldc], make_zero());
    % elif beta != 1:
        nt_store_c(&c[i + ${j}*ldc], gimmik_vmul(${beta}, nt_load_c(&c[i + ${j}*ldc])));
    % endif
  % endif
% endfor
    }
}
