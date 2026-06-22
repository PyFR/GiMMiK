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

    if (i < n)
    {
        ${dtype} bv, csub[${m}];

## Preload C values for rows which will receive a non-zero dot product
% for j, jx in enumerate(afix):
  % if jx != -1:
    % if beta == 0:
        csub[${j}] = make_zero();
    % elif beta == 1:
        csub[${j}] = nt_load_c(&c[i + ${j}*ldc]);
    % else:
        csub[${j}] = gimmik_vmul(${beta}, nt_load_c(&c[i + ${j}*ldc]));
    % endif
  % endif
% endfor

## Iterate through the used rows of B
% for kx in bix:
        bv = b[i + ${kx}*ldb];
  % for j, jx in enumerate(A[:, kx]):
    % if jx != 0:
        csub[${j}] = gimmik_vmadd(csub[${j}], ${jx}, bv);
    % endif
    ##
    % if kx == alix[j]:
        nt_store_c(&c[i + ${j}*ldc], csub[${j}]);
    % endif
  % endfor
% endfor

## Handle rows of A which are all zero
% for j, jx in enumerate(afix):
  % if jx == -1 and beta == 0:
        nt_store_c(&c[i + ${j}*ldc], make_zero());
  % elif jx == -1 and beta != 1:
        nt_store_c(&c[i + ${j}*ldc], gimmik_vmul(${beta}, nt_load_c(&c[i + ${j}*ldc])));
  % endif
% endfor
    }
}
