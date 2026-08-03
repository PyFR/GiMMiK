#include <sycl/sycl.hpp>

sycl::event
% if n is None:
${kname}(sycl::queue& q, int n,
         const ${dtype}* __restrict bp, int ldb,
         ${dtype}* __restrict cp, int ldc)
{
    const int nw = (n + ${width} - 1) / ${width};
  % if width > 1:
    ldb /= ${width};
    ldc /= ${width};
  % endif
% else:
${kname}(sycl::queue& q, const ${dtype}* __restrict bp, ${dtype}* __restrict cp)
{
    const int nw = ${-(-n // width)};
    const ${'long' if k*ldb >= width*2**31 else 'int'} ldb = ${ldb // width};
    const ${'long' if m*ldc >= width*2**31 else 'int'} ldc = ${ldc // width};
% endif
    const int gx = ((nw + ${blockx} - 1) / ${blockx}) * ${blockx};
    return q.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(gx), sycl::range<1>(${blockx})),
        [=](sycl::nd_item<1> it) [[sycl::reqd_work_group_size(${blockx})]] {
        const int i = it.get_global_id(0);
        if (i >= nw)
            return;
        // Re-assert non-aliasing: __restrict on the launcher's pointer
        // parameters is lost once they are captured by value into the kernel
        // lambda, so restore it on the pointers actually used in the body.
        const ${dtype}* __restrict b = bp;
        ${dtype}* __restrict c = cp;
        ${dtype} bv, csub[${m}];

## Iterate through the used rows of B
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
        c[i + ${j}*ldc] = ${dtype}(0);
  % elif jx == -1 and beta != 1:
        c[i + ${j}*ldc] *= ${beta};
  % endif
% endfor
    });
}
