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
% for j, jx in enumerate(A):
  % if beta == 0:
        c[i + ${j}*ldc] = ${dot(lambda kx: f'b[i + {kx}*ldb]', jx)};
  % elif beta == 1:
        c[i + ${j}*ldc] += ${dot(lambda kx: f'b[i + {kx}*ldb]', jx)};
  % else:
        c[i + ${j}*ldc] = ${dot(lambda kx: f'b[i + {kx}*ldb]', jx)}
                        + ${beta}*c[i + ${j}*ldc];
  % endif
% endfor
    });
}
