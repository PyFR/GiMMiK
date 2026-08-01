<%
mx = partition(A, into=msplit, by='rows')
bchunks = chunk(bix, bsz)
%>\
#include <sycl/sycl.hpp>

sycl::event
% if n is None:
${kname}(sycl::queue& q, int n,
         const ${dtype}* __restrict b, int ldb,
         ${dtype}* __restrict c, int ldc)
{
    const int nw = (n + ${width} - 1) / ${width};
  % if width > 1:
    ldb /= ${width};
    ldc /= ${width};
  % endif
% else:
${kname}(sycl::queue& q, const ${dtype}* __restrict b, ${dtype}* __restrict c)
{
    const int nw = ${-(-n // width)};
    const ${'long' if k*ldb >= width*2**31 else 'int'} ldb = ${ldb // width};
    const ${'long' if m*ldc >= width*2**31 else 'int'} ldc = ${ldc // width};
% endif
    const int gx = ((nw + ${blockx} - 1) / ${blockx}) * ${blockx};
    sycl::range<2> global(${msplit}, gx);
    sycl::range<2> local(${msplit}, ${blockx});

    return q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<${dtype}, 1> bsub(
            sycl::range<1>(${2 * bsz * blockx}), cgh);

        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> it) {
        const int i = it.get_global_id(1);
        const int lx = it.get_local_id(1), ly = it.get_local_id(0);

        ${dtype} bv, csub[${-(-m // msplit)}];

## Fill the initial shared memory block
% for cid in range(msplit):
        if (i < nw && ly == ${cid})
        {
  % for kx in bchunks[0]:
    % if loop.index % msplit == cid:
            bsub[${loop.index * blockx} + lx] = b[i + ${kx}*ldb];
    % endif
  % endfor
        }
% endfor
        it.barrier(sycl::access::fence_space::local_space);

## Iterate over each row-chunk of B
% for bb in range(len(bchunks)):
  ## Iterate over each row-chunk of C
  % for cid, mcx in enumerate(mx):
        if (i < nw && ly == ${cid})
        {
    ## Start filling the next shared memory block
    % if not loop.parent.last:
      % for kx in bchunks[bb + 1]:
        % if loop.index % msplit == cid:
            bsub[${((bb + 1) % 2) * bsz * blockx + loop.index * blockx} + lx] = b[i + ${kx}*ldb];
        % endif
      % endfor
    % endif
    ## Accumulate our dot products
    % for kx in bchunks[bb]:
            bv = bsub[${(bb % 2) * bsz * blockx + loop.index * blockx} + lx];
      % for j, jx in enumerate(A[mcx, kx]):
        % if jx != 0 and kx == afix[mcx[j]]:
            csub[${j}] = ${jx}*bv;
        % elif jx != 0:
            csub[${j}] += ${jx}*bv;
        % endif
        ## If we're done with this dot product then store to global
        % if kx == alix[mcx[j]] and beta == 0:
            c[i + ${mcx[j]}*ldc] = csub[${j}];
        % elif kx == alix[mcx[j]] and beta == 1:
            c[i + ${mcx[j]}*ldc] += csub[${j}];
        % elif kx == alix[mcx[j]]:
            c[i + ${mcx[j]}*ldc] = csub[${j}] + ${beta}*c[i + ${mcx[j]}*ldc];
        % endif
      % endfor
    % endfor
    ## Handle rows of A which are all zero
    % if loop.parent.last:
      % for j, jx in enumerate(afix):
        % if jx == -1 and j % msplit == cid and beta == 0:
            c[i + ${j}*ldc] = ${dtype}(0);
        % elif jx == -1 and j % msplit == cid and beta != 1:
            c[i + ${j}*ldc] *= ${beta};
        % endif
      % endfor
    % endif
        }
  % endfor
        it.barrier(sycl::access::fence_space::local_space);
% endfor
        });
    });
}
