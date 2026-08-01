<%
kparts = partition(A, ksplit, by='cols')
cchunks = chunk(range(m), csz)
loaded = set()
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
    sycl::range<2> global(${ksplit}, gx);
    sycl::range<2> local(${ksplit}, ${blockx});

    return q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<${dtype}, 1> csub(
            sycl::range<1>(${(ksplit - 1) * csz * blockx}), cgh);

        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> it) {
        const int i = it.get_global_id(1);
        const int lx = it.get_local_id(1), ly = it.get_local_id(0);

        ${dtype} cv[${-(-csz // ksplit)}], bv[${-(-k // ksplit)}], dotp;

## Iterate over the row-partitions of C
% for cchunk in cchunks:
  ## Iterate over the row-partitions of B
  % for bid, kbx in enumerate(kparts):
        if (i < nw && ly == ${bid})
        {
    ## Evaluate our partial dot products
    % for j in cchunk:
      ## Load in any missing parts of B
      % for kx in kbx:
        % if A[j, kx] != 0 and kx not in loaded:
            bv[${loop.index}] = b[i + ${kx}*ldb]; <% loaded.add(kx) %>
        % endif
      % endfor
      % if (dotex := dot(lambda kx: f'bv[{kx}]', A[j, kbx])) != '0.0':
            dotp = ${dotex};
      % else:
            dotp = ${dtype}(0);
      % endif
      ## Save to a register
      % if loop.index % ksplit == bid:
            cv[${loop.index // ksplit}] = dotp;
      ## Save to shared memory
      % else:
            csub[${(bid - (bid > loop.index % ksplit)) * csz * blockx} + ${loop.index * blockx} + lx] = dotp;
      % endif
    % endfor
        }
  % endfor
        it.barrier(sycl::access::fence_space::local_space);
  ## Iterate over the column-partitions of B
  % for bid, kbx in enumerate(kparts):
        if (i < nw && ly == ${bid})
        {
    ## Sum and output the final set of dot products
    % for j in cchunk:
      % if loop.index % ksplit == bid:
            dotp = cv[${loop.index // ksplit}] + ${' + '.join(f'csub[{i * csz * blockx + loop.index * blockx} + lx]'
                                                              for i in range(ksplit - 1))};
        % if beta == 0:
            c[i + ${j}*ldc] = dotp;
        % elif beta == 1:
            c[i + ${j}*ldc] += dotp;
        % else:
            c[i + ${j}*ldc] = dotp + ${beta}*c[i + ${j}*ldc];
        % endif
      % endif
    % endfor
        }
  % endfor
        it.barrier(sycl::access::fence_space::local_space);
% endfor
        });
    });
}
