<%inherit file='base'/>

<%
kparts = partition(A, ksplit, by='cols')
cchunks = chunk(range(m), csz)
loaded = set()
%>

__global__ void
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
    const int ldb = ${ldb // width};
    const int ldc = ${ldc // width};
% endif
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    ${dtype} cv[${-(-csz // ksplit)}], bv[${-(-k // ksplit)}], dotp;
    __shared__ ${dtype} csub[${ksplit - 1}][${csz}][${blockx}];

    if (i >= n)
        return;

## Iterate over the column-partitions of B
% for bid, kbx in enumerate(kparts):
    if (threadIdx.y == ${bid})
    {
  ## Iterate over the row-partitions of C
  % for cchunk in cchunks:
    ## Evaluate our partial dot products
    % for j in cchunk:
      ## Load in any missing parts of B
      % for kx in kbx:
        % if A[j, kx] != 0 and kx not in loaded:
        bv[${loop.index}] = __ldcg(b + i + ${kx}*ldb); <% loaded.add(kx) %>
        % endif
      % endfor
      % if (dotex := dot(lambda kx: f'bv[{kx}]', A[j, kbx])) != '0.0':
        dotp = ${dotex};
      % else:
        dotp = make_zero();
      % endif
      ## Save to a register
      % if loop.index % ksplit == bid:
        cv[${loop.index // ksplit}] = dotp;
      ## Save to shared memory
      % else:
        csub[${bid - (bid > loop.index % ksplit)}][${loop.index}][threadIdx.x] = dotp;
      % endif
    % endfor
        __barrier_sync(0);
    ## Sum and output the final set of dot products
    % for j in cchunk:
      % if loop.index % ksplit == bid:
        dotp = cv[${loop.index // ksplit}] + ${' + '.join(f'csub[{i}][{loop.index}][threadIdx.x]'
                                                          for i in range(ksplit - 1))};
        % if beta == 0:
        __stcg(c + i + ${j}*ldc, dotp);
        % elif beta == 1:
        c[i + ${j}*ldc] += dotp;
        % else:
        c[i + ${j}*ldc] = dotp + ${beta}*c[i + ${j}*ldc];
        % endif
      % endif
    % endfor
        __barrier_sync(0);
  % endfor
    }
% endfor
}
