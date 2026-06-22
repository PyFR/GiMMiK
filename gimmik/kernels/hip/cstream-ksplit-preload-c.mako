<%inherit file='base'/>

<%
kparts = partition(A, ksplit, by='cols')
cchunks = chunk(range(m), csz)
loaded = set()
%>

__global__ __launch_bounds__(${blockx*ksplit}) void
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
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    ${dtype} cv[${-(-csz // ksplit)}], bv[${-(-k // ksplit)}], dotp;
    __shared__ ${dtype} csub[${ksplit - 1}][${csz}][${blockx}];

## Iterate over the row-partitions of C
% for cchunk in cchunks:
  ## Iterate over the row-partitions of B
  % for bid, kbx in enumerate(kparts):
    if (i < n && threadIdx.y == ${bid})
    {
    ## Evaluate our partial dot products
    % for j in cchunk:
      ## Load in any missing parts of B
      % for kx in kbx:
        % if A[j, kx] != 0 and kx not in loaded:
        bv[${loop.index}] = b[i + ${kx}*ldb]; <% loaded.add(kx) %>
        % endif
      % endfor
      <%
      dotex = dot(lambda kx: f'bv[{kx}]', A[j, kbx])
      has_dotp = A[j].any()
      %>
      % if dotex != '0.0':
        dotp = ${dotex};
      % else:
        dotp = make_zero();
      % endif
      ## Save to a register
      % if loop.index % ksplit == bid:
        % if beta == 0:
        cv[${loop.index // ksplit}] = dotp;
        % elif beta == 1 and has_dotp:
        cv[${loop.index // ksplit}] = nt_load_c(&c[i + ${j}*ldc]);
        cv[${loop.index // ksplit}] += dotp;
        % elif has_dotp:
        cv[${loop.index // ksplit}] = ${beta}*nt_load_c(&c[i + ${j}*ldc]);
        cv[${loop.index // ksplit}] += dotp;
        % endif
      ## Save to shared memory
      % else:
        csub[${bid - (bid > loop.index % ksplit)}][${loop.index}][threadIdx.x] = dotp;
      % endif
    % endfor
    }
  % endfor
    __syncthreads();
  ## Iterate over the column-partitions of B
  % for bid, kbx in enumerate(kparts):
    if (i < n && threadIdx.y == ${bid})
    {
    ## Sum and output the final set of dot products
    % for j in cchunk:
      % if loop.index % ksplit == bid:
        <% has_dotp = A[j].any() %>
        % if beta == 0:
        dotp = cv[${loop.index // ksplit}] + ${' + '.join(f'csub[{i}][{loop.index}][threadIdx.x]'
                                                          for i in range(ksplit - 1))};
        nt_store_c(&c[i + ${j}*ldc], dotp);
        % elif beta == 1 and has_dotp:
        dotp = cv[${loop.index // ksplit}] + ${' + '.join(f'csub[{i}][{loop.index}][threadIdx.x]'
                                                          for i in range(ksplit - 1))};
        nt_store_c(&c[i + ${j}*ldc], dotp);
        % elif beta != 1 and has_dotp:
        dotp = cv[${loop.index // ksplit}] + ${' + '.join(f'csub[{i}][{loop.index}][threadIdx.x]'
                                                          for i in range(ksplit - 1))};
        nt_store_c(&c[i + ${j}*ldc], dotp);
        % elif beta != 1:
        nt_store_c(&c[i + ${j}*ldc], ${beta}*nt_load_c(&c[i + ${j}*ldc]));
        % endif
      % endif
    % endfor
    }
  % endfor
    __syncthreads();
% endfor
}
