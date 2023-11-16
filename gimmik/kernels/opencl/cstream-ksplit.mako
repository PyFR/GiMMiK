<%
kparts = partition(A, ksplit, by='cols')
cchunks = chunk(range(m), csz)
loaded = set()
%>

__kernel __attribute__((reqd_work_group_size(${blockx}, ${ksplit}, 1))) void
% if n is None:
${kname}(int n,
         __global const ${dtype}* restrict b, int ldb,
         __global ${dtype}* restrict c, int ldc)
{
  % if width > 1:
    n = ((n + ${width} - 1) / ${width}) * ${width};
    ldb /= ${width};
    ldc /= ${width};
  % endif
% else:
${kname}(__global const ${dtype}* restrict b, __global ${dtype}* restrict c)
{
    const int n = ${-(-n // width)};
    const ${'long' if k*ldb >= width*2**31 else 'int'} ldb = ${ldb // width};
    const ${'long' if m*ldc >= width*2**31 else 'int'} ldc = ${ldc // width};
% endif
    int i = get_global_id(0);
    int lx = get_local_id(0), ly = get_local_id(1);

    ${dtype} cv[${-(-csz // ksplit)}], bv[${-(-k // ksplit)}], dotp;
    __local ${dtype} csub[${ksplit - 1}][${csz}][${blockx}];

## Iterate over the row-partitions of C
% for cchunk in cchunks:
  ## Iterate over the row-partitions of B
  % for bid, kbx in enumerate(kparts):
    if (i < n && ly == ${bid})
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
        dotp = 0;
      % endif
      ## Save to a register
      % if loop.index % ksplit == bid:
        cv[${loop.index // ksplit}] = dotp;
      ## Save to shared memory
      % else:
        csub[${bid - (bid > loop.index % ksplit)}][${loop.index}][lx] = dotp;
      % endif
    % endfor
    }
  % endfor
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
  ## Iterate over the column-partitions of B
  % for bid, kbx in enumerate(kparts):
    if (i < n && ly == ${bid})
    {
    ## Sum and output the final set of dot products
    % for j in cchunk:
      % if loop.index % ksplit == bid:
        dotp = cv[${loop.index // ksplit}] + ${' + '.join(f'csub[{i}][{loop.index}][lx]'
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
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
% endfor
}
