<%inherit file='base'/>

<%
kparts = partition(A, ksplit, by='cols')
cchunks = chunk(range(m), csz)
loaded = set()
%>

kernel void
% if n is None:
${kname}(constant int& n_,
         device ${dtype}* b, constant int& ldb_,
         device ${dtype}* c, constant int& ldc_,
         uint2 tpig [[thread_position_in_grid]],
         uint2 tpitg [[thread_position_in_threadgroup]])
{
    const int n = ((n_ + ${width} - 1) / ${width}) * ${width};
    const int ldb = ldb_ / ${width};
    const int ldc = ldc_ / ${width};
% else:
${kname}(device const ${dtype}* b, device ${dtype}* c,
         uint2 tpig [[thread_position_in_grid]],
         uint2 tpitg [[thread_position_in_threadgroup]])
{
    const int n = ${-(-n // width)};
    const ${'long' if k*ldb >= width*2**31 else 'int'} ldb = ${ldb // width};
    const ${'long' if m*ldc >= width*2**31 else 'int'} ldc = ${ldc // width};
% endif
    const int i = tpig.x;

    ${dtype} cv[${-(-csz // ksplit)}], bv[${-(-k // ksplit)}], dotp;
    threadgroup ${dtype} csub[${ksplit - 1}][${csz}][${blockx}];

## Iterate over the row-partitions of C
% for cchunk in cchunks:
  ## Iterate over the row-partitions of B
  % for bid, kbx in enumerate(kparts):
    if (i < n && tpitg.y == ${bid})
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
        dotp = make_zero();
      % endif
      ## Save to a register
      % if loop.index % ksplit == bid:
        cv[${loop.index // ksplit}] = dotp;
      ## Save to shared memory
      % else:
        csub[${bid - (bid > loop.index % ksplit)}][${loop.index}][tpitg.x] = dotp;
      % endif
    % endfor
    }
  % endfor
    threadgroup_barrier(mem_flags::mem_threadgroup);
  ## Iterate over the column-partitions of B
  % for bid, kbx in enumerate(kparts):
    if (i < n && tpitg.y == ${bid})
    {
    ## Sum and output the final set of dot products
    % for j in cchunk:
      % if loop.index % ksplit == bid:
        dotp = cv[${loop.index // ksplit}] + ${' + '.join(f'csub[{i}][{loop.index}][tpitg.x]'
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
% endfor
}
