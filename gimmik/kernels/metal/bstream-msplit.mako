<%inherit file='base'/>

<%
mx = partition(A, into=msplit, by='rows')
bchunks = chunk(bix, bsz)
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

    ${dtype} bv, csub[${-(-m // msplit)}];
    threadgroup ${dtype} bsub[2][${bsz}][${blockx}];

## Fill the initial shared memory block
% for cid in range(msplit):
    if (i < n && tpitg.y == ${cid})
    {
  % for kx in bchunks[0]:
    % if loop.index % msplit == cid:
        bsub[0][${loop.index}][tpitg.x] = b[i + ${kx}*ldb];
    % endif
  % endfor
    }
% endfor
    threadgroup_barrier(mem_flags::mem_threadgroup);

## Iterate over each row-chunk of B
% for bb in range(len(bchunks)):
  ## Iterate over each row-chunk of C
  % for cid, mcx in enumerate(mx):
    if (i < n && tpitg.y == ${cid})
    {
    ## Start filling the next shared memory block
    % if not loop.parent.last:
      % for kx in bchunks[bb + 1]:
        % if loop.index % msplit == cid:
        bsub[${(bb + 1) % 2}][${loop.index}][tpitg.x] = b[i + ${kx}*ldb];
        % endif
      % endfor
    % endif
    ## Accumulate our dot products
    % for kx in bchunks[bb]:
        bv = bsub[${bb % 2}][${loop.index}][tpitg.x];
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
        c[i + ${j}*ldc] = make_zero();
        % elif jx == -1 and j % msplit == cid and beta != 1:
        c[i + ${j}*ldc] *= ${beta};
        % endif
      % endfor
    % endif
    }
  % endfor
    threadgroup_barrier(mem_flags::mem_threadgroup);
% endfor
}
