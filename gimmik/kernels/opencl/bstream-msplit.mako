<%
mx = partition(A, into=msplit, by='rows')
bchunks = chunk(bix, bsz)
%>

__kernel __attribute__((reqd_work_group_size(${blockx}, ${msplit}, 1))) void
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

    ${dtype} bv, csub[${-(-m // msplit)}];
    __local ${dtype} bsub[2][${bsz}][${blockx}];

## Fill the initial shared memory block
% for cid in range(msplit):
    if (i < n && ly == ${cid})
    {
  % for kx in bchunks[0]:
    % if loop.index % msplit == cid:
        bsub[0][${loop.index}][lx] = b[i + ${kx}*ldb];
    % endif
  % endfor
    }
% endfor
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

## Iterate over each row-chunk of B
% for bb in range(len(bchunks)):
  ## Iterate over each row-chunk of C
  % for cid, mcx in enumerate(mx):
    if (i < n && ly == ${cid})
    {
    ## Start filling the next shared memory block
    % if not loop.parent.last:
      % for kx in bchunks[bb + 1]:
        % if loop.index % msplit == cid:
        bsub[${(bb + 1) % 2}][${loop.index}][lx] = b[i + ${kx}*ldb];
        % endif
      % endfor
    % endif
    ## Accumulate our dot products
    % for kx in bchunks[bb]:
        bv = bsub[${bb % 2}][${loop.index}][lx];
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
        c[i + ${j}*ldc] = 0;
        % elif jx == -1 and j % msplit == cid and beta != 1:
        c[i + ${j}*ldc] *= ${beta};
        % endif
      % endfor
    % endif
    }
  % endfor
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
% endfor
}
