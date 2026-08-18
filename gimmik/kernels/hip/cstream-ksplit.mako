<%inherit file='base'/>

<%
kparts = partition(A, ksplit, by='cols')
cchunks = chunk(range(m), csz)
loaded = set()
preload = context.get('preload', False)
%>

${parent.prologue(blockx*ksplit)}\
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
      nzixs = A[j, kbx].nonzero()[0]
      terms = (f"{A[j, kbx[i]]}*bv[{i}]" for i in nzixs)
      dotex = ' + '.join(terms) or 'make_zero()'
      has_dotp = A[j].any()
      %>
        dotp = ${dotex};
      ## Save to a register
      % if loop.index % ksplit == bid:
        % if preload and has_dotp and beta == 1:
        cv[${loop.index // ksplit}] = nt_load(&c[i + ${j}*ldc]) + dotp;
        % elif preload and has_dotp and beta != 0:
        cv[${loop.index // ksplit}] = ${beta}*nt_load(&c[i + ${j}*ldc]) + dotp;
        % else:
        cv[${loop.index // ksplit}] = dotp;
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
        <%
        sum_expr = f"cv[{loop.index // ksplit}]"
        for s_idx in range(ksplit - 1):
            sum_expr = f"{sum_expr} + csub[{s_idx}][{loop.index}][threadIdx.x]"
        %>
        % if preload and beta == 0:
        dotp = ${sum_expr};
        nt_store(&c[i + ${j}*ldc], dotp);
        % elif preload and beta == 1 and has_dotp:
        dotp = ${sum_expr};
        nt_store(&c[i + ${j}*ldc], dotp);
        % elif preload and beta != 1 and has_dotp:
        dotp = ${sum_expr};
        nt_store(&c[i + ${j}*ldc], dotp);
        % elif preload and beta != 1:
        nt_store(&c[i + ${j}*ldc], ${beta}*nt_load(&c[i + ${j}*ldc]));
        % elif beta == 0:
        dotp = ${sum_expr};
        nt_store(&c[i + ${j}*ldc], dotp);
        % elif beta == 1:
        dotp = ${sum_expr};
        nt_store(&c[i + ${j}*ldc], nt_load(&c[i + ${j}*ldc]) + dotp);
        % else:
        dotp = ${sum_expr};
        nt_store(&c[i + ${j}*ldc], dotp + ${beta}*nt_load(&c[i + ${j}*ldc]));
        % endif
      % endif
    % endfor
    }
  % endfor
    __syncthreads();
% endfor
}
