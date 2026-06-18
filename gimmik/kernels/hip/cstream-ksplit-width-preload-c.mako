<%inherit file='base'/>

% if width == 2:
static inline __device__ ${dtype}
gimmik_vmul(${dtype[:-1]} a, ${dtype} b)
{
    return make_${dtype}(a*b.x, a*b.y);
}

static inline __device__ ${dtype}
gimmik_vadd(${dtype} a, ${dtype} b)
{
    return make_${dtype}(a.x + b.x, a.y + b.y);
}

static inline __device__ ${dtype}
gimmik_vmadd(${dtype} acc, ${dtype[:-1]} a, ${dtype} b)
{
    // Keep the multiply-add expression visible to the compiler.
    return make_${dtype}(acc.x + a*b.x, acc.y + a*b.y);
}
% elif width == 4:
static inline __device__ ${dtype}
gimmik_vmul(${dtype[:-1]} a, ${dtype} b)
{
    return make_${dtype}(a*b.x, a*b.y, a*b.z, a*b.w);
}

static inline __device__ ${dtype}
gimmik_vadd(${dtype} a, ${dtype} b)
{
    return make_${dtype}(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

static inline __device__ ${dtype}
gimmik_vmadd(${dtype} acc, ${dtype[:-1]} a, ${dtype} b)
{
    // Keep the multiply-add expression visible to the compiler.
    return make_${dtype}(acc.x + a*b.x, acc.y + a*b.y, acc.z + a*b.z, acc.w + a*b.w);
}
% else:
#error "cstream_ksplit_width_preload_c only supports width=2 or width=4"
% endif

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
    n = (n + ${width} - 1) / ${width};
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
  ## Iterate over the column-partitions of B
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

      ## Expand vectorized partial dot product
      <%
      nzixs = []
      for l_idx, kx in enumerate(kbx):
          if A[j, kx] != 0:
              nzixs.append((l_idx, kx))

      has_dotp = any(A[j, kx] != 0 for kx in range(k))
      if not nzixs:
          dotex = 'make_zero()'
      else:
          first_l_idx, first_kx = nzixs[0]
          dotex = f"gimmik_vmul({A[j, first_kx]}, bv[{first_l_idx}])"
          for l_idx, kx in nzixs[1:]:
              dotex = f"gimmik_vmadd({dotex}, {A[j, kx]}, bv[{l_idx}])"
      %>
      dotp = ${dotex};

      ## Save to a register
      % if loop.index % ksplit == bid:
        % if beta == 0:
        cv[${loop.index // ksplit}] = dotp;
        % elif beta == 1 and has_dotp:
        cv[${loop.index // ksplit}] = nt_load_c(&c[i + ${j}*ldc]);
        cv[${loop.index // ksplit}] = gimmik_vadd(cv[${loop.index // ksplit}], dotp);
        % elif has_dotp:
        cv[${loop.index // ksplit}] = gimmik_vmul(${beta}, nt_load_c(&c[i + ${j}*ldc]));
        cv[${loop.index // ksplit}] = gimmik_vadd(cv[${loop.index // ksplit}], dotp);
        % endif
      ## Save to shared memory
      % else:
        csub[${bid - (bid > loop.index % ksplit)}][${loop.index}][threadIdx.x] = dotp;
      % endif
    % endfor
    }
  % endfor
    __syncthreads();
    
  ## Sum and output the final set of dot products
  % for bid, kbx in enumerate(kparts):
    if (i < n && threadIdx.y == ${bid})
    {
    % for j in cchunk:
      % if loop.index % ksplit == bid:
        <%
        has_dotp = any(A[j, kx] != 0 for kx in range(k))
        sum_expr = f"cv[{loop.index // ksplit}]"
        for s_idx in range(ksplit - 1):
            sum_expr = f"gimmik_vadd({sum_expr}, csub[{s_idx}][{loop.index}][threadIdx.x])"
        %>
        % if beta == 0:
        dotp = ${sum_expr};
        nt_store_c(&c[i + ${j}*ldc], dotp);
        % elif beta == 1 and has_dotp:
        dotp = ${sum_expr};
        nt_store_c(&c[i + ${j}*ldc], dotp);
        % elif beta != 1 and has_dotp:
        dotp = ${sum_expr};
        nt_store_c(&c[i + ${j}*ldc], dotp);
        % elif beta != 1:
        nt_store_c(&c[i + ${j}*ldc], gimmik_vmul(${beta}, nt_load_c(&c[i + ${j}*ldc])));
        % endif
      % endif
    % endfor
    }
  % endfor
    __syncthreads();
% endfor
}
