<%inherit file='base'/>
##
## Dense double-precision GEMM on the CDNA Matrix Cores (MFMA).
##
## A is densified, padded and baked into the kernel in Matrix-Core fragment
## order; B is streamed; C is non-temporal stored; the epilogue is fully
## unrolled.  Two code paths:
##   msplit == 1  -- direct: one wavefront, B straight from global, compile-time
##                   zero-tile skipping (skip MMA + B load for all-zero tiles).
##   msplit  > 1  -- m-split + k-blocked: msplit wavefronts (block.y) each own a
##                   slice of the m-tiles; B is staged into LDS in chunks of kc
##                   active k-tiles (bounds LDS for any k) with a double2
##                   cooperative copy, and only the k-rows A uses (bix) are read.
##
## Operand lane layout for v_mfma_f64_16x16x4_f64 (wave64), g=lane/16, p=lane%16:
##   A  (16x4 ): A[i][kk]  i=p,        kk=g
##   B  (4x16 ): B[kk][j]  kk=g,       j=p
##   C/D(16x16): D[i][j]   j=p,        i=4*reg + g   (v4f64)
## Bake: Ag[(mt*k_tiles+kt)*64 + lane] = A_pad[mt*16 + lane%16][kt*4 + lane//16]
##
<%
    tiles = blockx // 16
    active_kt = [kt for kt in range(k_tiles)
                 if any(amask[mt][kt] for mt in range(m_tiles))]
    mtpg = -(-m_tiles // msplit)
%>
typedef ${dtype} gimmik_f64x4 __attribute__((ext_vector_type(4)));
typedef ${dtype} gimmik_f64x2 __attribute__((ext_vector_type(2)));

__device__ static const ${dtype} ${kname}_Ag[${m_tiles * k_tiles * 64}] = {
    ${', '.join(a_hex)}
};

__global__ __launch_bounds__(${blockx * msplit}) void
% if n is None:
${kname}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
% else:
${kname}(const ${dtype}* __restrict__ b, ${dtype}* __restrict__ c)
{
    const int n = ${n};
    const ${'long long' if k * ldb >= 2**31 else 'int'} ldb = ${ldb};
    const ${'long long' if m * ldc >= 2**31 else 'int'} ldc = ${ldc};
% endif
    const int lane = threadIdx.x;
    const int g = lane / 16;
    const int p = lane % 16;
    const int col_base = ${blockx}*blockIdx.x;

% if msplit == 1:
    ## ---- direct path: single wavefront, B straight from global ----
    ${dtype} a;
% for t in range(tiles):
    ${dtype} bv_${t};
% endfor
% for mt in range(m_tiles):
%  for t in range(tiles):
    gimmik_f64x4 acc_${mt}_${t} = {0.0, 0.0, 0.0, 0.0};
%  endfor
% endfor
% for kt in active_kt:
<% krow_guard = (kt + 1)*4 > k %>
%  for t in range(tiles):
    {
        const int col = col_base + ${t*16} + p;
        const int krow = ${kt*4} + g;
        bv_${t} = (col < n${' && krow < %d' % k if krow_guard else ''}) ? b[krow*ldb + col] : (${dtype})0;
    }
%  endfor
%  for mt in range(m_tiles):
%   if amask[mt][kt]:
    a = ${kname}_Ag[${(mt*k_tiles + kt)*64} + lane];
%    for t in range(tiles):
    acc_${mt}_${t} = __builtin_amdgcn_mfma_f64_16x16x4f64(a, bv_${t}, acc_${mt}_${t}, 0, 0, 0);
%    endfor
%   endif
%  endfor
% endfor
% for mt in range(m_tiles):
%  for t in range(tiles):
%   for reg in range(4):
    {
        const int row = ${mt*16 + 4*reg} + g;
        const int col = col_base + ${t*16} + p;
        if (row < ${m} && col < n)
% if beta == 0:
            store_c(&c[row*ldc + col], acc_${mt}_${t}[${reg}]);
% elif beta == 1:
            store_c(&c[row*ldc + col], gimmik_vadd(load_c(&c[row*ldc + col]), acc_${mt}_${t}[${reg}]));
% else:
            store_c(&c[row*ldc + col], gimmik_vadd(gimmik_vmul(${beta}, load_c(&c[row*ldc + col])), acc_${mt}_${t}[${reg}]));
% endif
    }
%   endfor
%  endfor
% endfor

% else:
    ## ---- m-split + k-blocked path ----
<%
    chunks = [active_kt[c:c+kc] for c in range(0, len(active_kt), kc)]
    nthreads = blockx * msplit
    half = blockx // 2
    mt_guard = (m_tiles % msplit != 0)
%>
    __shared__ __align__(16) ${dtype} ${kname}_Bs[${kc * 4 * blockx}];
    const int tid = threadIdx.y*${blockx} + threadIdx.x;
    const int wmt = threadIdx.y*${mtpg};
    ${dtype} a, ${', '.join('bv_%d' % t for t in range(tiles))};
% for j in range(mtpg):
%  for t in range(tiles):
    gimmik_f64x4 acc_${j}_${t} = {0.0, 0.0, 0.0, 0.0};
%  endfor
% endfor

% for ci, chunk in enumerate(chunks):
<%
    cpos = {kt: a for a, kt in enumerate(chunk)}
    bload = [(kr, cpos[kr // 4]*4 + kr % 4)
             for kr in bix_rows if (kr // 4) in cpos]
    nb = len(bload)
    need_zero = nb < len(chunk)*4
%>
%  if ci > 0:
    __syncthreads();
%  endif
%  if need_zero:
    for (int idx = tid; idx < ${len(chunk)*4*blockx}; idx += ${nthreads})
        ${kname}_Bs[idx] = (${dtype})0;
    __syncthreads();
%  endif
    {
        static const int bg[${nb}] = { ${', '.join(str(x) for x, _ in bload)} };
        static const int bl[${nb}] = { ${', '.join(str(x) for _, x in bload)} };
%  if vec2:
        for (int idx = tid; idx < ${nb * half}; idx += ${nthreads})
        {
            const int r = idx / ${half};
            const int cc = (idx % ${half}) * 2;
            const int col = col_base + cc;
            if (col + 1 < n)
                *(gimmik_f64x2*)&${kname}_Bs[bl[r]*${blockx} + cc] =
                    *(const gimmik_f64x2*)&b[bg[r]*ldb + col];
            else if (col < n)
                ${kname}_Bs[bl[r]*${blockx} + cc] = b[bg[r]*ldb + col];
        }
%  else:
        for (int idx = tid; idx < ${nb * blockx}; idx += ${nthreads})
        {
            const int r = idx / ${blockx};
            const int cc = idx % ${blockx};
            const int col = col_base + cc;
            if (col < n)
                ${kname}_Bs[bl[r]*${blockx} + cc] = b[bg[r]*ldb + col];
        }
%  endif
    }
    __syncthreads();
%  for a_pos, kt in enumerate(chunk):
%   for t in range(tiles):
    bv_${t} = ${kname}_Bs[(${a_pos*4} + g)*${blockx} + ${t*16} + p];
%   endfor
%   for j in range(mtpg):
%    if mt_guard:
    if (wmt + ${j} < ${m_tiles})
%    endif
    {
        a = ${kname}_Ag[((wmt + ${j})*${k_tiles} + ${kt})*64 + lane];
%    for t in range(tiles):
        acc_${j}_${t} = __builtin_amdgcn_mfma_f64_16x16x4f64(a, bv_${t}, acc_${j}_${t}, 0, 0, 0);
%    endfor
    }
%   endfor
%  endfor
% endfor

% for j in range(mtpg):
%  for t in range(tiles):
%   for reg in range(4):
%    if mt_guard:
    if (wmt + ${j} < ${m_tiles})
%    endif
    {
        const int row = (wmt + ${j})*16 + ${4*reg} + g;
        const int col = col_base + ${t*16} + p;
        if (row < ${m} && col < n)
% if beta == 0:
            store_c(&c[row*ldc + col], acc_${j}_${t}[${reg}]);
% elif beta == 1:
            store_c(&c[row*ldc + col], gimmik_vadd(load_c(&c[row*ldc + col]), acc_${j}_${t}[${reg}]));
% else:
            store_c(&c[row*ldc + col], gimmik_vadd(gimmik_vmul(${beta}, load_c(&c[row*ldc + col])), acc_${j}_${t}[${reg}]));
% endif
    }
%   endfor
%  endfor
% endfor
% endif
}
