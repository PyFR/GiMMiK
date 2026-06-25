<%inherit file='base'/>
##
## Dense double-precision GEMM on the CDNA Matrix Cores (MFMA).
##
## Mirrors the NVIDIA PTX dense path: the constant operand A is densified,
## padded and baked into the kernel in Matrix-Core fragment order, B is
## streamed, and the epilogue is fully unrolled.  Two knobs over v1:
##   * zero-tile skipping -- amask[mt][kt] marks 16x4 A-tiles with a non-zero;
##     all-zero tiles skip their MMA (and, on the direct path, the B load).
##   * m-splitting -- msplit wavefronts per block (in block.y) each own a
##     slice of the m-tiles, lowering per-wavefront accumulator pressure.
##     For msplit>1 the B tile is staged once in LDS and shared by the whole
##     block, so B is not re-read per wavefront.
##
## Operand lane layout for v_mfma_f64_16x16x4_f64 (wave64), g=lane/16, p=lane%16:
##   A  (16x4 ): A[i][kk]  i=p,        kk=g          (1 reg/lane)
##   B  (4x16 ): B[kk][j]  kk=g,       j=p           (1 reg/lane)
##   C/D(16x16): D[i][j]   j=p,        i=4*reg + g   (v4f64, 4 reg/lane)
## Bake: Ag[(mt*k_tiles+kt)*64 + lane] = A_pad[mt*16 + lane%16][kt*4 + lane//16]
##
<%
    tiles  = blockx // 16                      # 16-wide N-tiles per wavefront
    k_pad  = k_tiles * 4
    active_kt = [kt for kt in range(k_tiles)
                 if any(amask[mt][kt] for mt in range(m_tiles))]
    mtpg = -(-m_tiles // msplit)               # m-tiles per wavefront
    warp_mts = [[mt for mt in range(w*mtpg, min((w+1)*mtpg, m_tiles))]
                for w in range(msplit)]
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
    ## ---- m-split path: stage B in LDS once, share across msplit wavefronts ----
    ## Only the k-rows A actually uses (bix_rows) are read from global; holes
    ## and the padded tail are zeroed (A is 0 there, so an MMA against 0 needs a
    ## finite -- not NaN -- operand).  The global read is vectorized as f64x2
    ## when the layout is 2-aligned.
    ## LDS stores ONLY the active k-tiles (inactive 4-wide k-slabs are dropped):
    ## active kt at position a occupies LDS rows [a*4, a*4+4).  Only bix rows are
    ## read from global (into their tile slot); hole/pad rows are zeroed so the
    ## MMA never multiplies A=0 by an uninitialised (possibly NaN) operand.
<%
    compact_pos = {kt: a for a, kt in enumerate(active_kt)}
    n_akt = len(active_kt)
    # for each used k-row: (global row, compact LDS row)
    bload = [(kr, compact_pos[kr // 4]*4 + kr % 4)
             for kr in bix_rows if (kr // 4) in compact_pos]
    nb = len(bload)
    need_zero = nb < n_akt*4          # any hole/pad row inside an active tile
    nthreads = blockx * msplit
    half = blockx // 2
%>
    __shared__ __align__(16) ${dtype} ${kname}_Bs[${n_akt * 4 * blockx}];
    const int tid = threadIdx.y*${blockx} + threadIdx.x;
% if need_zero:
    for (int idx = tid; idx < ${n_akt * 4 * blockx}; idx += ${nthreads})
        ${kname}_Bs[idx] = (${dtype})0;
    __syncthreads();
% endif
    static const int ${kname}_bg[${nb}] = { ${', '.join(str(g) for g, _ in bload)} };
    static const int ${kname}_bl[${nb}] = { ${', '.join(str(l) for _, l in bload)} };
% if vec2:
    for (int idx = tid; idx < ${nb * half}; idx += ${nthreads})
    {
        const int r = idx / ${half};
        const int cc = (idx % ${half}) * 2;
        const int col = col_base + cc;
        if (col + 1 < n)
            *(gimmik_f64x2*)&${kname}_Bs[${kname}_bl[r]*${blockx} + cc] =
                *(const gimmik_f64x2*)&b[${kname}_bg[r]*ldb + col];
        else if (col < n)
            ${kname}_Bs[${kname}_bl[r]*${blockx} + cc] = b[${kname}_bg[r]*ldb + col];
    }
% else:
    for (int idx = tid; idx < ${nb * blockx}; idx += ${nthreads})
    {
        const int r = idx / ${blockx};
        const int cc = idx % ${blockx};
        const int col = col_base + cc;
        if (col < n)
            ${kname}_Bs[${kname}_bl[r]*${blockx} + cc] = b[${kname}_bg[r]*ldb + col];
    }
% endif
    __syncthreads();

% for w in range(msplit):
<% mts = warp_mts[w] %>
%  if mts:
    if (threadIdx.y == ${w})
    {
        ${dtype} a, ${', '.join('bv_%d' % t for t in range(tiles))};
%   for j in range(len(mts)):
%    for t in range(tiles):
        gimmik_f64x4 acc_${j}_${t} = {0.0, 0.0, 0.0, 0.0};
%    endfor
%   endfor
%   for kt in active_kt:
%    for t in range(tiles):
        bv_${t} = ${kname}_Bs[(${compact_pos[kt]*4} + g)*${blockx} + ${t*16} + p];
%    endfor
%    for j, mt in enumerate(mts):
%     if amask[mt][kt]:
        a = ${kname}_Ag[${(mt*k_tiles + kt)*64} + lane];
%      for t in range(tiles):
        acc_${j}_${t} = __builtin_amdgcn_mfma_f64_16x16x4f64(a, bv_${t}, acc_${j}_${t}, 0, 0, 0);
%      endfor
%     endif
%    endfor
%   endfor
%   for j, mt in enumerate(mts):
%    for t in range(tiles):
%     for reg in range(4):
        {
            const int row = ${mt*16 + 4*reg} + g;
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
%     endfor
%    endfor
%   endfor
    }
%  endif
% endfor
% endif
}
