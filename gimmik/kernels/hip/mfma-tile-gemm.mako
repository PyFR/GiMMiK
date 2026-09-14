<%inherit file='base'/>
##
## Dense tiled GEMM optimized for AMD wave-level MFMA:
##   1. Uses v_mfma_f64_16x16x4_f64 or v_mfma_f32_16x16x4_f32 for 16x16 tiles.
##   2. Takes A in MFMA lane-major order so each lane can vector-load the two
##      A operands used by a pair of consecutive MFMA K groups.
##   3. Loads A directly from global memory into VGPRs; A does not use LDS.
##   4. Loads B with normal cached global loads, stages B through LDS, and
##      reads LDS in the MFMA operand layout.
##   5. Uses PGR2/double buffering for both operands: A is double-buffered in
##      VGPR pairs, while B is double-buffered as global->VGPR->LDS.
##   6. Remaps block ids to a column-major/B-reuse tile order so neighboring
##      workgroups compute different M tiles for the same N tile, improving B
##      cache reuse.
##
## The two instructions hand back their accumulators differently: the f64 one
## gives lane 0 rows 0, 4, 8, 12 while the f32 one gives it rows 0, 1, 2, 3.
##
<%
    mfma_k = 4
    f64 = sdtype == 'double'
    apair_align = 16 if f64 else 8

    acc4_t, apair_t = f'{kname}_acc4', f'{kname}_apair'
    aligned_a = f'__builtin_assume_aligned(a, {apair_align})'

    nthreads = blockx*blocky

    if f64:
        mfma = '__builtin_amdgcn_mfma_f64_16x16x4f64'
    else:
        mfma = '__builtin_amdgcn_mfma_f32_16x16x4f32'

    def c_row_offset(reg):
        return f'{4*reg} + g' if f64 else f'4*g + {reg}'

    valid_m_tiles, n_tiles = min(-(-m // 16), MT // 16), NT // 16
    k_group_pairs, nwaves = KT // (2*mfma_k), nthreads // 64
    mtpg = -(-valid_m_tiles // nwaves)

    # Whether K spans more than one tile, so the buffers alternate
    multi_ktile = KT < k_pad

    b_tile_elems = KT*NT
    b_tile_full, b_tile_tail = divmod(b_tile_elems, nthreads)
    b_tile_iters = b_tile_full + bool(b_tile_tail)

    def m_tile_always_held(j):
        return (nwaves - 1)*mtpg + j < valid_m_tiles

    def acc(w, j, t):
        return f'acc_{j}_{t}' if width == 1 else f'acc_{w}_{j}_{t}'

    def bval(kg, t, w):
        return f'bv_{kg}_{t}' if width == 1 else f'bv_{kg}_{t}.{'xyzw'[w]}'

    def cval(j, t, reg):
        if width == 1:
            return f'{acc(0, j, t)}[{reg}]'
        else:
            parts = ', '.join(f'{acc(w, j, t)}[{reg}]' for w in range(width))
            return f'make_{dtype}({parts})'

    c_epilogue_indices = [
        (j, t, reg)
        for j in range(mtpg)
        for t in range(n_tiles)
        for reg in range(4)
    ]
%>
typedef ${sdtype} ${acc4_t} __attribute__((ext_vector_type(4)));
typedef ${sdtype} ${apair_t} __attribute__((ext_vector_type(2)));

## ---------------------------------------------------------------------------
## Tile-level global prefetch and LDS staging helpers
## ---------------------------------------------------------------------------

<%def name="a_pair_expr(j, kgp, kbase)">\
ap[((row_base / 16 + wmt + ${j})*${k_pad // 8} + ${kbase} + ${kgp})*64 + lane]\
</%def>

<%def name="a_operand_prefetch_tile(row_offset_expr, slot)">\
<%
    kbase = f'(({row_offset_expr}) / {KT})*{k_group_pairs}'
%>\
% for j in range(mtpg):
%  for kgp in range(k_group_pairs):
<%
    dst = f'a_pair_{slot}_{j}_{kgp}'
%>\
%   if m_tile_always_held(j):
        ${dst} = ${a_pair_expr(j, kgp, kbase)};
%   else:
        if (wmt + ${j} < ${valid_m_tiles})
            ${dst} = ${a_pair_expr(j, kgp, kbase)};
        else
            ${dst} = {(${sdtype})0.0, (${sdtype})0.0};
%   endif
%  endfor
% endfor
</%def>

<%def name="b_prefetch_tile_frag(row_offset_expr, slot, pp)">\
<%
    dst = f'b_next_{slot}_{pp}'
%>\
        {
            const int idx = tid + ${pp*nthreads};
% if pp == b_tile_full:
            if (idx < ${b_tile_elems})
% endif
            {
                const int kk = idx / ${NT};
                const int cc = idx % ${NT};
                const int krow = ${row_offset_expr} + kk;
                const int col = col_base + cc;
                const bool have = fast_b_tile || (krow < ${k} && col < n);

                ${dst} = have ? b[krow*ldb + col] : make_zero();
            }
        }
</%def>

<%def name="b_prefetch_tile(row_offset_expr, slot)">\
        {
        const bool krows_ok = ${row_offset_expr} + ${KT} <= ${k};
        const bool fast_b_tile = krows_ok && fast_col;
% for pp in range(b_tile_iters):
${b_prefetch_tile_frag(row_offset_expr, slot, pp)}\
% endfor
        }
</%def>

<%def name="b_write_tile_frag(buf_expr, slot, pp)">\
<%
    store = f'{kname}_Bs[{buf_expr} + idx] = b_next_{slot}_{pp};'
%>\
        {
            const int idx = tid + ${pp*nthreads};
% if pp == b_tile_full:
            if (idx < ${b_tile_elems})
                ${store}
% else:
            ${store}
% endif
        }
</%def>

<%def name="b_write_tile(buf_expr, slot)">\
% for pp in range(b_tile_iters):
${b_write_tile_frag(buf_expr, slot, pp)}\
% endfor
</%def>
## ---------------------------------------------------------------------------
## Wave-level MFMA accumulation
## ---------------------------------------------------------------------------

<%def name="mfma_k_group(kgp, which)">\
<%
    kg, acomp = 2*kgp + which, 'xy'[which]
%>\
% for t in range(n_tiles):
        const ${dtype} bv_${kg}_${t} = ${kname}_Bs[
            curbuf*${b_tile_elems} + (${kg*mfma_k} + g)*${NT} + ${t*16} + p];
% endfor
% for j in range(mtpg):
<%
    av0, av1 = (f'a_pair_{s}_{j}_{kgp}.{acomp}' for s in (0, 1))
    av = f'curbuf ? {av1} : {av0}' if multi_ktile else av0
%>\
%  if not m_tile_always_held(j):
        if (wmt + ${j} < ${valid_m_tiles})
%  endif
        {
            const ${sdtype} av = ${av};
%  for t in range(n_tiles):
%   for w in range(width):
            ${acc(w, j, t)} = ${mfma}(
                av, ${bval(kg, t, w)}, ${acc(w, j, t)}, 0, 0, 0);
%   endfor
%  endfor
        }
% endfor
</%def>

## ---------------------------------------------------------------------------
## C epilogue helpers
## ---------------------------------------------------------------------------

<%def name="c_epilogue_coords(j, t, reg)">\
            const int mt = wmt + ${j};
            const int row = row_base + mt*16 + ${c_row_offset(reg)};
            const int col = col_base + ${t*16} + p;
</%def>

<%def name="store_c_epilogue_beta1(guarded)">\
<%
    ind = ' '*16 if guarded else ' '*12
%>\
% for j, t, reg in c_epilogue_indices:
        ${dtype} c_old_${j}_${t}_${reg};
%  if guarded:
        bool c_valid_${j}_${t}_${reg};
%  endif
% endfor

% for j, t, reg in c_epilogue_indices:
        {
${c_epilogue_coords(j, t, reg)}\
%  if guarded:
            c_valid_${j}_${t}_${reg} = mt < ${valid_m_tiles} &&
                (fast_row || row < ${m}) && (fast_col || col < n);
            if (c_valid_${j}_${t}_${reg})
%  endif
${ind}c_old_${j}_${t}_${reg} = nt_load(&c[row*ldc + col]);
        }
% endfor

% for j, t, reg in c_epilogue_indices:
        {
${c_epilogue_coords(j, t, reg)}\
%  if guarded:
            if (c_valid_${j}_${t}_${reg})
%  endif
${ind}nt_store(&c[row*ldc + col],
${ind}         c_old_${j}_${t}_${reg} + ${cval(j, t, reg)});
        }
% endfor
</%def>

<%def name="store_c_epilogue_scalar(guarded)">\
% for j, t, reg in c_epilogue_indices:
<%
    tile_held = m_tile_always_held(j)
    ind = ' '*12 if tile_held and not guarded else ' '*16
%>\
        {
${c_epilogue_coords(j, t, reg)}\
%  if guarded:
            const bool row_ok = fast_row || row < ${m};
            const bool col_ok = fast_col || col < n;
            if (mt < ${valid_m_tiles} && row_ok && col_ok)
%  elif not tile_held:
            if (mt < ${valid_m_tiles})
%  endif
%  if beta == 0:
${ind}nt_store(&c[row*ldc + col], ${cval(j, t, reg)});
%  else:
${ind}nt_store(&c[row*ldc + col], ${beta}*nt_load(&c[row*ldc + col]) +
${ind}         ${cval(j, t, reg)});
%  endif
        }
% endfor
</%def>

<%def name="store_c_epilogue(guarded)">\
% if beta == 1:
${store_c_epilogue_beta1(guarded)}\
% else:
${store_c_epilogue_scalar(guarded)}\
% endif
</%def>

__global__ __launch_bounds__(${nthreads}) void
% if n is None:
${kname}(const ${sdtype}* __restrict__ a, int n,
         const ${dtype}* __restrict__ b, int ldb_,
         ${dtype}* __restrict__ c, int ldc_)
{
%  if width > 1:
    n = (n + ${width} - 1) / ${width};
    ldb_ /= ${width};
    ldc_ /= ${width};
%  endif
    const long long ldb = ldb_;
    const long long ldc = ldc_;
% else:
${kname}(const ${sdtype}* __restrict__ a,
         const ${dtype}* __restrict__ b, ${dtype}* __restrict__ c)
{
    const int n = ${-(-n // width)};
    const ${'long long' if k*ldb >= width*2**31 else 'int'} ldb = ${ldb // width};
    const ${'long long' if m*ldc >= width*2**31 else 'int'} ldc = ${ldc // width};
% endif
    // A comes in pre-packed as row16-tile, K-group-pair, lane, pair-element
    const ${apair_t}* ap = (const ${apair_t}*)${aligned_a};

    const int tid = threadIdx.y*${blockx} + threadIdx.x;
    const int lane = tid & 63;
    const int wave = tid >> 6;
    const int wmt = wave*${mtpg};

    const int g = lane / 16;   // MFMA K group / C row group
    const int p = lane % 16;   // MFMA row/column position

    const int logical_bid = blockIdx.y*gridDim.x + blockIdx.x;
    const int m_tile = logical_bid % gridDim.y;
    const int n_tile = logical_bid / gridDim.y;
    const int row_base = m_tile*${MT};
    const int col_base = n_tile*${NT};
    const bool fast_col = col_base + ${NT} <= n;
    const bool fast_row = row_base + ${MT} <= ${m};
% if m_tile_always_held(mtpg - 1):
    const bool fast_tile = fast_col && fast_row;
% else:
    const bool fast_tile = false;
% endif

    __shared__ __align__(16) ${dtype} ${kname}_Bs[${2*b_tile_elems}];

% for j in range(mtpg):
%  for t in range(n_tiles):
%   for w in range(width):
    ${acc4_t} ${acc(w, j, t)} = {0.0, 0.0, 0.0, 0.0};
%   endfor
%  endfor
% endfor

% for j in range(mtpg):
%  for kgp in range(k_group_pairs):
    ${apair_t} a_pair_0_${j}_${kgp};
%   if multi_ktile:
    ${apair_t} a_pair_1_${j}_${kgp};
%   endif
%  endfor
% endfor
% for pp in range(b_tile_iters):
    ${dtype} b_next_0_${pp};
%  if multi_ktile:
    ${dtype} b_next_1_${pp};
%  endif
% endfor

${a_operand_prefetch_tile('0', 0)}\
% if multi_ktile:
${a_operand_prefetch_tile(str(KT), 1)}\
% endif
${b_prefetch_tile('0', 0)}\
${b_write_tile('0', 0)}\
% if multi_ktile:
${b_prefetch_tile(str(KT), 1)}\
% endif
    __syncthreads();

    for (int k0 = 0; k0 < ${k_pad}; k0 += ${KT})
    {
        const int curbuf = (k0 / ${KT}) & 1;
% if multi_ktile:
        const int nextbuf = curbuf ^ 1;
        const int k_next = k0 + ${KT};
        const int k_next2 = k0 + ${2*KT};

        if (k_next < ${k_pad})
        {
            if (curbuf)
            {
${b_write_tile(f'nextbuf*{b_tile_elems}', 0)}\
            }
            else
            {
${b_write_tile(f'nextbuf*{b_tile_elems}', 1)}\
            }
        }

        if (k_next2 < ${k_pad})
        {
            if (curbuf)
            {
${b_prefetch_tile('k_next2', 1)}\
            }
            else
            {
${b_prefetch_tile('k_next2', 0)}\
            }
        }
% endif
% for kgp in range(k_group_pairs):
%  for which in range(2):
${mfma_k_group(kgp, which)}\
%  endfor
% endfor
% if multi_ktile:

        if (k_next < ${k_pad})
        {
            if (k_next2 < ${k_pad})
            {
                if (curbuf)
                {
${a_operand_prefetch_tile('k_next2', 1)}\
                }
                else
                {
${a_operand_prefetch_tile('k_next2', 0)}\
                }
            }
            __syncthreads();
        }
% endif
    }

    if (fast_tile)
    {
${store_c_epilogue(False)}\
    }
    else
    {
${store_c_epilogue(True)}\
    }
}
