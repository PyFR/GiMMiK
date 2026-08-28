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
    a_pair_groups = 2
    f64 = sdtype == 'double'
    apair_align = 16 if f64 else 8

    nthreads = blockx*blocky

    def check(bad, msg):
        if bad:
            raise ValueError(f'mfma-tile-gemm: {msg}')

    check(nthreads % 64, 'block must be a whole number of wave64 waves')
    check(MT % 16, 'MT must be a multiple of 16')
    check(NT % 16, 'NT must be a multiple of 16')
    check(KT % (mfma_k*a_pair_groups), 'KT must cover whole A pairs')
    check(blocky <= 0, 'blocky must be positive')
    check(width not in (1, 2, 4), 'width must be 1, 2 or 4')

    if f64:
        mfma = '__builtin_amdgcn_mfma_f64_16x16x4f64'
    else:
        mfma = '__builtin_amdgcn_mfma_f32_16x16x4f32'

    def c_row_offset(reg):
        if f64:
            return f'{4*reg} + g'
        else:
            return f'4*g + {reg}'

    valid_m_tiles = min(-(-m // 16), MT // 16)
    n_tiles = NT // 16
    k_groups = KT // mfma_k
    k_group_pairs = k_groups // 2
    nwaves = nthreads // 64
    mtpg = -(-valid_m_tiles // nwaves)
    b_tile_elems = KT*NT
    b_tile_iters = -(-b_tile_elems // nthreads)

    def m_tile_always_held(j):
        return (nwaves - 1)*mtpg + j < valid_m_tiles

    def acc(w, j, t):
        if width == 1:
            return f'acc_{j}_{t}'
        else:
            return f'acc_{w}_{j}_{t}'

    def bval(kg, t, w):
        if width == 1:
            return f'bv_{kg}_{t}'
        else:
            return f'bv_{kg}_{t}.{"xyzw"[w]}'

    def cval(j, t, reg):
        if width == 1:
            return f'acc_{j}_{t}[{reg}]'
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
typedef ${sdtype} gimmik_acc4 __attribute__((ext_vector_type(4)));
typedef ${sdtype} gimmik_apair __attribute__((ext_vector_type(2)));

## ---------------------------------------------------------------------------
## Tile-level global prefetch and LDS staging helpers
## ---------------------------------------------------------------------------

<%def name="a_pair_expr(j, kgp, kbase)">\
ap[((row_base / 16 + wmt + ${j})*${k_pad // 8} + ${kbase} + ${kgp})*64 + lane]\
</%def>

<%def name="a_operand_prefetch_tile(row_offset_expr, slot)">
<%
    kbase = f'(({row_offset_expr}) / {KT})*{k_group_pairs}'
%>
% for j in range(mtpg):
%  for kgp in range(k_group_pairs):
<%
    dst = f'a_pair_{slot}_{j}_{kgp}'
%>
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

<%def name="b_prefetch_tile_frag(row_offset_expr, slot, pp)">
        {
            const int idx = tid + ${pp*nthreads};
% if (pp + 1)*nthreads > b_tile_elems:
            if (idx < ${b_tile_elems})
% endif
            {
                const int kk = idx / ${NT};
                const int cc = idx % ${NT};
                const int krow = ${row_offset_expr} + kk;
                const int col = col_base + cc;
                const bool have = fast_b_tile || (krow < ${k} && col < n);

                b_next_${slot}_${pp} =
                    have ? b[krow*ldb + col] : make_zero();
            }
        }
</%def>

<%def name="b_prefetch_tile(row_offset_expr, slot)">
        {
        const bool fast_b_tile = (${row_offset_expr} + ${KT} <= ${k}) && fast_col;
% for pp in range(b_tile_iters):
${b_prefetch_tile_frag(row_offset_expr, slot, pp)}
% endfor
        }
</%def>

<%def name="b_write_tile_frag(buf_expr, slot, pp)">
        {
            const int idx = tid + ${pp*nthreads};
% if (pp + 1)*nthreads > b_tile_elems:
            if (idx < ${b_tile_elems})
% endif
            ${kname}_Bs[${buf_expr} + idx] = b_next_${slot}_${pp};
        }
</%def>

<%def name="b_write_tile(buf_expr, slot)">
% for pp in range(b_tile_iters):
${b_write_tile_frag(buf_expr, slot, pp)}
% endfor
</%def>

<%def name="b_prefetch_write_tile(row_offset_expr, buf_expr, slot)">
${b_prefetch_tile(row_offset_expr, slot)}
${b_write_tile(buf_expr, slot)}
</%def>

## ---------------------------------------------------------------------------
## Wave-level MFMA accumulation
## ---------------------------------------------------------------------------

<%def name="mfma_accumulate(j, kgp, kg, acomp)">
            const ${sdtype} av = curbuf ? a_pair_1_${j}_${kgp}.${acomp}
                                        : a_pair_0_${j}_${kgp}.${acomp};
% for t in range(n_tiles):
%  for w in range(width):
            ${acc(w, j, t)} = ${mfma}(
                av, ${bval(kg, t, w)}, ${acc(w, j, t)}, 0, 0, 0);
%  endfor
% endfor
</%def>

<%def name="mfma_k_group(kgp, which)">
<%
    kg = 2*kgp + which
    acomp = 'xy'[which]
%>
% for t in range(n_tiles):
        const ${dtype} bv_${kg}_${t} = ${kname}_Bs[
            curbuf*${b_tile_elems} + (${kg*4} + g)*${NT} + ${t*16} + p];
% endfor
% for j in range(mtpg):
% if not m_tile_always_held(j):
        if (wmt + ${j} < ${valid_m_tiles})
% endif
        {
${mfma_accumulate(j, kgp, kg, acomp)}
        }
% endfor
</%def>

## ---------------------------------------------------------------------------
## C epilogue helpers
## ---------------------------------------------------------------------------

<%def name="c_epilogue_coords(j, t, reg)">
            const int mt = wmt + ${j};
            const int row = row_base + mt*16 + ${c_row_offset(reg)};
            const int col = col_base + ${t*16} + p;
</%def>

<%def name="store_c_epilogue_beta1(guarded)">
% for j, t, reg in c_epilogue_indices:
        ${dtype} c_old_${j}_${t}_${reg};
%  if guarded:
        bool c_valid_${j}_${t}_${reg};
%  endif
% endfor

% for j, t, reg in c_epilogue_indices:
        {
${c_epilogue_coords(j, t, reg)}
%  if guarded:
            c_valid_${j}_${t}_${reg} = mt < ${valid_m_tiles} &&
                (fast_row || row < ${m}) && (fast_col || col < n);
            if (c_valid_${j}_${t}_${reg})
%  endif
            c_old_${j}_${t}_${reg} = nt_load(&c[row*ldc + col]);
        }
% endfor

% for j, t, reg in c_epilogue_indices:
        {
${c_epilogue_coords(j, t, reg)}
%  if guarded:
            if (c_valid_${j}_${t}_${reg})
%  endif
            nt_store(&c[row*ldc + col],
                     c_old_${j}_${t}_${reg} + ${cval(j, t, reg)});
        }
% endfor
</%def>

<%def name="store_c_epilogue_scalar(guarded)">
% for j, t, reg in c_epilogue_indices:
        {
${c_epilogue_coords(j, t, reg)}
%  if guarded:
            if (mt < ${valid_m_tiles} && (fast_row || row < ${m}) &&
                (fast_col || col < n))
%  elif not m_tile_always_held(j):
            if (mt < ${valid_m_tiles})
%  endif
%  if beta == 0:
            nt_store(&c[row*ldc + col], ${cval(j, t, reg)});
%  else:
            nt_store(&c[row*ldc + col], ${beta}*nt_load(&c[row*ldc + col])
                                        + ${cval(j, t, reg)});
%  endif
        }
% endfor
</%def>

<%def name="store_c_epilogue(guarded)">
% if beta == 1:
${store_c_epilogue_beta1(guarded)}
% else:
${store_c_epilogue_scalar(guarded)}
% endif
</%def>

__global__ __launch_bounds__(${blockx * blocky}) void
% if n is None:
${kname}(const ${sdtype}* __restrict__ a, int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
% if width > 1:
    n = (n + ${width} - 1) / ${width};
    ldb /= ${width};
    ldc /= ${width};
% endif
% else:
${kname}(const ${sdtype}* __restrict__ a,
         const ${dtype}* __restrict__ b, ${dtype}* __restrict__ c)
{
    const int n = ${-(-n // width)};
    const ${'long long' if k * ldb >= width*2**31 else 'int'} ldb = ${ldb // width};
    const ${'long long' if m * ldc >= width*2**31 else 'int'} ldc = ${ldc // width};
% endif
    // A comes in pre-packed as row16-tile, K-group-pair, lane, pair-element
    const gimmik_apair* __restrict__ ap =
        (const gimmik_apair*)__builtin_assume_aligned(a, ${apair_align});

    const int tid = threadIdx.y*${blockx} + threadIdx.x;
    const int lane = tid & 63;
    const int wave = tid >> 6;
    const int wmt = wave*${mtpg};

    const int g = lane / 16;   // MFMA K group / C row group
    const int p = lane % 16;   // MFMA row/column position

    const int logical_bid = blockIdx.y*gridDim.x + blockIdx.x;
    const int m_tile = logical_bid % gridDim.y;
    const int n_tile = logical_bid / gridDim.y;
    const int row_base = m_tile * ${MT};
    const int col_base = n_tile * ${NT};
    const bool fast_col = col_base + ${NT} <= n;
    const bool fast_row = row_base + ${MT} <= ${m};
    const bool fast_tile = fast_col & fast_row & (${nwaves * mtpg} <= ${valid_m_tiles});

    __shared__ __align__(16) ${dtype} ${kname}_Bs[${2*b_tile_elems}];

% for j in range(mtpg):
%  for t in range(n_tiles):
%   for w in range(width):
    gimmik_acc4 ${acc(w, j, t)} = {0.0, 0.0, 0.0, 0.0};
%   endfor
%  endfor
% endfor

% for j in range(mtpg):
%  for kgp in range(k_group_pairs):
    gimmik_apair a_pair_0_${j}_${kgp};
    gimmik_apair a_pair_1_${j}_${kgp};
%  endfor
% endfor
% for pp in range(b_tile_iters):
    ${dtype} b_next_0_${pp};
    ${dtype} b_next_1_${pp};
% endfor

${a_operand_prefetch_tile('0', 0)}
% if KT < k_pad:
${a_operand_prefetch_tile(str(KT), 1)}
% endif
${b_prefetch_write_tile('0', '0', 0)}
% if KT < k_pad:
${b_prefetch_tile(str(KT), 1)}
% endif
    __syncthreads();

    for (int k0 = 0; k0 < ${k_pad}; k0 += ${KT})
    {
        const int curbuf = (k0 / ${KT}) & 1;
        const int nextbuf = curbuf ^ 1;
        const int k_next = k0 + ${KT};
        const int k_next2 = k0 + ${2*KT};

        if (k_next < ${k_pad})
        {
            if (curbuf)
            {
${b_write_tile(f'nextbuf*{b_tile_elems}', 0)}
            }
            else
            {
${b_write_tile(f'nextbuf*{b_tile_elems}', 1)}
            }
        }

        if (k_next2 < ${k_pad})
        {
            if (curbuf)
            {
${b_prefetch_tile('k_next2', 1)}
            }
            else
            {
${b_prefetch_tile('k_next2', 0)}
            }
        }
% for kgp in range(k_group_pairs):
%  for which in range(2):
${mfma_k_group(kgp, which)}
%  endfor
% endfor

        if (k_next < ${k_pad})
        {
            if (k_next2 < ${k_pad})
            {
                if (curbuf)
                {
${a_operand_prefetch_tile('k_next2', 1)}
                }
                else
                {
${a_operand_prefetch_tile('k_next2', 0)}
                }
            }
            __syncthreads();
        }
    }

    if (fast_tile)
    {
${store_c_epilogue(False)}
    }
    else
    {
${store_c_epilogue(True)}
    }
}
