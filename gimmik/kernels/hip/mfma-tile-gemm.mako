<%inherit file='base'/>
##
## Dense FP64 tiled GEMM optimized for AMD wave-level MFMA:
##   1. Uses v_mfma_f64_16x16x4_f64 for 16x16 FP64 wave tiles.
##   2. Packs baked A in MFMA lane-major order so each lane can vector-load
##      the two FP64 A operands used by a pair of consecutive MFMA K groups.
##   3. Loads A directly from global/constant memory into VGPRs; A does not
##      use LDS.
##   4. Loads B with normal cached global loads, stages B through LDS, and
##      reads LDS in the MFMA operand layout.
##   5. Uses PGR2/double buffering for both operands: A is double-buffered in
##      VGPR pairs, while B is double-buffered as global->VGPR->LDS.
##   6. Remaps block ids to a column-major/B-reuse tile order so neighboring
##      workgroups compute different M tiles for the same N tile, improving B
##      cache reuse.
##
<%
    sdtype = context.get('sdtype', dtype)
    mfma_k = 4
    a_pair_groups = 2

    nthreads = blockx * blocky
    if nthreads % 64:
        raise ValueError('mfma-tile-gemm expects a whole number of wave64 waves')
    if MT % 16:
        raise ValueError('mfma-tile-gemm expects MT to be a multiple of 16')
    if NT % 16:
        raise ValueError('mfma-tile-gemm expects NT to be a multiple of 16')
    if KT % (mfma_k * a_pair_groups):
        raise ValueError('mfma-tile-gemm expects KT to cover whole A pairs')
    if blocky <= 0:
        raise ValueError('mfma-tile-gemm expects blocky > 0')
    if width not in (1, 2):
        raise ValueError('mfma-tile-gemm expects width to be 1 or 2')

    valid_m_tiles = min(-(-m // 16), MT // 16)
    n_tiles = NT // 16
    k_groups = KT // mfma_k
    k_group_pairs = k_groups // 2
    nwaves = nthreads // 64
    mtpg = -(-valid_m_tiles // nwaves)
    b_tile_iters = -(-(KT * NT) // nthreads)
    c_epilogue_indices = [
        (j, t, reg)
        for j in range(mtpg)
        for t in range(n_tiles)
        for reg in range(4)
    ]
%>
typedef ${sdtype} gimmik_f64x4 __attribute__((ext_vector_type(4)));
typedef ${sdtype} gimmik_a_f64x2 __attribute__((ext_vector_type(2)));

// A is packed as row16-tile, K-group-pair, lane, pair-element.
__device__ static const __align__(16) ${sdtype} ${kname}_Ag[${m_pad * k_pad}] = {
    ${', '.join(a_hex)}
};

## ---------------------------------------------------------------------------
## Tile-level global prefetch and LDS staging helpers
## ---------------------------------------------------------------------------

<%def name="a_operand_prefetch_tile(row_offset_expr, slot)">
% for j in range(mtpg):
<%
    mt = f'wmt + {j}'
    full_valid_mt = (nwaves - 1) * mtpg + j < valid_m_tiles
%>
%  for kgp in range(k_group_pairs):
        {
%   if full_valid_mt:
            a_pair_${slot}_${j}_${kgp} = *(const gimmik_a_f64x2*)&${kname}_Ag[
                (((row_base / 16 + (${mt}))*${k_pad // 8}
                  + ((${row_offset_expr}) / ${KT})*${k_group_pairs} + ${kgp})*64
                 + lane)*2
            ];
%   else:
            if (${mt} < ${valid_m_tiles})
            {
                a_pair_${slot}_${j}_${kgp} = *(const gimmik_a_f64x2*)&${kname}_Ag[
                    (((row_base / 16 + (${mt}))*${k_pad // 8}
                      + ((${row_offset_expr}) / ${KT})*${k_group_pairs} + ${kgp})*64
                     + lane)*2
                ];
            }
            else
            {
                a_pair_${slot}_${j}_${kgp} = {(${sdtype})0.0, (${sdtype})0.0};
            }
%   endif
        }
%  endfor
% endfor
</%def>

<%def name="b_prefetch_tile_frag(row_offset_expr, slot, pp)">
        {
            const int idx = tid + ${pp * nthreads};
% if (pp + 1) * nthreads <= KT * NT:
            const int kk = idx / ${NT};
            const int cc = idx % ${NT};
            const int krow = ${row_offset_expr} + kk;
            const int col = col_base + cc;
            b_next_${slot}_${pp} = fast_b_tile ? b[krow*ldb + col]
                : ((krow < ${k} && col < n) ? b[krow*ldb + col] : make_zero());
% else:
            if (idx < ${KT * NT})
            {
                const int kk = idx / ${NT};
                const int cc = idx % ${NT};
                const int krow = ${row_offset_expr} + kk;
                const int col = col_base + cc;
                b_next_${slot}_${pp} = fast_b_tile ? b[krow*ldb + col]
                    : ((krow < ${k} && col < n) ? b[krow*ldb + col] : make_zero());
            }
% endif
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
            const int idx = tid + ${pp * nthreads};
% if (pp + 1) * nthreads <= KT * NT:
            ${kname}_Bs[${buf_expr} + idx] = b_next_${slot}_${pp};
% else:
            if (idx < ${KT * NT})
                ${kname}_Bs[${buf_expr} + idx] = b_next_${slot}_${pp};
% endif
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
## C epilogue helpers
## ---------------------------------------------------------------------------

<%def name="c_epilogue_coords(j, t, reg)">
            const int mt = wmt + ${j};
            const int row = row_base + mt*16 + ${4 * reg} + g;
            const int col = col_base + ${t * 16} + p;
</%def>

<%def name="c_epilogue_acc_expr(j, t, reg)">
% if width == 1:
acc_${j}_${t}[${reg}]\
% else:
make_${dtype}(acc_0_${j}_${t}[${reg}], acc_1_${j}_${t}[${reg}])\
% endif
</%def>

<%def name="store_c_epilogue_beta1(guarded)">
% for j, t, reg in c_epilogue_indices:
        ${dtype} c_old_${j}_${t}_${reg};
%    if guarded:
        bool c_valid_${j}_${t}_${reg};
%    endif
% endfor

% for j, t, reg in c_epilogue_indices:
        {
${c_epilogue_coords(j, t, reg)}
%    if guarded:
            c_valid_${j}_${t}_${reg} =
                mt < ${valid_m_tiles} && (fast_row || row < ${m}) && (fast_col || col < n);
            if (c_valid_${j}_${t}_${reg})
%    endif
            c_old_${j}_${t}_${reg} = nt_load(&c[row*ldc + col]);
        }
% endfor

% for j, t, reg in c_epilogue_indices:
        {
${c_epilogue_coords(j, t, reg)}
%    if guarded:
            if (c_valid_${j}_${t}_${reg})
%    endif
            nt_store(&c[row*ldc + col], c_old_${j}_${t}_${reg} + ${c_epilogue_acc_expr(j, t, reg)});
        }
% endfor
</%def>

<%def name="store_c_epilogue_scalar(guarded)">
% for j in range(mtpg):
<%
    full_valid_mt = (nwaves - 1) * mtpg + j < valid_m_tiles
%>
%  for t in range(n_tiles):
%   for reg in range(4):
        {
${c_epilogue_coords(j, t, reg)}
%    if guarded:
            if (mt < ${valid_m_tiles} && (fast_row || row < ${m}) && (fast_col || col < n))
%    elif not full_valid_mt:
            if (mt < ${valid_m_tiles})
%    endif
% if width == 1:
% if beta == 0:
                nt_store(&c[row*ldc + col], acc_${j}_${t}[${reg}]);
% else:
                nt_store(&c[row*ldc + col], ${beta}*nt_load(&c[row*ldc + col]) + acc_${j}_${t}[${reg}]);
% endif
% else:
% if beta == 0:
                nt_store(&c[row*ldc + col], ${c_epilogue_acc_expr(j, t, reg)});
% else:
                nt_store(&c[row*ldc + col], ${beta}*nt_load(&c[row*ldc + col]) + ${c_epilogue_acc_expr(j, t, reg)});
% endif
% endif
        }
%   endfor
%  endfor
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
    const ${'long long' if k * ldb >= width*2**31 else 'int'} ldb = ${ldb // width};
    const ${'long long' if m * ldc >= width*2**31 else 'int'} ldc = ${ldc // width};
% endif
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

    __shared__ __align__(16) ${dtype} ${kname}_Bs[${2 * KT * NT}];

% for j in range(mtpg):
%  for t in range(n_tiles):
%   for w in range(width):
%    if width == 1:
    gimmik_f64x4 acc_${j}_${t} = {0.0, 0.0, 0.0, 0.0};
%    else:
    gimmik_f64x4 acc_${w}_${j}_${t} = {0.0, 0.0, 0.0, 0.0};
%    endif
%   endfor
%  endfor
% endfor

% for j in range(mtpg):
%  for kgp in range(k_group_pairs):
    gimmik_a_f64x2 a_pair_0_${j}_${kgp};
    gimmik_a_f64x2 a_pair_1_${j}_${kgp};
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
        const int k_next2 = k0 + ${2 * KT};

        if (k_next < ${k_pad})
        {
            if ((k0 / ${KT}) & 1)
            {
${b_write_tile('nextbuf*' + str(KT * NT), 0)}
            }
            else
            {
${b_write_tile('nextbuf*' + str(KT * NT), 1)}
            }
        }

        if (k_next2 < ${k_pad})
        {
            if ((k0 / ${KT}) & 1)
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
<%
        kg = 2*kgp + which
        acomp = 'xy'[which]
%>
%   for t in range(n_tiles):
        const ${dtype} bv_${kg}_${t} =
            ${kname}_Bs[curbuf*${KT * NT} + (${kg * 4} + g)*${NT} + ${t * 16} + p];
%   endfor
%   for j in range(mtpg):
<%
        mt = f'wmt + {j}'
        full_valid_mt = (nwaves - 1) * mtpg + j < valid_m_tiles
%>
% if full_valid_mt:
        {
            const ${sdtype} av =
                curbuf ? a_pair_1_${j}_${kgp}.${acomp} : a_pair_0_${j}_${kgp}.${acomp};
%   for t in range(n_tiles):
%    if width == 1:
            acc_${j}_${t} =
                __builtin_amdgcn_mfma_f64_16x16x4f64(av, bv_${kg}_${t}, acc_${j}_${t}, 0, 0, 0);
%    else:
%     for w in range(width):
            acc_${w}_${j}_${t} =
                __builtin_amdgcn_mfma_f64_16x16x4f64(av, bv_${kg}_${t}.${'xy'[w]}, acc_${w}_${j}_${t}, 0, 0, 0);
%     endfor
%    endif
%   endfor
        }
% else:
        if (${mt} < ${valid_m_tiles})
        {
            const ${sdtype} av =
                curbuf ? a_pair_1_${j}_${kgp}.${acomp} : a_pair_0_${j}_${kgp}.${acomp};
%   for t in range(n_tiles):
%    if width == 1:
            acc_${j}_${t} =
                __builtin_amdgcn_mfma_f64_16x16x4f64(av, bv_${kg}_${t}, acc_${j}_${t}, 0, 0, 0);
%    else:
%     for w in range(width):
            acc_${w}_${j}_${t} =
                __builtin_amdgcn_mfma_f64_16x16x4f64(av, bv_${kg}_${t}.${'xy'[w]}, acc_${w}_${j}_${t}, 0, 0, 0);
%     endfor
%    endif
%   endfor
        }
% endif
%   endfor
%  endfor
% endfor

        if (k_next < ${k_pad})
        {
            if (k_next2 < ${k_pad})
            {
                if ((k0 / ${KT}) & 1)
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
