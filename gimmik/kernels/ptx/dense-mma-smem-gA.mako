<%inherit file='base'/>

<%!
import struct
import math
%>

<%
assert dtype == "double"
assert n is not None and ldb is not None and ldc is not None

M, K_ = A.shape
assert K_ == k
M_PAD   = -(-M // 8) * 8
M_TILES = M_PAD // 8
K_REM   = k % 4
K_PAD   = k if K_REM == 0 else k + (4 - K_REM)
K_ITERS = K_PAD // 4

# A in fragment-layout (same as dense-mma-smem-nn)
a_u64 = []
for m_tile in range(M_TILES):
    for k_iter in range(K_ITERS):
        for lane in range(32):
            r_div4 = lane // 4
            r_mod4 = lane % 4
            i = m_tile * 8 + r_div4
            j = k_iter * 4 + r_mod4
            v = float(A[i, j]) if (i < M and j < k) else 0.0
            u = struct.unpack('<Q', struct.pack('<d', v))[0]
            a_u64.append(f'0x{u:016x}')

WARPS_PER_CTA = warps_per_cta
NN = nn
BLOCKX     = 32 * WARPS_PER_CTA
N_PER_WARP = 8 * NN
N_PER_CTA  = WARPS_PER_CTA * N_PER_WARP
A_ELEMS    = M_TILES * K_ITERS * 32
# v2 load: 2 f64 per thread per iter -> 2*BLOCKX elements per copy iter
A_PAIRS    = A_ELEMS // 2                # number of f64x2 pairs
A_PAIRS_TAIL = A_ELEMS % 2               # 0 if even, 1 if odd
COPY_V2_ITERS = math.ceil(A_PAIRS / BLOCKX)

FRAG_STRIDE_BYTES = 32 * 8
B_KITER_STRIDE    = 4 * ldb * 8
B_NTILE_STRIDE    = 8 * 8
C_MTILE_STRIDE    = 8 * ldc * 8
C_NTILE_STRIDE    = 8 * 8
%>

.global .align 16 .b64 ${kname}_Ag[${A_ELEMS}] = {
    ${', '.join(a_u64)}
};
.shared .align 16 .b64 ${kname}_As[${A_ELEMS}];

.visible .entry ${kname}(.param .u64 _b,
                         .param .u64 _c)
{
    .reg .u32  tid, warp, lane, r_mod4, r_div4;
    .reg .u64  b_ptr, c_ptr;
    .reg .u32  warp_n_base;
    .reg .u64  as_thr_base, b_thr_base, c_thr_base;
    .reg .pred pwarp_exit;
    .reg .f64  a_frag;
% for nt in range(NN):
    .reg .u32  b_col_${nt}, c_col0_${nt}, c_col1_${nt};
    .reg .pred pvalid_bcol_${nt}, pvalid_c0col_${nt}, pvalid_c1col_${nt};
    .reg .f64  b_frag_${nt};
    .reg .f64  c0_${nt}_<${M_TILES}>, c1_${nt}_<${M_TILES}>;
% endfor

    ld.param.u64 b_ptr, [_b];
    ld.param.u64 c_ptr, [_c];
    cvta.to.global.u64 b_ptr, b_ptr;
    cvta.to.global.u64 c_ptr, c_ptr;

    mov.u32 tid, %tid.x;
    shr.u32 warp, tid, 5;
    and.b32 lane, tid, 31;
    shr.u32 r_div4, lane, 2;
    and.b32 r_mod4, lane, 3;

    // ---- Cooperative copy A from .global to .shared using v2 loads ----
    {
        .reg .u64 a_glb_base, a_smem_base;
        mov.u64 a_glb_base,  ${kname}_Ag;
        cvta.to.global.u64 a_glb_base, a_glb_base;
        mov.u64 a_smem_base, ${kname}_As;
% for ci in range(COPY_V2_ITERS):
<%
    base_pair = ci * BLOCKX
    is_last = ci == COPY_V2_ITERS - 1
    pairs_this = min(BLOCKX, A_PAIRS - base_pair)
%>
        {
            .reg .u32 pidx;
            .reg .u64 off64, gaddr, saddr;
            .reg .f64 v0, v1;
% if is_last and pairs_this < BLOCKX:
            .reg .pred plast;
            add.u32 pidx, tid, ${base_pair};
            setp.lt.u32 plast, pidx, ${A_PAIRS};
            mul.wide.u32 off64, pidx, 16;
            add.u64 gaddr, a_glb_base,  off64;
            add.u64 saddr, a_smem_base, off64;
            @plast ld.global.nc.v2.f64 {v0, v1}, [gaddr];
            @plast st.shared.v2.f64     [saddr], {v0, v1};
% else:
            add.u32 pidx, tid, ${base_pair};
            mul.wide.u32 off64, pidx, 16;
            add.u64 gaddr, a_glb_base,  off64;
            add.u64 saddr, a_smem_base, off64;
            ld.global.nc.v2.f64 {v0, v1}, [gaddr];
            st.shared.v2.f64     [saddr], {v0, v1};
% endif
        }
% endfor
% if A_PAIRS_TAIL:
        // Odd element at the very end (rare; A_ELEMS odd)
        {
            .reg .pred plast;
            .reg .u64 gaddr, saddr;
            .reg .f64 v;
            setp.eq.u32 plast, tid, 0;
            add.u64 gaddr, a_glb_base,  ${(A_ELEMS-1) * 8};
            add.u64 saddr, a_smem_base, ${(A_ELEMS-1) * 8};
            @plast ld.global.nc.f64 v, [gaddr];
            @plast st.shared.f64    [saddr], v;
        }
% endif
    }
    bar.sync 0;

    {
        .reg .u32 cta;
        mov.u32    cta, %ctaid.x;
        mul.lo.u32 cta, cta, ${N_PER_CTA};
        mul.lo.u32 warp_n_base, warp, ${N_PER_WARP};
        add.u32    warp_n_base, warp_n_base, cta;
    }
    setp.ge.u32 pwarp_exit, warp_n_base, ${n};
    @pwarp_exit bra $L_EXIT;

% for nt in range(NN):
    add.u32 b_col_${nt}, warp_n_base, ${nt * 8};
    add.u32 b_col_${nt}, b_col_${nt}, r_div4;
    {
        .reg .u32 t;
        shl.b32 t, r_mod4, 1;
        add.u32 c_col0_${nt}, warp_n_base, ${nt * 8};
        add.u32 c_col0_${nt}, c_col0_${nt}, t;
        add.u32 c_col1_${nt}, c_col0_${nt}, 1;
    }
    setp.lt.u32 pvalid_bcol_${nt},  b_col_${nt},  ${n};
    setp.lt.u32 pvalid_c0col_${nt}, c_col0_${nt}, ${n};
    setp.lt.u32 pvalid_c1col_${nt}, c_col1_${nt}, ${n};
% endfor

    {
        .reg .u64 t64, a_smem_base, lane64;
        mov.u64      a_smem_base, ${kname}_As;
        cvt.u64.u32  lane64, lane;
        shl.b64      t64, lane64, 3;
        add.u64      as_thr_base, a_smem_base, t64;
    }

    {
        .reg .u64 t64, bcol64;
        mul.wide.u32 t64, r_mod4, ${ldb};
        cvt.u64.u32  bcol64, b_col_0;
        add.u64      t64, t64, bcol64;
        shl.b64      t64, t64, 3;
        add.u64      b_thr_base, b_ptr, t64;
    }

    {
        .reg .u64 t64, ccol64;
        mul.wide.u32 t64, r_div4, ${ldc};
        cvt.u64.u32  ccol64, c_col0_0;
        add.u64      t64, t64, ccol64;
        shl.b64      t64, t64, 3;
        add.u64      c_thr_base, c_ptr, t64;
    }

% for mt in range(M_TILES):
    .reg .pred pm_${mt};
    {
        .reg .u32 crow;
        add.u32 crow, r_div4, ${mt * 8};
        setp.lt.u32 pm_${mt}, crow, ${M};
    }
% endfor

% for nt in range(NN):
% for mt in range(M_TILES):
% if beta == 0:
    mov.f64 c0_${nt}_${mt}, 0d0000000000000000;
    mov.f64 c1_${nt}_${mt}, 0d0000000000000000;
% else:
    {
        .reg .u64 caddr;
        .reg .pred p0, p1;
        add.u64      caddr, c_thr_base, ${mt * C_MTILE_STRIDE + nt * C_NTILE_STRIDE};
        and.pred     p0, pm_${mt}, pvalid_c0col_${nt};
        and.pred     p1, pm_${mt}, pvalid_c1col_${nt};
        mov.f64      c0_${nt}_${mt}, 0d0000000000000000;
        mov.f64      c1_${nt}_${mt}, 0d0000000000000000;
        @p0 ld.global.f64 c0_${nt}_${mt}, [caddr];
        @p1 ld.global.f64 c1_${nt}_${mt}, [caddr + 8];
    }
% endif
% endfor
% endfor

% for ki in range(K_ITERS):
% for nt in range(NN):
    {
        .reg .u64 baddr;
        .reg .pred pb_load;
        add.u64 baddr, b_thr_base, ${ki * B_KITER_STRIDE + nt * B_NTILE_STRIDE};
% if K_REM != 0 and ki == K_ITERS - 1:
        {
            .reg .u32 brow;
            .reg .pred pbrow;
            add.u32 brow, r_mod4, ${ki * 4};
            setp.lt.u32 pbrow, brow, ${k};
            and.pred pb_load, pbrow, pvalid_bcol_${nt};
        }
% else:
        and.pred pb_load, pvalid_bcol_${nt}, pvalid_bcol_${nt};
% endif
        mov.f64 b_frag_${nt}, 0d0000000000000000;
        @pb_load ld.global.nc.f64 b_frag_${nt}, [baddr];
    }
% endfor
% for mt in range(M_TILES):
    ld.shared.f64 a_frag, [as_thr_base + ${(mt * K_ITERS + ki) * FRAG_STRIDE_BYTES}];
% for nt in range(NN):
    mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64
        {c0_${nt}_${mt}, c1_${nt}_${mt}},
        {a_frag},
        {b_frag_${nt}},
        {c0_${nt}_${mt}, c1_${nt}_${mt}};
% endfor
% endfor
% endfor

% for nt in range(NN):
% for mt in range(M_TILES):
    {
        .reg .u64 caddr;
        .reg .pred p0, p1;
        add.u64  caddr, c_thr_base, ${mt * C_MTILE_STRIDE + nt * C_NTILE_STRIDE};
        and.pred p0, pm_${mt}, pvalid_c0col_${nt};
        and.pred p1, pm_${mt}, pvalid_c1col_${nt};
        @p0 st.global.f64 [caddr],     c0_${nt}_${mt};
        @p1 st.global.f64 [caddr + 8], c1_${nt}_${mt};
    }
% endfor
% endfor

$L_EXIT:
    ret;
}
