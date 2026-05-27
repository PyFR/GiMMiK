<%inherit file='base'/>

.global .align 16 .b64 ${kname}_Ag[${32 * m_tiles * k_tiles}] = {
    ${', '.join(a_u64)}
};

.visible .entry ${kname}(.param .u64 _b,
                         .param .u64 _c)
{
    .reg .u32  tid, warp, lane, r_mod4, r_div4;
    .reg .u64  b_ptr, c_ptr;
    .reg .u32  warp_n_base;
    .reg .u64  ag_thr_base, b_thr_base, c_thr_base;
    .reg .pred pwarp_exit;
    .reg .${pftype}  a_frag;
% for nt in range(nn):
    .reg .u32  b_col_${nt}, c_col0_${nt}, c_col1_${nt};
%  if not n_col_aligned:
    .reg .pred pvalid_bcol_${nt}, pvalid_c0col_${nt}, pvalid_c1col_${nt};
%  endif
    .reg .${pftype}  b_frag_${nt};
    .reg .${pftype}  c0_${nt}_<${m_tiles}>, c1_${nt}_<${m_tiles}>;
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

    {
        .reg .u32 cta;
        mov.u32    cta, %ctaid.x;
        mul.lo.u32 cta, cta, ${n_per_cta};
        mul.lo.u32 warp_n_base, warp, ${n_per_warp};
        add.u32    warp_n_base, warp_n_base, cta;
    }
    setp.ge.u32 pwarp_exit, warp_n_base, ${n};
    @pwarp_exit bra $L_EXIT;

% for nt in range(nn):
    add.u32 b_col_${nt}, warp_n_base, ${8 * nt};
    add.u32 b_col_${nt}, b_col_${nt}, r_div4;
    {
        .reg .u32 t;
        shl.b32 t, r_mod4, 1;
        add.u32 c_col0_${nt}, warp_n_base, ${8 * nt};
        add.u32 c_col0_${nt}, c_col0_${nt}, t;
        add.u32 c_col1_${nt}, c_col0_${nt}, 1;
    }
%  if not n_col_aligned:
    setp.lt.u32 pvalid_bcol_${nt},  b_col_${nt},  ${n};
    setp.lt.u32 pvalid_c0col_${nt}, c_col0_${nt}, ${n};
    setp.lt.u32 pvalid_c1col_${nt}, c_col1_${nt}, ${n};
%  endif
% endfor

    // A thread base: &Ag[0] + lane*8
    {
        .reg .u64 t64, a_glb_base, lane64;
        mov.u64      a_glb_base, ${kname}_Ag;
        cvta.to.global.u64 a_glb_base, a_glb_base;
        cvt.u64.u32  lane64, lane;
        shl.b64      t64, lane64, 3;
        add.u64      ag_thr_base, a_glb_base, t64;
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

% for mt in range(m_tiles):
%  if pm_runtime(mt):
    .reg .pred pm_${mt};
    {
        .reg .u32 crow;
        add.u32 crow, r_div4, ${8 * mt};
        setp.lt.u32 pm_${mt}, crow, ${m};
    }
%  endif
% endfor

% for nt in range(nn):
%  for mt in range(m_tiles):
%   if beta_zero:
    mov.${pftype} c0_${nt}_${mt}, ${fzero};
    mov.${pftype} c1_${nt}_${mt}, ${fzero};
%   else:
<%
    pm = f'pm_{mt}' if pm_runtime(mt) else None
    pvc0 = f'pvalid_c0col_{nt}' if not n_col_aligned else None
    pvc1 = f'pvalid_c1col_{nt}' if not n_col_aligned else None
    needs_zero_init = pm is not None or pvc0 is not None or pvc1 is not None
%>
    {
        .reg .u64 caddr;
        add.u64      caddr, c_thr_base, ${mt * c_mtile_stride + nt * c_ntile_stride};
%    if needs_zero_init:
        mov.${pftype}      c0_${nt}_${mt}, ${fzero};
        mov.${pftype}      c1_${nt}_${mt}, ${fzero};
%    endif
        ${pred_emit(f'ld.weak.global.cg.{pftype} c0_{nt}_{mt}, [caddr];', pm, pvc0, pred_reg=f'p0_{nt}_{mt}')}
        ${pred_emit(f'ld.weak.global.cg.{pftype} c1_{nt}_{mt}, [caddr + {dwidth_i}];', pm, pvc1, pred_reg=f'p1_{nt}_{mt}')}
    }
%   endif
%  endfor
% endfor

% for ki in range(k_tiles):
%  for nt in range(nn):
<%
    pvb = f'pvalid_bcol_{nt}' if not n_col_aligned else None
    k_tail = (k_rem != 0 and loop.parent.last)
    needs_zero = pvb is not None or k_tail
    pbrow = 'pbrow' if k_tail else None
%>
    {
        .reg .u64 baddr;
        add.u64 baddr, b_thr_base, ${ki * b_kiter_stride + nt * b_ntile_stride};
%   if needs_zero:
        mov.${pftype} b_frag_${nt}, ${fzero};
%   endif
%   if k_tail:
        .reg .pred pbrow;
        {
            .reg .u32 brow;
            add.u32 brow, r_mod4, ${4 * ki};
            setp.lt.u32 pbrow, brow, ${k};
        }
%   endif
        ${pred_emit(f'ld.weak.global.cg.{pftype} b_frag_{nt}, [baddr];', pbrow, pvb, pred_reg=f'pb_{ki}_{nt}')}
    }
%  endfor
%  for mt in range(m_tiles):
    ld.weak.global.${pftype} a_frag, [ag_thr_base + ${(mt * k_tiles + ki) * frag_stride_bytes}];
%   for nt in range(nn):
    mma.sync.aligned.m8n8k4.row.col.${pftype}.${pftype}.${pftype}.${pftype}
        {c0_${nt}_${mt}, c1_${nt}_${mt}},
        {a_frag},
        {b_frag_${nt}},
        {c0_${nt}_${mt}, c1_${nt}_${mt}};
%   endfor
%  endfor
% endfor

% for nt in range(nn):
%  for mt in range(m_tiles):
<%
    pm = f'pm_{mt}' if pm_runtime(mt) else None
    pvc0 = f'pvalid_c0col_{nt}' if not n_col_aligned else None
    pvc1 = f'pvalid_c1col_{nt}' if not n_col_aligned else None
%>
    {
        .reg .u64 caddr;
        add.u64  caddr, c_thr_base, ${mt * c_mtile_stride + nt * c_ntile_stride};
        ${pred_emit(f'st.weak.global.{pftype} [caddr], c0_{nt}_{mt};', pm, pvc0, pred_reg=f'p0s_{nt}_{mt}')}
        ${pred_emit(f'st.weak.global.{pftype} [caddr + {dwidth_i}], c1_{nt}_{mt};', pm, pvc1, pred_reg=f'p1s_{nt}_{mt}')}
    }
%  endfor
% endfor

$L_EXIT:
    ret;
}
