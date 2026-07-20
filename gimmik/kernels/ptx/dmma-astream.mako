<%inherit file='base'/>
/*
  dmma-astream-v1

  Dense FP64 kernel using configurable warp-level DMMA tiles. The tiles of A
  are precomputed and put in global memory within this compilation unit. Then
  tiles of B and A are streamed from global into registers. This kernel uses
  scalar loads/stores for C.
 */

.global .align 16 .b64 ${kname}_Ag[${a_elems}] = {
    ${', '.join(a_u64)}
};

.visible .entry ${kname}(.param .u64 _b,
                         .param .u64 _c)
{
    .reg .u32 tid, warp, lane, r_mod4, r_div4;
    .reg .u64 b_ptr, c_ptr;
    .reg .u32 warp_n_base;
    .reg .u64 ag_thr_base, b_thr_base, c_thr_base;
    .reg .pred pwarp_exit;
    .reg .${pftype} a_frag_<${a_regs}>;
% for nt in range(nn):
    .reg .u32 b_col_${nt}, c_col0_${nt}, c_col1_${nt};
%  if not n_col_aligned:
    .reg .pred pvalid_bcol_${nt}, pvalid_c0col_${nt}, pvalid_c1col_${nt};
%  endif
    .reg .${pftype} b_frag_${nt}_<${b_regs}>;
%  for mt in range(m_tiles):
    .reg .${pftype} c_${nt}_${mt}_<${c_regs}>;
%  endfor
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
        mov.u32 cta, %ctaid.x;
        mul.lo.u32 cta, cta, ${n_per_cta};
        mul.lo.u32 warp_n_base, warp, ${n_per_warp};
        add.u32 warp_n_base, warp_n_base, cta;
    }
    setp.ge.u32 pwarp_exit, warp_n_base, ${n};
    @pwarp_exit bra $L_EXIT;

% for nt in range(nn):
    add.u32 b_col_${nt}, warp_n_base, ${tile_n * nt};
    add.u32 b_col_${nt}, b_col_${nt}, r_div4;
    {
        .reg .u32 t;
        shl.b32 t, r_mod4, 1;
        add.u32 c_col0_${nt}, warp_n_base, ${tile_n * nt};
        add.u32 c_col0_${nt}, c_col0_${nt}, t;
        add.u32 c_col1_${nt}, c_col0_${nt}, 1;
    }
%  if not n_col_aligned:
    setp.lt.u32 pvalid_bcol_${nt}, b_col_${nt}, ${n};
    setp.lt.u32 pvalid_c0col_${nt}, c_col0_${nt}, ${n};
    setp.lt.u32 pvalid_c1col_${nt}, c_col1_${nt}, ${n};
%  endif
% endfor

    // A thread base: &Ag[0] + lane*sizeof(f64)
    {
        .reg .u64 t64, a_glb_base, lane64;
        mov.u64 a_glb_base, ${kname}_Ag;
        cvta.to.global.u64 a_glb_base, a_glb_base;
        cvt.u64.u32 lane64, lane;
        shl.b64 t64, lane64, 3;
        add.u64 ag_thr_base, a_glb_base, t64;
    }

    {
        .reg .u64 t64, bcol64;
        mul.wide.u32 t64, r_mod4, ${ldb};
        cvt.u64.u32 bcol64, b_col_0;
        add.u64 t64, t64, bcol64;
        shl.b64 t64, t64, 3;
        add.u64 b_thr_base, b_ptr, t64;
    }

    {
        .reg .u64 t64, ccol64;
        mul.wide.u32 t64, r_div4, ${ldc};
        cvt.u64.u32 ccol64, c_col0_0;
        add.u64 t64, t64, ccol64;
        shl.b64 t64, t64, 3;
        add.u64 c_thr_base, c_ptr, t64;
    }

% for mt in range(m_tiles):
%  for mg in range(m_groups):
%   if pm_runtime(mt, mg):
    .reg .pred pm_${mt}_${mg};
    {
        .reg .u32 crow;
        add.u32 crow, r_div4, ${tile_m * mt + 8 * mg};
        setp.lt.u32 pm_${mt}_${mg}, crow, ${m};
    }
%   endif
%  endfor
% endfor

% for nt in range(nn):
%  for mt in range(m_tiles):
%   if beta_zero:
%    for ci in range(c_regs):
    mov.${pftype} c_${nt}_${mt}_${ci}, ${fzero};
%    endfor
%   else:
%    for mg in range(m_groups):
<%
    pm = f'pm_{mt}_{mg}' if pm_runtime(mt, mg) else None
    pvc0 = f'pvalid_c0col_{nt}' if not n_col_aligned else None
    pvc1 = f'pvalid_c1col_{nt}' if not n_col_aligned else None
    needs_zero_init = pm is not None or pvc0 is not None or pvc1 is not None
    c0 = f'c_{nt}_{mt}_{2*mg}'
    c1 = f'c_{nt}_{mt}_{2*mg + 1}'
%>
    {
        .reg .u64 caddr;
        add.u64 caddr, c_thr_base, ${mt * c_mtile_stride + mg * c_mgroup_stride + nt * c_ntile_stride};
%     if needs_zero_init:
        mov.${pftype} ${c0}, ${fzero};
        mov.${pftype} ${c1}, ${fzero};
%     endif
        ${pred_emit(f'ld.weak.global.cg.{pftype} {c0}, [caddr];', pm, pvc0, pred_reg=f'p0_{nt}_{mt}_{mg}')}
        ${pred_emit(f'ld.weak.global.cg.{pftype} {c1}, [caddr + {dwidth_i}];', pm, pvc1, pred_reg=f'p1_{nt}_{mt}_{mg}')}
    }
%    endfor
%   endif
%  endfor
% endfor

% for ki in range(k_tiles):
<%
    ki_used = any(a_tile_nz[mt][ki] for mt in range(m_tiles))
%>
%  if ki_used:
%   for nt in range(nn):
%    for kg in range(k_groups):
<%
    pvb = f'pvalid_bcol_{nt}' if not n_col_aligned else None
    k_tail = (k_rem != 0 and loop.parent.parent.last)
    needs_zero = pvb is not None or k_tail
    pbrow = f'pbrow_{kg}' if k_tail else None
%>
    {
        .reg .u64 baddr;
        add.u64 baddr, b_thr_base, ${ki * b_kiter_stride + kg * b_kgroup_stride + nt * b_ntile_stride};
%     if needs_zero:
        mov.${pftype} b_frag_${nt}_${kg}, ${fzero};
%     endif
%     if k_tail:
        .reg .pred ${pbrow};
        {
            .reg .u32 brow;
            add.u32 brow, r_mod4, ${tile_k * ki + 4 * kg};
            setp.lt.u32 ${pbrow}, brow, ${k};
        }
%     endif
        ${pred_emit(f'ld.weak.global.cg.{pftype} b_frag_{nt}_{kg}, [baddr];', pbrow, pvb, pred_reg=f'pb_{ki}_{nt}_{kg}')}
    }
%    endfor
%   endfor
%  endif
%  for mt in range(m_tiles):
%   if a_tile_nz[mt][ki]:
%    for ai in range(a_regs):
    ld.weak.global.${pftype} a_frag_${ai}, [ag_thr_base + ${a_tile_idx[mt][ki] * frag_stride_bytes + 32 * ai * dwidth_i}];
%    endfor
%    for nt in range(nn):
    mma.sync.aligned.${ptx_mma_shape}.row.col.${pftype}.${pftype}.${pftype}.${pftype}
        ${reg_list(f'c_{nt}_{mt}', c_regs)},
        ${reg_list('a_frag', a_regs)},
        ${reg_list(f'b_frag_{nt}', b_regs)},
        ${reg_list(f'c_{nt}_{mt}', c_regs)};
%    endfor
%   endif
%  endfor
% endfor

% for mt in range(m_tiles):
%  for nt in range(nn):
%   for mg in range(m_groups):
<%
    pm = f'pm_{mt}_{mg}' if pm_runtime(mt, mg) else None
    pvc0 = f'pvalid_c0col_{nt}' if not n_col_aligned else None
    pvc1 = f'pvalid_c1col_{nt}' if not n_col_aligned else None
    c0 = f'c_{nt}_{mt}_{2*mg}'
    c1 = f'c_{nt}_{mt}_{2*mg + 1}'
%>
    {
        .reg .u64 caddr;
        add.u64 caddr, c_thr_base, ${mt * c_mtile_stride + mg * c_mgroup_stride + nt * c_ntile_stride};
        ${pred_emit(f'st.weak.global.{pftype} [caddr], {c0};', pm, pvc0, pred_reg=f'p0s_{nt}_{mt}_{mg}')}
        ${pred_emit(f'st.weak.global.{pftype} [caddr + {dwidth_i}], {c1};', pm, pvc1, pred_reg=f'p1s_{nt}_{mt}_{mg}')}
    }
%   endfor
%  endfor
% endfor

$L_EXIT:
    ret;
}
