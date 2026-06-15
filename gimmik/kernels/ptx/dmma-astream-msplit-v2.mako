<%inherit file='base'/>

.global .align 16 .b64 ${kname}_Ag[${a_elems}] = {
    ${', '.join(a_u64)}
};
.extern .shared .align 128 .b8 ${kname}_dynm[];

.visible .entry ${kname}(.param .u64 b_desc,
                         .param .u64 _c)
.maxntid ${blockx_total}, 1, 1
{
    .reg .u32 tid, warp, lane, r_mod4, r_div4;
    .reg .u32 ctaid_x, n_start_cta, warp_n, warp_m, warp_n_base;
    .reg .u64 bdesc_addr, c_ptr;
    .reg .u64 ag_thr_base, c_thr_base;
    .reg .u32 b_smem, b_thr_base, tma_mbar;
    .reg .pred p_tid0, pwarp_exit, p_load_warp, p_warp_lead;
% for nt in range(nn):
    .reg .u32 b_col_${nt}, c_col0_${nt}, c_col1_${nt};
%  if not n_col_aligned:
    .reg .pred pvalid_bcol_${nt}, pvalid_c0col_${nt}, pvalid_c1col_${nt};
%  endif
% endfor

    ld.param.u64 bdesc_addr, [b_desc];
    ld.param.u64 c_ptr, [_c];
    cvta.to.global.u64 c_ptr, c_ptr;

    mov.u32 tid, %tid.x;
    shr.u32 warp, tid, 5;
    and.b32 lane, tid, 31;
    shr.u32 r_div4, lane, 2;
    and.b32 r_mod4, lane, 3;
    mov.u32 ctaid_x, %ctaid.x;
    mul.lo.u32 n_start_cta, ctaid_x, ${n_per_cta};

    {
        .reg .u32 t;
        div.u32 warp_n, warp, ${msplit};
        mad.lo.u32 t, warp_n, ${msplit}, 0;
        sub.u32 warp_m, warp, t;
    }

    {
        .reg .u32 dynm_base;
        mov.u32 dynm_base, ${kname}_dynm;
        add.u32 b_smem, dynm_base, ${b_off};
        add.u32 tma_mbar, dynm_base, ${tma_mbar_off};
    }

    setp.eq.u32 p_tid0, tid, 0;
    setp.eq.u32 p_load_warp, warp, 0;
    {
        .reg .b32 _elect_lane;
        elect.sync _elect_lane|p_warp_lead, 0xffffffff;
    }

    @p_tid0 mbarrier.init.shared::cta.b64 [tma_mbar], 32;
    @p_tid0 fence.proxy.async.shared::cta;
    bar.sync 0;

    @!p_load_warp bra $L_AFTER_B_TMA;
    {
        @p_warp_lead cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes
            [b_smem], [bdesc_addr, {n_start_cta, 0}], [tma_mbar];
        @p_warp_lead mbarrier.expect_tx.relaxed.cta.shared::cta.b64
            [tma_mbar], ${b_tile_bytes};
        bar.warp.sync 0xffffffff;
        .reg .b64 state;
        .reg .pred p1;
        mbarrier.arrive.shared::cta.b64 state, [tma_mbar];
$L_TMA_WAIT:
        mbarrier.try_wait.shared::cta.b64 p1, [tma_mbar], state, ${mbar_maxwait};
        @!p1 bra.uni $L_TMA_WAIT;
    }
$L_AFTER_B_TMA:
    bar.sync 0;

    {
        .reg .u32 t;
        mul.lo.u32 t, warp_n, ${n_per_warp};
        add.u32 warp_n_base, n_start_cta, t;
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
        .reg .u32 bcol_local, t, row_off;
        mad.lo.u32 bcol_local, warp_n, ${n_per_warp}, r_div4;
        mul.lo.u32 t, bcol_local, ${dwidth_i};
        mul.lo.u32 row_off, r_mod4, ${n_per_cta * dwidth_i};
        add.u32 t, t, row_off;
        add.u32 b_thr_base, b_smem, t;
    }

    {
        .reg .u64 t64, ccol64;
        mul.wide.u32 t64, r_div4, ${ldc};
        cvt.u64.u32 ccol64, c_col0_0;
        add.u64 t64, t64, ccol64;
        shl.b64 t64, t64, 3;
        add.u64 c_thr_base, c_ptr, t64;
    }

% for wm in range(msplit):
<%
    owned_mts = [mt for mt in range(m_tiles) if mt % msplit == wm]
%>
%  if owned_mts:
    {
        .reg .pred p_this_msplit;
        setp.ne.u32 p_this_msplit, warp_m, ${wm};
        @p_this_msplit bra $L_SKIP_MS_${wm};
    }
    {
        .reg .${pftype} a_frag_<${a_regs}>;
%   for nt in range(nn):
        .reg .${pftype} b_frag_${nt}_<${b_regs}>;
%   endfor
%   for nt in range(nn):
%    for mt in owned_mts:
        .reg .${pftype} c_${nt}_${mt}_<${c_regs}>;
%    endfor
%   endfor
%   for mt in owned_mts:
%    for mg in range(m_groups):
%     if pm_runtime(mt, mg):
        .reg .pred pm_${mt}_${mg};
        {
            .reg .u32 crow;
            add.u32 crow, r_div4, ${tile_m * mt + 8 * mg};
            setp.lt.u32 pm_${mt}_${mg}, crow, ${m};
        }
%     endif
%    endfor
%   endfor

%   for nt in range(nn):
%    for mt in owned_mts:
%     if beta_zero:
%      for ci in range(c_regs):
        mov.${pftype} c_${nt}_${mt}_${ci}, ${fzero};
%      endfor
%     else:
%      for mg in range(m_groups):
<%
    pm = f'pm_{mt}_{mg}' if pm_runtime(mt, mg) else None
    c0 = f'c_{nt}_{mt}_{2*mg}'
    c1 = f'c_{nt}_{mt}_{2*mg + 1}'
    cpair = f'{c0}, {c1}'
%>
        {
            .reg .u64 caddr;
            add.u64 caddr, c_thr_base, ${mt * c_mtile_stride + mg * c_mgroup_stride + nt * c_ntile_stride};
%       if pm is not None:
            mov.${pftype} ${c0}, ${fzero};
            mov.${pftype} ${c1}, ${fzero};
%       endif
            ${pred_emit(f'ld.weak.global.cg.v2.{pftype} {{{cpair}}}, [caddr];', pm, pred_reg=f'p01_{wm}_{nt}_{mt}_{mg}', indent=' ' * 12)}
        }
%      endfor
%     endif
%    endfor
%   endfor

%   for ki in range(k_tiles):
%    for nt in range(nn):
%     for kg in range(k_groups):
<%
    pvb = f'pvalid_bcol_{nt}' if not n_col_aligned else None
    k_tail = (k_rem != 0 and loop.parent.parent.last)
    needs_zero = pvb is not None or k_tail
    pbrow = f'pbrow_{kg}' if k_tail else None
%>
        {
            .reg .u32 baddr;
            add.u32 baddr, b_thr_base, ${ki * b_smem_kiter_stride + kg * b_smem_kgroup_stride + nt * b_smem_ntile_stride};
%      if needs_zero:
            mov.${pftype} b_frag_${nt}_${kg}, ${fzero};
%      endif
%      if k_tail:
            .reg .pred ${pbrow};
            {
                .reg .u32 brow;
                add.u32 brow, r_mod4, ${tile_k * ki + 4 * kg};
                setp.lt.u32 ${pbrow}, brow, ${k};
            }
%      endif
            ${pred_emit(f'ld.shared.{pftype} b_frag_{nt}_{kg}, [baddr];', pbrow, pvb, pred_reg=f'pb_{wm}_{ki}_{nt}_{kg}', indent=' ' * 12)}
        }
%     endfor
%    endfor
%    for mt in owned_mts:
%     for ai in range(a_regs):
        ld.weak.global.${pftype} a_frag_${ai}, [ag_thr_base + ${(mt * k_tiles + ki) * frag_stride_bytes + 32 * ai * dwidth_i}];
%     endfor
%     for nt in range(nn):
        mma.sync.aligned.${ptx_mma_shape}.row.col.${pftype}.${pftype}.${pftype}.${pftype}
            ${reg_list(f'c_{nt}_{mt}', c_regs)},
            ${reg_list('a_frag', a_regs)},
            ${reg_list(f'b_frag_{nt}', b_regs)},
            ${reg_list(f'c_{nt}_{mt}', c_regs)};
%     endfor
%    endfor
%   endfor

%   for mt in owned_mts:
%    for nt in range(nn):
%     for mg in range(m_groups):
<%
    pm = f'pm_{mt}_{mg}' if pm_runtime(mt, mg) else None
    c0 = f'c_{nt}_{mt}_{2*mg}'
    c1 = f'c_{nt}_{mt}_{2*mg + 1}'
    cpair = f'{c0}, {c1}'
%>
        {
            .reg .u64 caddr;
            add.u64 caddr, c_thr_base, ${mt * c_mtile_stride + mg * c_mgroup_stride + nt * c_ntile_stride};
            ${pred_emit(f'st.weak.global.v2.{pftype} [caddr], {{{cpair}}};', pm, pred_reg=f'p01s_{wm}_{nt}_{mt}_{mg}', indent=' ' * 12)}
        }
%     endfor
%    endfor
%   endfor
    }
$L_SKIP_MS_${wm}:
%  endif
% endfor

$L_EXIT:
    ret;
}
