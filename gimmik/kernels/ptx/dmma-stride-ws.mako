<%inherit file='base'/>

<%def name="producer_init_setup()">
    // Producer warp: initial A bulk-copy + first B load
    @!p_prod bra.uni $L_AFTER_INIT_B;
    {
        .reg .b32 n_start0;
        .reg .u64 a_glb;
        mul.lo.u32 n_start0, ctaid_x, ${n_per_cta};
        mov.u64 a_glb, ${kname}_Ag;
        cvta.to.global.u64 a_glb, a_glb;
        @p_warp_lead cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes
            [a_smem], [a_glb], ${a_elems * dwidth_i}, [s_tma_mbar];
        @p_warp_lead cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes
            [b1_smem], [bdesc_addr, {n_start0, 0}], [s_tma_mbar];
        @p_warp_lead mbarrier.expect_tx.relaxed.cta.shared::cta.b64
            [s_tma_mbar], ${b_tile_bytes + a_elems * dwidth_i};
        bar.warp.sync 0xffffffff;
        .reg .b64 state;
        .reg .pred p1;
        mbarrier.arrive.shared::cta.b64 state, [s_tma_mbar];
$L_TMA_INIT_W:
        mbarrier.try_wait.shared::cta.b64 p1, [s_tma_mbar], state, ${mbar_maxwait};
        @!p1 bra.uni $L_TMA_INIT_W;
        .reg .b64 _state2;
        @p_warp_lead mbarrier.arrive.shared::cta.b64 _state2, [s_bready_mbar];
    }
$L_AFTER_INIT_B:
</%def>

<%def name="compute_warp_body()">
    // --- Compute Warps
    @!p_compute bra.uni $L_AFTER_COMPUTE;

    // Wait on the current B tile.
    {
        .reg .pred p1;
$L_WAIT_BRDY:
        mbarrier.try_wait.parity.shared::cta.b64 p1, [s_bready_mbar], phase, ${mbar_maxwait};
        @!p1 bra.uni $L_WAIT_BRDY;
    }

    // MMA
    {
        .reg .b32 b_sm_a;
        .reg .pred p_ph;
        setp.ne.u32 p_ph, phase, 0;
        selp.b32 b_sm_a, b2_smem, b1_smem, p_ph;

        .reg .b32 a_thr_a;
        {
            .reg .b32 t;
            shl.b32 t, lane, 3;
            add.u32 a_thr_a, a_smem, t;
        }
% for nt in range(nn):
        .reg .b32 b_thr_a_${nt};
        {
            .reg .b32 bcol_g, t_off;
            add.u32 bcol_g, base_bcol, ${tile_n * nt};
            shl.b32 t_off, bcol_g, 3;
            add.u32 b_thr_a_${nt}, b_sm_a, t_off;
        }
% endfor

% if beta_zero:
        // beta=0: skip shared-staging entirely; compute warps store MMA
        // outputs straight to global C with N-tail predication.
        .reg .u64 c_glob_addr;
        ld.param.u64 c_glob_addr, [c_desc];
        cvta.to.global.u64 c_glob_addr, c_glob_addr;
% else:
        .reg .b32 c_thr_smem;
        {
            .reg .b32 t1, ccol_b;
            mul.lo.u32 t1, base_crow, ${n_per_cta * dwidth_i};
            shl.b32 ccol_b, base_ccol, 3;
            add.u32 c_thr_smem, c_smem, t1;
            add.u32 c_thr_smem, c_thr_smem, ccol_b;
        }
% endif

        // Zero accumulators
% for mt in range(m_tiles):
%  for nt in range(nn):
        .reg .${pftype} d_${mt}_${nt}_<${c_regs}>;
%   for ci in range(c_regs):
        mov.${pftype} d_${mt}_${nt}_${ci}, ${fzero};
%   endfor
%  endfor
% endfor

        .reg .${pftype} a_frag_<${a_regs}>;
% for nt in range(nn):
        .reg .${pftype} b_frag_${nt}_<${b_regs}>;
% endfor
% for kt in range(k_tiles):
<%
    kt_used = any(a_tile_nz[mt][kt] for mt in range(m_tiles))
%>
%  if kt_used:
%   for nt in range(nn):
%    for kg in range(k_groups):
<%
    k_tail = (k_rem != 0 and loop.parent.parent.last)
    pbrow = f'pbrow_{nt}_{kg}' if k_tail else None
%>
        {
            .reg .b32 b_a, b_row, b_off;
            add.u32 b_row, base_brow, ${tile_k * kt + 4 * kg};
            mul.lo.u32 b_off, b_row, ${n_per_cta * dwidth_i};
            add.u32 b_a, b_thr_a_${nt}, b_off;
%     if k_tail:
            .reg .pred ${pbrow};
            setp.lt.u32 ${pbrow}, b_row, ${k};
            mov.${pftype} b_frag_${nt}_${kg}, ${fzero};
            @${pbrow} ld.shared.${pftype} b_frag_${nt}_${kg}, [b_a];
%     else:
            ld.shared.${pftype} b_frag_${nt}_${kg}, [b_a];
%     endif
        }
%    endfor
%   endfor
%  endif
%  for mt in range(m_tiles):
%   if a_tile_nz[mt][kt]:
%    for ai in range(a_regs):
        ld.shared.${pftype} a_frag_${ai}, [a_thr_a + ${a_tile_idx[mt][kt] * frag_stride_bytes + 32 * ai * dwidth_i}];
%    endfor
%    for nt in range(nn):
        mma.sync.aligned.${ptx_mma_shape}.row.col.${pftype}.${pftype}.${pftype}.${pftype}
            ${reg_list(f'd_{mt}_{nt}', c_regs)},
            ${reg_list('a_frag', a_regs)},
            ${reg_list(f'b_frag_{nt}', b_regs)},
            ${reg_list(f'd_{mt}_{nt}', c_regs)};
%    endfor
%   endif
%  endfor
% endfor

% if beta_zero:
        .reg .u64 c_thr_glob_base;
        {
            .reg .u32 thr_col_off, thr_addr_off_lo;
            add.u32 thr_col_off, base_ccol, n_start_curr;
            mad.lo.u32 thr_addr_off_lo, base_crow, ${ldc}, thr_col_off;
            .reg .u64 thr_byte_off;
            mul.wide.u32 thr_byte_off, thr_addr_off_lo, ${dwidth_i};
            add.u64 c_thr_glob_base, c_glob_addr, thr_byte_off;
        }
%  for mt in range(m_tiles):
%   for mg in range(m_groups):
<%
    row_tail = pm_runtime(mt, mg)
%>
%    if row_tail:
        .reg .pred p_row_${mt}_${mg};
        {
            .reg .b32 crow;
            add.u32 crow, base_crow, ${tile_m * mt + 8 * mg};
            setp.lt.u32 p_row_${mt}_${mg}, crow, ${m};
        }
%    endif
%    for nt in range(nn):
        {
            .reg .pred p_st;
            .reg .u32 g_ccol;
            add.u32 g_ccol, base_ccol, ${tile_n * nt};
            add.u32 g_ccol, g_ccol, n_start_curr;
            setp.lt.u32 p_st, g_ccol, ${n};
%     if row_tail:
            and.pred p_st, p_st, p_row_${mt}_${mg};
%     endif
            .reg .u64 _c_addr;
            add.u64 _c_addr, c_thr_glob_base, ${((tile_m * mt + 8 * mg) * ldc + tile_n * nt) * dwidth_i};
            @p_st st.weak.global.v2.${pftype} [_c_addr], {d_${mt}_${nt}_${2*mg}, d_${mt}_${nt}_${2*mg + 1}};
        }
%    endfor
%   endfor
%  endfor
% else:
        // Wait until producer's prev-iter TMA-store of C has drained.
        {
            .reg .pred p1;
$L_WAIT_CSTORE:
            mbarrier.try_wait.parity.shared::cta.b64 p1, [s_cstored_mbar], phase, ${mbar_maxwait};
            @!p1 bra.uni $L_WAIT_CSTORE;
        }

        // Vector-store accumulator pairs to csmem. M-tail / N-tail OOB rows
        // are dropped by the C tensor map.
%  for mt in range(m_tiles):
%   for mg in range(m_groups):
%    for nt in range(nn):
        {
            .reg .b32 csaddr;
            add.u32 csaddr, c_thr_smem, ${mt * c_mtile_smem_stride + mg * c_mgroup_smem_stride + nt * c_ntile_smem_stride};
            st.shared.v2.${pftype} [csaddr], {d_${mt}_${nt}_${2*mg}, d_${mt}_${nt}_${2*mg + 1}};
        }
%    endfor
%   endfor
%  endfor
% endif

% if not beta_zero:
        bar.sync 1, ${comp_threads};
        fence.proxy.async.shared::cta;
        {
            .reg .b64 _state;
            @p_tid0 mbarrier.arrive.shared::cta.b64 _state, [s_cready_mbar];
        }
% endif

        // Match the stealing kernel's "work used" point: retire the current
        // phase only after compute-side work for the tile is complete.
        {
            .reg .b64 _bconsumed_state;
            @p_warp_lead mbarrier.arrive.shared::cta.b64 _bconsumed_state, [s_bconsumed_mbar];
        }

    }
$L_AFTER_COMPUTE:
</%def>

<%def name="data_warp_body()">
    // --- Data Movement Warp
    @!p_prod bra.uni $L_AFTER_DATA;
    {
        .reg .b32 n_c_store;
        mul.lo.u32 n_c_store, block_idx_x, ${n_per_cta};

        .reg .pred p_next_work;
        setp.ne.u32 p_next_work, next_work, 0;

        // Issue the next B load into the alternate buffer before retiring
        // the current phase.
        .reg .b64 b_state;
        @!p_next_work bra.uni $L_SKIP_NEXT_B_ISSUE;
        {
            mul.lo.u32 n_start_next, next_block_idx_x, ${n_per_cta};
            .reg .b32 b_next;
            .reg .pred p_ph;
            setp.ne.u32 p_ph, phase, 0;
            selp.b32 b_next, b1_smem, b2_smem, p_ph;
            @p_warp_lead cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes
                [b_next], [bdesc_addr, {n_start_next, 0}], [s_tma_mbar];
            @p_warp_lead mbarrier.expect_tx.relaxed.cta.shared::cta.b64
                [s_tma_mbar], ${b_tile_bytes};
            @p_warp_lead cp.async.bulk.commit_group;
            bar.warp.sync 0xffffffff;
            mbarrier.arrive.shared::cta.b64 b_state, [s_tma_mbar];
        }
$L_SKIP_NEXT_B_ISSUE:

% if not beta_zero:
        // TMA reduce+store of C (beta=1 only; beta=0 uses direct global
        // stores from compute warps, so the producer does no C work).
        {
            .reg .pred p1;
            .reg .b64 _c_state;
$L_WAIT_CRDY:
            mbarrier.try_wait.parity.shared::cta.b64 p1, [s_cready_mbar], phase, ${mbar_maxwait};
            @!p1 bra.uni $L_WAIT_CRDY;
            @p_warp_lead cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group
                [cdesc_addr, {n_c_store, 0}], [c_smem];
            @p_warp_lead cp.async.bulk.commit_group;
            @p_warp_lead cp.async.bulk.wait_group 0;
            @p_warp_lead mbarrier.arrive.shared::cta.b64 _c_state, [s_cstored_mbar];
        }
% endif

        // Wait for the next B load to complete, then mark it ready for the
        // next compute iteration.
        @!p_next_work bra.uni $L_SKIP_NEXT_B_READY;
        {
            .reg .b64 _bready_state;
            .reg .pred p1;
$L_WAIT_TMA:
            mbarrier.try_wait.shared::cta.b64 p1, [s_tma_mbar], b_state, ${mbar_maxwait};
            @!p1 bra.uni $L_WAIT_TMA;

$L_WAIT_BCONSUMED:
            mbarrier.try_wait.parity.shared::cta.b64 p1, [s_bconsumed_mbar], phase, ${mbar_maxwait};
            @!p1 bra.uni $L_WAIT_BCONSUMED;

            @p_warp_lead mbarrier.arrive.shared::cta.b64 _bready_state, [s_bready_mbar];
        }
$L_SKIP_NEXT_B_READY:
    }
$L_AFTER_DATA:
</%def>

.global .align 16 .b64 ${kname}_Ag[${a_elems}] = {
    ${', '.join(a_u64)}
};
.visible .entry ${kname}(.param .u64 b_desc,
                         .param .u64 c_desc)
.maxntid ${blockx_total}, 1, 1
{
    .shared .align 128 .b8 s_b1[${b_tile_bytes}];
    .shared .align 128 .b8 s_b2[${b_tile_bytes}];
% if not beta_zero:
    .shared .align 128 .b8 s_c[${8 * m_tiles * tile_m * n_per_cta}];
% endif
    .shared .align 128 .b8 s_a[${a_elems * dwidth_i}];
    .shared .align 8 .b64 s_tma_mbar;
    .shared .align 8 .b64 s_bready_mbar;
    .shared .align 8 .b64 s_bconsumed_mbar;
    .shared .align 8 .b64 s_cready_mbar;
    .shared .align 8 .b64 s_cstored_mbar;
    .reg .b32 tid, warp, lane, phase, ctaid_x;
    .reg .b32 base_brow, base_bcol, base_crow, base_ccol;
    .reg .b32 work, next_work, iter, next_iter;
    .reg .b32 block_idx_x, next_block_idx_x, n_start_curr, n_start_next;
    .reg .u64 bdesc_addr, cdesc_addr;
    .reg .b32 a_smem, b1_smem, b2_smem, c_smem;
    .reg .pred p_compute, p_prod;
    .reg .pred p_warp_lead;
    .reg .pred p_done;
    .reg .pred p_tid0;

    mov.u32 tid, %tid.x;
    shr.u32 warp, tid, 5;
    and.b32 lane, tid, 31;
    mov.u32 ctaid_x, %ctaid.x;

    mov.u32 b1_smem, s_b1;
    mov.u32 b2_smem, s_b2;
% if not beta_zero:
    mov.u32 c_smem, s_c;
% endif
    mov.u32 a_smem, s_a;

    ld.param.u64 bdesc_addr, [b_desc];
    ld.param.u64 cdesc_addr, [c_desc];

    setp.eq.u32 p_tid0, tid, 0;

    setp.lt.u32 p_compute, warp, ${n_comp_warps};
    setp.eq.u32 p_prod, warp, ${prod_warp};

    {
        .reg .b32 _elect_lane;
        elect.sync _elect_lane|p_warp_lead, 0xffffffff;
    }

    // mbarrier init (tid 0 only); pre-arrive cstored so compute iter 0
    // can write csmem immediately when beta != 0.
    {
        .reg .pred p_init;
        setp.eq.u32 p_init, tid, 0;
        .reg .b64 _state;
        @p_init mbarrier.init.shared::cta.b64 [s_tma_mbar], 32;
        @p_init mbarrier.init.shared::cta.b64 [s_bready_mbar], 1;
        @p_init mbarrier.init.shared::cta.b64 [s_bconsumed_mbar], ${n_comp_warps};
        @p_init mbarrier.init.shared::cta.b64 [s_cready_mbar], 1;
        @p_init mbarrier.init.shared::cta.b64 [s_cstored_mbar], 1;
        @p_init mbarrier.arrive.shared::cta.b64 _state, [s_cstored_mbar];
        @p_init fence.proxy.async.shared::cta;
    }
    bar.sync 0;

    // Compute-warp lane geometry
    {
        .reg .b32 t, w_n_base;
        and.b32 base_brow, lane, 3;
        shr.u32 base_crow, lane, 2;
        mul.lo.u32 w_n_base, warp, ${n_per_warp};
        add.u32 base_bcol, base_crow, w_n_base;
        shl.b32 t, base_brow, 1;
        add.u32 base_ccol, t, w_n_base;
    }

    ${producer_init_setup()}

    mov.u32 block_idx_x, ctaid_x;
    mov.u32 work, 1;
    mov.u32 iter, 0;
    mov.u32 phase, 0;

    // Bounded grid-stride loop: block_idx_x = ctaid.x + iter*grid_stride.
$L_LOOP:
    setp.eq.u32 p_done, work, 0;
    @p_done bra.uni $L_EXIT;

    mul.lo.u32 n_start_curr, block_idx_x, ${n_per_cta};

    {
        .reg .pred p_iter, p_block, p_next;
        add.u32 next_iter, iter, 1;
        add.u32 next_block_idx_x, block_idx_x, ${grid_stride};
        setp.lt.u32 p_iter, next_iter, ${stride_iters};
        setp.lt.u32 p_block, next_block_idx_x, ${work_blocks};
        and.pred p_next, p_iter, p_block;
        selp.b32 next_work, 1, 0, p_next;
    }

    ${compute_warp_body()}

    ${data_warp_body()}

    mov.u32 block_idx_x, next_block_idx_x;
    mov.u32 work, next_work;
    mov.u32 iter, next_iter;
    xor.b32 phase, phase, 1;
    bra.uni $L_LOOP;

$L_EXIT:
    ret;
}
