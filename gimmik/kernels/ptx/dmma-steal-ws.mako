<%inherit file='base'/>

<%def name="producer_init_setup()">
    // Producer warp: initial A bulk-copy + B load for ctaid_x's work
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

    // Wait on B
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
            add.u32 bcol_g, base_bcol, ${8 * nt};
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
        .reg .${pftype} d_x_${mt}_${nt}, d_y_${mt}_${nt};
        mov.${pftype} d_x_${mt}_${nt}, ${fzero};
        mov.${pftype} d_y_${mt}_${nt}, ${fzero};
%  endfor
% endfor

        .reg .${pftype} a_f;
% for mt in range(m_tiles):
%  for kt in range(k_tiles):
<%
    k_tail = (k_rem != 0 and loop.last)
%>
%   if a_tile_nz[mt][kt]:
        {
            .reg .b32 a_a;
            add.u32 a_a, a_thr_a, ${32 * a_tile_idx[mt][kt] * dwidth_i};
            ld.shared.${pftype} a_f, [a_a];
%    if k_tail:
            .reg .pred pbrow_${mt}_${kt};
            {
                .reg .b32 brow;
                add.u32 brow, base_brow, ${4 * kt};
                setp.lt.u32 pbrow_${mt}_${kt}, brow, ${k};
            }
%    endif
%    for nt in range(nn):
            {
                .reg .b32 b_a, b_row;
                .reg .${pftype} b_f;
                add.u32 b_row, base_brow, ${4 * kt};
                mul.lo.u32 b_row, b_row, ${n_per_cta * dwidth_i};
                add.u32 b_a, b_thr_a_${nt}, b_row;
%     if k_tail:
                mov.${pftype} b_f, ${fzero};
                @pbrow_${mt}_${kt} ld.shared.${pftype} b_f, [b_a];
%     else:
                ld.shared.${pftype} b_f, [b_a];
%     endif
                mma.sync.aligned.m8n8k4.row.col.${pftype}.${pftype}.${pftype}.${pftype}
                    {d_x_${mt}_${nt}, d_y_${mt}_${nt}}, {a_f}, {b_f},
                    {d_x_${mt}_${nt}, d_y_${mt}_${nt}};
            }
%    endfor
        }
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
<%
    row_tail = pm_runtime(mt)
%>
%   if row_tail:
        .reg .pred p_row_${mt};
        {
            .reg .b32 crow;
            add.u32 crow, base_crow, ${8 * mt};
            setp.lt.u32 p_row_${mt}, crow, ${m};
        }
%   endif
%   for nt in range(nn):
        {
            .reg .pred p_st;
            .reg .u32 g_ccol;
            add.u32 g_ccol, base_ccol, ${8 * nt};
            add.u32 g_ccol, g_ccol, n_start_curr;
            setp.lt.u32 p_st, g_ccol, ${n};
%    if row_tail:
            and.pred p_st, p_st, p_row_${mt};
%    endif
            .reg .u64 _c_addr;
            add.u64 _c_addr, c_thr_glob_base, ${(8 * mt * ldc + 8 * nt) * dwidth_i};
            @p_st st.weak.global.v2.${pftype} [_c_addr], {d_x_${mt}_${nt}, d_y_${mt}_${nt}};
        }
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

        // Vector-store {d_x, d_y} pairs to csmem.  M-tail / N-tail OOB rows
        // are dropped by the C tensor map.
%  for mt in range(m_tiles):
%   for nt in range(nn):
        {
            .reg .b32 csaddr;
            add.u32 csaddr, c_thr_smem, ${mt * c_mtile_smem_stride + nt * c_ntile_smem_stride};
            st.shared.v2.${pftype} [csaddr], {d_x_${mt}_${nt}, d_y_${mt}_${nt}};
        }
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

    }

        // Wait for new work and unpack
        {
            .reg .pred p1, p_have;
            .reg .b32 pipe_idx, pipe_phase;
            .reg .b32 pipe_addr, pipe_ready_mbar, pipe_used_mbar, pipe_base;
            and.b32 pipe_idx, pipe_seq, 3;
            shr.u32 pipe_phase, pipe_seq, 2;
            and.b32 pipe_phase, pipe_phase, 1;
            shl.b32 pipe_addr, pipe_idx, 2;
            mov.u32 pipe_base, s_block_pipe;
            add.u32 pipe_addr, pipe_base, pipe_addr;
            shl.b32 pipe_ready_mbar, pipe_idx, 3;
            mov.u32 pipe_base, s_pipe_ready_mbar;
            add.u32 pipe_ready_mbar, pipe_base, pipe_ready_mbar;
            shl.b32 pipe_used_mbar, pipe_idx, 3;
            mov.u32 pipe_base, s_pipe_used_mbar;
            add.u32 pipe_used_mbar, pipe_base, pipe_used_mbar;
$L_WAIT_WNEW_C:
            mbarrier.try_wait.parity.shared::cta.b64 p1, [pipe_ready_mbar], pipe_phase, ${mbar_maxwait};
            @!p1 bra.uni $L_WAIT_WNEW_C;

            ld.shared::cta.s32 block_idx_x, [pipe_addr];
            setp.ge.s32 p_have, block_idx_x, 0;
            selp.b32 work, 1, 0, p_have;

            .reg .b64 _state;
            @p_warp_lead mbarrier.arrive.shared::cta.b64 _state, [pipe_used_mbar];
        }
$L_AFTER_COMPUTE:
</%def>

<%def name="data_warp_body()">
    // --- Data Movement Warp
    @!p_prod bra.uni $L_AFTER_DATA;
    {
        .reg .b32 n_c_store;
        mul.lo.u32 n_c_store, block_idx_x, ${n_per_cta};

        // Wait for new work and unpack
        {
            .reg .pred p1, p_have;
            .reg .b32 pipe_idx, pipe_phase;
            .reg .b32 pipe_addr, pipe_ready_mbar, pipe_used_mbar;
            .reg .b32 pipe_base;
            and.b32 pipe_idx, pipe_seq, 3;
            shr.u32 pipe_phase, pipe_seq, 2;
            and.b32 pipe_phase, pipe_phase, 1;
            shl.b32 pipe_addr, pipe_idx, 2;
            mov.u32 pipe_base, s_block_pipe;
            add.u32 pipe_addr, pipe_base, pipe_addr;
            shl.b32 pipe_ready_mbar, pipe_idx, 3;
            mov.u32 pipe_base, s_pipe_ready_mbar;
            add.u32 pipe_ready_mbar, pipe_base, pipe_ready_mbar;
            shl.b32 pipe_used_mbar, pipe_idx, 3;
            mov.u32 pipe_base, s_pipe_used_mbar;
            add.u32 pipe_used_mbar, pipe_base, pipe_used_mbar;
$L_WAIT_WNEW_D:
            mbarrier.try_wait.parity.shared::cta.b64 p1, [pipe_ready_mbar], pipe_phase, ${mbar_maxwait};
            @!p1 bra.uni $L_WAIT_WNEW_D;

            ld.shared::cta.s32 block_idx_x, [pipe_addr];
            setp.ge.s32 p_have, block_idx_x, 0;
            selp.b32 work, 1, 0, p_have;
            .reg .b64 _state;
            @p_warp_lead mbarrier.arrive.shared::cta.b64 _state, [pipe_used_mbar];
        }

        // TMA loads of next B
        {
            mul.lo.u32 n_start_next, block_idx_x, ${n_per_cta};
            .reg .b32 b_next;
            .reg .pred p_ph;
            setp.ne.u32 p_ph, phase, 0;
            selp.b32 b_next, b1_smem, b2_smem, p_ph;
            @p_warp_lead cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes
                [b_next], [bdesc_addr, {n_start_next, 0}], [s_tma_mbar];
            @p_warp_lead mbarrier.expect_tx.relaxed.cta.shared::cta.b64
                [s_tma_mbar], ${b_tile_bytes};
            @p_warp_lead cp.async.bulk.commit_group;
        }
        bar.warp.sync 0xffffffff;

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

        // Wait for next B to be ready, then signal B and C ready
        {
            .reg .b64 b_state, _bready_state, _c_state;
            .reg .pred p1;
            mbarrier.arrive.shared::cta.b64 b_state, [s_tma_mbar];
$L_WAIT_TMA:
            mbarrier.try_wait.shared::cta.b64 p1, [s_tma_mbar], b_state, ${mbar_maxwait};
            @!p1 bra.uni $L_WAIT_TMA;

            @p_warp_lead mbarrier.arrive.shared::cta.b64 _bready_state, [s_bready_mbar];
        }

        // Keep the producer at most one B tile ahead of compute.
        {
            .reg .pred p1;
            .reg .b32 pipe_idx, pipe_phase, pipe_used_mbar, pipe_base;
            and.b32 pipe_idx, pipe_seq, 3;
            shr.u32 pipe_phase, pipe_seq, 2;
            and.b32 pipe_phase, pipe_phase, 1;
            shl.b32 pipe_used_mbar, pipe_idx, 3;
            mov.u32 pipe_base, s_pipe_used_mbar;
            add.u32 pipe_used_mbar, pipe_base, pipe_used_mbar;
$L_WAIT_PIPE_USED_D:
            mbarrier.try_wait.parity.shared::cta.b64 p1, [pipe_used_mbar], pipe_phase, ${mbar_maxwait};
            @!p1 bra.uni $L_WAIT_PIPE_USED_D;
        }
    }
$L_AFTER_DATA:
</%def>

<%def name="ctrl_warp_body()">
    // --- Controller Warp
    @!p_steal bra.uni $L_AFTER_CTRL;
    {
        .reg .pred p1, p2, p_canc;
        .reg .b64 _state;
        .reg .b128 resp;
        .reg .b32 pipe_idx, pipe_phase;
        .reg .b32 pipe_addr, pipe_ready_mbar, pipe_used_mbar, pipe_base;
        and.b32 pipe_idx, pipe_seq, 3;
        shr.u32 pipe_phase, pipe_seq, 2;
        and.b32 pipe_phase, pipe_phase, 1;
        shl.b32 pipe_addr, pipe_idx, 2;
        mov.u32 pipe_base, s_block_pipe;
        add.u32 pipe_addr, pipe_base, pipe_addr;
        shl.b32 pipe_ready_mbar, pipe_idx, 3;
        mov.u32 pipe_base, s_pipe_ready_mbar;
        add.u32 pipe_ready_mbar, pipe_base, pipe_ready_mbar;
        shl.b32 pipe_used_mbar, pipe_idx, 3;
        mov.u32 pipe_base, s_pipe_used_mbar;
        add.u32 pipe_used_mbar, pipe_base, pipe_used_mbar;

        // A slot is reusable only after every consumer has released it.
        setp.ge.u32 p2, pipe_seq, 4;
        @!p2 bra.uni $L_PIPE_FREE;
        .reg .b32 pipe_used_phase;
        xor.b32 pipe_used_phase, pipe_phase, 1;
$L_WAIT_WUSED:
        mbarrier.try_wait.parity.shared::cta.b64 p1, [pipe_used_mbar], pipe_used_phase, ${mbar_maxwait};
        @!p1 bra.uni $L_WAIT_WUSED;
$L_PIPE_FREE:

        @p_warp_lead fence.proxy.async.shared::cta;
        @p_warp_lead clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128
            [s_wid], [s_steal_mbar];
        @p_warp_lead mbarrier.arrive.expect_tx.shared::cta.b64
            _state, [s_steal_mbar], 16;

$L_WAIT_STEAL:
        mbarrier.try_wait.parity.shared::cta.b64 p1, [s_steal_mbar], phase, ${mbar_maxwait};
        @!p1 bra.uni $L_WAIT_STEAL;

        // Query if there's new work
        ld.shared::cta.b128 resp, [s_wid];
        clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p_canc, resp;
        @p_canc clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 block_idx_x, resp;
        @!p_canc mov.u32 block_idx_x, 0xffffffff;
        selp.b32 work, 1, 0, p_canc;

        // Publish this slot.
        @p_warp_lead st.shared::cta.u32 [pipe_addr], block_idx_x;
        @p_warp_lead mbarrier.arrive.shared::cta.b64 _state, [pipe_ready_mbar];
    }
$L_AFTER_CTRL:
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
    .shared .align 16 .b8 s_wid[16];
    .shared .align 16 .b32 s_block_pipe[4];
    .shared .align 8 .b64 s_tma_mbar;
    .shared .align 8 .b64 s_bready_mbar;
    .shared .align 8 .b64 s_cready_mbar;
    .shared .align 8 .b64 s_cstored_mbar;
    .shared .align 8 .b64 s_steal_mbar;
    .shared .align 8 .b64 s_pipe_ready_mbar[4];
    .shared .align 8 .b64 s_pipe_used_mbar[4];
    .reg .b32 tid, warp, lane, phase, ctaid_x;
    .reg .b32 base_brow, base_bcol, base_crow, base_ccol;
    .reg .b32 work, block_idx_x, n_start_curr, n_start_next;
    .reg .b32 pipe_seq;
    .reg .u64 bdesc_addr, cdesc_addr;
    .reg .b32 a_smem, b1_smem, b2_smem, c_smem;
    .reg .pred p_compute, p_prod, p_steal;
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
    setp.eq.u32 p_steal, warp, ${steal_warp};

    {
        .reg .b32 _elect_lane;
        elect.sync _elect_lane|p_warp_lead, 0xffffffff;
    }

    // mbarrier init (tid 0 only); pre-arrive csmem_free so compute iter 0
    // can write csmem immediately.
    {
        .reg .pred p_init;
        setp.eq.u32 p_init, tid, 0;
        .reg .b64 _state;
        @p_init mbarrier.init.shared::cta.b64 [s_tma_mbar], 32;
        @p_init mbarrier.init.shared::cta.b64 [s_bready_mbar], 1;
        @p_init mbarrier.init.shared::cta.b64 [s_cready_mbar], 1;
        @p_init mbarrier.init.shared::cta.b64 [s_cstored_mbar], 1;
        @p_init mbarrier.init.shared::cta.b64 [s_steal_mbar], 1;
% for pi in range(4):
        @p_init mbarrier.init.shared::cta.b64 [s_pipe_ready_mbar + ${8 * pi}], 1;
        @p_init mbarrier.init.shared::cta.b64 [s_pipe_used_mbar + ${8 * pi}], ${n_comp_warps + 1};
% endfor
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
    mov.u32 phase, 0;
    mov.u32 pipe_seq, 0;

$L_LOOP:
    setp.eq.u32 p_done, work, 0;
    @p_done bra.uni $L_EXIT;

    mul.lo.u32 n_start_curr, block_idx_x, ${n_per_cta};
    ${compute_warp_body()}

    ${data_warp_body()}

    ${ctrl_warp_body()}

    add.u32 pipe_seq, pipe_seq, 1;
    xor.b32 phase, phase, 1;
    bra.uni $L_LOOP;

$L_EXIT:
    ret;
}
