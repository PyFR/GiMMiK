<%inherit file='base'/>
<%
assert dtype == "double"
assert n is not None and ldb is not None and ldc is not None
mbar_maxwait = '0x989680'
%>

.global .align 16 .b64 ${kname}_Ag[${a_elems}] = {
    ${', '.join(a_u64)}
};
.extern .shared .align 128 .b8 ${kname}_dynm[];
.const  .align 64  .b8 ${kname}_bdesc[128];
.const  .align 64  .b8 ${kname}_cdesc[128];

.visible .entry ${kname}(.param .u64 _b,
                         .param .u64 _c)
.maxntid ${blockx_total}, 1, 1
{
    .reg .b32 tid, warp, lane, phase, ctaid_x;
    .reg .b32 base_brow, base_bcol, base_crow, base_ccol;
    .reg .b32 work, block_idx_x, n_start_curr, n_start_next;
    .reg .u64 bdesc_addr, cdesc_addr;
    .reg .b32 a_smem, b1_smem, b2_smem, c_smem;
    .reg .b32 tma_mbar, wid_new_mbar, bready_mbar, cready_mbar, cstored_mbar, steal_mbar;
    .reg .b32 wid_used_mbar, wid_smem;
    .reg .pred p_compute, p_prod, p_steal;
    .reg .pred p_warp_lead;
    .reg .pred p_done;
    .reg .pred p_tid0;

    mov.u32 tid, %tid.x;
    shr.u32 warp, tid, 5;
    and.b32 lane, tid, 31;
    mov.u32 ctaid_x, %ctaid.x;

    .reg .b32 dynm_base;
    mov.u32 dynm_base, ${kname}_dynm;
    add.u32 b1_smem, dynm_base, ${b1_off};
    add.u32 b2_smem, dynm_base, ${b2_off};
    add.u32 c_smem, dynm_base, ${c_off};
    add.u32 a_smem, dynm_base, ${a_off};
    add.u32 wid_smem, dynm_base, ${wid_off};

    add.u32 tma_mbar, dynm_base, ${tma_mbar_off};
    add.u32 bready_mbar, dynm_base, ${bready_mbar_off};
    add.u32 cready_mbar, dynm_base, ${cready_mbar_off};
    add.u32 cstored_mbar, dynm_base, ${cstored_mbar_off};
    add.u32 steal_mbar, dynm_base, ${steal_mbar_off};
    add.u32 wid_new_mbar, dynm_base, ${wid_new_mbar_off};
    add.u32 wid_used_mbar, dynm_base, ${wid_used_mbar_off};

    cvta.const.u64 bdesc_addr, ${kname}_bdesc;
    cvta.const.u64 cdesc_addr, ${kname}_cdesc;

    setp.eq.u32 p_tid0, tid, 0;

    setp.lt.u32 p_compute, warp, ${n_comp_warps};
    setp.eq.u32 p_prod,    warp, ${prod_warp};
    setp.eq.u32 p_steal,   warp, ${steal_warp};

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
        @p_init mbarrier.init.shared::cta.b64 [tma_mbar], 32;
        @p_init mbarrier.init.shared::cta.b64 [bready_mbar], 1;
        @p_init mbarrier.init.shared::cta.b64 [cready_mbar], 1;
        @p_init mbarrier.init.shared::cta.b64 [cstored_mbar], 1;
        @p_init mbarrier.init.shared::cta.b64 [steal_mbar], 1;
        @p_init mbarrier.init.shared::cta.b64 [wid_used_mbar], ${n_comp_warps + 1};
        @p_init mbarrier.init.shared::cta.b64 [wid_new_mbar], 1;
        @p_init mbarrier.arrive.shared::cta.b64 _state, [cstored_mbar];
        @p_init fence.proxy.async.shared::cta;
    }
    bar.sync 0;

    // Cooperative copy A: .global -> a_smem (ld.global.nc.v2.f64)
    {
        .reg .u64 a_glb_base;
        .reg .b32 pidx;
        .reg .f64 av0, av1;
        mov.u64            a_glb_base, ${kname}_Ag;
        cvta.to.global.u64 a_glb_base, a_glb_base;
% for ci in range(copy_v2_iters):
<%
    base_pair = ci * blockx_total
    is_last = ci == copy_v2_iters - 1
    pairs_this = min(blockx_total, a_pairs - base_pair)
    needs_guard = is_last and pairs_this < blockx_total
%>
        {
            .reg .u64 ofs64, gaddr;
            .reg .b32 saddr;
            add.u32      pidx, tid, ${base_pair};
% if needs_guard:
            .reg .pred p_load;
            setp.lt.u32  p_load, pidx, ${a_pairs};
% endif
            mul.wide.u32 ofs64, pidx, 16;
            add.u64      gaddr, a_glb_base, ofs64;
            cvt.u32.u64  saddr, ofs64;
            add.u32      saddr, saddr, a_smem;
% if needs_guard:
            @p_load ld.global.nc.v2.f64 {av0, av1}, [gaddr];
            @p_load st.shared.v2.f64    [saddr], {av0, av1};
% else:
            ld.global.nc.v2.f64 {av0, av1}, [gaddr];
            st.shared.v2.f64    [saddr], {av0, av1};
% endif
        }
% endfor
% if a_pairs_tail:
        {
            .reg .pred p_tail;
            .reg .u64 gaddr;
            .reg .b32 saddr;
            .reg .f64 v;
            setp.eq.u32 p_tail, tid, 0;
            add.u64 gaddr, a_glb_base, ${(a_elems - 1) * 8};
            mov.u32 saddr, ${(a_elems - 1) * 8};
            add.u32 saddr, saddr, a_smem;
            @p_tail ld.global.nc.f64 v, [gaddr];
            @p_tail st.shared.f64    [saddr], v;
        }
% endif
    }
    bar.sync 0;

    // Compute-warp lane geometry (cheap; all warps execute uniformly)
    {
        .reg .b32 t, w_n_base;
        and.b32    base_brow, lane, 3;
        shr.u32    base_crow, lane, 2;
        mul.lo.u32 w_n_base, warp, ${n_per_warp};
        add.u32    base_bcol, base_crow, w_n_base;
        shl.b32    t, base_brow, 1;
        add.u32    base_ccol, t, w_n_base;
    }

    // Producer warp: initial B load for ctaid_x's work
    @!p_prod bra.uni $L_AFTER_INIT_B;
    {
        .reg .b32 n_start0;
        mul.lo.u32 n_start0, ctaid_x, ${n_per_cta};
        @p_warp_lead cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes
            [b1_smem], [bdesc_addr, {n_start0, 0}], [tma_mbar];
        @p_warp_lead mbarrier.expect_tx.relaxed.cta.shared::cta.b64
            [tma_mbar], ${b_tile_bytes};
        bar.warp.sync 0xffffffff;
        .reg .b64 state;
        .reg .pred p1;
        mbarrier.arrive.shared::cta.b64 state, [tma_mbar];
$L_TMA_INIT_W:
        mbarrier.try_wait.shared::cta.b64 p1, [tma_mbar], state, ${mbar_maxwait};
        @!p1 bra.uni $L_TMA_INIT_W;
        .reg .b64 _state2;
        @p_warp_lead mbarrier.arrive.shared::cta.b64 _state2, [bready_mbar];
    }
$L_AFTER_INIT_B:

    mov.u32 block_idx_x, ctaid_x;
    mov.u32 work, 1;
    mov.u32 phase, 0;

$L_LOOP:
    setp.eq.u32 p_done, work, 0;
    @p_done bra.uni $L_EXIT;

    mul.lo.u32 n_start_curr, block_idx_x, ${n_per_cta};

    // --- Compute Warps
    @!p_compute bra.uni $L_AFTER_COMPUTE;

    // Wait on B
    {
        .reg .pred p1;
$L_WAIT_BRDY:
        mbarrier.try_wait.parity.shared::cta.b64 p1, [bready_mbar], phase, ${mbar_maxwait};
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
            add.u32     bcol_g, base_bcol, ${8 * nt};
            shl.b32     t_off,  bcol_g, 3;
            add.u32     b_thr_a_${nt}, b_sm_a, t_off;
        }
% endfor

        .reg .b32 c_thr_smem;
        {
            .reg .b32 t1, ccol_b;
            mul.lo.u32  t1,     base_crow, ${n_per_cta * 8};
            shl.b32     ccol_b, base_ccol, 3;
            add.u32     c_thr_smem, c_smem, t1;
            add.u32     c_thr_smem, c_thr_smem, ccol_b;
        }

        // Zero accumulators
% for mt in range(m_tiles):
% for nt in range(nn):
        .reg .f64 d_x_${mt}_${nt}, d_y_${mt}_${nt};
        mov.f64 d_x_${mt}_${nt}, 0d0000000000000000;
        mov.f64 d_y_${mt}_${nt}, 0d0000000000000000;
% endfor
% endfor

        .reg .f64 a_f;
% for mt in range(m_tiles):
% for kt in range(k_iters):
<%
    k_tail = (k_rem != 0 and kt == k_iters - 1)
%>
        {
            .reg .b32 a_a;
            add.u32       a_a, a_thr_a, ${(kt * 32 + mt * 32 * k_iters) * 8};
            ld.shared.f64 a_f, [a_a];
% if k_tail:
            .reg .pred pbrow_${mt}_${kt};
            {
                .reg .b32 brow;
                add.u32     brow, base_brow, ${4 * kt};
                setp.lt.u32 pbrow_${mt}_${kt}, brow, ${k};
            }
% endif
% for nt in range(nn):
            {
                .reg .b32 b_a, b_row;
                .reg .f64 b_f;
                add.u32       b_row, base_brow, ${4 * kt};
                mul.lo.u32    b_row, b_row, ${n_per_cta * 8};
                add.u32       b_a, b_thr_a_${nt}, b_row;
% if k_tail:
                mov.f64 b_f, 0d0000000000000000;
                @pbrow_${mt}_${kt} ld.shared.f64 b_f, [b_a];
% else:
                ld.shared.f64 b_f, [b_a];
% endif
                mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64
                    {d_x_${mt}_${nt}, d_y_${mt}_${nt}}, {a_f}, {b_f},
                    {d_x_${mt}_${nt}, d_y_${mt}_${nt}};
            }
% endfor
        }
% endfor
% endfor

        // Wait until producer's prev-iter TMA-store of C has drained.
        {
            .reg .pred p1;
$L_WAIT_CSTORE:
            mbarrier.try_wait.parity.shared::cta.b64 p1, [cstored_mbar], phase, ${mbar_maxwait};
            @!p1 bra.uni $L_WAIT_CSTORE;
        }

        // Vector-store {d_x, d_y} pairs to csmem.  M-tail / N-tail OOB rows
        // are dropped by the C tensor map.
% for mt in range(m_tiles):
% for nt in range(nn):
        {
            .reg .b32 csaddr;
            add.u32 csaddr, c_thr_smem, ${mt * c_mtile_smem_stride + nt * c_ntile_smem_stride};
            st.shared.v2.f64 [csaddr], {d_x_${mt}_${nt}, d_y_${mt}_${nt}};
        }
% endfor
% endfor

        bar.sync 1, ${comp_threads};
        fence.proxy.async.shared::cta;
        {
            .reg .b64 _state;
            @p_tid0 mbarrier.arrive.shared::cta.b64 _state, [cready_mbar];
        }

        // Wait for new work and unpack
        {
            .reg .pred p1, p_canc;
            .reg .b128 resp;
$L_WAIT_WNEW_C:
            mbarrier.try_wait.parity.shared::cta.b64 p1, [wid_new_mbar], phase, ${mbar_maxwait};
            @!p1 bra.uni $L_WAIT_WNEW_C;

            ld.shared::cta.b128 resp, [wid_smem];
            clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p_canc, resp;
            @p_canc clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 block_idx_x, resp;
            selp.b32 work, 1, 0, p_canc;

            .reg .b64 _state;
            @p_warp_lead mbarrier.arrive.shared::cta.b64 _state, [wid_used_mbar];
        }
    }
$L_AFTER_COMPUTE:

    // --- Data Movement Warp
    @!p_prod bra.uni $L_AFTER_DATA;
    {
        .reg .b32 n_c_store;
        mul.lo.u32 n_c_store, block_idx_x, ${n_per_cta};

        // Wait for new work and unpack
        {
            .reg .pred p1, p_canc;
            .reg .b128 resp;
$L_WAIT_WNEW_D:
            mbarrier.try_wait.parity.shared::cta.b64 p1, [wid_new_mbar], phase, ${mbar_maxwait};
            @!p1 bra.uni $L_WAIT_WNEW_D;

            ld.shared::cta.b128 resp, [wid_smem];
            clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p_canc, resp;
            @p_canc clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 block_idx_x, resp;
            selp.b32 work, 1, 0, p_canc;
            .reg .b64 _state;
            @p_warp_lead mbarrier.arrive.shared::cta.b64 _state, [wid_used_mbar];
        }

        // TMA loads of next B
        {
            mul.lo.u32 n_start_next, block_idx_x, ${n_per_cta};
            .reg .b32 b_next;
            .reg .pred p_ph;
            setp.ne.u32 p_ph, phase, 0;
            selp.b32 b_next, b1_smem, b2_smem, p_ph;
            @p_warp_lead cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes
                [b_next], [bdesc_addr, {n_start_next, 0}], [tma_mbar];
            @p_warp_lead mbarrier.expect_tx.relaxed.cta.shared::cta.b64
                [tma_mbar], ${b_tile_bytes};
            @p_warp_lead cp.async.bulk.commit_group;
        }
        bar.warp.sync 0xffffffff;

        // TMA store/reduce+store of a C
        {
            .reg .pred p1;
            .reg .b64 _c_state;
$L_WAIT_CRDY:
            mbarrier.try_wait.parity.shared::cta.b64 p1, [cready_mbar], phase, ${mbar_maxwait};
            @!p1 bra.uni $L_WAIT_CRDY;
% if beta == 0:
            @p_warp_lead cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group
                [cdesc_addr, {n_c_store, 0}], [c_smem];
% else:
            @p_warp_lead cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group
                [cdesc_addr, {n_c_store, 0}], [c_smem];
% endif
            @p_warp_lead cp.async.bulk.commit_group;
            @p_warp_lead cp.async.bulk.wait_group 0;
            @p_warp_lead mbarrier.arrive.shared::cta.b64 _c_state, [cstored_mbar];
        }

        // Wait for next B to be ready, then signal B and C ready
        {
            .reg .b64 b_state, _bready_state, _c_state;
            .reg .pred p1;
            mbarrier.arrive.shared::cta.b64 b_state, [tma_mbar];
$L_WAIT_TMA:
            mbarrier.try_wait.shared::cta.b64 p1, [tma_mbar], b_state, ${mbar_maxwait};
            @!p1 bra.uni $L_WAIT_TMA;

            @p_warp_lead mbarrier.arrive.shared::cta.b64 _bready_state, [bready_mbar];
        }
    }
$L_AFTER_DATA:

    // --- Controller Warp
    @!p_steal bra.uni $L_AFTER_CTRL;
    {
        .reg .pred p1, p2, p_canc;
        .reg .b64 _state;
        .reg .b128 resp;
        @p_warp_lead fence.proxy.async.shared::cta;
        @p_warp_lead clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128
            [wid_smem], [steal_mbar];
        @p_warp_lead mbarrier.arrive.expect_tx.shared::cta.b64
            _state, [steal_mbar], 16;

$L_WAIT_STEAL:
        mbarrier.try_wait.parity.shared::cta.b64 p1, [steal_mbar], phase, ${mbar_maxwait};
        @!p1 bra.uni $L_WAIT_STEAL;

        // Signal new work
        @p_warp_lead mbarrier.arrive.shared::cta.b64 _state, [wid_new_mbar];

        // Query if there's new work
        ld.shared::cta.b128 resp, [wid_smem];
        clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p_canc, resp;
        selp.b32 work, 1, 0, p_canc;

        // Wait for old work to be used
$L_WAIT_WUSED:
        mbarrier.try_wait.parity.shared::cta.b64 p2, [wid_used_mbar], phase, ${mbar_maxwait};
        @!p2 bra.uni $L_WAIT_WUSED;
    }
$L_AFTER_CTRL:

    xor.b32 phase, phase, 1;
    bra.uni $L_LOOP;

$L_EXIT:
    ret;
}
