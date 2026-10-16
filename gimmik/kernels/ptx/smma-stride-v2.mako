<%inherit file='base'/>
<%!
# The native 4x4 FFMA ordering
order = [(0, 1), (0, 0), (1, 1), (1, 0), (1, 3), (1, 2), (0, 3), (0, 2)]
%>

<%def name="prepare_output()">
% if not beta_zero and preload_c:
    // Keep beta*C independent of the dot product until the original epilogue.
    .reg .f32 oldc_<${mt*nt}>;
    {
        .reg .b32 pc, row, last, oc;
        .reg .b64 addr;
        .reg .pred full, pr, tail;
%  for r in range(mt):
        mad.lo.u32 row, group, ${mt}, ${r};
        setp.lt.u32 pr, row, ${m};
%   for v in range(nt//2):
        mad.lo.u32 pc, worker, 2, col;
        add.u32 pc, pc, ${2*v*workers};
        add.u32 last, pc, 1;
        setp.lt.u32 full, last, nn;
        and.pred full, full, pr;
        mad.wide.u32 addr, pc, 4, c_row_${r};
        @full ld.global.v2.f32 {${', '.join(f'oldc_{r*nt + 2*v + j}' for j in range(2))}}, [addr];
%    if n is None or n % 2:
%     for j in range(2):
        add.u32 oc, pc, ${j};
        setp.lt.u32 tail, oc, nn;
        and.pred tail, tail, pr;
        and.pred tail, tail, !full;
        @tail ld.global.f32 oldc_${r*nt + 2*v + j}, [addr + ${4*j}];
%     endfor
%    endif
%   endfor
%  endfor
    }
% elif not beta_zero and prefetch_c:
    // Fetch beta*C into L1 while the dot product runs, without extra registers.
    {
        .reg .b32 pc, row;
        .reg .b64 addr;
        .reg .pred valid, pr;
%  for r in range(mt):
        mad.lo.u32 row, group, ${mt}, ${r};
        setp.lt.u32 pr, row, ${m};
%   for v in range(nt//2):
        mad.lo.u32 pc, worker, 2, col;
        add.u32 pc, pc, ${2*v*workers};
        setp.lt.u32 valid, pc, nn;
        and.pred valid, valid, pr;
        mad.wide.u32 addr, pc, 4, c_row_${r};
        @valid prefetch.global.L1 [addr];
%   endfor
%  endfor
    }
% endif
</%def>

<%def name="add_c(r, v)">
% if preload_c:
%  for j in range(2):
    @full add.f32 acc_${r*nt + 2*v + j}, acc_${r*nt + 2*v + j}, oldc_${r*nt + 2*v + j};
%  endfor
% else:
    .reg .f32 old_<2>;
    @full ld.global.v2.f32 {old_0, old_1}, [addr];
%  for j in range(2):
    @full add.f32 acc_${r*nt + 2*v + j}, acc_${r*nt + 2*v + j}, old_${j};
%  endfor
% endif
</%def>
<%def name="add_c_tail(r, v, j)">
% if preload_c:
    @valid add.f32 acc_${r*nt + 2*v + j}, acc_${r*nt + 2*v + j}, oldc_${r*nt + 2*v + j};
% else:
    .reg .f32 tail_${j};
    @valid ld.global.f32 tail_${j}, [addr + ${4*j}];
    @valid add.f32 acc_${r*nt + 2*v + j}, acc_${r*nt + 2*v + j}, tail_${j};
% endif
</%def>

<%def name="compute_panel(kb)">
% for first in range(0, kt, 4):
<% mask = zero_chunks[(kb*kt + first)//4] %>
%  if mask:
    // Skip a zero 4-column operator tile without shortening the TMA panel.
    {
        .reg .b32 masked;
        .reg .pred zero_chunk;
        and.b32 masked, group_bit, ${hex(mask)};
        setp.ne.u32 zero_chunk, masked, 0;
        @zero_chunk bra CHUNK_DONE_${kb}_${first};
    }
%  endif
    ${self.compute_range(kb, first, first + 4)}
%  if mask:
    CHUNK_DONE_${kb}_${first}:
%  endif
% endfor
</%def>

<%def name="compute_range(kb, first, last)">
% for ik in range(first, last):
%  if kb*kt + ik < k and k_used[kb*kt + ik]:
    {
        .reg .f32 av_<${mt}>, bv_<2>;
%   if ffma2:
        .reg .b64 aa_<${mt}>, bb, packed;
%   endif
%   for r in range(0, mt, 4):
        ld.shared.v4.f32 {av_${r}, av_${r + 1}, av_${r + 2}, av_${r + 3}}, [ap + ${4*((kb*kt + ik)*mpad + r)}];
%   endfor
%   if ffma2:
%    for r in range(mt):
        mov.b64 aa_${r}, {av_${r}, av_${r}};
%    endfor
%   endif
%   for v in range(nt//2):
        ld.shared.v2.f32 {bv_0, bv_1}, [bp + ${4*(ik*panel + 2*v*workers)}];
%    if ffma2:
        mov.b64 bb, {bv_0, bv_1};
%     for r in range(0, mt, 4):
%      for y in [1, 0, 3, 2]:
<% idx = (r + y)*nt + 2*v %>
        mov.b64 packed, {acc_${idx}, acc_${idx + 1}};
        fma.rn.f32x2 packed, bb, aa_${r + y}, packed;
        mov.b64 {acc_${idx}, acc_${idx + 1}}, packed;
%      endfor
%     endfor
%    else:
%     for r in range(0, mt, 4):
%      for x, y in order:
        fma.rn.f32 acc_${(r + y)*nt + 2*v + x}, bv_${x}, av_${r + y}, acc_${(r + y)*nt + 2*v + x};
%      endfor
%     endfor
%    endif
%   endfor
    }
%  endif
% endfor
</%def>

.global .align 16 .b32 ${kname}_Ag[${len(a_u32)}] = {${', '.join(a_u32)}};
.visible .entry ${kname}(
% if n is None:
    .param .u32 _n,
% endif
    .param .u64 b_desc, .param .u64 _c
% if n is None:
    , .param .u32 _ldc
% endif
)
.maxntid ${workers*groups}, 1, 1
{
    .shared .align 128 .b8 s_a[${abytes}];
    .shared .align 128 .b8 s_b[${2*bbytes}];
    .shared .align 8 .b64 mb;
    .reg .b32 tid, worker, group, col, step, nn, phase, buff;
    .reg .b32 sa, sb, barrier, ap, bp, nextcol;
    .reg .b64 desc, c, ag, state;
% if n is None:
    .reg .b32 ldc;
    .reg .b64 ldc64;
% endif
    .reg .pred leader, done, more, ready;
    mov.u32 tid, %tid.x;
    rem.u32 worker, tid, ${workers};
    div.u32 group, tid, ${workers};
    mov.u32 sa, s_a;
    mov.u32 sb, s_b;
    mov.u32 barrier, mb;
    ld.param.u64 desc, [b_desc];
    ld.param.u64 c, [_c];
% if n is not None:
    mov.u32 nn, ${n};
% else:
    ld.param.u32 nn, [_n];
    ld.param.u32 ldc, [_ldc];
    cvt.u64.u32 ldc64, ldc;
% endif
    mov.u32 col, %ctaid.x;
    mul.lo.u32 col, col, ${panel};
    mov.u32 step, %nctaid.x;
    mul.lo.u32 step, step, ${panel};
    setp.ge.u32 done, col, nn;
    @done bra.uni EXIT;

    setp.eq.u32 leader, tid, 0;
    @leader mbarrier.init.shared::cta.b64 [barrier], 1;
    @leader fence.proxy.async.shared::cta;
    bar.sync 0;

    @!leader bra INIT_DONE;
    mov.u64 ag, ${kname}_Ag;
    cvta.to.global.u64 ag, ag;
    mbarrier.arrive.expect_tx.shared::cta.b64 state, [barrier], ${abytes + bbytes};
    cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [sa], [ag], ${abytes}, [barrier];
    cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [sb], [desc, {col, 0}], [barrier];
    INIT_DONE:
    mov.u32 phase, 0;
    WAIT_INIT:
    mbarrier.try_wait.parity.shared::cta.b64 ready, [barrier], phase;
    @!ready bra.uni WAIT_INIT;
    xor.b32 phase, phase, 1;
    bar.sync 0;

    mad.lo.u32 ap, group, ${4*mt}, sa;
    .reg .b32 group_bit;
    shl.b32 group_bit, 1, group;
    .reg .b64 c_row_<${mt}>;
    {
        .reg .b32 row;
        .reg .b64 off;
% for r in range(mt):
        mad.lo.u32 row, group, ${mt}, ${r};
%  if n is not None:
        mad.wide.u32 c_row_${r}, row, ${4*ldc}, c;
%  else:
        mul.wide.u32 off, row, 4;
        mul.lo.u64 off, off, ldc64;
        add.u64 c_row_${r}, c, off;
%  endif
% endfor
    }

    mov.u32 buff, 0;
    N_TILE_LOOP:
    .reg .f32 acc_<${mt*nt}>;
% for i in range(mt*nt):
    mov.f32 acc_${i}, ${fzero};
% endfor
    ${self.prepare_output()}
% for kb in range(kpad//kt):
    // Prefetch the following K panel, carrying the pipeline across N tiles.
%  if (kb + 1)*kt < kpad:
    mov.u32 nextcol, col;
    mov.pred more, 1;
%  else:
    add.u32 nextcol, col, step;
    setp.lt.u32 more, nextcol, nn;
%  endif
    @!more bra.uni PREFETCH_DONE_${kb};
    @!leader bra PREFETCH_DONE_${kb};
    {
        .reg .b32 dst;
        xor.b32 dst, buff, ${bbytes};
        add.u32 dst, dst, sb;
        mbarrier.arrive.expect_tx.shared::cta.b64 state, [barrier], ${bbytes};
        cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes
            [dst], [desc, {nextcol, ${(kb + 1)*kt if (kb + 1)*kt < kpad else 0}}], [barrier];
    }
    PREFETCH_DONE_${kb}:
    mad.lo.u32 bp, worker, 8, sb;
    add.u32 bp, bp, buff;
%  if zero_groups[kb]:
    // Skip zero tiles
    {
        .reg .pred zero_tile, zg;
        mov.pred zero_tile, 0;
%   for g in zero_groups[kb]:
        setp.eq.u32 zg, group, ${g};
        or.pred zero_tile, zero_tile, zg;
%   endfor
        @zero_tile bra COMPUTE_DONE_${kb};
    }
%  endif
    ${self.compute_panel(kb)}
%  if zero_groups[kb]:
    COMPUTE_DONE_${kb}:
%  endif
    @!more bra.uni WAIT_DONE_${kb};
    WAIT_${kb}:
    mbarrier.try_wait.parity.shared::cta.b64 ready, [barrier], phase;
    @!ready bra.uni WAIT_${kb};
    xor.b32 phase, phase, 1;
    WAIT_DONE_${kb}:
    bar.sync 0;
    xor.b32 buff, buff, ${bbytes};
% endfor
    {
        .reg .b32 outcol, row;
        .reg .b64 addr;
        .reg .pred valid;
% for r in range(mt):
        mad.lo.u32 row, group, ${mt}, ${r};
%  for v in range(nt//2):
        mad.lo.u32 outcol, worker, 2, col;
        add.u32 outcol, outcol, ${2*v*workers};
        mad.wide.u32 addr, outcol, 4, c_row_${r};
        {
            .reg .b32 oc;
            .reg .pred pr, full;
%   if m == mpad:
            mov.pred pr, 1;
%   else:
            setp.lt.u32 pr, row, ${m};
%   endif
%   if n is not None and n % panel == 0:
            mov.pred full, 1;
%   else:
            add.u32 oc, outcol, 1;
            setp.lt.u32 full, oc, nn;
%   endif
            and.pred full, full, pr;
%   if not beta_zero:
            ${self.add_c(r, v)}
%   endif
            @full st.global.v2.f32 [addr], {${', '.join(f'acc_{r*nt + 2*v + j}' for j in range(2))}};
%   if n is None or n % 2:
%    for j in range(2):
            add.u32 oc, outcol, ${j};
            setp.lt.u32 valid, oc, nn;
            and.pred valid, valid, pr;
            and.pred valid, valid, !full;
%     if not beta_zero:
            ${self.add_c_tail(r, v, j)}
%     endif
            @valid st.global.f32 [addr + ${4*j}], acc_${r*nt + 2*v + j};
%    endfor
%   endif
        }
%  endfor
% endfor
    }
    add.u32 col, col, step;
    setp.lt.u32 more, col, nn;
    @more bra.uni N_TILE_LOOP;

    bar.sync 0;
    @leader mbarrier.inval.shared::cta.b64 [barrier];
    EXIT:
    ret;
}
