<%inherit file='base'/>

<%
mx = partition(A, into=msplit, by='rows')
bchunks = chunk(bix, bsz)
m_per_group = max(len(mcx) for mcx in mx)
bsub_bytes = 2 * bsz * blockx * dwidth_i
def bsub_off(buf, idx):
    return (buf * bsz + idx) * blockx * dwidth_i
%>

% if n is None:
.visible .entry ${kname}(.param .u32 _n,
                         .param .u64 _b,
                         .param .u32 _ldb,
                         .param .u64 _c,
                         .param .u32 _ldc)
{
    .reg .u32 ldb, ldc;
    ld.param.u32 ldb, [_ldb];
    ld.param.u32 ldc, [_ldc];
% else:
.visible .entry ${kname}(.param .u64 _b,
                         .param .u64 _c)
{
% endif
    .reg .u32 n, id, tid_x, tid_y;
    .reg .u64 b, c, b_base, c_base, bsub_thread;
% if use_cpasync:
    .reg .u32 bsub_sm_thread;
% endif
    .reg .${pftype} bv, csub<${m_per_group}>;
    .reg .pred p1, p_skip;
    .shared .align 8 .b8 _bsub[${bsub_bytes}];

% if n is None:
    ld.param.u32 n, [_n];
% else:
    mov.u32 n, ${n};
% endif
    ld.param.u64 b, [_b];
    ld.param.u64 c, [_c];

    {
        .reg .u32 _ctaid_x;
        mov.u32 _ctaid_x, %ctaid.x;
        mov.u32 tid_x, %tid.x;
        mov.u32 tid_y, %tid.y;
        mad.lo.u32 id, _ctaid_x, ${blockx}, tid_x;
    }

    setp.ge.u32 p1, id, n;
    @p1 bra $L_EXIT;

    cvta.to.global.u64 b, b;
    cvta.to.global.u64 c, c;

    {
        .reg .u64 _id64;
        cvt.u64.u32 _id64, id;
        mad.lo.u64 b_base, _id64, ${dwidth_i}, b;
        mad.lo.u64 c_base, _id64, ${dwidth_i}, c;
    }

    {
        .reg .u64 _tx_off;
        mul.wide.u32 _tx_off, tid_x, ${dwidth_i};
        mov.u64 bsub_thread, _bsub;
        add.u64 bsub_thread, bsub_thread, _tx_off;
    }
% if use_cpasync:
    {
        .reg .u64 _sm64;
        cvta.to.shared.u64 _sm64, bsub_thread;
        cvt.u32.u64 bsub_sm_thread, _sm64;
    }
% endif

% for cid, mcx in enumerate(mx):
## cid = ${cid}, rows ${mcx}
    setp.ne.u32 p_skip, tid_y, ${cid};
    @p_skip bra $L_END_CID_${cid};

## Zero accumulators
%  for j, row_j in enumerate(mcx):
%   if afix[row_j] != -1:
    mov.${pftype} csub${j}, ${fzero};
%   endif
%  endfor

## Pre-fill double buffer
%  if use_cpasync:
## Async fill of chunk 0
%   for idx, kx in [(i, k) for i, k in enumerate(bchunks[0]) if i % msplit == cid]:
%    if n is None:
    {
        .reg .u64 _bptr;
        mad.wide.u32 _bptr, ldb, ${kx * dwidth_i}, b_base;
        cp.async.ca.shared::cta.global [bsub_sm_thread + ${bsub_off(0, idx)}], [_bptr], ${dwidth_i};
    }
%    else:
    cp.async.ca.shared::cta.global [bsub_sm_thread + ${bsub_off(0, idx)}], [b_base + ${ldb*kx*dwidth_i}], ${dwidth_i};
%    endif
%   endfor
    cp.async.commit_group;
    cp.async.wait_all;
    bar.sync 0;
%  else:
## Sync fill of chunk 0
%   for idx, kx in [(i, k) for i, k in enumerate(bchunks[0]) if i % msplit == cid]:
    {
        .reg .${pftype} _bv;
%    if n is None:
        .reg .u64 _bptr;
        mad.wide.u32 _bptr, ldb, ${kx * dwidth_i}, b_base;
        ld.weak.global.cg.${pftype} _bv, [_bptr];
%    else:
        ld.weak.global.cg.${pftype} _bv, [b_base + ${ldb*kx*dwidth_i}];
%    endif
        st.shared.${pftype} [bsub_thread + ${bsub_off(0, idx)}], _bv;
    }
%   endfor
    bar.sync 0;
%  endif

## Main loop over B-chunks (double-buffered)
%  for bb in range(len(bchunks)):
<%
        buf_cur = bb % 2
        buf_next = (bb + 1) % 2
%>
%   if not loop.last:
%    for idx, kx in [(i, k) for i, k in enumerate(bchunks[bb + 1]) if i % msplit == cid]:
%     if use_cpasync:
%      if n is None:
    {
        .reg .u64 _bptr;
        mad.wide.u32 _bptr, ldb, ${kx * dwidth_i}, b_base;
        cp.async.ca.shared::cta.global [bsub_sm_thread + ${bsub_off(buf_next, idx)}], [_bptr], ${dwidth_i};
    }
%      else:
    cp.async.ca.shared::cta.global [bsub_sm_thread + ${bsub_off(buf_next, idx)}], [b_base + ${ldb*kx*dwidth_i}], ${dwidth_i};
%      endif
%     else:
    {
        .reg .${pftype} _bv;
%      if n is None:
        .reg .u64 _bptr;
        mad.wide.u32 _bptr, ldb, ${kx * dwidth_i}, b_base;
        ld.weak.global.cg.${pftype} _bv, [_bptr];
%      else:
        ld.weak.global.cg.${pftype} _bv, [b_base + ${ldb*kx*dwidth_i}];
%      endif
        st.shared.${pftype} [bsub_thread + ${bsub_off(buf_next, idx)}], _bv;
    }
%     endif
%    endfor
%    if use_cpasync:
    cp.async.commit_group;
%    endif
%   endif

%   for idx, kx in enumerate(bchunks[bb]):
%    if any(A[row_j, kx] for row_j in mcx):
    ld.shared.${pftype} bv, [bsub_thread + ${bsub_off(buf_cur, idx)}];
%    endif
%    for j, row_j in enumerate(mcx):
%     if A[row_j, kx] != 0:
    fma.rn.${pftype} csub${j}, bv, ${A[row_j, kx]}, csub${j};
%     endif
%    endfor
%    for j, row_j in enumerate(mcx):
%     if kx == alix[row_j]:
%      if beta_zero:
%       if n is None:
    {
        .reg .u64 _cptr;
        mad.wide.u32 _cptr, ldc, ${row_j * dwidth_i}, c_base;
        st.weak.global.cg.${pftype} [_cptr], csub${j};
    }
%       else:
    st.weak.global.cg.${pftype} [c_base + ${ldc*row_j*dwidth_i}], csub${j};
%       endif
%      else:
    {
        .reg .${pftype} _ctmp;
%       if n is None:
        .reg .u64 _cptr;
        mad.wide.u32 _cptr, ldc, ${row_j * dwidth_i}, c_base;
        ld.weak.global.cg.${pftype} _ctmp, [_cptr];
        fma.rn.${pftype} _ctmp, _ctmp, ${float(beta)}, csub${j};
        st.weak.global.${pftype} [_cptr], _ctmp;
%       else:
        ld.weak.global.cg.${pftype} _ctmp, [c_base + ${ldc*row_j*dwidth_i}];
        fma.rn.${pftype} _ctmp, _ctmp, ${float(beta)}, csub${j};
        st.weak.global.${pftype} [c_base + ${ldc*row_j*dwidth_i}], _ctmp;
%       endif
    }
%      endif
%     endif
%    endfor
%   endfor
%   if use_cpasync:
%    if not loop.last:
    cp.async.wait_all;
%    endif
%   endif
    bar.sync 0;
%  endfor
## End of Main loop over B-chunks

## Handle zero rows in this cid's group
%  if has_zero_rows:
%   for row_j in mcx:
%    if afix[row_j] == -1:
%     if beta_zero:
    {
        .reg .${pftype} _tmp;
        mov.${pftype} _tmp, ${fzero};
%      if n is None:
        .reg .u64 _cptr;
        mad.wide.u32 _cptr, ldc, ${row_j * dwidth_i}, c_base;
        st.weak.global.cg.${pftype} [_cptr], _tmp;
%      else:
        st.weak.global.cg.${pftype} [c_base + ${ldc*row_j*dwidth_i}], _tmp;
%      endif
    }
%     elif beta != 1:
    {
        .reg .${pftype} _tmp;
%      if n is None:
        .reg .u64 _cptr;
        mad.wide.u32 _cptr, ldc, ${row_j * dwidth_i}, c_base;
        ld.weak.global.cg.${pftype} _tmp, [_cptr];
        mul.${pftype} _tmp, _tmp, ${float(beta)};
        st.weak.global.${pftype} [_cptr], _tmp;
%      else:
        ld.weak.global.cg.${pftype} _tmp, [c_base + ${ldc*row_j*dwidth_i}];
        mul.${pftype} _tmp, _tmp, ${float(beta)};
        st.weak.global.${pftype} [c_base + ${ldc*row_j*dwidth_i}], _tmp;
%      endif
    }
%     endif
%    endif
%   endfor
%  endif

$L_END_CID_${cid}:
% endfor

$L_EXIT:
    ret;
}
