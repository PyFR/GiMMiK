<%inherit file='base'/>

<%
mx = partition(A, into=msplit, by='rows')
bchunks = chunk(bix, bsz)
m_per_group = max(len(mcx) for mcx in mx)
bsub_bytes = 2 * bsz * blockx * 2 * dwidth_i
def bsub_off(buf, idx):
    return (buf * bsz + idx) * blockx * 2 * dwidth_i
%>

.visible .entry ${kname}(.param .u64 _b,
                         .param .u64 _c)
{
    .reg .u32 n, id, tid_x, tid_y;
    .reg .u64 b, c, b_base, c_base, bsub_thread;
% if use_cpasync:
    .reg .u32 bsub_sm_thread;
% endif
    .reg .${pftype} bv_a, bv_b, csub_a<${m_per_group}>, csub_b<${m_per_group}>;
    .reg .pred p1, p_skip;
    .shared .align 16 .b8 _bsub[${bsub_bytes}];

    mov.u32 n, ${-(-n // 2)};
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
        mad.lo.u64 b_base, _id64, ${2*dwidth_i}, b;
        mad.lo.u64 c_base, _id64, ${2*dwidth_i}, c;
    }

    {
        .reg .u64 _tx_off;
        mul.wide.u32 _tx_off, tid_x, ${2*dwidth_i};
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

% if beta_zero or not preload_c:
## Zero accumulators
%  for j, row_j in enumerate(mcx):
%   if afix[row_j] != -1:
    mov.${pftype} csub_a${j}, ${fzero};
    mov.${pftype} csub_b${j}, ${fzero};
%   endif
%  endfor
% else:
## Pre-load C and scale by beta so per-row completion is a plain store
%  for j, row_j in enumerate(mcx):
%   if afix[row_j] != -1:
    ld.weak.global.cg.v2.${pftype} {csub_a${j}, csub_b${j}}, [c_base + ${ldc*row_j*dwidth_i}];
    mul.${pftype} csub_a${j}, csub_a${j}, ${float(beta)};
    mul.${pftype} csub_b${j}, csub_b${j}, ${float(beta)};
%   endif
%  endfor
% endif

## Pre-fill double buffer
%  if use_cpasync:
%   for idx, kx in [(i, k) for i, k in enumerate(bchunks[0]) if i % msplit == cid]:
    cp.async.ca.shared::cta.global [bsub_sm_thread + ${bsub_off(0, idx)}], [b_base + ${ldb*kx*dwidth_i}], ${2*dwidth_i};
%   endfor
    cp.async.commit_group;
    cp.async.wait_all;
    bar.sync 0;
%  else:
%   for idx, kx in [(i, k) for i, k in enumerate(bchunks[0]) if i % msplit == cid]:
    {
        .reg .${pftype} _bva, _bvb;
        ld.weak.global.cg.v2.${pftype} {_bva, _bvb}, [b_base + ${ldb*kx*dwidth_i}];
        st.shared.v2.${pftype} [bsub_thread + ${bsub_off(0, idx)}], {_bva, _bvb};
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
    cp.async.ca.shared::cta.global [bsub_sm_thread + ${bsub_off(buf_next, idx)}], [b_base + ${ldb*kx*dwidth_i}], ${2*dwidth_i};
%     else:
    {
        .reg .${pftype} _bva, _bvb;
        ld.weak.global.cg.v2.${pftype} {_bva, _bvb}, [b_base + ${ldb*kx*dwidth_i}];
        st.shared.v2.${pftype} [bsub_thread + ${bsub_off(buf_next, idx)}], {_bva, _bvb};
    }
%     endif
%    endfor
%    if use_cpasync:
    cp.async.commit_group;
%    endif
%   endif

%   for idx, kx in enumerate(bchunks[bb]):
%    if any(A[row_j, kx] for row_j in mcx):
    ld.shared.v2.${pftype} {bv_a, bv_b}, [bsub_thread + ${bsub_off(buf_cur, idx)}];
%    endif
%    for j, row_j in enumerate(mcx):
%     if A[row_j, kx] != 0:
    fma.rn.${pftype} csub_a${j}, bv_a, ${A[row_j, kx]}, csub_a${j};
    fma.rn.${pftype} csub_b${j}, bv_b, ${A[row_j, kx]}, csub_b${j};
%     endif
%    endfor
%    for j, row_j in enumerate(mcx):
%     if kx == alix[row_j]:
%      if beta_zero:
    st.weak.global.cg.v2.${pftype} [c_base + ${ldc*row_j*dwidth_i}], {csub_a${j}, csub_b${j}};
%      elif preload_c:
    st.weak.global.v2.${pftype} [c_base + ${ldc*row_j*dwidth_i}], {csub_a${j}, csub_b${j}};
%      else:
    {
        .reg .${pftype} _ca, _cb;
        ld.weak.global.cg.v2.${pftype} {_ca, _cb}, [c_base + ${ldc*row_j*dwidth_i}];
        fma.rn.${pftype} _ca, _ca, ${float(beta)}, csub_a${j};
        fma.rn.${pftype} _cb, _cb, ${float(beta)}, csub_b${j};
        st.weak.global.v2.${pftype} [c_base + ${ldc*row_j*dwidth_i}], {_ca, _cb};
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

## Handle zero rows in this cid's group
%  if has_zero_rows:
%   for row_j in mcx:
%    if afix[row_j] == -1:
%     if beta_zero:
    {
        .reg .${pftype} _z;
        mov.${pftype} _z, ${fzero};
        st.weak.global.cg.v2.${pftype} [c_base + ${ldc*row_j*dwidth_i}], {_z, _z};
    }
%     elif beta != 1:
    {
        .reg .${pftype} _ca, _cb;
        ld.weak.global.cg.v2.${pftype} {_ca, _cb}, [c_base + ${ldc*row_j*dwidth_i}];
        mul.${pftype} _ca, _ca, ${float(beta)};
        mul.${pftype} _cb, _cb, ${float(beta)};
        st.weak.global.v2.${pftype} [c_base + ${ldc*row_j*dwidth_i}], {_ca, _cb};
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
