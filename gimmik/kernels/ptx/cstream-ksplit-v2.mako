<%inherit file='base'/>

<%
kparts = partition(A, ksplit, by='cols')
cchunks = chunk(list(range(m)), csz)
cv_per_thread = -(-csz // ksplit)
bv_per_thread = max(len(kbx) for kbx in kparts)
csub_bytes = (ksplit - 1) * csz * blockx * 2 * dwidth_i
%>

.visible .entry ${kname}(.param .u64 _b,
                         .param .u64 _c)
{
    .reg .u32 n, id, tid_x, tid_y;
    .reg .u64 b, c, b_base, c_base, csub_thread;
    .reg .${pftype} bv_a<${bv_per_thread}>, bv_b<${bv_per_thread}>;
    .reg .${pftype} cv_a<${cv_per_thread}>, cv_b<${cv_per_thread}>;
    .reg .${pftype} dotp_a, dotp_b;
    .reg .pred p1, p_skip;
    .shared .align 16 .b8 _csub[${csub_bytes}];

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
        mov.u64 csub_thread, _csub;
        add.u64 csub_thread, csub_thread, _tx_off;
    }

% for bid, kbx in enumerate(kparts):
## bid = ${bid}: ${len(kbx)} B columns, ksplit=${ksplit}
    setp.ne.u32 p_skip, tid_y, ${bid};
    @p_skip bra $L_END_BID_${bid};

<%  loaded = set() %>

%  for cchunk_i, cchunk in enumerate(cchunks):
## Chunk ${cchunk_i}: partial dot-product
%   for row_idx, j in enumerate(cchunk):
<%      owner_bid = row_idx % ksplit %>
%    for kxi, kx in enumerate(kbx):
%     if A[j, kx] != 0 and kx not in loaded:
    ld.weak.global.cg.v2.${pftype} {bv_a${kxi}, bv_b${kxi}}, [b_base + ${ldb*kx*dwidth_i}];
<%      loaded.add(kx) %>
%     endif
%    endfor
    mov.${pftype} dotp_a, ${fzero};
    mov.${pftype} dotp_b, ${fzero};
%    for kxi, kx in enumerate(kbx):
%     if A[j, kx] != 0:
    fma.rn.${pftype} dotp_a, bv_a${kxi}, ${A[j, kx]}, dotp_a;
    fma.rn.${pftype} dotp_b, bv_b${kxi}, ${A[j, kx]}, dotp_b;
%     endif
%    endfor
%    if owner_bid == bid:
    mov.${pftype} cv_a${row_idx // ksplit}, dotp_a;
    mov.${pftype} cv_b${row_idx // ksplit}, dotp_b;
%    else:
<%        csub_idx = bid - (1 if bid > owner_bid else 0) %>
    st.shared.v2.${pftype} [csub_thread + ${(csub_idx * csz + row_idx) * blockx * 2 * dwidth_i}], {dotp_a, dotp_b};
%    endif
%   endfor
    bar.sync 0;

## Combine phase (owned rows only)
%   for row_idx, j in enumerate(cchunk):
%    if row_idx % ksplit == bid:
    mov.${pftype} dotp_a, cv_a${row_idx // ksplit};
    mov.${pftype} dotp_b, cv_b${row_idx // ksplit};
%     for other_bid in range(ksplit):
%      if other_bid != bid:
<%            csub_idx = other_bid - (1 if other_bid > (row_idx % ksplit) else 0) %>
    {
        .reg .${pftype} _ta, _tb;
        ld.shared.v2.${pftype} {_ta, _tb}, [csub_thread + ${(csub_idx * csz + row_idx) * blockx * 2 * dwidth_i}];
        add.${pftype} dotp_a, dotp_a, _ta;
        add.${pftype} dotp_b, dotp_b, _tb;
    }
%      endif
%     endfor
%     if beta_zero:
    st.weak.global.cg.v2.${pftype} [c_base + ${ldc*j*dwidth_i}], {dotp_a, dotp_b};
%     else:
    {
        .reg .${pftype} _ca, _cb;
        ld.weak.global.cg.v2.${pftype} {_ca, _cb}, [c_base + ${ldc*j*dwidth_i}];
        fma.rn.${pftype} _ca, _ca, ${float(beta)}, dotp_a;
        fma.rn.${pftype} _cb, _cb, ${float(beta)}, dotp_b;
        st.weak.global.v2.${pftype} [c_base + ${ldc*j*dwidth_i}], {_ca, _cb};
    }
%     endif

%    endif
%   endfor
    bar.sync 0;
%  endfor

$L_END_BID_${bid}:
% endfor

$L_EXIT:
    ret;
}
