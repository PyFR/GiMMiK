<%inherit file='base'/>

<%
pftype = 'f32' if dtype == 'float' else 'f64'
dwidth_i = 4 if dtype == 'float' else 8
fzero = '0f00000000' if dtype == 'float' else '0d0000000000000000'
kparts = partition(A, ksplit, by='cols')
cchunks = chunk(list(range(m)), csz)
cv_per_thread = -(-csz // ksplit)
bv_per_thread = max(len(kbx) for kbx in kparts)
csub_bytes = (ksplit - 1) * csz * blockx * dwidth_i
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
    .reg .u64 b, c, b_base, c_base, csub_thread;
    .reg .${pftype} bv<${bv_per_thread}>, cv<${cv_per_thread}>, dotp;
    .reg .pred p1, p_skip;
    .shared .align 8 .b8 _csub[${csub_bytes}];

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
    mov.u64 csub_thread, _csub;
    add.u64 csub_thread, csub_thread, _tx_off;
    }

% for bid, kbx in enumerate(kparts):
## bid = ${bid}: ${len(kbx)} B columns, ksplit=${ksplit}
    setp.ne.u32 p_skip, tid_y, ${bid};
    @p_skip bra $L_END_BID_${bid};

<%
    loaded = set()
    kbx_idx = {kx: i for i, kx in enumerate(kbx)}
%>

%   for cchunk_i, cchunk in enumerate(cchunks):
## Chunk ${cchunk_i}: partial dot-product
%     for row_idx, j in enumerate(cchunk):
<%
        nz = [(kbx_idx[kx], kx, A[j, kx]) for kx in kbx if A[j, kx] != 0]
        owner_bid = row_idx % ksplit
%>
%       for (kxi, kx, jx) in nz:
%         if kx not in loaded:
% if n is None:
    {
    .reg .u32 _boff;
    .reg .u64 _bptr;
    mul.lo.u32 _boff, ldb, ${kx};
    mad.wide.u32 _bptr, ${dwidth_i}, _boff, b_base;
    ld.global.nc.${pftype} bv${kxi}, [_bptr];
    }
% else:
    ld.global.nc.${pftype} bv${kxi}, [b_base + ${ldb*kx*dwidth_i}];
% endif
<%          loaded.add(kx) %>
%         endif
%       endfor
%       if nz:
%         for i, (kxi, kx, jx) in enumerate(nz):
%           if i == 0:
    mul.${pftype} dotp, bv${kxi}, ${jx};
%           else:
    fma.rn.${pftype} dotp, bv${kxi}, ${jx}, dotp;
%           endif
%         endfor
%       else:
    mov.${pftype} dotp, ${fzero};
%       endif
%       if owner_bid == bid:
    mov.${pftype} cv${row_idx // ksplit}, dotp;
%       else:
<%        csub_idx = bid - (1 if bid > owner_bid else 0) %>
    st.shared.${pftype} [csub_thread + ${(csub_idx * csz + row_idx) * blockx * dwidth_i}], dotp;
%       endif
%     endfor
    bar.sync 0;

## Combine phase (owned rows only)
%     for row_idx, j in enumerate(cchunk):
%       if row_idx % ksplit == bid:
    mov.${pftype} dotp, cv${row_idx // ksplit};
%         for other_bid in range(ksplit):
%           if other_bid != bid:
<%            csub_idx = other_bid - (1 if other_bid > (row_idx % ksplit) else 0) %>
    {
    .reg .${pftype} _tmp;
    ld.shared.${pftype} _tmp, [csub_thread + ${(csub_idx * csz + row_idx) * blockx * dwidth_i}];
    add.${pftype} dotp, dotp, _tmp;
    }
%           endif
%         endfor
% if beta == 0:
% if n is None:
    {
    .reg .u32 _coff;
    .reg .u64 _cptr;
    mul.lo.u32 _coff, ldc, ${j};
    mad.wide.u32 _cptr, ${dwidth_i}, _coff, c_base;
    st.weak.global.cg.${pftype} [_cptr], dotp;
    }
% else:
    st.weak.global.cg.${pftype} [c_base + ${ldc*j*dwidth_i}], dotp;
% endif
% else:
    {
    .reg .${pftype} _ctmp;
% if n is None:
    .reg .u32 _coff;
    .reg .u64 _cptr;
    mul.lo.u32 _coff, ldc, ${j};
    mad.wide.u32 _cptr, ${dwidth_i}, _coff, c_base;
    ld.global.${pftype} _ctmp, [_cptr];
    fma.rn.${pftype} _ctmp, _ctmp, ${float(beta)}, dotp;
    st.global.${pftype} [_cptr], _ctmp;
% else:
    ld.global.${pftype} _ctmp, [c_base + ${ldc*j*dwidth_i}];
    fma.rn.${pftype} _ctmp, _ctmp, ${float(beta)}, dotp;
    st.global.${pftype} [c_base + ${ldc*j*dwidth_i}], _ctmp;
% endif
    }
% endif

%       endif
%     endfor
    bar.sync 0;
%   endfor

$L_END_BID_${bid}:
% endfor

$L_EXIT:
    ret;
}
