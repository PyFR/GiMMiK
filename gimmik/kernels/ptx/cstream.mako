<%inherit file='base'/>

<%
pftype = "f32" if dtype == "float" else "f64"
dwidth_i = 4 if dtype == "float" else 8
fzero = "0f00000000" if dtype == "float" else "0d0000000000000000"
bix_list = list(bix)
bix_pos = {kx: i for i, kx in enumerate(bix_list)}
K_used = len(bix_list)
row_nz = [[(kx, A[j, kx]) for kx in range(k) if A[j, kx] != 0] for j in range(m)]
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
    .reg .u32 n, id;
    .reg .u64 b, c, b_base, c_base;
    .reg .${pftype} bv<${K_used}>, dotp;
    .reg .pred p1;

% if n is None:
    ld.param.u32 n, [_n];
% else:
    mov.u32 n, ${n};
% endif
    ld.param.u64 b, [_b];
    ld.param.u64 c, [_c];

    {
    .reg .u32 _grd<3>;
    mov.u32 _grd0, %ntid.x;
    mov.u32 _grd1, %ctaid.x;
    mov.u32 _grd2, %tid.x;
    mad.lo.u32 id, _grd0, _grd1, _grd2;
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

## Batch-load active B columns
%for i, kx in enumerate(bix_list):
% if n is None:
    {
    .reg .u32 _boff;
    .reg .u64 _bptr;
    mul.lo.u32 _boff, ldb, ${kx};
    mad.wide.u32 _bptr, ${dwidth_i}, _boff, b_base;
    ld.global.nc.${pftype} bv${i}, [_bptr];
    }
% else:
    ld.global.nc.${pftype} bv${i}, [b_base + ${ldb*kx*dwidth_i}];
% endif
%endfor

## Compute and store each output row
%for j in range(m):
%  if row_nz[j]:
%   for i_nz, (kx, jx) in enumerate(row_nz[j]):
%    if i_nz == 0:
    mul.${pftype} dotp, bv${bix_pos[kx]}, ${jx};
%    else:
    fma.rn.${pftype} dotp, bv${bix_pos[kx]}, ${jx}, dotp;
%    endif
%   endfor
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

%  else:
## Zero row of A
% if beta == 0:
    {
    .reg .${pftype} _tmp;
    mov.${pftype} _tmp, ${fzero};
% if n is None:
    .reg .u32 _coff;
    .reg .u64 _cptr;
    mul.lo.u32 _coff, ldc, ${j};
    mad.wide.u32 _cptr, ${dwidth_i}, _coff, c_base;
    st.weak.global.cg.${pftype} [_cptr], _tmp;
% else:
    st.weak.global.cg.${pftype} [c_base + ${ldc*j*dwidth_i}], _tmp;
% endif
    }
% elif beta != 1:
    {
    .reg .${pftype} _tmp;
% if n is None:
    .reg .u32 _coff;
    .reg .u64 _cptr;
    mul.lo.u32 _coff, ldc, ${j};
    mad.wide.u32 _cptr, ${dwidth_i}, _coff, c_base;
    ld.global.${pftype} _tmp, [_cptr];
    mul.${pftype} _tmp, _tmp, ${float(beta)};
    st.global.${pftype} [_cptr], _tmp;
% else:
    ld.global.${pftype} _tmp, [c_base + ${ldc*j*dwidth_i}];
    mul.${pftype} _tmp, _tmp, ${float(beta)};
    st.global.${pftype} [c_base + ${ldc*j*dwidth_i}], _tmp;
% endif
    }
% endif
%  endif
%endfor

$L_EXIT:
    ret;
}
