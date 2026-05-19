<%inherit file='base'/>

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
    .reg .${pftype} csub<${m}>, bv<${len(bix_list)}>;
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
% for i, kx in enumerate(bix_list):
%  if n is None:
    {
        .reg .u32 _boff;
        .reg .u64 _bptr;
        mul.lo.u32 _boff, ldb, ${kx};
        mad.wide.u32 _bptr, ${dwidth_i}, _boff, b_base;
        ld.weak.global.cg.${pftype} bv${i}, [_bptr];
    }
%  else:
    ld.weak.global.cg.${pftype} bv${i}, [b_base + ${ldb*kx*dwidth_i}];
%  endif
% endfor

% if not beta_zero:
## Pre-load C so per-row completion is a plain store
%  for j in range(m):
%   if afix[j] != -1:
%    if n is None:
    {
        .reg .u32 _coff;
        .reg .u64 _cptr;
        mul.lo.u32 _coff, ldc, ${j};
        mad.wide.u32 _cptr, ${dwidth_i}, _coff, c_base;
        ld.weak.global.cg.${pftype} csub${j}, [_cptr];
    }
%    else:
    ld.weak.global.cg.${pftype} csub${j}, [c_base + ${ldc*j*dwidth_i}];
%    endif
%   endif
%  endfor
%  for j in range(m):
%   if afix[j] != -1:
    mul.${pftype} csub${j}, csub${j}, ${float(beta)};
%   endif
%  endfor
% endif

## Main compute
% for kx in bix_list:
%  for j, jx in enumerate(A[:, kx]):
%   if jx != 0:
%    if beta_zero and kx == afix[j]:
    mul.${pftype} csub${j}, bv${bix_pos[kx]}, ${jx};
%    else:
    fma.rn.${pftype} csub${j}, bv${bix_pos[kx]}, ${jx}, csub${j};
%    endif
%   endif
%   if kx == alix[j]:
%    if n is None:
    {
        .reg .u32 _coff;
        .reg .u64 _cptr;
        mul.lo.u32 _coff, ldc, ${j};
        mad.wide.u32 _cptr, ${dwidth_i}, _coff, c_base;
        st.weak.global.cg.${pftype} [_cptr], csub${j};
    }
%    else:
    st.weak.global.cg.${pftype} [c_base + ${ldc*j*dwidth_i}], csub${j};
%    endif

%   endif
%  endfor
% endfor

% if has_zero_rows:
    {
        .reg .${pftype} _tmp;
        mov.${pftype} _tmp, ${fzero};
%  for j, jx in enumerate(afix):
%   if jx == -1 and beta_zero:
%    if n is None:
        {
            .reg .u32 _coff;
            .reg .u64 _cptr;
            mul.lo.u32 _coff, ldc, ${j};
            mad.wide.u32 _cptr, ${dwidth_i}, _coff, c_base;
            st.weak.global.cg.${pftype} [_cptr], _tmp;
        }
%    else:
        st.weak.global.cg.${pftype} [c_base + ${ldc*j*dwidth_i}], _tmp;
%    endif

%   elif jx == -1:
%    if n is None:
        {
            .reg .u32 _coff;
            .reg .u64 _cptr;
            mul.lo.u32 _coff, ldc, ${j};
            mad.wide.u32 _cptr, ${dwidth_i}, _coff, c_base;
            ld.weak.global.cg.${pftype} _tmp, [_cptr];
            mul.${pftype} _tmp, _tmp, ${float(beta)};
            st.weak.global.cg.${pftype} [_cptr], _tmp;
        }
%    else:
        ld.weak.global.cg.${pftype} _tmp, [c_base + ${ldc*j*dwidth_i}];
        mul.${pftype} _tmp, _tmp, ${float(beta)};
        st.weak.global.cg.${pftype} [c_base + ${ldc*j*dwidth_i}], _tmp;
%    endif
%   endif
%  endfor
    }
% endif

$L_EXIT:
    ret;
}
