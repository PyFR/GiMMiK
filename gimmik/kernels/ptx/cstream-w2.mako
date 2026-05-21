<%inherit file='base'/>

.visible .entry ${kname}(.param .u64 _b,
                         .param .u64 _c)
{
    .reg .u32 n, id;
    .reg .u64 b, c, b_base, c_base;
    .reg .f64 bv_a<${len(bix)}>, bv_b<${len(bix)}>, dotp_a, dotp_b;
    .reg .pred p1;

    mov.u32 n, ${-(-n // 2)};
    ld.param.u64 b, [_b];
    ld.param.u64 c, [_c];

    {
        .reg .u32 _ctaid_x, _tid_x;
        mov.u32 _ctaid_x, %ctaid.x;
        mov.u32 _tid_x, %tid.x;
        mad.lo.u32 id, _ctaid_x, ${blockx}, _tid_x;
    }

    setp.ge.u32 p1, id, n;
    @p1 bra $L_EXIT;

    cvta.to.global.u64 b, b;
    cvta.to.global.u64 c, c;

    {
        .reg .u64 _id64;
        cvt.u64.u32 _id64, id;
        mad.lo.u64 b_base, _id64, 16, b;
        mad.lo.u64 c_base, _id64, 16, c;
    }

## Batch-load B column pairs
% for i, kx in enumerate(bix):
    ld.weak.global.cg.v2.f64 {bv_a${i}, bv_b${i}}, [b_base + ${ldb*kx*dwidth_i}];
% endfor

## Main compute: two parallel dot-product streams per thread
% for j in range(m):
%  if row_nz[j]:
%   for kx, jx in row_nz[j]:
%    if loop.first:
    mul.f64 dotp_a, bv_a${bix[kx]}, ${jx};
    mul.f64 dotp_b, bv_b${bix[kx]}, ${jx};
%    else:
    fma.rn.f64 dotp_a, bv_a${bix[kx]}, ${jx}, dotp_a;
    fma.rn.f64 dotp_b, bv_b${bix[kx]}, ${jx}, dotp_b;
%    endif
%   endfor
%   if beta_zero:
    st.weak.global.cg.v2.f64 [c_base + ${ldc*j*dwidth_i}], {dotp_a, dotp_b};
%   else:
    {
        .reg .f64 _ca, _cb;
        ld.weak.global.cg.v2.f64 {_ca, _cb}, [c_base + ${ldc*j*dwidth_i}];
        fma.rn.f64 _ca, _ca, ${float(beta)}, dotp_a;
        fma.rn.f64 _cb, _cb, ${float(beta)}, dotp_b;
        st.weak.global.v2.f64 [c_base + ${ldc*j*dwidth_i}], {_ca, _cb};
    }
%   endif

%  else:
## Zero row of A
%   if beta_zero:
    {
        .reg .f64 _z;
        mov.f64 _z, ${fzero};
        st.weak.global.cg.v2.f64 [c_base + ${ldc*j*dwidth_i}], {_z, _z};
    }
%   elif beta != 1:
    {
        .reg .f64 _ca, _cb;
        ld.weak.global.cg.v2.f64 {_ca, _cb}, [c_base + ${ldc*j*dwidth_i}];
        mul.f64 _ca, _ca, ${float(beta)};
        mul.f64 _cb, _cb, ${float(beta)};
        st.weak.global.v2.f64 [c_base + ${ldc*j*dwidth_i}], {_ca, _cb};
    }
%   endif
%  endif
% endfor

$L_EXIT:
    ret;
}
