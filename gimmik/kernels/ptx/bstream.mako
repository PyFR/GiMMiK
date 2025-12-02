<%inherit file='base'/>

<%
pftype = "f32" if dtype == "float" else "f64"
putype = "u32" if dtype == "float" else "u64"
pbtype = "b32" if dtype == "float" else "b64"
rtype = "f" if dtype == "float" else "fd"
dwidth = "4" if dtype == "float" else "8"
%>

% if n is None:
.visible .entry ${kname}(.param .u32 _n,
                         .param .u64 _b,
                         .param .u32 _ldb,
                         .param .u64 _c,
                         .param .u32 _ldc)
{
    .reg .u32 n, ldb, ldc;
    ld.param.u32 n, [_n];
    ld.param.u32 ldb, [_ldb];
    ld.param.u32 ldc, [_ldc];
% else:
.visible .entry ${kname}(.param .u64 _b,
                         .param .u64 _c)
{
    .reg .u32 n;
    mov.u32 n, ${n};
%endif
    .reg .u32 id;
    .reg .u64 b, c;
    .reg .${pftype} csub<${m}>;
    .reg .${pftype} ctmp<${m}>;
    .reg .pred p1;
    ld.param.u64 b, [_b];
    ld.param.u64 c, [_c];

    {
    .reg .u32 grid<3>;
    mov.u32 grid0, %ntid.x;
    mov.u32 grid1, %ctaid.x;
    mov.u32 grid2, %tid.x;
    mad.lo.u32 id, grid0, grid1, grid2;
    }
    setp.ge.u32	p1, id, n;
    @p1 bra $L_EXIT;
    cvta.to.global.u64 b, b;
    cvta.to.global.u64 c, c;

    {
    .reg .${pftype} bv;
    .reg .u32 boff<${len(bix)}>, coff;
    .reg .u64 bptr<${len(bix)}>, cptr;
%for kx in bix:
%  if n is None:
    mul.lo.u32 boff${kx}, ldb, ${kx};
    ${address((f"bptr{kx}", "u64"), ("b", "u64"), dwidth, (f"boff{kx}", "u32"), ("id", "u32"))}
%  else:
    ${address((f"bptr{kx}", "u64"), ("b", "u64"), dwidth, (f"{ldb*kx}", "u32"), ("id", "u32"))}
%  endif
     ld.weak.global.cg.${pftype} bv, [bptr${kx}];

%  for j, jx in enumerate(A[:, kx]):
%    if jx != 0 and kx == afix[j]:
    mul.${pftype} csub${j}, bv, ${jx};
%    elif jx != 0:
    fma.rn.${pftype} csub${j}, bv, ${jx}, csub${j};
%    endif

%    if kx == alix[j] and beta == 0:
%      if n is None:
    mul.lo.u32 coff, ldc, ${j};
    ${address(("cptr", "u64"), ("c", "u64"), dwidth, ("coff", "u32"), ("id", "u32"))}
%      else:
    ${address(("cptr", "u64"), ("c", "u64"), dwidth, (f"{ldc*j}", "u32"), ("id", "u32"))}
%      endif:
    st.weak.global.cg.${pftype} [cptr], csub${j};

%    elif kx == alix[j] and beta == 1:
%      if n is None:
    mul.lo.u32 coff, ldc, ${j};
    ${address(("cptr", "u64"), ("c", "u64"), dwidth, ("coff", "u32"), ("id", "u32"))}
%      else:
    ${address(("cptr", "u64"), ("c", "u64"), dwidth, (f"{ldc*j}", "u32"), ("id", "u32"))}
%      endif:
    ld.weak.global.${pftype} ctmp${j}, [cptr];
    add.${pftype} ctmp${j}, ctmp${j}, csub${j};
    st.weak.global.cg.${pftype} [cptr], ctmp${j};

%    elif kx == alix[j]:
% if n is None:
    mul.lo.u32 coff, ldc, ${j};
    ${address(("cptr", "u64"), ("c", "u64"), dwidth, ("coff", "u32"), ("id", "u32"))}
% else:
    ${address(("cptr", "u64"), ("c", "u64"), dwidth, (f"{ldc*j}", "u32"), ("id", "u32"))}
% endif:
    ld.weak.global.${pftype} ctmp${j}, [cptr];
    fma.rn.${pftype} ctmp${j}, ctmp${j}, ${beta}, csub${j};
    st.weak.global.cg.${pftype} [cptr], csub${j};
%    endif
%  endfor
%endfor
    }

  {
    .reg .u32 coff;
    .reg .u64 cptr;
    .reg .${pftype} fz;
    .reg .${putype} uz;
    .reg .${pftype} cin, cout;
    mov.${putype} uz, 0;
    mov.${pbtype} fz, uz;

%for j, jx in enumerate(afix):
%  if jx == -1 and beta == 0:
%    if n is None:
    mul.lo.u32 coff, ldc, ${j};
    ${address(("cptr", "u64"), ("c", "u64"), dwidth, ("coff", "u32"), ("id", "u32"))}
%     else:
    ${address(("cptr", "u64"), ("c", "u64"), dwidth, (f"{ldc*j}", "u32"), ("id", "u32"))}
%     endif:
    st.weak.global.cg.${pftype} [cptr], fz;

%  elif jx == -1 and beta != 1:
%    if n is None:
    mul.lo.u32 coff, ldc, ${j};
    ${address(("cptr", "u64"), ("c", "u64"), dwidth, ("coff", "u32"), ("id", "u32"))}
%    else:
    ${address(("cptr", "u64"), ("c", "u64"), dwidth, (f"{ldc*j}", "u32"), ("id", "u32"))}
%    endif:
    ld.weak.global.cg.${pftype} cin, [cptr];
    mul.${pftype} cout, cin, ${beta};
    st.weak.globla.cg.${pftype} [cptr], cout;
%    endif
%endfor
  }

$L_EXIT:
    ret;
}