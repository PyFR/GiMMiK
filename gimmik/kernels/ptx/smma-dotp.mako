<%inherit file='smma-tt'/>

<%def name="compute_init()">
    // Keep the small constant operator in registers across grid strides.
    .reg .f32 ar_<${mt*k}>;
% for r in range(mt):
% for ik in range(k):
% if k_used[ik]:
    ld.shared.f32 ar_${r*k+ik}, [ap + ${(ik*mpad+r)*4}];
% endif
% endfor
% endfor
</%def>

<%def name="compute_panel(kb)">
% for ik in range(0,kt,4):
% for v in range(nt//4):
    {
        .reg .f32 bv_<16>;
% for x in range(4):
<% ix = kb*kt+ik+x %>
% if ix < k and k_used[ix]:
        ld.shared.v4.f32 {bv_${x*4}, bv_${x*4+1}, bv_${x*4+2}, bv_${x*4+3}}, [bp + ${((ik+x)*panel+v*4*workers)*4}];
% endif
% endfor
% for r in range(mt):
% for x in range(4):
<% ix = kb*kt+ik+x %>
% if ix < k and k_used[ix]:
% for y in range(4):
        fma.rn.f32 acc_${r*nt+v*4+y}, ar_${r*k+ix}, bv_${x*4+y}, acc_${r*nt+v*4+y};
% endfor
% endif
% endfor
% endfor
    }
% endfor
% endfor
</%def>
