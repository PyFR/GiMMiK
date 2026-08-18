#include <metal_stdlib>

using namespace metal;

% if dtype.endswith('4'):
static inline ${dtype} make_zero()
{ return ${dtype}(0, 0, 0, 0); }
% elif dtype.endswith('2'):
static inline ${dtype} make_zero()
{ return ${dtype}(0, 0); }
% else:
static inline ${dtype} make_zero()
{ return 0; }
% endif

## Kernel signature along with the leading dimension setup
<%def name="prologue(tpos)">\
<% tpos = ',\n         '.join(tpos) %>\
kernel void
% if n is None:
${kname}(constant int& n_,
         device ${dtype}* b, constant int& ldb_,
         device ${dtype}* c, constant int& ldc_,
         ${tpos})
{
  % if width > 1:
    const int n = (n_ + ${width} - 1) / ${width};
    const long ldb = ldb_ / ${width};
    const long ldc = ldc_ / ${width};
  % else:
    const int n = n_;
    const long ldb = ldb_;
    const long ldc = ldc_;
  % endif
% else:
${kname}(device const ${dtype}* b, device ${dtype}* c,
         ${tpos})
{
    const int n = ${-(-n // width)};
    const ${'long' if k*ldb >= width*2**31 else 'int'} ldb = ${ldb // width};
    const ${'long' if m*ldc >= width*2**31 else 'int'} ldc = ${ldc // width};
% endif
</%def>

${next.body()}
