## Kernel signature along with the leading dimension setup
<%def name="prologue(wgs=None)">\
% if wgs is not None:
__kernel __attribute__((reqd_work_group_size(${wgs[0]}, ${wgs[1]}, 1))) void
% else:
__kernel void
% endif
% if n is None:
${kname}(int n,
         __global const ${dtype}* restrict b, int ldb_,
         __global ${dtype}* restrict c, int ldc_)
{
  % if width > 1:
    n = (n + ${width} - 1) / ${width};
    ldb_ /= ${width};
    ldc_ /= ${width};
  % endif
    const long ldb = ldb_;
    const long ldc = ldc_;
% else:
${kname}(__global const ${dtype}* restrict b, __global ${dtype}* restrict c)
{
    const int n = ${-(-n // width)};
    const ${'long' if k*ldb >= width*2**31 else 'int'} ldb = ${ldb // width};
    const ${'long' if m*ldc >= width*2**31 else 'int'} ldc = ${ldc // width};
% endif
</%def>

${next.body()}
