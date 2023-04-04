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

${next.body()}
