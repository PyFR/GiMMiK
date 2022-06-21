% if dtype.endswith('4'):
static inline __device__ ${dtype} make_zero()
{ return make_${dtype}(0, 0, 0, 0); }
% elif dtype.endswith('2'):
static inline __device__ ${dtype} make_zero()
{ return make_${dtype}(0, 0); }
% else:
static inline __device__ ${dtype} make_zero()
{ return 0; }
% endif

${next.body()}
