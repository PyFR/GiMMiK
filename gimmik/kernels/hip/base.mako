% if dtype.endswith('4'):
inline __device__ ${dtype} make_zero()
{ return make_${dtype}(0, 0, 0, 0); }
% elif dtype.endswith('2'):
inline __device__ ${dtype} make_zero()
{ return make_${dtype}(0, 0); }
% else:
inline __device__ ${dtype} make_zero()
{ return 0; }
% endif

${next.body()}
