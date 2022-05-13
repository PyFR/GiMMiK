% if dtype.endswith('4'):
inline __device__ ${dtype} operator+(${dtype} a, ${dtype} b)
{ return make_${dtype}(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }

inline __device__ ${dtype} operator*(${dtype[:-1]} a, ${dtype} b)
{ return make_${dtype}(a*b.x, a*b.y, a*b.z, a*b.w); }

inline __device__ void operator+=(${dtype} &a, ${dtype} b)
{ a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

inline __device__ ${dtype} make_zero()
{ return make_${dtype}(0, 0, 0, 0); }
% elif dtype.endswith('2'):
inline __device__ ${dtype} operator+(${dtype} a, ${dtype} b)
{ return make_${dtype}(a.x + b.x, a.y + b.y); }

inline __device__ ${dtype} operator*(${dtype[:-1]} a, ${dtype} b)
{ return make_${dtype}(a*b.x, a*b.y); }

inline __device__ void operator+=(${dtype} &a, ${dtype} b)
{ a.x += b.x; a.y += b.y; }

inline __device__ ${dtype} make_zero()
{ return make_${dtype}(0, 0); }
% else:
inline __device__ ${dtype} make_zero()
{ return 0; }
% endif

${next.body()}
