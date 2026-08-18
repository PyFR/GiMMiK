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

## Kernel signature along with the leading dimension setup
<%def name="prologue()">\
__global__ void
% if n is None:
${kname}(int n,
         const ${dtype}* __restrict__ b, int ldb_,
         ${dtype}* __restrict__ c, int ldc_)
{
  % if width > 1:
    n = (n + ${width} - 1) / ${width};
    ldb_ /= ${width};
    ldc_ /= ${width};
  % endif
    const long long ldb = ldb_;
    const long long ldc = ldc_;
% else:
${kname}(const ${dtype}* __restrict__ b, ${dtype}* __restrict__ c)
{
    const int n = ${-(-n // width)};
    const ${'long long' if k*ldb >= width*2**31 else 'int'} ldb = ${ldb // width};
    const ${'long long' if m*ldc >= width*2**31 else 'int'} ldc = ${ldc // width};
% endif
</%def>

${next.body()}
