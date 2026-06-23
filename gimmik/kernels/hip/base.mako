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

% if width == 1:
static inline __device__ ${dtype}
gimmik_vmul(${dtype} a, ${dtype} b)
{
    return a*b;
}

static inline __device__ ${dtype}
gimmik_vadd(${dtype} a, ${dtype} b)
{
    return a + b;
}

static inline __device__ ${dtype}
gimmik_vmadd(${dtype} acc, ${dtype} a, ${dtype} b)
{
    // Keep the multiply-add expression visible to the compiler.
    return acc + a*b;
}
% elif width == 2:
static inline __device__ ${dtype}
gimmik_vmul(${dtype[:-1]} a, ${dtype} b)
{
    return make_${dtype}(a*b.x, a*b.y);
}

static inline __device__ ${dtype}
gimmik_vadd(${dtype} a, ${dtype} b)
{
    return make_${dtype}(a.x + b.x, a.y + b.y);
}

static inline __device__ ${dtype}
gimmik_vmadd(${dtype} acc, ${dtype[:-1]} a, ${dtype} b)
{
    // Keep the multiply-add expression visible to the compiler.
    return make_${dtype}(acc.x + a*b.x, acc.y + a*b.y);
}
% elif width == 4:
static inline __device__ ${dtype}
gimmik_vmul(${dtype[:-1]} a, ${dtype} b)
{
    return make_${dtype}(a*b.x, a*b.y, a*b.z, a*b.w);
}

static inline __device__ ${dtype}
gimmik_vadd(${dtype} a, ${dtype} b)
{
    return make_${dtype}(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

static inline __device__ ${dtype}
gimmik_vmadd(${dtype} acc, ${dtype[:-1]} a, ${dtype} b)
{
    // Keep the multiply-add expression visible to the compiler.
    return make_${dtype}(acc.x + a*b.x, acc.y + a*b.y, acc.z + a*b.z, acc.w + a*b.w);
}
% else:
#error "HIP vector helpers only support width=2 or width=4"
% endif

static inline __device__ void
nt_store_c(${dtype}* p, ${dtype} v)
{
% if dtype.endswith('4'):
    __builtin_nontemporal_store(v.x, &p->x);
    __builtin_nontemporal_store(v.y, &p->y);
    __builtin_nontemporal_store(v.z, &p->z);
    __builtin_nontemporal_store(v.w, &p->w);
% elif dtype.endswith('2'):
    __builtin_nontemporal_store(v.x, &p->x);
    __builtin_nontemporal_store(v.y, &p->y);
% else:
    __builtin_nontemporal_store(v, p);
% endif
}

static inline __device__ ${dtype}
nt_load_c(const ${dtype}* p)
{
% if dtype.endswith('4'):
    return make_${dtype}(__builtin_nontemporal_load(&p->x),
                         __builtin_nontemporal_load(&p->y),
                         __builtin_nontemporal_load(&p->z),
                         __builtin_nontemporal_load(&p->w));
% elif dtype.endswith('2'):
    return make_${dtype}(__builtin_nontemporal_load(&p->x),
                         __builtin_nontemporal_load(&p->y));
% else:
    return __builtin_nontemporal_load(p);
% endif
}

<% nt_c = context.get('nt_c', True) %>

static inline __device__ void
store_c(${dtype}* p, ${dtype} v)
{
% if nt_c:
    nt_store_c(p, v);
% else:
    *p = v;
% endif
}

static inline __device__ ${dtype}
load_c(const ${dtype}* p)
{
% if nt_c:
    return nt_load_c(p);
% else:
    return *p;
% endif
}

${next.body()}
