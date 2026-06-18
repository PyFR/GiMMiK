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

${next.body()}
