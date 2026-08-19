#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

% if skip:
constant ulong amask[] = { ${', '.join(f'{v}ul' for v in amask)} };
% endif

[[max_total_threads_per_threadgroup(${nthread})]]
kernel void ${kname}(device const ${sdtype}* a,
                     constant int& n_,
                     device const ${sdtype}* b, constant int& ldb_,
                     device ${sdtype}* c, constant int& ldc_,
                     uint tgpos [[threadgroup_position_in_grid]],
                     uint tidtg [[thread_index_in_threadgroup]],
                     uint sgid [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]])
{
    const long n = n_, ldb = ldb_, ldc = ldc_;

    threadgroup ${sdtype} bp[${kp*bs}];
    threadgroup ${sdtype} ix[64];

    const long col0 = (long)tgpos*${w};
    const device ${sdtype}* bcol = b + col0;

## A panel which lies wholly inside B needs no bounds checking at all
    const bool full = col0 + ${w} <= n;

## Rows of the panel past the end of B contribute nothing
% if kp != k:
    for (uint i = ${k*bs} + tidtg; i < ${kp*bs}; i += ${nthread})
        bp[i] = 0.0;

% endif
    if (full)
    {
        for (uint i = tidtg; i < ${k*w}; i += ${nthread})
        {
            uint r = i / ${w}, cc = i - r*${w};
            bp[r*${bs} + cc] = bcol[r*ldb + cc];
        }
    }
    else
    {
        for (uint i = tidtg; i < ${k*w}; i += ${nthread})
        {
            uint r = i / ${w}, cc = i - r*${w};
            bp[r*${bs} + cc] = (col0 + cc < n) ? bcol[r*ldb + cc] : 0.0;
        }
    }

## Each lane learns which element of the fragment it holds
    for (uint i = tidtg; i < 64; i += ${nthread})
        ix[i] = (${sdtype})i;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_${sdtype}8x8 im;
    simdgroup_load(im, ix, 8);

    const uint eo = (uint)im.thread_elements()[0];
    const ushort lrow = eo >> 3;
    const ushort lcol = eo & 7;

    for (uint mt = sgid; mt < ${ntm}; mt += ${ns})
    {
        simdgroup_${sdtype}8x8 acc[${nw}];

        #pragma clang loop unroll(full)
        for (ushort j = 0; j < ${nw}; j++)
            acc[j] = simdgroup_${sdtype}8x8(0);

% if skip:
        const ulong msk = amask[mt];

% endif
        #pragma clang loop unroll(full)
        for (ushort kt = 0; kt < ${ntk}; kt++)
        {
% if skip:
            if (!(msk & (1ul << kt)))
                continue;

% endif
            const uint ao = (mt*${ntk} + kt)*64 + eo;

            simdgroup_${sdtype}8x8 af;
            af.thread_elements()[0] = a[ao];
            af.thread_elements()[1] = a[ao + 1];

            #pragma clang loop unroll(full)
            for (ushort j = 0; j < ${nw}; j++)
            {
                simdgroup_${sdtype}8x8 bf;
                simdgroup_load(bf, bp + kt*${8*bs} + j*8, ${bs});
                simdgroup_multiply_accumulate(acc[j], af, bf, acc[j]);
            }
        }

        const uint grow = mt*8 + lrow;

## Whole 8x8 stores need every row and column of the tile to be live
        if (${cond})
        {
            #pragma clang loop unroll(full)
            for (ushort j = 0; j < ${nw}; j++)
            {
                device ${sdtype}* cp = c + mt*8*ldc + col0 + j*8;
${store}
            }
        }
        else if (grow < ${m})
        {
            #pragma clang loop unroll(full)
            for (ushort j = 0; j < ${nw}; j++)
            {
                const long cc = col0 + j*8 + lcol;
                device ${sdtype}* q = c + grow*ldc + cc;

                if (cc < n)
                    q[0] = ${e0};

                if (cc + 1 < n)
                    q[1] = ${e1};
            }
        }
    }
}
