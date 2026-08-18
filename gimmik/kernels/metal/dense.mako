#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

% if skip:
constant ulong amask[] = { ${', '.join(f'{v}ul' for v in amask)} };
% endif

[[max_total_threads_per_threadgroup(${nthread})]]
kernel void ${kname}(device const ${sdtype}* a,
                     device const ${sdtype}* b, device ${sdtype}* c,
                     uint tgpos [[threadgroup_position_in_grid]],
                     uint tidtg [[thread_index_in_threadgroup]],
                     uint sgid [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]])
{
    threadgroup ${sdtype} bp[${kp*bs}];
    threadgroup ${sdtype} ix[64];

    const long col0 = (long)tgpos*${w};
    const device ${sdtype}* bcol = b + col0;

## Rows of the panel past the end of B contribute nothing
% if kp != k:
    for (uint i = ${k*bs} + tidtg; i < ${kp*bs}; i += ${nthread})
        bp[i] = 0.0;

% endif
    for (uint i = tidtg; i < ${k*w}; i += ${nthread})
    {
        uint r = i / ${w}, cc = i - r*${w};
        bp[r*${bs} + cc] = ${bload};
    }

## Each lane learns which element of the fragment it holds
    for (uint i = tidtg; i < 64; i += ${nthread})
        ix[i] = (${sdtype})i;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_${sdtype}8x8 im;
    simdgroup_load(im, ix, 8);

    const uint eo = (uint)im.thread_elements()[0];
% if cond:
    const ushort lrow = eo >> 3;
    const ushort lcol = eo & 7;
% endif

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

% if cond:
        const uint grow = mt*8 + lrow;

% endif
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < ${nw}; j++)
        {
% if cond:
            if (${cond})
            {
                device ${sdtype}* cp = c + mt*${8*ldc}L + col0 + j*8;
${store}
            }
            else if (grow < ${m})
            {
                const long cc = col0 + j*8 + lcol;
                device ${sdtype}* q = c + (long)grow*${ldc}L + cc;

                if (cc < ${n})
                    q[0] = ${e0};

                if (cc + 1 < ${n})
                    q[1] = ${e1};
            }
% else:
            device ${sdtype}* cp = c + mt*${8*ldc}L + col0 + j*8;
${store}
% endif
        }
    }
}
