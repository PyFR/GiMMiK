<%inherit file='base'/>
##
## Dense f64 GEMM on the CDNA Matrix Cores (MFMA) -- software-pipelined variant.
##
## Same maths as mfma-dense (msplit=1 direct path): A densified + baked in
## fragment order, B streamed from global, C non-temporal stored, epilogue
## fully unrolled, zero-tile skipping via amask.  The only difference: B for
## the NEXT k-tile is issued before the MFMAs of the CURRENT k-tile, so the
## global-load latency overlaps the Matrix-Core work (double-buffered B in
## registers, buffers 0/1 alternated per k-tile).
##
## Operand lane layout for v_mfma_f64_16x16x4_f64 (wave64), g=lane/16, p=lane%16:
##   A  (16x4 ): A[i][kk]  i=p,        kk=g
##   B  (4x16 ): B[kk][j]  kk=g,       j=p
##   C/D(16x16): D[i][j]   j=p,        i=4*reg + g   (v4f64)
##
<%
    tiles = blockx // 16
    active_kt = [kt for kt in range(k_tiles)
                 if any(amask[mt][kt] for mt in range(m_tiles))]
%>
typedef ${dtype} gimmik_f64x4 __attribute__((ext_vector_type(4)));

__device__ static const ${dtype} ${kname}_Ag[${m_tiles * k_tiles * 64}] = {
    ${', '.join(a_hex)}
};

__global__ __launch_bounds__(${blockx}) void
% if n is None:
${kname}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
% else:
${kname}(const ${dtype}* __restrict__ b, ${dtype}* __restrict__ c)
{
    const int n = ${n};
    const ${'long long' if k * ldb >= 2**31 else 'int'} ldb = ${ldb};
    const ${'long long' if m * ldc >= 2**31 else 'int'} ldc = ${ldc};
% endif
    const int lane = threadIdx.x;
    const int g = lane / 16;
    const int p = lane % 16;
    const int col_base = ${blockx}*blockIdx.x;

    ${dtype} a;
    ${dtype} ${', '.join('bv0_%d' % t for t in range(tiles))};
    ${dtype} ${', '.join('bv1_%d' % t for t in range(tiles))};
% for mt in range(m_tiles):
%  for t in range(tiles):
    gimmik_f64x4 acc_${mt}_${t} = {0.0, 0.0, 0.0, 0.0};
%  endfor
% endfor

% if active_kt:
<% kt0 = active_kt[0]; g0 = (kt0 + 1)*4 > k %>
    // prefetch the first k-tile into buffer 0
%  for t in range(tiles):
    {
        const int col = col_base + ${t*16} + p;
        const int krow = ${kt0*4} + g;
        bv0_${t} = (col < n${' && krow < %d' % k if g0 else ''}) ? b[krow*ldb + col] : (${dtype})0;
    }
%  endfor

%  for i, kt in enumerate(active_kt):
<%
    cur = i % 2
    nxt = (i + 1) % 2
    has_next = i + 1 < len(active_kt)
%>
%   if has_next:
<% knext = active_kt[i+1]; gN = (knext + 1)*4 > k %>
    // prefetch k-tile ${knext} into buffer ${nxt}
%    for t in range(tiles):
    {
        const int col = col_base + ${t*16} + p;
        const int krow = ${knext*4} + g;
        bv${nxt}_${t} = (col < n${' && krow < %d' % k if gN else ''}) ? b[krow*ldb + col] : (${dtype})0;
    }
%    endfor
%   endif
%   for mt in range(m_tiles):
%    if amask[mt][kt]:
    a = ${kname}_Ag[${(mt*k_tiles + kt)*64} + lane];
%     for t in range(tiles):
    acc_${mt}_${t} = __builtin_amdgcn_mfma_f64_16x16x4f64(a, bv${cur}_${t}, acc_${mt}_${t}, 0, 0, 0);
%     endfor
%    endif
%   endfor
%  endfor
% endif

% for mt in range(m_tiles):
%  for t in range(tiles):
%   for reg in range(4):
    {
        const int row = ${mt*16 + 4*reg} + g;
        const int col = col_base + ${t*16} + p;
        if (row < ${m} && col < n)
% if beta == 0:
            store_c(&c[row*ldc + col], acc_${mt}_${t}[${reg}]);
% elif beta == 1:
            store_c(&c[row*ldc + col], gimmik_vadd(load_c(&c[row*ldc + col]), acc_${mt}_${t}[${reg}]));
% else:
            store_c(&c[row*ldc + col], gimmik_vadd(gimmik_vmul(${beta}, load_c(&c[row*ldc + col])), acc_${mt}_${t}[${reg}]));
% endif
    }
%   endfor
%  endfor
% endfor
}
