<%inherit file='base'/>
##
## Dense double-precision GEMM using the CDNA Matrix Cores (MFMA).
##
## Modelled on the NVIDIA PTX "dmma-astream" kernel (mma.sync.aligned.m8n8k4):
## the constant operand A is densified, padded and *baked* into the kernel in
## Matrix-Core fragment order, B is streamed straight from global memory and C
## is written with non-temporal stores.  The NVIDIA tensor-core tile is
## m8n8k4; the CDNA f64 Matrix-Core tile is m16n16k4, computed by
## __builtin_amdgcn_mfma_f64_16x16x4f64 over a single 64-lane wavefront.
##
## ---------------------------------------------------------------------------
## Operand lane layout for v_mfma_f64_16x16x4_f64 (wave64), with
##     g = lane / 16   (0..3)      p = lane % 16   (0..15)
##   A  (16x4, 1 reg/lane):  A[i][kk]  with  i = p,        kk = g
##   B  (4x16, 1 reg/lane):  B[kk][j]  with  kk = g,       j  = p
##   C/D(16x16, 4 reg/lane): D[i][j]   with  j = p,        i  = 4*g + reg
## The baked A array (built in hip.py) uses EXACTLY this mapping:
##   Ag[(mt*k_tiles + kt)*64 + lane] = A_padded[mt*16 + (lane%16)][kt*4 + (lane//16)]
## If an on-device accuracy check fails, this single mapping (here + in
## _mfma_dense_bake) is the place to revisit -- A, B and C are all derived
## from it consistently.
## ---------------------------------------------------------------------------
##
<%
    tiles = blockx // 16          # 16-wide N-tiles swept per wavefront (= cols/block / 16)
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
    const int g = lane / 16;                 // 0..3
    const int p = lane % 16;                 // 0..15
    const int col_base = ${blockx}*blockIdx.x;

    ${dtype} a;
% for mt in range(m_tiles):
%  for t in range(tiles):
    gimmik_f64x4 acc_${mt}_${t} = {0.0, 0.0, 0.0, 0.0};
%  endfor
% endfor

% for kt in range(k_tiles):
<%
    krow_guard = (kt + 1)*4 > k
%>
%  for t in range(tiles):
    ${dtype} bv_${t};
    {
        const int col = col_base + ${t*16} + p;
        const int krow = ${kt*4} + g;
        bv_${t} = (col < n${' && krow < %d' % k if krow_guard else ''}) ? b[krow*ldb + col] : (${dtype})0;
    }
%  endfor
%  for mt in range(m_tiles):
    a = ${kname}_Ag[${(mt*k_tiles + kt)*64} + lane];
%   for t in range(tiles):
    acc_${mt}_${t} = __builtin_amdgcn_mfma_f64_16x16x4f64(a, bv_${t}, acc_${mt}_${t}, 0, 0, 0);
%   endfor
%  endfor
% endfor

% for mt in range(m_tiles):
%  for t in range(tiles):
%   for reg in range(4):
    {
        const int row = ${mt*16 + reg} + 4*g;
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
