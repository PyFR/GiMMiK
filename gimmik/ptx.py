# -*- coding: utf-8 -*-

import struct

import numpy as np

from gimmik.base import MatMul


class PTXMatMul(MatMul):
    platform = 'ptx'
    basemeta = {'block': (128, 1, 1), 'width': 1, 'shared': 0,
                'dynamic_shared': 0}

    def _kernel_generators(self, dtype, dsize, *, compute_capability=None,
                           trim_a=False):
        base_args = {'cc': compute_capability,
                     'pred_emit': self._pred_emit,
                     'trim_a': bool(trim_a) and dtype == 'double'}

        yield from self._sparse_kernel_generators(dtype, dsize, base_args)
        yield from self._dense_kernel_generators(dtype, dsize, base_args)

    def _sparse_kernel_generators(self, dtype, dsize, base_args):
        arr = self.A
        nnz = int(np.count_nonzero(arr))
        nuq = int(len(np.unique(np.abs(arr))))
        density = nnz / arr.size
        if not ((nuq <= 28) or (density <= 0.15)):
            return

        # B loading, C streaming kernel
        yield ('cstream', base_args | {}, {'desc': 'cstream'})

        # B streaming, C accumulation kernel
        yield ('bstream', base_args | {}, {'desc': 'bstream'})

        # Four-way m-split B streaming, C accumulation kernel
        ms, bsz, blkx = 4, 24, 32
        args = base_args | {'msplit': ms, 'bsz': bsz, 'blockx': blkx}
        meta = {'block': (blkx, ms, 1), 'shared': 2*bsz*blkx*dsize,
                'desc': f'bstream-msplit/m{ms}-b{bsz}-x{blkx}'}
        yield ('bstream-msplit', args, meta)

        # Single-warp LDGSTS variant for medium-M beta=0 large-K cases
        if self.beta == 0 and self.m <= 320 and len(self.bix) >= 64:
            ms, bsz, blkx = 1, 32, 64
            args = base_args | {'msplit': ms, 'bsz': bsz, 'blockx': blkx}
            meta = {'block': (blkx, ms, 1),
                    'shared': 2*bsz*blkx*dsize,
                    'desc': f'bstream-msplit/m{ms}-b{bsz}-x{blkx}'}
            yield ('bstream-msplit', args, meta)

        # Two-way k-split B loading, C streaming kernel
        ks, csz, blkx = 2, 24, 32
        args = base_args | {'ksplit': ks, 'csz': csz, 'blockx': blkx}
        meta = {'block': (blkx, ks, 1), 'shared': (ks - 1)*csz*blkx*dsize,
                'desc': f'cstream-ksplit/k{ks}-c{csz}-x{blkx}'}
        yield ('cstream-ksplit', args, meta)

        # Four-way k-split for large K
        K_used = len(self.bix)
        if K_used > 500:
            ks, csz, blkx = 4, 20, 32
            args = base_args | {'ksplit': ks, 'csz': csz, 'blockx': blkx}
            meta = {'block': (blkx, ks, 1),
                    'shared': (ks - 1)*csz*blkx*dsize,
                    'desc': f'cstream-ksplit/k{ks}-c{csz}-x{blkx}'}
            yield ('cstream-ksplit', args, meta)

        # Width-2 vector cstream for fp64 small-K
        if (dtype == 'double' and self.n is not None and self.n % 2 == 0
                and K_used <= 100
                and (self.aligne is None or self.aligne % 2 == 0)):
            blkx = 128
            args = base_args | {'blockx': blkx}
            meta = {'block': (blkx, 1, 1), 'width': 2,
                    'desc': f'cstream-w2/x{blkx}'}
            yield ('cstream-w2', args, meta)

    def _dense_kernel_generators(self, dtype, dsize, base_args):
        cc = base_args['cc'] or (0, 0)
        if not (dtype == 'double' and cc >= (9, 0) and self.n is not None
                and self.m <= 128 and self.k <= 128):
            return

        # Dense DMMA m8n8k4; block stealing default on sm_100+ for gA
        bs_default = cc >= (10, 0)
        dense_configs = [
            ('dense-mma-smem-gA', 1, 8),
            ('dense-mma-smem-gA', 2, 4),
            ('dense-mma-smem-gA', 4, 4),
            ('dense-mma-gAd',     2, 2),
            ('dense-mma-gAd',     4, 2),
        ]
        for tpl, nn, w in dense_configs:
            blkx = 32 * w
            n_per_cta = 8 * nn * w
            if n_per_cta > self.n:
                continue
            bs = (tpl == 'dense-mma-smem-gA') and bs_default
            setup = self._dense_mma_setup(nn=nn, warps_per_cta=w)
            args = (base_args | {'warps_per_cta': w, 'nn': nn,
                                 'block_stealing': bs} | setup)
            meta = {
                'block': (blkx, 1, 1),
                'grid': (-(-self.n // n_per_cta), 1, 1),
                'desc': f'{tpl}/nn{nn}-w{w}{"-bs" if bs else ""}',
            }
            yield (tpl, args, meta)

    def _dense_mma_setup(self, *, nn, warps_per_cta):
        a = self.A
        m, k = a.shape
        m_tiles = -(-m // 8)
        k_rem   = k % 4
        k_iters = (k + (4 - k_rem if k_rem else 0)) // 4

        # A in fragment layout: lane l -> A[m_tile*8 + l/4][k_iter*4 + l%4]
        a_u64 = []
        for m_tile in range(m_tiles):
            for k_iter in range(k_iters):
                for lane in range(32):
                    i = m_tile * 8 + lane // 4
                    j = k_iter * 4 + lane % 4
                    v = float(a[i, j]) if (i < m and j < k) else 0.0
                    u = struct.unpack('<Q', struct.pack('<d', v))[0]
                    a_u64.append(f'0x{u:016x}')

        n_per_warp = 8 * nn
        n_per_cta  = warps_per_cta * n_per_warp
        a_elems    = m_tiles * k_iters * 32

        # Predicate-elision flags
        n_col_aligned = (self.n is not None and self.n % n_per_warp == 0)
        def pm_runtime(mt):
            return (mt + 1) * 8 > m

        return {
            'm_tiles': m_tiles,
            'k_rem': k_rem, 'k_iters': k_iters,
            'a_u64': a_u64,
            'n_per_warp': n_per_warp, 'n_per_cta': n_per_cta,
            'a_elems': a_elems,
            'frag_stride_bytes': 32 * 8,
            'b_kiter_stride': 4 * (self.ldb or 0) * 8,
            'b_ntile_stride': 8 * 8,
            'c_mtile_stride': 8 * (self.ldc or 0) * 8,
            'c_ntile_stride': 8 * 8,
            'n_col_aligned': n_col_aligned,
            'pm_runtime': pm_runtime,
        }

    @staticmethod
    def _pred_emit(instr, *preds, pred_reg=None, indent=' ' * 8):
        actual = [p for p in preds if p is not None]
        if not actual:
            return instr
        if len(actual) == 1:
            return f'@{actual[0]} {instr}'
        if pred_reg is None:
            raise ValueError('pred_reg required when combining multiple '
                             'predicates')
        lines = [f'.reg .pred {pred_reg};',
                 f'and.pred {pred_reg}, {actual[0]}, {actual[1]};']
        for p in actual[2:]:
            lines.append(f'and.pred {pred_reg}, {pred_reg}, {p};')
        lines.append(f'@{pred_reg} {instr}')
        return f'\n{indent}'.join(lines)

    def _process_meta(self, meta):
        if self.n is not None and 'grid' not in meta:
            div = meta['block'][0]*meta['width']
            meta['grid'] = (-(-self.n // div), 1, 1)
