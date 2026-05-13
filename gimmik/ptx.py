# -*- coding: utf-8 -*-

import struct

import numpy as np

from gimmik.base import MatMul


class PTXMatMul(MatMul):
    platform = 'ptx'
    basemeta = {'block': (128, 1, 1), 'width': 1, 'shared': 0,
                'dynamic_shared': 0}

    @staticmethod
    def is_sparse_suitable(arr):
        nnz = int(np.count_nonzero(arr))
        nuq = int(len(np.unique(np.abs(arr))))
        density = nnz / arr.size
        return (nuq <= 28) or (density <= 0.15)

    # Shape/arch gate for dense DMMA; n/ldb/ldc are validated at generate time
    @staticmethod
    def is_dense_suitable(arr, dtype, cc):
        return (np.dtype(dtype) == np.float64
                and cc is not None and cc >= (9, 0)
                and arr.shape[0] <= 128 and arr.shape[1] <= 128)

    @classmethod
    def is_suitable(cls, arr, dtype, cc):
        return (cls.is_sparse_suitable(arr)
                or cls.is_dense_suitable(arr, dtype, cc))

    def _kernel_generators(self, dtype, dsize, *, compute_capability=None):
        base_args = {'cc': compute_capability,
                     'pred_emit': self._pred_emit}

        yield from self._sparse_kernel_generators(dtype, dsize, base_args)
        yield from self._dense_kernel_generators(dtype, dsize, base_args)

    def _sparse_kernel_generators(self, dtype, dsize, base_args):
        if not self.is_sparse_suitable(self.A):
            return

        # B loading, C streaming kernel
        yield ('cstream', base_args, {'desc': 'cstream'})

        # B streaming, C accumulation kernel
        yield ('bstream', base_args, {'desc': 'bstream'})

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
        if not (self.is_dense_suitable(self.A, dtype, cc)
                and self.n is not None):
            return

        # Some kernels can optional steal blocks
        bs_default = cc >= (10, 0)

        if cc >= (10, 0):
            # Warp specialised is uniformly better on sm_100+, so no need to JIT
            # other versions
            dense_configs = [('dense-mma-smem-gA', 4, 4)]
        else:
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

        # Warp-specialised dense DMMA
        if cc >= (10, 0):
            yield from self._dense_ws_kernel_generators(dtype, dsize, base_args)

    def _dense_ws_kernel_generators(self, dtype, dsize, base_args):
        m_pad = -(-self.m // 8) * 8
        k_pad = -(-self.k // 4) * 4

        # (nn, w_compute) -- block has w_compute + 2 warps (producer, stealer)
        ws_configs = [(1, 4), (2, 4), (4, 4)]
        for nn, w in ws_configs:
            n_per_cta = 8 * nn * w
            if n_per_cta > self.n:
                continue
            blkx = 32 * (w + 2)
            setup = self._dense_mma_setup(nn=nn, warps_per_cta=w)
            ws_layout = self._dense_ws_layout(
                n_comp_warps=w, n_per_cta=n_per_cta,
                m_pad=m_pad, k_pad=k_pad, a_elems=setup['a_elems']
            )

            if ws_layout['dynm_total_bytes'] > 200 * 1024:
                continue

            args = (base_args
                    | {'warps_per_cta': w, 'nn': nn}
                    | setup | ws_layout)
            meta = {
                'block': (blkx, 1, 1),
                'grid': (-(-self.n // n_per_cta), 1, 1),
                'desc': f'dense-mma-ws/nn{nn}-w{w}',
                'ws_tensor_map': True,
                'ws_n_per_cta': n_per_cta,
                'ws_k_pad': k_pad,
                'ws_m_pad': m_pad,
                'dynamic_shared': ws_layout['dynm_total_bytes'],
            }
            yield ('dense-mma-ws', args, meta)

    @staticmethod
    def _dense_ws_layout(*, n_comp_warps, n_per_cta, m_pad, k_pad, a_elems):
        n_total_warps   = n_comp_warps + 2
        blockx_total    = 32 * n_total_warps

        b_tile_bytes = k_pad * n_per_cta * 8
        c_tile_bytes = m_pad * n_per_cta * 8
        a_bytes      = a_elems * 8

        smem_size = {'b1': b_tile_bytes, 'b2': b_tile_bytes, 'c': c_tile_bytes,
                     'a': a_bytes, 'wid': 16}
        smem_off, off = {}, 0
        for k, v in smem_size.items():
            off = (off + 15) & ~15
            smem_off[f'{k}_off'] = off
            off += v

        mbar_names = ('tma', 'bready', 'cready', 'cstored',
                      'steal', 'wid_new', 'wid_used')
        for k in mbar_names:
            smem_off[f'{k}_mbar_off'] = off
            off += 8

        # Pad total to 16-byte multiple
        dynm_total_bytes = (off + 15) & ~15

        params = {'n_comp_warps': n_comp_warps,
                  'blockx_total': blockx_total,
                  'prod_warp': n_comp_warps,
                  'steal_warp': n_comp_warps + 1,
                  'comp_threads': 32 * n_comp_warps,
                  'm_pad': m_pad,
                  'k_pad': k_pad,
                  'b_tile_doubles': k_pad * n_per_cta,
                  'b_tile_bytes': b_tile_bytes,
                  'c_tile_doubles': m_pad * n_per_cta,
                  'c_mtile_smem_stride': 8 * n_per_cta * 8,
                  'c_ntile_smem_stride': 8 * 8,
                  'dynm_total_bytes': dynm_total_bytes,
                  }
        params |= smem_off
        return params

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
