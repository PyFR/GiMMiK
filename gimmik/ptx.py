import numpy as np

from gimmik.base import MatMul


class PTXMatMul(MatMul):
    platform = 'ptx'
    basemeta = {
        'block': (128, 1, 1),
        'width': 1,
        'shared': 0,
        'dynamic_shared': 0
    }

    # Map Supported CC -> Minimum PTX version
    PTX_SM = {(8, 0): (7, 0), (9, 0): (8, 0), (10, 0): (8, 7), (10, 3): (8, 7),
              (12, 0): (8, 7), (12, 1): (8, 7)}

    @classmethod
    def is_sparse_suitable(cls, arr, cc):
        nnz = np.count_nonzero(arr)
        nuq = len(np.unique(np.abs(arr)))
        density = nnz / arr.size
        return ((nuq <= 28) or (density <= 0.15)) and cc in cls.PTX_SM

    @classmethod
    def is_dense_suitable(cls, arr, cc):
        cc_appropriate = cc in cls.PTX_SM and cc >= (9, 0)
        return (arr.dtype == np.float64 and cc_appropriate
                and arr.shape[0] <= 128 and arr.shape[1] <= 128)

    @classmethod
    def is_suitable(cls, arr, cc):
        return cls.is_sparse_suitable(arr, cc) or cls.is_dense_suitable(arr, cc)

    def _kernel_generators(self, dtype, dsize, *, compute_capability=None,
                           smem_info=None):
        cc = compute_capability or (0, 0)
        ptx = self.PTX_SM.get(cc, (0, 0))
        smem_info = smem_info or (48*1024, 48*1024)
        base_args = {
            'ptx': ptx,
            'cc': cc,
            'smem_info': smem_info,
            'pred_emit': self._pred_emit,
            'pftype': 'f32' if dtype == 'float' else 'f64',
            'dwidth_i': 4 if dtype == 'float' else 8,
            'fzero': ('0f00000000' if dtype == 'float'
                      else '0d0000000000000000'),
            'beta_zero': self.beta == 0,
            'mbar_maxwait': '0x989680',
        }

        if self.is_sparse_suitable(self.A, cc):
            yield from self._sparse_kernel_generators(dtype, dsize, base_args)

        if self.is_dense_suitable(self.A, cc):
            yield from self._dense_kernel_generators(dtype, dsize, base_args)

    def _sparse_kernel_generators(self, dtype, dsize, base_args):
        # Sparse-shared template constants
        base_args = base_args | {
            'has_zero_rows': bool(self.has_zero_rows),
            'row_nz': [[(kx, self.A[j, kx]) for kx in range(self.k)
                        if self.A[j, kx] != 0] for j in range(self.m)],
        }

        # B loading, C streaming kernel
        yield ('cstream', base_args, {'desc': 'cstream'})

        # B streaming, C accumulation kernel
        yield ('bstream', base_args, {'desc': 'bstream'})

        # Four-way m-split B streaming, C accumulation kernel
        ms, bsz, blkx = 4, 24, 32
        args = base_args | {'msplit': ms, 'bsz': bsz, 'blockx': blkx}
        meta = {
            'block': (blkx, ms, 1),
            'shared': 2*bsz*blkx*dsize,
            'desc': f'bstream-msplit/m{ms}-b{bsz}-x{blkx}',
        }
        yield ('bstream-msplit', args, meta)

        # Single-warp LDGSTS variant for medium-M beta=0 large-K cases
        if self.beta == 0 and self.m <= 320 and len(self.bix) >= 64:
            ms, bsz, blkx = 1, 32, 64
            args = base_args | {'msplit': ms, 'bsz': bsz, 'blockx': blkx}
            meta = {
                'block': (blkx, ms, 1),
                'shared': 2*bsz*blkx*dsize,
                'desc': f'bstream-msplit/m{ms}-b{bsz}-x{blkx}',
            }
            yield ('bstream-msplit', args, meta)

        # Two-way k-split B loading, C streaming kernel
        ks, csz, blkx = 2, 24, 32
        args = base_args | {'ksplit': ks, 'csz': csz, 'blockx': blkx}
        meta = {
            'block': (blkx, ks, 1),
            'shared': (ks - 1)*csz*blkx*dsize,
            'desc': f'cstream-ksplit/k{ks}-c{csz}-x{blkx}',
        }
        yield ('cstream-ksplit', args, meta)

        # Four-way k-split for large K
        K_used = len(self.bix)
        if K_used > 500:
            ks, csz, blkx = 4, 20, 32
            args = base_args | {'ksplit': ks, 'csz': csz, 'blockx': blkx}
            meta = {
                'block': (blkx, ks, 1),
                'shared': (ks - 1)*csz*blkx*dsize,
                'desc': f'cstream-ksplit/k{ks}-c{csz}-x{blkx}',
            }
            yield ('cstream-ksplit', args, meta)

        # Width-2 vector cstream for fp64 small-K
        if (dtype == 'double' and self.n is not None and self.n % 2 == 0
                and K_used <= 100
                and (self.aligne is None or self.aligne % 2 == 0)):
            blkx = 128
            args = base_args | {'blockx': blkx}
            meta = {
                'block': (blkx, 1, 1),
                'width': 2,
                'desc': f'cstream-w2/x{blkx}',
            }
            yield ('cstream-w2', args, meta)

    def _dense_kernel_generators(self, dtype, dsize, base_args):
        cc = base_args['cc'] or (0, 0)

        # Block stealing requires sm_100+
        block_steal = cc >= (10, 0)
        if block_steal:
            dense_configs = [('dense-mma-smem-gA', 4, 4)]
        else:
            dense_configs = [
                ('dense-mma-smem-gA', 1, 8),
                ('dense-mma-smem-gA', 2, 4),
                ('dense-mma-smem-gA', 4, 4),
                ('dense-mma-gAd', 2, 2),
                ('dense-mma-gAd', 4, 2),
            ]

        for tpl, nn, w in dense_configs:
            if (n_per_cta := 8 * nn * w) > self.n:
                continue
            setup = self._dense_mma_setup(nn, w, block_steal)
            blkx = 32 * w
            args = base_args | setup
            meta = {
                'block': (blkx, 1, 1),
                'grid': (-(-self.n // n_per_cta), 1, 1),
                'desc': f'{tpl}/nn{nn}-w{w}{'-bs' if block_steal else ''}',
            }
            yield (tpl, args, meta)

        # Warp-specialised dense DMMA, required block stealing
        if block_steal:
            yield from self._dense_ws_kernel_generators(dtype, dsize, base_args)

    def _dense_ws_kernel_generators(self, dtype, dsize, base_args):
        static_max, dynamic_max = base_args['smem_info']

        # (nn, compute) -- block has compute + 2 warps (producer, stealer)
        ws_configs = [(1, 4), (2, 4), (4, 4)]
        for nn, w in ws_configs:
            if (n_per_cta := 8 * nn * w) > self.n:
                continue

            setup = self._dense_mma_setup(nn, w, True)
            ws_setup = self._dense_ws_setup(setup, w)

            if ws_setup['dynm_total_bytes'] > dynamic_max:
                continue

            blkx = 32 * (w + 2)
            args = base_args | setup | ws_setup
            meta = {
                'block': (blkx, 1, 1),
                'grid': (-(-self.n // n_per_cta), 1, 1),
                'desc': f'dense-mma-ws/nn{nn}-w{w}',
                'ws_b_tile': (n_per_cta, setup['k_pad']),
                'dynamic_shared': ws_setup['dynm_total_bytes'],
            }
            if self.beta != 0:
                meta |= {'ws_out_tile': (n_per_cta, setup['m_pad'])}
            yield ('dense-mma-ws', args, meta)

    @staticmethod
    def _dsmem_alloc(regions, mbars, align=16):
        out, off = {}, 0
        for name, size in regions:
            off = (off + align - 1) & ~(align - 1)
            out[f'{name}_off'] = off
            off += size
        for name in mbars:
            out[f'{name}_mbar_off'] = off
            off += 8
        total = (off + align - 1) & ~(align - 1)
        return out, total

    @classmethod
    def _dense_ws_setup(cls, setup, n_comp_warps):
        n_per_cta = setup['n_per_cta']
        b_tile_bytes = setup['k_pad'] * n_per_cta * 8
        c_tile_bytes = setup['m_pad'] * n_per_cta * 8
        a_bytes = setup['m_tiles'] * setup['k_tiles'] * 32 * 8

        regions = [('b1', b_tile_bytes), ('b2', b_tile_bytes),
                   ('c', c_tile_bytes), ('a', a_bytes), ('wid', 16)]
        mbars = ('tma', 'bready', 'cready', 'cstored',
                 'steal', 'wid_new', 'wid_used')
        offsets, dynm_total_bytes = cls._dsmem_alloc(regions, mbars)

        args = {
            'n_comp_warps': n_comp_warps,
            'blockx_total': 32 * (n_comp_warps + 2),
            'prod_warp': n_comp_warps,
            'steal_warp': n_comp_warps + 1,
            'comp_threads': 32 * n_comp_warps,
            'b_tile_bytes': b_tile_bytes,
            'c_mtile_smem_stride': 8 * n_per_cta * 8,
            'c_ntile_smem_stride': 8 * 8,
            'dynm_total_bytes': dynm_total_bytes,
        }

        return offsets | args

    def _dense_mma_setup(self, nn, warps_per_cta, block_steal):
        a = self.A
        m, k = a.shape
        m_tiles = (m + 7) // 8
        k_tiles = (k + 3) // 4
        k_rem = k % 4

        # A in DMMA-fragment layout: lane l -> A[mt*8 + l//4][kt*4 + l%4]
        # i.e. an (m_tiles, k_tiles) grid of row-major 8x4 tiles, packed as
        # uint64
        a_pad = np.zeros((m_tiles*8, k_tiles*4))
        a_pad[:m, :k] = a
        tiles = a_pad.reshape(m_tiles, 8, k_tiles, 4).swapaxes(1, 2)
        a_u64 = [f'0x{u:016x}' for u in tiles.view(np.uint64).ravel()]

        n_per_warp = 8 * nn
        n_per_cta  = warps_per_cta * n_per_warp

        # Predicate-elision flags
        n_col_aligned = (self.n is not None and self.n % n_per_warp == 0)
        def pm_runtime(mt):
            return (mt + 1) * 8 > m

        return {
            'warps_per_cta': warps_per_cta,
            'nn': nn,
            'm_tiles': m_tiles,
            'k_tiles': k_tiles,
            'k_rem': k_rem,
            'm_pad': m_tiles * 8,
            'k_pad': k_tiles * 4,
            'a_u64': a_u64,
            'n_per_warp': n_per_warp,
            'n_per_cta': n_per_cta,
            'frag_stride_bytes': 32 * 8,
            'b_kiter_stride': 4 * (self.ldb or 0) * 8,
            'b_ntile_stride': 8 * 8,
            'c_mtile_stride': 8 * (self.ldc or 0) * 8,
            'c_ntile_stride': 8 * 8,
            'n_col_aligned': n_col_aligned,
            'pm_runtime': pm_runtime,
            'block_stealing': block_steal,
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
