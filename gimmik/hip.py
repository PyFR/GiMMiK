# -*- coding: utf-8 -*-

from gimmik.base import MatMul

import numpy as np


class HIPMatMul(MatMul):
    platform = 'hip'
    basemeta = {'block': (128, 1, 1), 'width': 1, 'shared': 0}

    def _candidate_specs(self, dtype, dsize, *, gcn_arch=None, warp_size=64):
        max_block_threads = 1024
        max_shared = 64*1024

        def emit(name, args, meta):
            block = meta.get('block', self.basemeta['block'])
            shared = meta.get('shared', self.basemeta['shared'])
            threads = block[0]*block[1]*block[2]

            if threads <= max_block_threads and shared <= max_shared:
                yield (name, args, meta)

        def emit_preload(name, args, meta):
            yield from emit(name, args | {'preload': True}, meta)

        ms, bsz, blkx = 4, 24, 64
        args = {'msplit': ms, 'bsz': bsz, 'blockx': blkx}
        meta = {
            'block': (blkx, ms, 1), 'shared': 2*bsz*blkx*dsize,
            'desc': f'bstream-msplit/m{ms}-b{bsz}-x{blkx}'
        }
        yield from emit('bstream-msplit', args, meta)

        ks, csz, blkx = 2, 24, 64
        args = {'ksplit': ks, 'csz': csz, 'blockx': blkx}
        meta = {
            'block': (blkx, ks, 1), 'shared': (ks - 1)*csz*blkx*dsize,
            'desc': f'cstream-ksplit/k{ks}-c{csz}-x{blkx}'
        }
        yield from emit('cstream-ksplit', args, meta)

        # Tuned HIP variants
        msplits, ksplits = [8, 4], [4, 2]
        bsz, csz, blkx = 8, 8, 64
        widths = [1]
        if self.aligne is not None and self.aligne % 2 == 0:
            widths.insert(0, 2)

        for width in widths:
            wargs = ({'dtype': f'{dtype}{width}', 'width': width}
                     if width > 1 else {})
            wmeta = {'width': width} if width > 1 else {}
            wpfx = f'w{width}-' if width > 1 else ''

            for ms in msplits:
                # m-split B streaming, C accumulation kernel
                args = {'msplit': ms, 'bsz': bsz, 'blockx': blkx} | wargs
                shared = 2*bsz*blkx*dsize*width
                meta = {
                    'block': (blkx, ms, 1), 'shared': shared,
                    'desc': f'bstream-msplit/{wpfx}m{ms}-b{bsz}-x{blkx}'
                } | wmeta
                yield from emit('bstream-msplit', args, meta)

            for ms in msplits:
                # m-split B streaming, C preloading, C accumulation kernel
                args = {'msplit': ms, 'bsz': bsz, 'blockx': blkx} | wargs
                shared = 2*bsz*blkx*dsize*width
                meta = {
                    'block': (blkx, ms, 1), 'shared': shared,
                    'desc': (
                        f'bstream-msplit-preload-c/'
                        f'{wpfx}m{ms}-b{bsz}-x{blkx}'
                    )
                } | wmeta
                yield from emit_preload('bstream-msplit', args, meta)

            for ks in ksplits:
                # k-split B loading, C preloading, C streaming kernel
                args = {'ksplit': ks, 'csz': csz, 'blockx': blkx} | wargs
                shared = (ks - 1)*csz*blkx*dsize*width
                meta = {
                    'block': (blkx, ks, 1), 'shared': shared,
                    'desc': (
                        f'cstream-ksplit-preload-c/'
                        f'{wpfx}k{ks}-c{csz}-x{blkx}'
                    )
                } | wmeta
                yield from emit_preload('cstream-ksplit', args, meta)

        if dsize == 8:
            # ── mfma-tile-gemm ────────────────────────────────────────────
            # Packed Direct-A MFMA path with B-reuse workgroup mapping and
            # cached B loads.  NT is the scalar output-column tile; width
            # converts it to vector columns before rendering the kernel.
            packed_mfma_tiles = [
                (64, 64, 8, 64, 4),
                (128, 64, 8, 64, 4),
                (64, 128, 8, 64, 4),
                (128, 128, 8, 64, 4),
            ]

            widths = [2] if self.aligne is not None and self.aligne % 2 == 0 else [1]

            for width in widths:
                for MT, NT, KT, blockx, blocky in packed_mfma_tiles:
                    if NT % width:
                        raise ValueError('mfma-tile-gemm width expects NT divisible by width')

                    block = (blockx, blocky, 1)
                    shared = 2*(KT*NT)*dsize
                    threads = block[0]*block[1]*block[2]

                    if threads <= max_block_threads and shared <= max_shared:
                        yield ('mfma-tile-gemm',
                               (width, MT, NT, KT, blockx, blocky, dsize))

    def _render_candidate_spec(self, dtype, kname, spec):
        if len(spec) == 2 and spec[0] == 'mfma-tile-gemm':
            spec = self._expand_mfma_candidate_spec(dtype, spec)

        return super()._render_candidate_spec(dtype, kname, spec)

    def _expand_mfma_candidate_spec(self, dtype, spec):
        name, mspec = spec
        width, MT, NT, KT, blockx, blocky, dsize = mspec
        vNT = NT // width
        block = (blockx, blocky, 1)
        a_packed_hex, m_pad, k_pad = self._dense_mfma_lane_bake(MT, KT)
        direct_a_shared = 2*(KT*NT)*dsize

        wargs = ({'dtype': f'{dtype}{width}', 'width': width,
                  'sdtype': dtype} if width > 1 else {})
        width_meta = {'width': width} if width > 1 else {}
        wpfx = f'w{width}-' if width > 1 else ''
        bpfx = f'b{blocky}-' if blocky != 4 else ''

        args = {
            'MT': MT, 'NT': vNT, 'KT': KT,
            'blockx': block[0], 'blocky': block[1],
            'a_hex': a_packed_hex, 'm_pad': m_pad,
            'k_pad': k_pad,
        } | wargs
        meta = {
            'block': block, 'shared': direct_a_shared,
            'bm': MT, 'ncols': vNT,
            'desc': (
                f'mfma-tile-gemm/'
                f'{wpfx}{bpfx}mt{MT}-nt{NT}-kt{KT}'
            ),
        } | width_meta

        return name, args, meta

    def _kernel_generators(self, dtype, dsize, *, gcn_arch=None, warp_size=64):
        for spec in self._candidate_specs(dtype, dsize, gcn_arch=gcn_arch,
                                          warp_size=warp_size):
            if len(spec) == 2 and spec[0] == 'mfma-tile-gemm':
                spec = self._expand_mfma_candidate_spec(dtype, spec)

            yield spec

    def _dense_mfma_lane_bake(self, BM, BK):
        # Pack A so each lane can vector-load the two FP64 operands it consumes
        # across a pair of consecutive 16x16x4 MFMA K groups:
        #   Apg[row16_tile][kg_pair][lane][which]
        # where lane = g*16 + p and which selects kg_pair*2 + {0, 1}.
        if BK % 8:
            raise ValueError('mfma lane-packed A expects BK to be a multiple of 8')

        m, k = self.A.shape
        m_pad = -(-m // BM) * BM
        k_pad = -(-k // BK) * BK
        a_pad = np.zeros((m_pad, k_pad), dtype=np.float64)
        a_pad[:m, :k] = self.A

        packed = []
        kg_pairs = BK // 8
        for row16 in range(m_pad // 16):
            row_base = row16*16
            for ktile in range(k_pad // BK):
                k_base = ktile*BK
                for kgp in range(kg_pairs):
                    for lane in range(64):
                        g = lane // 16
                        p = lane % 16
                        row = row_base + p
                        for which in range(2):
                            kg = 2*kgp + which
                            packed.append(a_pad[row, k_base + kg*4 + g])

        return [float(x).hex() for x in packed], m_pad, k_pad

    def _process_meta(self, meta):
        bm = meta.get('bm')
        if bm is not None:
            meta['grid_y'] = -(-self.A.shape[0] // bm)

        if self.n is not None:
            div = meta.get('ncols', meta['block'][0])*meta['width']
            gy = meta.get('grid_y', 1)
            meta['grid'] = (-(-self.n // div), gy, 1)
