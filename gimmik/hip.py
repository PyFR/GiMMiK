from gimmik.base import MatMul

import numpy as np


class HIPMatMul(MatMul):
    platform = 'hip'
    basemeta = {'block': (128, 1, 1), 'width': 1, 'shared': 0}

    # The HIP kernels tolerate a wider spread of unique values
    max_unique = 128

    def _kernel_generators(self, dtype, dsize, *, sigs, gcn_arch=None,
                           warp_size=64):
        arch = gcn_arch.partition(':')[0] if gcn_arch is not None else None
        mfma_supported = dsize == 8 and arch in {'gfx942', 'gfx950'}

        if not self._unrolled_viable() and not mfma_supported:
            return

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
            'variant': f'bstream-msplit/m{ms}-b{bsz}-x{blkx}'
        }
        yield from emit('bstream-msplit', args, meta)

        ks, csz, blkx = 2, 24, 64
        args = {'ksplit': ks, 'csz': csz, 'blockx': blkx}
        meta = {
            'block': (blkx, ks, 1), 'shared': (ks - 1)*csz*blkx*dsize,
            'variant': f'cstream-ksplit/k{ks}-c{csz}-x{blkx}'
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
                    'variant': f'bstream-msplit/{wpfx}m{ms}-b{bsz}-x{blkx}'
                } | wmeta
                yield from emit('bstream-msplit', args, meta)

            for ms in msplits:
                # m-split B streaming, C preloading, C accumulation kernel
                args = {'msplit': ms, 'bsz': bsz, 'blockx': blkx} | wargs
                shared = 2*bsz*blkx*dsize*width
                meta = {
                    'block': (blkx, ms, 1), 'shared': shared,
                    'variant': (
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
                    'variant': (
                        f'cstream-ksplit-preload-c/'
                        f'{wpfx}k{ks}-c{csz}-x{blkx}'
                    )
                } | wmeta
                yield from emit_preload('cstream-ksplit', args, meta)

        if mfma_supported:
            packed_mfma_tiles = [
                (64, 64, 8, 64, 4),
                (128, 64, 8, 64, 4),
                (64, 128, 8, 64, 4),
                (128, 128, 8, 64, 4),
            ]
            widths = ([2] if self.aligne is not None and self.aligne % 2 == 0
                      else [1])

            for width in widths:
                wargs = ({'dtype': f'{dtype}{width}', 'width': width,
                          'sdtype': dtype} if width > 1 else {})
                wpfx = f'w{width}-' if width > 1 else ''

                for mt, nt, kt, blockx, blocky in packed_mfma_tiles:
                    a_hex, m_pad, k_pad = self._dense_mfma_lane_bake(mt, kt)
                    block = (blockx, blocky, 1)
                    args = {
                        'MT': mt, 'NT': nt // width, 'KT': kt,
                        'blockx': blockx, 'blocky': blocky,
                        'a_hex': a_hex, 'm_pad': m_pad, 'k_pad': k_pad,
                    } | wargs
                    wmeta = {'width': width} if width > 1 else {}
                    meta = {
                        'block': block, 'shared': 2*kt*nt*dsize,
                        'launch': {
                            'grid': ({'div': nt}, -(-self.m // mt), 1)
                        },
                        'variant': f'mfma-tile-gemm/{wpfx}'
                                   f'mt{mt}-nt{nt}-kt{kt}',
                    } | wmeta
                    yield from emit('mfma-tile-gemm', args, meta)

    def _dense_mfma_lane_bake(self, mt, kt):
        # Pack A in the lane order consumed by pairs of MFMA K groups.
        if kt % 8:
            raise ValueError('MFMA K tile must be a multiple of 8')

        m_pad = -(-self.m // mt)*mt
        k_pad = -(-self.k // kt)*kt
        a_pad = np.zeros((m_pad, k_pad), dtype=np.float64)
        a_pad[:self.m, :self.k] = self.A

        packed = []
        for row16 in range(m_pad // 16):
            for ktile in range(k_pad // kt):
                for kgp in range(kt // 8):
                    for lane in range(64):
                        group, row = divmod(lane, 16)
                        for pair_offset in range(2):
                            kidx = ktile*kt + (2*kgp + pair_offset)*4 + group
                            packed.append(a_pad[row16*16 + row, kidx])

        return [float(x).hex() for x in packed], m_pad, k_pad

    def _launch_description(self, meta):
        div = meta['block'][0]*meta['width']

        return {'grid': ({'div': div}, 1, 1), 'block': meta['block']}
