# -*- coding: utf-8 -*-

from gimmik.base import MatMul

import numpy as np


class HIPMatMul(MatMul):
    platform = 'hip'
    basemeta = {'block': (128, 1, 1), 'width': 1, 'shared': 0}

    def _kernel_generators(self, dtype, dsize, *, gcn_arch=None, warp_size=64):
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

        if dsize == 8:
            blkx = 64
            a_hex, m_tiles, k_tiles, amask = self._mfma_dense_bake()
            bix_rows = sorted(self.bix)          # k-rows A actually uses
            vec2_opts = [(False, '')]
            if self.aligne is not None and self.aligne % 2 == 0:
                vec2_opts.insert(0, (True, 'w2-'))

            for vec2, wpfx in vec2_opts:
                for kc in [8, 16]:
                    shared = kc*4*blkx*dsize
                    for ms in [8, 16]:
                        args = {
                            'blockx': blkx, 'a_hex': a_hex,
                            'm_tiles': m_tiles, 'k_tiles': k_tiles,
                            'amask': amask, 'msplit': ms,
                            'bix_rows': bix_rows, 'vec2': vec2, 'kc': kc
                        }
                        meta = {
                            'block': (blkx, ms, 1), 'shared': shared,
                            'desc': (
                                f'mfma-dense-msplit/{wpfx}'
                                f'm{m_tiles}-k{k_tiles}-s{ms}-kc{kc}-x{blkx}'
                            )
                        }
                        yield from emit('mfma-dense-msplit', args, meta)

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

    def _mfma_dense_bake(self):
        # Densify, pad and reorder A into v_mfma_f64_16x16x4 fragment order:
        #   Ag[(mt*k_tiles + kt)*64 + lane]
        #       = A_pad[mt*16 + lane%16][kt*4 + lane//16]
        # i.e. with lane = g*16 + p, operand A wants i = p, kk = g.
        # amask[mt][kt] flags 16x4 A-tiles that contain a non-zero, so the
        # kernel can skip the MMA (and, on the direct path, the B load) for
        # all-zero tiles -- structural zero-tile skipping.
        m, k = self.A.shape
        m_tiles = -(-m // 16)
        k_tiles = -(-k // 4)
        a_pad = np.zeros((m_tiles*16, k_tiles*4), dtype=np.float64)
        a_pad[:m, :k] = self.A
        a_hex = []
        for mt in range(m_tiles):
            for kt in range(k_tiles):
                for lane in range(64):
                    i = mt*16 + (lane % 16)
                    kk = kt*4 + (lane // 16)
                    a_hex.append(float(a_pad[i, kk]).hex())
        amask = [[bool(np.any(a_pad[mt*16:mt*16+16, kt*4:kt*4+4]))
                  for kt in range(k_tiles)] for mt in range(m_tiles)]
        return a_hex, m_tiles, k_tiles, amask

    def _process_meta(self, meta):
        if self.n is not None:
            div = meta['block'][0]*meta['width']
            meta['grid'] = (-(-self.n // div), 1, 1)
