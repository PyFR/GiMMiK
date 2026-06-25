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

        blkx = self.basemeta['block'][0]

        # B loading, C streaming kernel
        yield from emit('cstream', {'blockx': blkx}, {})

        # B streaming, C accumulation kernel
        yield from emit('bstream', {'blockx': blkx}, {})

        # Four-way m-split B streaming, C accumulation kernel
        ms, bsz, blkx = 4, 24, 64
        args = {'msplit': ms, 'bsz': bsz, 'blockx': blkx}
        meta = {'block': (blkx, ms, 1), 'shared': 2*bsz*blkx*dsize}
        yield from emit('bstream-msplit', args, meta)

        # Two-way k-split B loading, C streaming kernel
        ks, csz, blkx = 2, 24, 64
        args = {'ksplit': ks, 'csz': csz, 'blockx': blkx}
        meta = {'block': (blkx, ks, 1), 'shared': (ks - 1)*csz*blkx*dsize}
        yield from emit('cstream-ksplit', args, meta)

        # Dense f64 GEMM via the CDNA Matrix Cores (MFMA); see mfma-dense.mako.
        # Modelled on the NVIDIA DMMA dense path: A is densified + baked in
        # Matrix-Core fragment order, B is streamed, C is non-temporal stored.
        # Densifying means it only pays off for reasonably dense operands, and
        # the MFMA intrinsic is CDNA3-only (gfx94x).
        if self._is_cdna3(gcn_arch) and self._mfma_dense_ok(dsize):
            blkx = 64
            a_hex, m_tiles, k_tiles, amask = self._mfma_dense_bake()
            k_pad = k_tiles*4
            bix_rows = sorted(self.bix)          # k-rows A actually uses
            vec2 = self.aligne is not None and self.aligne % 2 == 0
            for ms in self._mfma_msplits(m_tiles):
                # msplit goes in block.y (cf. bstream-msplit) so block.x stays
                # 64 = one wavefront = the cols-per-block grid contract.
                shared = k_pad*blkx*dsize if ms > 1 else 0
                args = {'blockx': blkx, 'a_hex': a_hex, 'm_tiles': m_tiles,
                        'k_tiles': k_tiles, 'amask': amask, 'msplit': ms,
                        'bix_rows': bix_rows, 'vec2': vec2}
                meta = {'block': (blkx, ms, 1), 'shared': shared,
                        'desc': f'mfma-dense/m{m_tiles}-k{k_tiles}-s{ms}-x{blkx}'}
                yield from emit('mfma-dense', args, meta)

            # Software-pipelined (double-buffered B) direct variant: prefetch
            # next k-tile's B while the current k-tile's MFMAs run.
            args = {'blockx': blkx, 'a_hex': a_hex, 'm_tiles': m_tiles,
                    'k_tiles': k_tiles, 'amask': amask}
            meta = {'block': (blkx, 1, 1), 'shared': 0,
                    'desc': f'mfma-dense-pipe/m{m_tiles}-k{k_tiles}-x{blkx}'}
            yield from emit('mfma-dense-pipe', args, meta)

        # Only emit tuned variants on architectures they have been validated for.
        base_arch = gcn_arch.split(':', 1)[0] if gcn_arch else None
        if base_arch not in {'gfx90a', 'gfx942'} or warp_size != 64:
            return

        # Tuned HIP variants
        msplits, ksplits = [4, 8], [2, 4]
        bsz, csz, blkx = 8, 8, 64
        widths = [1]
        if self.aligne is not None and self.aligne % 2 == 0:
            widths.append(2)

        for width in widths:
            wargs = ({'dtype': f'{dtype}{width}', 'width': width}
                     if width > 1 else {})
            wmeta = {'width': width} if width > 1 else {}
            wpfx = f'w{width}-' if width > 1 else ''

            # B loading, C streaming kernel
            args = {'blockx': blkx} | wargs
            meta = {'block': (blkx, 1, 1),
                    'desc': f'cstream/{wpfx}x{blkx}'} | wmeta
            yield from emit('cstream', args, meta)

            # B streaming, C accumulation kernel
            meta = {'block': (blkx, 1, 1),
                    'desc': f'bstream/{wpfx}x{blkx}'} | wmeta
            yield from emit('bstream', args, meta)

            for ms in msplits:
                # m-split B streaming, C accumulation kernel
                args = {'msplit': ms, 'bsz': bsz, 'blockx': blkx} | wargs
                shared = 2*bsz*blkx*dsize*width
                meta = {
                    'block': (blkx, ms, 1), 'shared': shared,
                    'desc': f'bstream-msplit/{wpfx}m{ms}-b{bsz}-x{blkx}'
                } | wmeta
                yield from emit('bstream-msplit', args, meta)

                # non-temporal B-load variant: B is read-once -> skip L2 alloc
                nmeta = {
                    'block': (blkx, ms, 1), 'shared': shared,
                    'desc': f'bstream-msplit-ntb/{wpfx}m{ms}-b{bsz}-x{blkx}'
                } | wmeta
                yield from emit('bstream-msplit', args | {'ntload': True}, nmeta)

            for ks in ksplits:
                # k-split B loading, C streaming kernel
                args = {'ksplit': ks, 'csz': csz, 'blockx': blkx} | wargs
                shared = (ks - 1)*csz*blkx*dsize*width
                meta = {
                    'block': (blkx, ks, 1), 'shared': shared,
                    'desc': f'cstream-ksplit/{wpfx}k{ks}-c{csz}-x{blkx}'
                } | wmeta
                yield from emit('cstream-ksplit', args, meta)

            # B loading, C preloading, C streaming kernel
            args = {'blockx': blkx} | wargs
            meta = {'block': (blkx, 1, 1),
                    'desc': f'cstream-preload-c/{wpfx}x{blkx}'} | wmeta
            yield from emit('cstream-preload-c', args, meta)

            # B streaming, C preloading, C accumulation kernel
            meta = {'block': (blkx, 1, 1),
                    'desc': f'bstream-preload-c/{wpfx}x{blkx}'} | wmeta
            yield from emit('bstream-preload-c', args, meta)

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
                yield from emit('bstream-msplit-preload-c', args, meta)

                # non-temporal B-load variant
                nmeta = {
                    'block': (blkx, ms, 1), 'shared': shared,
                    'desc': (
                        f'bstream-msplit-preload-c-ntb/'
                        f'{wpfx}m{ms}-b{bsz}-x{blkx}'
                    )
                } | wmeta
                yield from emit('bstream-msplit-preload-c', args | {'ntload': True}, nmeta)

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
                yield from emit('cstream-ksplit-preload-c', args, meta)

    @staticmethod
    def _is_cdna3(gcn_arch):
        base = gcn_arch.split(':', 1)[0] if gcn_arch else None
        return base in {'gfx940', 'gfx941', 'gfx942'}

    def _mfma_dense_ok(self, dsize):
        # f64 Matrix Cores only (that is the only hard requirement of the
        # mfma_f64_16x16x4 instruction).  The kernel densifies A and is left
        # for the autotuner to accept or reject on speed; the earlier
        # m,k <= 128 and density >= 0.5 gates were too strict and hid it from
        # real PyFR tet operators.  Large m increases register pressure (each
        # wavefront keeps m_tiles*4 v4f64 accumulators live) -> m-splitting is
        # the natural follow-up if that becomes the bottleneck.
        return dsize == 8

    def _mfma_msplits(self, m_tiles):
        # m-split factors to offer (placed in block.y).  Each wavefront keeps
        # m_tiles/msplit * 4 v4f64 accumulators live, so splitting m lowers
        # register pressure / raises occupancy on large-m operators.  msplit=1
        # is the direct (no-LDS) path; msplit>1 stages B once in LDS and shares
        # it across the block (so B is not re-read per wavefront).
        return [ms for ms in (1, 2, 4) if ms == 1 or ms <= m_tiles]

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
