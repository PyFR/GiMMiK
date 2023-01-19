# -*- coding: utf-8 -*-

from gimmik.base import MatMul


class CUDAMatMul(MatMul):
    platform = 'cuda'
    basemeta = {'block': (128, 1, 1), 'width': 1, 'shared': 0,
                'dynamic_shared': 0}

    def _kernel_generators(self, dtype, dsize, *, compute_capability=None):
        # B loading, C streaming kernel
        yield ('cstream', {}, {})

        # B streaming, C accumulation kernel
        yield ('bstream', {}, {})

        # Four-way m-split B streaming, C accumulation kernel
        ms, bsz, blkx = 4, 24, 32
        args = {'msplit': ms, 'bsz': bsz, 'blockx': blkx}
        meta = {'block': (blkx, ms, 1), 'shared': 2*bsz*blkx*dsize}
        yield ('bstream-msplit', args, meta)

        # Two-way k-split B loading, C streaming kernel
        ks, csz, blkx = 2, 24, 32
        args = {'ksplit': ks, 'csz': csz, 'blockx': blkx}
        meta = {'block': (blkx, ks, 1), 'shared': (ks - 1)*csz*blkx*dsize}
        yield ('cstream-ksplit', args, meta)

        # At single precision also consider vectorized kernels
        if (dtype == 'float' and
            self.aligne is not None and self.aligne % 2 == 0):
            # Vector B loading, C streaming kernel
            args = {'dtype': 'float2', 'width': 2}
            meta = {'width': 2}
            yield ('cstream', args, meta)

            # Vector four-way m-split B streaming, C accumulation kernel
            ms, bsz, blkx = 4, 16, 32
            args = {'dtype': 'float2', 'width': 2, 'msplit': ms,
                    'bsz': bsz, 'blockx': blkx}
            meta = {'block': (blkx, ms, 1), 'width': 2,
                    'shared': 2*blkx*bsz*2*dsize}
            yield ('bstream-msplit', args, meta)

            # Vector two-way k-split B loading, C streaming kernel
            ks, csz, blkx = 2, 24, 32
            args = {'dtype': 'float2', 'width': 2, 'ksplit': ks,
                    'csz': csz, 'blockx': blkx}
            meta = {'block': (blkx, ks, 1), 'width': 2,
                    'shared': 2*(ks - 1)*csz*blkx*dsize}
            yield ('cstream-ksplit', args, meta)

    def _process_meta(self, meta):
        if self.n is not None:
            div = meta['block'][0]*meta['width']
            meta['grid'] = (-(-self.n // div), 1, 1)
