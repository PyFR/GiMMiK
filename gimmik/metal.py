# -*- coding: utf-8 -*-

from gimmik.base import MatMul


class MetalMatMul(MatMul):
    platform = 'metal'
    basemeta = {'threadgroup': (128, 1, 1), 'threadgroup_mem_size': 0,
                'width': 1}

    def _kernel_generators(self, dtype, dsize):
        # B loading, C streaming kernel
        yield ('cstream', {}, {})

        # B streaming, C accumulation kernel
        yield ('bstream', {}, {})

        # Four-way m-split B streaming, C accumulation kernel
        ms, bsz, blkx = 4, 16, 32
        args = {'msplit': ms, 'blockx': blkx, 'bsz': bsz}
        meta = {'threadgroup': (blkx, ms, 1),
                'threadgroup_mem_size': 2*blkx*bsz*dsize}
        yield ('bstream-msplit', args, meta)

        # Four-way m-split B streaming, C accumulation kernel
        ms, bsz, blkx = 4, 20, 32
        args = {'msplit': ms, 'blockx': blkx, 'bsz': bsz}
        meta = {'threadgroup': (blkx, ms, 1),
                'threadgroup_mem_size': 2*blkx*bsz*dsize}
        yield ('bstream-msplit', args, meta)

        # Two-way k-split B loading, C streaming kernel
        ks, csz, blkx = 2, 20, 32
        args = {'ksplit': ks, 'csz': csz, 'blockx': blkx}
        meta = {'threadgroup': (blkx, ks, 1),
                'threadgroup_mem_size': (ks - 1)*csz*blkx*dsize}
        yield ('cstream-ksplit', args, meta)

        if self.aligne is not None and self.aligne % 2 == 0:
            # Vector B loading, C streaming kernel
            args = {'dtype': 'float2', 'width': 2}
            meta = {'width': 2}
            yield ('cstream', args, meta)

            # Vector B streaming, C accumulation kernel
            yield ('bstream', args, meta)

            # Vector four-way m-split B streaming, C accumulation kernel
            ms, bsz, blkx = 4, 16, 32
            args = {'dtype': 'float2', 'width': 2, 'msplit': ms,
                    'blockx': blkx, 'bsz': bsz}
            meta = {'threadgroup': (blkx, ms, 1),
                    'threadgroup_mem_size': 2*blkx*bsz*dsize, 'width': 2}
            yield ('bstream-msplit', args, meta)

    def _process_meta(self, meta):
        if self.n is not None:
            tg = meta['threadgroup']
            meta['grid'] = (-(-self.n // meta['width']), tg[1], 1)
