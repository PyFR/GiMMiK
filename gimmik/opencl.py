# -*- coding: utf-8 -*-

from gimmik.base import MatMul


class OpenCLMatMul(MatMul):
    platform = 'opencl'
    basemeta = {'local_work_size': None, 'local_mem_size': 0, 'width': 1}

    def _kernel_generators(self, dtype, dsize, *, local_mem_size=None):
        max_local_mem = local_mem_size or 1024**3

        # B loading, C streaming kernel
        yield ('cstream', {}, {})

        # B streaming, C accumulation kernel
        yield ('bstream', {}, {})

        # Four-way m-split B streaming, C accumulation kernel
        ms, bsz, blkx = 4, 16, 64
        args = {'msplit': ms, 'blockx': blkx, 'bsz': bsz}
        meta = {'local_work_size': (blkx, ms),
                'local_mem_size': 2*blkx*bsz*dsize}
        if meta['local_mem_size'] < max_local_mem:
            yield ('bstream-msplit', args, meta)

        # Two-way k-split B loading, C streaming kernel
        ks, csz, blkx = 2, 32, 64
        args = {'ksplit': ks, 'csz': csz, 'blockx': blkx}
        meta = {'local_work_size': (blkx, ks),
                'local_mem_size': (ks - 1)*csz*blkx*dsize}
        if meta['local_mem_size'] < max_local_mem:
            yield ('cstream-ksplit', args, meta)

        # At single precision also consider vectorized kernels
        if (dtype == 'float' and
            self.aligne is not None and self.aligne % 2 == 0):
            # Vector B loading, C streaming kernel
            args = {'dtype': 'float2', 'width': 2}
            meta = {'width': 2}
            yield ('cstream', args, meta)

            # Vector four-way m-split B streaming, C accumulation kernel
            ms, bsz, blkx = 4, 16, 64
            args = {'dtype': 'float2', 'width': 2, 'msplit': ms,
                    'blockx': blkx, 'bsz': bsz}
            meta = {'local_work_size': (blkx, ms),
                    'local_mem_size': 2*blkx*bsz*dsize, 'width': 2}
            if meta['local_mem_size'] < max_local_mem:
                yield ('bstream-msplit', args, meta)

    def _process_meta(self, meta):
        if self.n is not None:
            lws, width = meta['local_work_size'], meta['width']
            if lws is not None:
                meta['global_work_size'] = (-(-self.n // width), lws[1])
            else:
                meta['global_work_size'] = (-(-self.n // width),)
