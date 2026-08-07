# -*- coding: utf-8 -*-

from gimmik.base import MatMul


class SYCLMatMul(MatMul):
    platform = 'sycl'
    basemeta = {'local_work_size': None, 'local_mem_size': 0, 'width': 1}

    def _kernel_generators(self, dtype, dsize, *, local_mem_size=None):
        max_local_mem = local_mem_size or 1024**3

        # Default 1D work-group size for the non-tiled kernels.  Launching
        # these via an explicit nd_range (rather than a basic parallel_for)
        # avoids the SYCL runtime auto-picking a poor work-group size.  256
        # (four gfx90a wavefronts / eight NVIDIA warps) hides global-memory
        # latency better than 128 on these bandwidth-bound kernels.
        sblkx = 256

        # Consider a vector width of two whenever the leading dimension is
        # suitably aligned.  Unlike CUDA the SYCL vector types (sycl::vec)
        # provide native arithmetic operators, so width is a plain template
        # knob shared by every kernel at both single and double precision.
        widths = [1]
        if self.aligne is not None and self.aligne % 2 == 0:
            widths.append(2)

        for width in widths:
            if width > 1:
                wargs = {'dtype': f'sycl::{dtype}{width}', 'width': width}
                wmeta = {'width': width}
            else:
                wargs = wmeta = {}

            # B loading, C streaming kernel
            yield ('cstream', {'blockx': sblkx} | wargs,
                   {'local_work_size': (sblkx,)} | wmeta)

            # B streaming, C accumulation kernel
            yield ('bstream', {'blockx': sblkx} | wargs,
                   {'local_work_size': (sblkx,)} | wmeta)

            # Four-way m-split B streaming, C accumulation kernel
            ms, bsz, blkx = 4, 16, 64
            args = {'msplit': ms, 'blockx': blkx, 'bsz': bsz} | wargs
            local_mem = 2*blkx*bsz*dsize*width
            meta = {'local_work_size': (blkx, ms),
                    'local_mem_size': local_mem} | wmeta
            if local_mem < max_local_mem:
                yield ('bstream-msplit', args, meta)

                # Preloading C up-front only alters the beta != 0 path, so it
                # is only worth emitting as an extra candidate there.
                if self.beta != 0:
                    yield ('bstream-msplit', args | {'preload': True},
                           meta | {'preload': True})

            # Two-way k-split B loading, C streaming kernel
            ks, csz, blkx = 2, 32, 64
            args = {'ksplit': ks, 'csz': csz, 'blockx': blkx} | wargs
            local_mem = (ks - 1)*csz*blkx*dsize*width
            meta = {'local_work_size': (blkx, ks),
                    'local_mem_size': local_mem} | wmeta
            if local_mem < max_local_mem:
                yield ('cstream-ksplit', args, meta)

                if self.beta != 0:
                    yield ('cstream-ksplit', args | {'preload': True},
                           meta | {'preload': True})

    def _process_meta(self, meta):
        if self.n is not None:
            lws, width = meta['local_work_size'], meta['width']
            nx = -(-self.n // width)
            if lws is None:
                meta['global_work_size'] = (nx,)
            elif len(lws) == 1:
                meta['global_work_size'] = (-(-nx // lws[0]) * lws[0],)
            else:
                meta['global_work_size'] = (nx, lws[1])
