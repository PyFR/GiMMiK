# -*- coding: utf-8 -*-

from gimmik.base import MatMul


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

        # Only emit tuned variants on architectures they have been validated for.
        base_arch = gcn_arch.split(':', 1)[0] if gcn_arch else None
        if base_arch not in {'gfx90a', 'gfx942'} or warp_size != 64:
            return

        # Tuned HIP variants
        msplits, ksplits = [4, 8], [2, 4]
        bsz, csz, blkx = 8, 8, 64
        width = 2 if self.aligne is not None and self.aligne % 2 == 0 else 1

        # B loading, C streaming kernel
        args = {'blockx': blkx}
        meta = {'block': (blkx, 1, 1), 'desc': f'cstream/x{blkx}'}
        yield from emit('cstream', args, meta)

        # B streaming, C accumulation kernel
        meta = {'block': (blkx, 1, 1), 'desc': f'bstream/x{blkx}'}
        yield from emit('bstream', args, meta)

        for ms in msplits:
            # m-split B streaming, C accumulation kernel
            args = {'msplit': ms, 'bsz': bsz, 'blockx': blkx}
            shared = 2*bsz*blkx*dsize
            meta = {'block': (blkx, ms, 1), 'shared': shared,
                    'desc': f'bstream-msplit/m{ms}-b{bsz}-x{blkx}'}
            yield from emit('bstream-msplit', args, meta)

        for ks in ksplits:
            # k-split B loading, C streaming kernel
            args = {'ksplit': ks, 'csz': csz, 'blockx': blkx}
            shared = (ks - 1)*csz*blkx*dsize
            meta = {'block': (blkx, ks, 1), 'shared': shared,
                    'desc': f'cstream-ksplit/k{ks}-c{csz}-x{blkx}'}
            yield from emit('cstream-ksplit', args, meta)

        # B loading, C preloading, C streaming kernel
        args = {'blockx': blkx}
        meta = {'block': (blkx, 1, 1), 'desc': f'cstream-preload-c/x{blkx}'}
        yield from emit('cstream-preload-c', args, meta)

        # B streaming, C preloading, C accumulation kernel
        meta = {'block': (blkx, 1, 1), 'desc': f'bstream-preload-c/x{blkx}'}
        yield from emit('bstream-preload-c', args, meta)

        if width > 1:
            args = {'dtype': f'{dtype}{width}', 'width': width,
                    'blockx': blkx}
            meta = {'block': (blkx, 1, 1), 'width': width,
                    'desc': f'cstream-width-preload-c/w{width}-x{blkx}'}
            yield from emit('cstream-width-preload-c', args, meta)

            meta = {'block': (blkx, 1, 1), 'width': width,
                    'desc': f'bstream-width-preload-c/w{width}-x{blkx}'}
            yield from emit('bstream-width-preload-c', args, meta)

        for ms in msplits:
            # m-split B streaming, C preloading, C accumulation kernel
            args = {'msplit': ms, 'bsz': bsz, 'blockx': blkx}
            shared = 2*bsz*blkx*dsize
            meta = {'block': (blkx, ms, 1), 'shared': shared,
                    'desc': f'bstream-msplit-preload-c/m{ms}-b{bsz}-x{blkx}'}
            yield from emit('bstream-msplit-preload-c', args, meta)

            if width > 1:
                args = {'msplit': ms, 'bsz': bsz, 'blockx': blkx,
                        'dtype': f'{dtype}{width}', 'width': width}
                meta = {
                    'block': (blkx, ms, 1), 'shared': shared*width,
                    'width': width,
                    'desc': (
                        f'bstream-msplit-width-preload-c/w{width}-'
                        f'm{ms}-b{bsz}-x{blkx}'
                    )
                }
                yield from emit('bstream-msplit-width-preload-c', args, meta)

        for ks in ksplits:
            # k-split B loading, C preloading, C streaming kernel
            args = {'ksplit': ks, 'csz': csz, 'blockx': blkx}
            shared = (ks - 1)*csz*blkx*dsize
            meta = {
                'block': (blkx, ks, 1), 'shared': shared,
                'desc': f'cstream-ksplit-preload-c/k{ks}-c{csz}-x{blkx}'
            }
            yield from emit('cstream-ksplit-preload-c', args, meta)

            if width > 1:
                args = {'ksplit': ks, 'csz': csz, 'blockx': blkx,
                        'dtype': f'{dtype}{width}', 'width': width}
                meta = {
                    'block': (blkx, ks, 1), 'shared': shared*width,
                    'width': width,
                    'desc': (
                        f'cstream-ksplit-width-preload-c/w{width}-'
                        f'k{ks}-c{csz}-x{blkx}'
                    )
                }
                yield from emit('cstream-ksplit-width-preload-c', args, meta)

    def _process_meta(self, meta):
        if self.n is not None:
            div = meta['block'][0]*meta['width']
            meta['grid'] = (-(-self.n // div), 1, 1)
