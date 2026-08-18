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

    def _launch_description(self, meta):
        div = meta['block'][0]*meta['width']

        return {'grid': ({'div': div}, 1, 1)}
