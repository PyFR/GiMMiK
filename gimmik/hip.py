import numpy as np

from gimmik.base import SIG_ABC, SIG_BC, MatMul, sig_of


class HIPMatMul(MatMul):
    platform = 'hip'
    basemeta = {'block': (128, 1, 1), 'width': 1, 'shared': 0}

    sigs = frozenset({SIG_BC, SIG_ABC})

    # The HIP kernels tolerate a wider spread of unique values
    max_unique = 128

    # Largest block and static LDS allocation the runtime will dispatch
    max_threads = 1024
    max_shared = 64*1024

    def _kernel_generators(self, dtype, dsize, *, sigs, gcn_arch=None,
                           warp_size=64):
        arch = gcn_arch.partition(':')[0] if gcn_arch is not None else None

        config = self._platform_config(dtype, arch)
        stats = self._matmul_stats(dtype, arch, warp_size)
        sparse_ok = self._unrolled_viable()

        for kcfg in config['kernels']:
            if not self._usable_config(kcfg, sigs, stats, sparse_ok):
                continue

            prepared = self._get_render_args(kcfg, dtype, dsize)

            if prepared is not None:
                yield prepared

    def _platform_config(self, dtype, arch):
        # Fall back on the default config when the arch has none of its own
        if arch is not None:
            try:
                return self._get_config(f'{arch}-{dtype}')
            except FileNotFoundError:
                pass

        return self._get_config(f'default-{dtype}')

    def _matmul_stats(self, dtype, arch, warp_size):
        nnz = np.count_nonzero(self.A)

        return {
            'dtype': dtype,
            'm': self.m,
            'k': self.k,
            'n': self.n,
            'beta': self.beta,
            'beta-zero': self.beta == 0,
            'aligne': self.aligne,
            'nnz': nnz,
            'density': nnz / self.A.size,
            'unique-abs': len(np.unique(np.abs(self.A))),
            'k-used': len(self.bix),
            'gcn-arch': arch,
            'warp-size': warp_size
        }

    def _usable_config(self, kcfg, sigs, stats, sparse_ok):
        if sig_of(kcfg) not in sigs:
            return False
        elif kcfg['family'] == 'sparse' and not sparse_ok:
            return False

        condition = kcfg.get('conditions')

        if condition is None:
            return True
        else:
            return self._eval_condition(condition, stats)

    def _get_render_args(self, kcfg, dtype, dsize):
        tpl, width = kcfg['template'], kcfg['width']
        params = kcfg.get('params', {})
        block = tuple(kcfg['block'])

        args = {'width': width, 'blockx': block[0]}
        meta = {'width': width, 'block': block, 'variant': kcfg['variant']}

        # Vector kernels move B and C through a wide element type
        if width > 1:
            args['dtype'] = f'{dtype}{width}'

        match kcfg['family']:
            case 'sparse':
                prepared = self._sparse_args(tpl, params, block, dsize, args,
                                             meta)
            case 'dense':
                prepared = self._dense_args(tpl, params, block, dtype, dsize,
                                            args, meta)
            case _:
                raise ValueError(f'Unknown HIP kernel family for {tpl}')

        if prepared is not None and self._fits(prepared[2]):
            return prepared
        else:
            return None

    def _fits(self, meta):
        bx, by, bz = meta['block']
        shared = meta.get('shared', 0)

        return bx*by*bz <= self.max_threads and shared <= self.max_shared

    def _sparse_args(self, tpl, params, block, dsize, args, meta):
        width, blkx = args['width'], block[0]
        preload = bool(params.get('preload-c', False))

        match tpl:
            # B loading, C streaming and B streaming, C accumulating kernels
            case 'cstream' | 'bstream':
                pass
            # M-split B streaming, C accumulation kernel
            case 'bstream-msplit':
                ms, bsz = block[1], params['bsz']
                args |= {'msplit': ms, 'bsz': bsz, 'preload': preload}
                meta['shared'] = 2*bsz*blkx*dsize*width
            # K-split B loading, C streaming kernel
            case 'cstream-ksplit':
                ks, csz = block[1], params['csz']
                args |= {'ksplit': ks, 'csz': csz, 'preload': preload}
                meta['shared'] = (ks - 1)*csz*blkx*dsize*width
            case _:
                raise ValueError(f'Unknown HIP sparse template {tpl}')

        return tpl, args, meta

    def _dense_args(self, tpl, params, block, dtype, dsize, args, meta):
        if tpl != 'mfma-tile-gemm':
            raise ValueError(f'Unknown HIP dense template {tpl}')

        width = args['width']
        mt, nt, kt = params['mt'], params['nt'], params['kt']

        # The kernel walks whole tiles, so A is padded out to cover them
        m_pad, k_pad = -(-self.m // mt)*mt, -(-self.k // kt)*kt

        # The MFMA takes its A fragments in the precision of B and C
        adtype = np.dtype(np.float32 if dtype == 'float' else np.float64)

        # Each lane reads its pair of A values as one vector of two
        align = 2*adtype.itemsize

        args |= {'MT': mt, 'NT': nt // width, 'KT': kt, 'k_pad': k_pad,
                 'blocky': block[1]}
        meta |= {
            'sig': SIG_ABC, 'shared': 2*kt*nt*dsize,
            'launch': {'grid': ({'div': nt}, -(-self.m // mt), 1)},
            'operands': {
                'a': {'dtype': adtype, 'align': align,
                      'nbytes': m_pad*k_pad*adtype.itemsize}
            },
            '_packer': self._dense_packer(m_pad, k_pad, adtype)
        }

        return tpl, args, meta

    def _dense_packer(self, m_pad, k_pad, adtype):
        m, k = self.m, self.k

        def pack(a):
            apad = np.zeros((m_pad, k_pad), dtype=adtype)
            apad[:m, :k] = a

            # Split A into 16 row by 8 column blocks, one per MFMA pair group
            t = apad.reshape(m_pad // 16, 16, k_pad // 8, 2, 4)

            # Then order each block by lane, so a lane reads its pair at once
            return np.ascontiguousarray(t.transpose(0, 2, 4, 1, 3)).reshape(-1)

        return pack

    def _launch_description(self, meta):
        div = meta['block'][0]*meta['width']

        return {'grid': ({'div': div}, 1, 1), 'block': meta['block']}
