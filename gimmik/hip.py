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

            if args := self._get_render_args(kcfg, dtype, dsize):
                yield args

    def _platform_config(self, dtype, arch):
        # Fall back on the default config when the arch has none of its own
        if arch is not None:
            try:
                return self._get_config(f'{arch}-{dtype}')
            except FileNotFoundError:
                pass

        return self._get_config(f'default-{dtype}')

    def _matmul_stats(self, dtype, arch, warp_size):
        return {
            'dtype': dtype,
            'm': self.m,
            'k': self.k,
            'n': self.n,
            'beta': self.beta,
            'beta-zero': self.beta == 0,
            'aligne': self.aligne,
            'nnz': self.nnz,
            'density': self.nnz / self.A.size,
            'unique-abs': self.unique_abs,
            'k-used': len(self.bix),
            'gcn-arch': arch,
            'warp-size': warp_size
        }

    def _usable_config(self, kcfg, sigs, stats, sparse_ok):
        conditions = kcfg.get('conditions')

        # Pass over kernels whose signature the caller can not invoke
        if sig_of(kcfg) not in sigs:
            return False
        # Pass over the unrolled kernels when A is too dense or too varied
        elif kcfg['family'] == 'sparse' and not sparse_ok:
            return False
        # Take a kernel which places no demands on the operator
        elif conditions is None:
            return True
        # Otherwise decide on the conditions the kernel gives
        else:
            return self._eval_condition(conditions, stats)

    def _get_render_args(self, kcfg, dtype, dsize):
        tpl, width = kcfg['template'], kcfg['width']
        family, params = kcfg['family'], kcfg['params']
        block = tuple(kcfg['block'])

        args = {'width': width, 'blockx': block[0]}
        meta = {'width': width, 'block': block, 'variant': kcfg['variant'],
                'sig': sig_of(kcfg)}

        # Vector kernels move B and C through a wide element type
        if width > 1:
            args['dtype'] = f'{dtype}{width}'

        match family, tpl:
            case 'sparse', 'bstream-msplit' | 'cstream-ksplit':
                meth = self._sparse_args
            case 'dense', 'mfma-tile-gemm':
                meth = self._dense_args
            case _:
                raise ValueError(f'Unknown HIP kernel {family}/{tpl}')

        meth(tpl, params, block, width, dsize, args, meta)

        if self._fits_device_limits(meta):
            return tpl, args, meta
        else:
            return None

    def _fits_device_limits(self, meta):
        bx, by, bz = meta['block']
        shared = meta['shared']

        return bx*by*bz <= self.max_threads and shared <= self.max_shared

    def _sparse_args(self, tpl, params, block, width, dsize, args, meta):
        blkx = block[0]
        preload = params.get('preload-c', False)

        match tpl:
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

    def _dense_args(self, tpl, params, block, width, dsize, args, meta):
        mt, nt, kt = params['mt'], params['nt'], params['kt']

        # Pad A out to a whole number of tiles
        m_pad, k_pad = -(-self.m // mt)*mt, -(-self.k // kt)*kt

        # The MFMA takes its A fragments in the precision of B and C
        adtype = np.dtype(np.float32 if dsize == 4 else np.float64)

        # Each lane reads its pair of A values as one vector of two
        align = 2*adtype.itemsize

        args |= {'MT': mt, 'NT': nt // width, 'KT': kt, 'k_pad': k_pad,
                 'blocky': block[1]}
        meta |= {
            'shared': 2*kt*nt*dsize,
            'launch': {'grid': ({'div': nt}, m_pad // mt, 1)},
            'operands': {
                'a': {'dtype': adtype, 'align': align,
                      'nbytes': m_pad*k_pad*adtype.itemsize}
            },
            '_packer': self._dense_packer(m_pad, k_pad, adtype)
        }

    def _dense_packer(self, m_pad, k_pad, adtype):
        m, k = self.m, self.k

        def pack(a):
            apad = np.zeros((m_pad, k_pad), dtype=adtype)
            apad[:m, :k] = a

            # Split A into 16 row by 8 column blocks, one per MFMA pair group
            t = apad.reshape(-1, 16, k_pad // 8, 2, 4)

            # Then order each block by lane, so a lane reads its pair at once
            return np.ascontiguousarray(t.transpose(0, 2, 4, 1, 3)).reshape(-1)

        return pack

    def _launch_description(self, meta):
        div = meta['block'][0]*meta['width']

        return {'grid': ({'div': div}, 1, 1), 'block': meta['block']}
