import numpy as np

from gimmik.base import MatMul, sig_of


class MetalMatMul(MatMul):
    platform = 'metal'
    basemeta = {'threadgroup': (128, 1, 1), 'threadgroup_mem_size': 0,
                'width': 1}

    def _kernel_generators(self, dtype, dsize, *, sigs, gpu_family=None,
                           max_nnz=None, **kwargs):
        config = self._platform_config(dtype, gpu_family)
        stats = self._matmul_stats(dtype, gpu_family)

        if max_nnz is None:
            nnz_max = config['max_nnz']
        else:
            nnz_max = max_nnz

        # Whether the operator is sparse enough for the unrolled kernels
        npr_max = config['max_nnz_per_row']
        sparse_ok = stats['nnz'] <= nnz_max and stats['npr'] <= npr_max

        for kcfg in config['kernels']:
            if not self._usable_config(kcfg, sigs, stats, sparse_ok):
                continue

            prepared = self._get_render_args(kcfg, dtype, dsize)

            if prepared is not None:
                yield prepared

    def _platform_config(self, dtype, gpu_family):
        # Fall back on the default config when the family has none of its own
        if gpu_family is not None:
            try:
                return self._get_config(f'family{gpu_family}_{dtype}')
            except FileNotFoundError:
                pass

        return self._get_config(f'default_{dtype}')

    def _matmul_stats(self, dtype, gpu_family):
        nnz = np.count_nonzero(self.A)

        return {
            'dtype': dtype,
            'm': self.m,
            'k': self.k,
            'n': self.n,
            'beta': self.beta,
            'beta_zero': self.beta == 0,
            'aligne': self.aligne,
            'nnz': nnz,
            'npr': nnz / self.m,
            'density': nnz / self.A.size,
            'aspect': self.m / self.k,
            'unique_abs': len(np.unique(np.abs(self.A))),
            'k_used': len(self.bix),
            'gpu_family': gpu_family
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
        tg = tuple(kcfg['threadgroup'])

        args = {'width': width}
        meta = {'width': width, 'threadgroup': tg,
                'variant': kcfg['variant']}

        # Vector kernels move B and C through a wide element type
        if width > 1:
            args['dtype'] = f'{dtype}{width}'

        match kcfg['family']:
            case 'sparse':
                return self._sparse_args(tpl, params, tg, dsize, args, meta)
            case _:
                raise ValueError(f'Unknown Metal kernel family for {tpl}')

    def _sparse_args(self, tpl, params, tg, dsize, args, meta):
        width, blkx = args['width'], tg[0]

        match tpl:
            # B loading, C streaming and B streaming, C accumulating kernels
            case 'cstream' | 'bstream':
                pass
            # M-split B streaming, C accumulation kernel
            case 'bstream-msplit':
                ms, bsz = tg[1], params['bsz']
                args |= {'msplit': ms, 'bsz': bsz, 'blockx': blkx}
                meta['threadgroup_mem_size'] = 2*blkx*bsz*dsize*width
            # K-split B loading, C streaming kernel
            case 'cstream-ksplit':
                ks, csz = tg[1], params['csz']
                args |= {'ksplit': ks, 'csz': csz, 'blockx': blkx}
                meta['threadgroup_mem_size'] = (ks - 1)*csz*blkx*dsize*width
            case _:
                raise ValueError(f'Unknown Metal sparse template {tpl}')

        return tpl, args, meta

    def _launch_description(self, meta):
        tg = meta['threadgroup']

        return {'grid': ({'div': meta['width']}, tg[1], 1)}
