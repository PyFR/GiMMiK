import numpy as np

from gimmik.base import SIG_ABC, SIG_BC, MatMul, sig_of


class MetalMatMul(MatMul):
    platform = 'metal'
    basemeta = {'threadgroup': (128, 1, 1), 'threadgroup_mem_size': 0,
                'width': 1}

    sigs = frozenset({SIG_BC, SIG_ABC})

    # Threadgroup memory to assume when the caller does not say
    default_tgmem = 32768

    # Largest threadgroup the Metal runtime will dispatch
    max_threads = 1024

    def _kernel_generators(self, dtype, dsize, *, sigs, gpu_family=None,
                           max_nnz=None, tgmem_max=None, **kwargs):
        config = self._platform_config(dtype, gpu_family)

        nz = self._tile_nz()
        stats = self._matmul_stats(dtype, gpu_family, nz)

        if max_nnz is None:
            nnz_max = config['max-nnz']
        else:
            nnz_max = max_nnz

        # Whether the operator is sparse enough for the unrolled kernels
        npr_max = config['max-nnz-per-row']
        sparse_ok = stats['nnz'] <= nnz_max and stats['npr'] <= npr_max

        tgmem = tgmem_max or self.default_tgmem

        for kcfg in config['kernels']:
            if not self._usable_config(kcfg, sigs, stats, sparse_ok):
                continue

            prepared = self._get_render_args(kcfg, dtype, dsize, nz, tgmem)

            if prepared is not None:
                yield prepared

    def _platform_config(self, dtype, gpu_family):
        # Fall back on the default config when the family has none of its own
        if gpu_family is not None:
            try:
                return self._get_config(f'family{gpu_family}-{dtype}')
            except FileNotFoundError:
                pass

        return self._get_config(f'default-{dtype}')

    def _matmul_stats(self, dtype, gpu_family, nz):
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
            'npr': nnz / self.m,
            'density': nnz / self.A.size,
            'tile-density': float(nz.mean()),
            'aspect': self.m / self.k,
            'unique-abs': len(np.unique(np.abs(self.A))),
            'k-used': len(self.bix),
            'gpu-family': gpu_family
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

    def _get_render_args(self, kcfg, dtype, dsize, nz, tgmem_max):
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
            case 'dense':
                return self._dense_args(tpl, params, tg, dsize, nz,
                                        tgmem_max, args, meta)
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

    def _tile_nz(self):
        # 8x8 tiles of A which hold at least one non-zero
        ntm, ntk = -(-self.m // 8), -(-self.k // 8)

        Apad = np.zeros((ntm*8, ntk*8), dtype=self.A.dtype)
        Apad[:self.m, :self.k] = self.A

        T = Apad.reshape(ntm, 8, ntk, 8).transpose(0, 2, 1, 3)

        return np.count_nonzero(T.reshape(ntm, ntk, -1), axis=2) > 0

    def _dense_args(self, tpl, params, tg, dsize, nz, tgmem_max, args, meta):
        # Panel width, simdgroups and panel padding for the dense kernel
        w, pad, nthread = params['w'], params['pad'], tg[0]

        if nthread % 32:
            raise ValueError('Dense threadgroups must be a simdgroup multiple')

        ns = nthread // 32

        ntm, ntk = -(-self.m // 8), -(-self.k // 8)
        kp, bs = ntk*8, w + pad
        tgmem = (kp*bs + 64)*dsize

        # Configs the operator or the device cannot accommodate are dropped
        if tgmem > tgmem_max or nthread > self.max_threads:
            return None

        amask = [int(sum(1 << j for j in range(ntk) if nz[i, j]))
                 for i in range(ntm)]
        skip = any(v != (1 << ntk) - 1 for v in amask)

        # Fragments are single precision whatever the kernel computes in
        adtype = np.dtype(np.float32)

        args |= self._dense_tplargs(w, ns, pad, ntm, ntk, kp, amask, skip)
        meta |= {
            'sig': SIG_ABC, 'threadgroup_mem_size': tgmem, 'nbaked': False,
            'launch': {'grid': ({'div': w, 'mul': nthread}, 1, 1)},
            'operands': {
                'a': {'dtype': adtype, 'align': 16,
                      'nbytes': ntm*ntk*64*adtype.itemsize}
            },
            '_packer': self._dense_packer(ntm, ntk, nz)
        }

        return tpl, args, meta

    def _dense_tplargs(self, w, ns, pad, ntm, ntk, kp, amask, skip):
        m = self.m

        # A tile short of rows has to be stored an element at a time
        if m % 8:
            cond = f'mt + 1 < {ntm} && full'
        else:
            cond = 'full'

        if self.beta == 0:
            body = ['simdgroup_store(acc[j], cp, ldc);']
            e0, e1 = 'V0', 'V1'
        else:
            scale = '' if self.beta == 1 else f'{self.beta}*'
            acc = 'acc[j].thread_elements()'
            body = ['simdgroup_float8x8 cf;',
                    'simdgroup_load(cf, cp, ldc);',
                    f'{acc} += {scale}cf.thread_elements();',
                    'simdgroup_store(acc[j], cp, ldc);']
            e0, e1 = f'V0 + {scale}q[0]', f'V1 + {scale}q[1]'

        e0 = e0.replace('V0', 'acc[j].thread_elements()[0]')
        e1 = e1.replace('V1', 'acc[j].thread_elements()[1]')

        store = '\n'.join(' '*16 + l for l in body)

        return {
            'w': w, 'ns': ns, 'bs': w + pad, 'nw': w // 8, 'nthread': 32*ns,
            'ntm': ntm, 'ntk': ntk, 'kp': kp, 'amask': amask, 'skip': skip,
            'cond': cond, 'store': store, 'e0': e0, 'e1': e1
        }

    def _dense_packer(self, ntm, ntk, nz):
        m, k = self.m, self.k
        keep = np.array([[bool(nz[i, j]) for j in range(ntk)]
                         for i in range(ntm)])

        def pack(a):
            Apad = np.zeros((ntm*8, ntk*8), dtype=np.float32)
            Apad[:m, :k] = a

            T = Apad.reshape(ntm, 8, ntk, 8).transpose(0, 2, 1, 3)
            anz = np.count_nonzero(T.reshape(ntm, ntk, -1), axis=2) > 0

            # The skipped tiles are baked in, so a must respect them
            if (anz & ~keep).any():
                raise ValueError('a is non-zero in a tile the kernel skips')

            return np.ascontiguousarray(T).reshape(-1)

        return pack
