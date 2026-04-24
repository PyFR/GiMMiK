# -*- coding: utf-8 -*-

import numpy as np

from gimmik.base import MatMul


class PTXSource:
    def __init__(self):
        self._src = ""

    def __iadd__(self, other):
        self._src = f"{self}\n\t{other}"
        return self

    def __str__(self):
        return self._src

    def __repr__(self):
        return self._src


class PTXMatMul(MatMul):
    platform = 'ptx'
    basemeta = {'block': (128, 1, 1), 'width': 1, 'shared': 0,
                'dynamic_shared': 0}

    def _address(self, out, base, size, *offs):
        src = PTXSource()
        out_type = out[1]
        if out_type != base[1]:
            raise RuntimeError("out and base must have the same type")

        if offs:
            off_type = offs[0][1]
            if not all(off[1] == off_type for off in offs):
                raise RuntimeError("offsets must all have the same tpye")

            if len(offs) == 1:
                off = offs[0]
                mad_type = "lo" if out_type == off_type else "wide"
                src += f"mad.{mad_type}.{off_type} {out[0]}, {size}, {off[0]}, {base[0]};"
            else:
                src += f".reg .{off_type} _addrs_acum;"
                src += f"add.{off_type} _addrs_acum, {offs[0][0]}, {offs[1][0]};"
                for off in offs[2:]:
                    src += f"add.{off_type} _addrs_acum, _addrs_acum, {off[0]};"
                mad_type = "lo" if out_type == off_type else "wide"
                src += f"mad.{mad_type}.{off_type} {out[0]}, {size}, _addrs_acum, {base[0]};"
        else:
            src += f"mov.{out_type} {out[0]}, {base[0]};"
        return f"{{{src}\n\t}}"

    def _kernel_generators(self, dtype, dsize, *, compute_capability=None):
        base_args = {'address': lambda o, b, s, *off: self._address(o, b, s,
        *off), 'cc': compute_capability}

        # Matrix-property gates
        arr = self.A
        nnz = int(np.count_nonzero(arr))
        nuq = int(len(np.unique(np.abs(arr))))
        density = nnz / arr.size
        sparse_suitable = (nuq <= 28) or (density <= 0.15)

        cc = compute_capability or (0, 0)
        dense_suitable = (
            dtype == 'double'
            and cc >= (9, 0)
            and self.n is not None
            and self.m <= 128
            and self.k <= 128
        )

        if sparse_suitable:
            yield ('cstream', base_args | {}, {})

            yield ('bstream', base_args | {}, {})

            ms, bsz, blkx = 4, 24, 32
            args = base_args | {'msplit': ms, 'bsz': bsz, 'blockx': blkx}
            meta = {'block': (blkx, ms, 1), 'shared': 2*bsz*blkx*dsize}
            yield ('bstream-msplit', args, meta)

            ms, bsz, blkx = 1, 16, 128
            args = base_args | {'msplit': ms, 'bsz': bsz, 'blockx': blkx}
            meta = {'block': (blkx, ms, 1), 'shared': 2*bsz*blkx*dsize}
            yield ('bstream-msplit', args, meta)

            ks, csz, blkx = 2, 24, 32
            args = base_args | {'ksplit': ks, 'csz': csz, 'blockx': blkx}
            meta = {'block': (blkx, ks, 1), 'shared': (ks - 1)*csz*blkx*dsize}
            yield ('cstream-ksplit', args, meta)

            K_used = len(self.bix)
            if K_used > 500:
                ks, csz, blkx = 4, 20, 32
                args = base_args | {'ksplit': ks, 'csz': csz, 'blockx': blkx}
                meta = {'block': (blkx, ks, 1),
                        'shared': (ks - 1)*csz*blkx*dsize}
                yield ('cstream-ksplit', args, meta)

            if (dtype == 'double' and self.n is not None and self.n % 2 == 0
                    and K_used <= 100
                    and (self.aligne is None or self.aligne % 2 == 0)):
                blkx = 128
                args = base_args | {'blockx': blkx}
                meta = {'block': (blkx, 1, 1), 'width': 2}
                yield ('cstream-w2', args, meta)

        if dense_suitable:
            # Dense DMMA m8n8k4 templates. Yields a small cover of the nn × w
            # space that empirically spans the autotune winners seen on tet
            # p=3,4 at N=500k. The PyFR wrapper's _benchmark picks the fastest.
            for tpl in ('dense-mma-smem-gA', 'dense-mma-gAd'):
                for nn in (1, 2, 4):
                    for w in (2, 4, 8):
                        blkx = 32 * w
                        n_per_cta = 8 * nn * w
                        if n_per_cta > self.n:
                            continue
                        args = base_args | {'warps_per_cta': w, 'nn': nn}
                        meta = {
                            'block': (blkx, 1, 1),
                            'grid': (-(-self.n // n_per_cta), 1, 1),
                        }
                        yield (tpl, args, meta)

            # Extra fine-grained nn for shapes where a specific nn usually
            # wins (p3/tet/m132, p4/tet/m132).
            for tpl in ('dense-mma-smem-gA', 'dense-mma-gAd'):
                for nn in (6,):
                    for w in (1, 4):
                        blkx = 32 * w
                        n_per_cta = 8 * nn * w
                        if n_per_cta > self.n:
                            continue
                        args = base_args | {'warps_per_cta': w, 'nn': nn}
                        meta = {
                            'block': (blkx, 1, 1),
                            'grid': (-(-self.n // n_per_cta), 1, 1),
                        }
                        yield (tpl, args, meta)

    def _process_meta(self, meta):
        if self.n is not None and 'grid' not in meta:
            div = meta['block'][0]*meta['width']
            meta['grid'] = (-(-self.n // div), 1, 1)
