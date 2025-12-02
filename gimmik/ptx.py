# -*- coding: utf-8 -*-

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

        # B streaming, C accumulation kernel
        args = base_args | {}
        yield ('bstream', args, {})

    def _process_meta(self, meta):
        if self.n is not None:
            div = meta['block'][0]*meta['width']
            meta['grid'] = (-(-self.n // div), 1, 1)
