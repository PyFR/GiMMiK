# -*- coding: utf-8 -*-

from gimmik.base import MatMul


class ISPCMatMul(MatMul):
    platform = 'ispc'
    basemeta = {}

    def _kernel_generators(self, dtype, dsize):
        yield ('cstream', {}, {})
