# -*- coding: utf-8 -*-

from gimmik.base import MatMul


class CMatMul(MatMul):
    platform = 'c'
    basemeta = {}

    def _kernel_generators(self, dtype, dsize):
        yield ('cstream', {}, {})
