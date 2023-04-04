# -*- coding: utf-8 -*-

from gimmik.base import MatMul


class COpenMPMatMul(MatMul):
    platform = 'c-openmp'
    basemeta = {}

    def _kernel_generators(self, dtype, dsize):
        yield ('cstream', {}, {})

