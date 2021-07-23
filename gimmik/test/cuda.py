# -*- coding: utf-8 -*-

import numpy as np
from pyfr.backends.cuda.provider import (CUDAKernelProvider,
                                         get_grid_for_block)
from gimmik.test.base import BaseTest


class CUDATest(BaseTest, CUDAKernelProvider):
    def __init__(self, platform):
        BaseTest.__init__(self, platform)
        CUDAKernelProvider.__init__(self, self.backend)

    def _make_kernel(self, src, b, out):
        # Build
        fun = self._build_kernel('gimmik_mm', src,
                                 [np.int32, np.intp]*2 + [np.int32])

        # Determine the grid/block
        block = (128, 1, 1)
        grid = get_grid_for_block(block, b.ncol)

        return fun, block, grid

    def mul_test_time(self, src, mat):
        self.test_malloc(mat)

        fun, block, grid = self._make_kernel(src, self._xin, self._xout)
        
        stream_comp = self.backend.cuda.create_stream()

        fun.exec_async(grid, block, stream_comp, self._xin.ncol, self._xin,
                       self._xin.leaddim, self._xout, self._xout.leaddim)
