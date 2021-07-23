# -*- coding: utf-8 -*-

from gimmik.test.base import BaseTest
import numpy as np
from pyfr.backends.hip.provider import (HIPKernelProvider,
                                         get_grid_for_block)
from statistics import geometric_mean, stdev
import time


class HIPTest(BaseTest, HIPKernelProvider):
    name = 'hip'

    def __init__(self, platform, cfg):
        BaseTest.__init__(self, platform, cfg)
        HIPKernelProvider.__init__(self, self.backend)

    def _make_kernel(self, src, b):
        # Build
        fun = self._build_kernel('gimmik_mm', src,
                                 [np.int32, np.intp]*2 + [np.int32])

        # Determine the grid/block
        block = (128, 1, 1)
        grid = get_grid_for_block(block, b.ncol)

        return fun, block, grid

    def mul_time(self, src, mat, n_runs=100):
        self.test_malloc(mat)

        fun, block, grid = self._make_kernel(src, self._xin)
        
        stream_comp = self.backend.hip.create_stream()
        
        run_times = []
        for i in range(n_runs):
            start = time.time()
            fun.exec_async(grid, block, stream_comp, self._xin.ncol, self._xin,
                self._xin.leaddim, self._xout, self._xout.leaddim)
            stream_comp.synchronize()
            end = time.time()

            run_times.append(end - start)
        
        return geometric_mean(run_times), stdev(run_times)
