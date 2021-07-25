# -*- coding: utf-8 -*-

from gimmik.test.base import BaseTest
import numpy as np
from pyfr.backends.cuda.provider import (CUDAKernelProvider,
                                         get_grid_for_block)
import time


class CUDATest(BaseTest, CUDAKernelProvider):
    name = 'cuda'

    def __init__(self, platform, cfg):
        BaseTest.__init__(self, platform, cfg)
        CUDAKernelProvider.__init__(self, self.backend)

    def _make_kernel(self, src, b, block_dim=128):
        # Build
        fun = self._build_kernel('gimmik_mm', src,
                                 [np.int32, np.intp]*2 + [np.int32])

        # Determine the grid/block
        block = (block_dim, 1, 1)
        grid = get_grid_for_block(block, b.ncol)

        return fun, block, grid

    def mul_profile(self, src, mat, dtype, n_runs=30):
        self.test_malloc(mat)

        fun, block, grid = self._make_kernel(src, self._xin)
        
        stream_comp = self.backend.cuda.create_stream()
        
        run_times = []
        for i in range(n_runs):
            start = time.time()
            fun.exec_async(grid, block, stream_comp, self._xin.ncol, self._xin,
                self._xin.leaddim, self._xout, self._xout.leaddim)
            stream_comp.synchronize()
            end = time.time()
            run_times.append(end - start)

        return self.profile_stats(run_times, mat, dtype)
