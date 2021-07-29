# -*- coding: utf-8 -*-

from gimmik.test.base import BaseTest
import numpy as np
from pyfr.backends.cuda.provider import (CUDAKernelProvider,
                                         get_grid_for_block)


class CUDATest(BaseTest):
    name = 'cuda'

    def __init__(self, platform, cfg):
        super(CUDATest, self).__init__(platform, cfg)
        self.provider = CUDAKernelProvider(self.backend)

    def _make_kernel_prof(self, src, b, out, stream):
        # Build
        fun = self.provider._build_kernel('gimmik_mm', src,
                                 [np.int32, np.intp]*2 + [np.int32])

        # Determine the grid/block
        block_dim = self.cfg.getint('gimmik-profile', 'block_dim', 128)
        block = (block_dim, 1, 1)
        grid = get_grid_for_block(block, b.ncol)

        class GimmikKernel(object):
            def run_sync(self):
                fun.exec_async(grid, block, stream, b.ncol, b, b.leaddim, out,
                               out.leaddim)
                stream.synchronize()

        return GimmikKernel()

    def mul_profile(self, src, mat):
        self.test_malloc(mat)

        self.stream = self.backend.cuda.create_stream()
        kernel = self._make_kernel_prof(src, self._xin, self._xout,
                                        self.stream)
        
        return self.profile_kernel(kernel, mat)
