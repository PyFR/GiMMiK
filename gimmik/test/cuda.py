# -*- coding: utf-8 -*-

from gimmik.test.base import BaseTest
import numpy as np
from pyfr.backends.cuda.cublas import CUDACUBLASKernels
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
            def run_sync(iself):
                fun.exec_async(grid, block, stream, b.ncol, b, b.leaddim, out,
                               out.leaddim)
                stream.synchronize()

            def run_async(iself):
                fun.exec_async(grid, block, stream, b.ncol, b, b.leaddim, out,
                               out.leaddim)

            def sync(iself):
                stream.synchronize()

        return GimmikKernel()

    def mul_cublas_profile(self, mat, alpha=1.0, beta=0.0):
        (n1, n2) = np.shape(mat)
        A = self.malloc(n2, n1, x0=mat)

        self.single_malloc(1, n2, name='cublas', rinit=True)
        if 'out' not in self._x:
            self.single_malloc(1, n1, name='out')

        cublas = CUDACUBLASKernels(self.backend)
        async_kernel = cublas.mul(A, self._x['cublas'], self._x['out'],
                                  alpha, beta)

        self.stream_cublas = self.backend.cuda.create_stream()

        class QueueWrapper(object):
            def __init__(iself, stream):
                iself.stream_comp = stream
        queue = QueueWrapper(self.stream_cublas)

        class GimmikKernel(object):
            def run_sync(iself):
                async_kernel.run(queue)
                self.stream_cublas.synchronize()

            def run_async(iself):
                async_kernel.run(queue)
                self.stream_cublas.synchronize()

            def sync(iself):
                self.stream_cublas.synchronize()

        return self.profile_kernel(GimmikKernel(), mat, self._x['cublas'],
                                   self._x['out'])

    def mul_profile(self, src, mat):
        self.prof_malloc(mat)

        self.stream_gimmik = self.backend.cuda.create_stream()
        kernel = self._make_kernel_prof(src, self._x['in'], self._x['out'],
                                        self.stream_gimmik)
        
        return self.profile_kernel(kernel, mat, self._x['in'], self._x['out'])
