# -*- coding: utf-8 -*-

from gimmik.test.base import BaseTest
import numpy as np
from pyfr.backends.hip.rocblas import HIPRocBLASKernels
from pyfr.backends.hip.provider import (HIPKernelProvider,
                                         get_grid_for_block)


class HIPTest(BaseTest):
    name = 'hip'

    def __init__(self, platform, cfg):
        super(HIPTest, self).__init__(platform, cfg)
        self.provider = HIPKernelProvider(self.backend)

    def _make_kernel_prof(self, src, b, out, stream):
        # Determine the grid/block
        block_dim = self.cfg.getint('gimmik-profile', 'block_dim', 128)
        block = (block_dim, 1, 1)
        grid = get_grid_for_block(block, b.ncol)

        # CUDA -> HIP find and replace
        src = src.replace('__global__ void',
                          f'__launch_bounds__({block[0]}) __global__ void')
        src = src.replace('blockDim.x', 'hipBlockDim_x')
        src = src.replace('blockIdx.x', 'hipBlockIdx_x')
        src = src.replace('threadIdx.x', 'hipThreadIdx_x')

        # Build
        fun = self.provider._build_kernel('gimmik_mm', src,
                                 [np.int32, np.intp]*2 + [np.int32])

        class GimmikKernel(object):
            def run_sync(self):
                fun.exec_async(grid, block, stream, b.ncol, b, b.leaddim, out,
                               out.leaddim)
                stream.synchronize()

        return GimmikKernel()

    def mul_rocblas_profile(self, mat, alpha=1.0, beta=0.0):
        (n1, n2) = np.shape(mat)
        A = self.malloc(n2, n1, x0=mat)

        self.single_malloc(1, n2, name='rocblas', rinit=True)
        if 'out' not in self._x:
            self.single_malloc(1, n1, name='out')

        rocblas = HIPRocBLASKernels(self.backend)
        async_kernel = rocblas.mul(A, self._x['rocblas'], self._x['out'],
                                  alpha, beta)

        self.stream_rocblas = self.backend.hip.create_stream()

        class QueueWrapper(object):
            def __init__(iself, stream):
                iself.stream_comp = stream
        queue = QueueWrapper(self.stream_rocblas)

        class GimmikKernel(object):
            def run_sync(iself):
                async_kernel.run(queue)
                self.stream_rocblas.synchronize()

            def run_async(iself):
                async_kernel.run(queue)
            
            def sync(iself):
                self.stream_rocblas.synchronize()

        return self.profile_kernel(GimmikKernel(), mat, self._x['rocblas'],
                                   self._x['out'])

    def mul_profile(self, src, mat):
        self.prof_malloc(mat)

        self.stream = self.backend.hip.create_stream()
        kernel = self._make_kernel_prof(src, self._x['in'], self._x['out'],
                                        self.stream)
        
        return self.profile_kernel(kernel, mat, self._x['in'], self._x['out'])