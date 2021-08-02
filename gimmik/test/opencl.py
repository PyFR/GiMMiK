# -*- coding: utf-8 -*-

from gimmik.test.base import BaseTest
import numpy as np
from pyfr.backends.opencl.provider import (OpenCLKernelProvider)
import pyopencl as cl


class OpenCLTest(BaseTest):
    name = 'opencl'

    def __init__(self, platform, cfg):
        super(OpenCLTest, self).__init__(platform, cfg)
        self.provider = OpenCLKernelProvider(self.backend)

    def _make_kernel_prof(self, src, b, out, queue):
        # Build
        fun = self.provider._build_kernel('gimmik_mm', src,
                                 [np.int32, np.intp]*2 + [np.int32])

        class GimmikKernel(object):
            def run_sync(self):
                fun(queue, (b.ncol,), None, b.ncol, b.data, b.leaddim,
                    out.data, out.leaddim)
                queue.finish()

            def run_async(self):
                fun(queue, (b.ncol,), None, b.ncol, b.data, b.leaddim,
                    out.data, out.leaddim)
                
            def sync(self):
                queue.finish()

        return GimmikKernel()

    def mul_profile(self, src, mat):
        self.prof_malloc(mat)
        
        self.queue = cl.CommandQueue(self.backend.ctx)
        kernel = self._make_kernel_prof(src, self._x['in'], self._x['out'],
                                        self.queue)
        
        return self.profile_kernel(kernel, mat, self._x['in'], self._x['out'])