# -*- coding: utf-8 -*-

from gimmik.test.base import BaseTest
import numpy as np
from pyfr.backends.opencl.provider import (OpenCLKernelProvider)
import pyopencl as cl
from statistics import geometric_mean, stdev
import time


class OpenCLTest(BaseTest, OpenCLKernelProvider):
    name = 'opencl'

    def __init__(self, platform, cfg):
        BaseTest.__init__(self, platform, cfg)
        OpenCLKernelProvider.__init__(self, self.backend)

    def _make_kernel(self, src, b):
        # Build
        fun = self._build_kernel('gimmik_mm', src,
                                 [np.int32, np.intp]*2 + [np.int32])

        return fun

    def mul_time(self, src, mat, n_runs=30):
        self.test_malloc(mat)

        fun = self._make_kernel(src, self._xin)
        
        queue = cl.CommandQueue(self.backend.ctx)
        
        run_times = []
        for i in range(n_runs):
            start = time.time()
            fun(queue, (self._xin.ncol,), None, self._xin.ncol,
                self._xin.data, self._xin.leaddim, self._xout.data, 
                self._xout.leaddim)
            queue.finish()
            end = time.time()

            run_times.append(end - start)
        
        return geometric_mean(run_times), stdev(run_times)
