# -*- coding: utf-8 -*-

from ctypes import cast, c_void_p
from gimmik.test.base import BaseTest
import numpy as np
from pyfr.backends.openmp.provider import (OpenMPKernelProvider)
from statistics import geometric_mean, stdev
import time


class OpenMPTest(BaseTest, OpenMPKernelProvider):
    name = 'openmp'

    def __init__(self, platform, cfg):
        BaseTest.__init__(self, platform, cfg)
        OpenMPKernelProvider.__init__(self, self.backend)

    def _make_kernel(self, src):
        # Build
        fun = self._build_kernel('gimmik_mm', src,
                                 [np.int32, np.intp]*2 + [np.int32])

        ptr = cast(fun, c_void_p).value

        # Render our parallel wrapper kernel
        src = self.backend.lookup.get_template('batch-gemm').render(
            lib='gimmik'
        )

        # Argument types for batch_gemm
        argt = [np.intp] + [np.int32]*2 + [np.intp, np.int32]*2

        # Build
        batch = self._build_kernel('batch_gemm', src, argt)

        return batch, ptr

    def mul_time(self, src, mat, n_runs=100):
        self.test_malloc(mat)

        batch, ptr = self._make_kernel(src)
                
        run_times = []
        for i in range(n_runs):
            start = time.time()
            batch(ptr, self._xin.leaddim, self._xin.nblocks, self._xin, 
                self._xin.blocksz, self._xout, self._xout.blocksz)
            end = time.time()

            run_times.append(end - start)
        
        return geometric_mean(run_times), stdev(run_times)
