# -*- coding: utf-8 -*-

from ctypes import cast, c_void_p
from gimmik.test.base import BaseTest
import numpy as np
from pyfr.backends.openmp.provider import (OpenMPKernelProvider)


class OpenMPTest(BaseTest):
    name = 'openmp'

    def __init__(self, platform, cfg):
        super(OpenMPTest, self).__init__(platform, cfg)
        self.provider = OpenMPKernelProvider(self.backend)

    def _make_kernel_prof(self, src, b, out):
        # Build
        fun = self.provider._build_kernel('gimmik_mm', src,
                                 [np.int32, np.intp]*2 + [np.int32])

        ptr = cast(fun, c_void_p).value

        # Render our parallel wrapper kernel
        src = self.backend.lookup.get_template('batch-gemm').render(
            lib='gimmik'
        )

        # Argument types for batch_gemm
        argt = [np.intp] + [np.int32]*2 + [np.intp, np.int32]*2

        # Build
        batch = self.provider._build_kernel('batch_gemm', src, argt)

        class GimmikKernel(object):
            def run_sync(self):
                batch(ptr, b.leaddim, b.nblocks, b, b.blocksz, out,
                      out.blocksz)

        return GimmikKernel()

    def mul_profile(self, src, mat):
        self.prof_malloc(mat)

        kernel = self._make_kernel_prof(src, self._x['in'], self._x['out'])

        return self.profile_kernel(kernel, mat, self._x['in'], self._x['out'])
