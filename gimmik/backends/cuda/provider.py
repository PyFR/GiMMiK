# -*- coding: utf-8 -*-

from gimmik.backends.base import (BaseKernelProvider)
from gimmik.backends.cuda.compiler import SourceModule
from gimmik.util import memoize


def get_grid_for_block(block, nrow, ncol=1):
    return (int((nrow + (-nrow % block[0])) // block[0]),
            int((ncol + (-ncol % block[1])) // block[1]), 1)


class CUDAKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes):
        # Compile the source code and retrieve the function
        mod = SourceModule(self.backend, src)
        return mod.get_function(name, argtypes)
