# -*- coding: utf-8 -*-

from gimmik.backends.base import (BaseKernelProvider)
from gimmik.backends.openmp.compiler import SourceModule
from gimmik.util import memoize


class OpenMPKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes, restype=None):
        mod = SourceModule(src, self.backend.cfg)
        return mod.function(name, restype, argtypes)
