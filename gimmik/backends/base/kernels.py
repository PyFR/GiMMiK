# -*- coding: utf-8 -*-

import itertools as it
import re
import types


class _BaseKernel(object):
    @property
    def retval(self):
        return None

    def run(self, queue, *args, **kwargs):
        pass


class ComputeKernel(_BaseKernel):
    ktype = 'compute'


class NullComputeKernel(ComputeKernel):
    pass


class _MetaKernel(object):
    def __init__(self, kernels):
        self._kernels = list(kernels)

    def run(self, queue, *args, **kwargs):
        for k in self._kernels:
            k.run(queue, *args, **kwargs)


class ComputeMetaKernel(_MetaKernel, ComputeKernel):
    pass


class BaseKernelProvider(object):
    def __init__(self, backend):
        self.backend = backend


class NotSuitableError(Exception):
    pass
