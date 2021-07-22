# -*- coding: utf-8 -*-

import numpy as np

import gimmik.backends.base as base
from gimmik.util import make_pybuf


class _CUDAMatrixCommon(object):
    @property
    def _as_parameter_(self):
        return self.data


class CUDAMatrixBase(_CUDAMatrixCommon, base.MatrixBase):
    def onalloc(self, basedata, offset):
        self.basedata = basedata
        self.data = int(self.basedata) + offset
        self.offset = offset

        # Process any initial value
        if self._initval is not None:
            self._set(self._initval)

        # Remove
        del self._initval

    def _get(self):
        # Allocate an empty buffer
        buf = np.empty((self.nrow, self.leaddim), dtype=self.dtype)

        # Copy
        self.backend.cuda.memcpy(buf, self.data, self.nbytes)

        # Unpack
        return self._unpack(buf[None, :, :])

    def _set(self, ary):
        buf = self._pack(ary)

        # Copy
        self.backend.cuda.memcpy(self.data, buf, self.nbytes)


class CUDAMatrix(CUDAMatrixBase, base.Matrix):
    pass


class CUDAMatrixSlice(_CUDAMatrixCommon, base.MatrixSlice):
    def _init_data(self, mat):
        return (int(mat.basedata) + mat.offset +
                (self.ra*self.leaddim + self.ca)*self.itemsize)


class CUDAMatrixBank(base.MatrixBank):
    pass


class CUDAConstMatrix(CUDAMatrixBase, base.ConstMatrix):
    pass


class CUDAView(base.View):
    pass


class CUDAQueue(base.Queue):
    def __init__(self, backend):
        super().__init__(backend)

        # CUDA streams
        self.stream_comp = backend.cuda.create_stream()
        self.stream_copy = backend.cuda.create_stream()

    def _wait(self):
        self.stream_comp.synchronize()
        self.stream_copy.synchronize()

        self._last_ktype = None

    def _at_sequence_point(self, item):
        return self._last_ktype != item.ktype

    @staticmethod
    def runall(queues):
        # First run any items which will not result in an implicit wait
        for q in queues:
            q._exec_nowait()

        # So long as there are items remaining in the queues
        while any(queues):
            # Execute a (potentially) blocking item from each queue
            for q in filter(None, queues):
                q._exec_next()
                q._exec_nowait()

        # Wait for all tasks to complete
        for q in queues:
            q._wait()
