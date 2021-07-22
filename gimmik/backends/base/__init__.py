# -*- coding: utf-8 -*-

from gimmik.backends.base.backend import BaseBackend
from gimmik.backends.base.kernels import (BaseKernelProvider,
                                        ComputeKernel, ComputeMetaKernel,
                                        NotSuitableError, NullComputeKernel)
from gimmik.backends.base.types import (ConstMatrix, Matrix, MatrixBank,
                                      MatrixBase, MatrixSlice, Queue, View)
