# -*- coding: utf-8 -*-

import re

from gimmik.backends.base import BaseBackend

class CUDABackend(BaseBackend):
    name = 'cuda'
    blocks = False

    def __init__(self, cfg):
        super().__init__(cfg)

        from gimmik.backends.cuda.compiler import NVRTC
        from gimmik.backends.cuda.driver import CUDA, CUDAError

        # Load and wrap CUDA and NVRTC
        self.cuda = CUDA()
        self.nvrtc = NVRTC()

        # Try each device until we find one that works
        for i in range(self.cuda.device_count()):
            try:
                self.cuda.set_device(i)
                break
            except CUDAError:
                pass
        else:
            raise RuntimeError('Unable to create a CUDA context')
        
        # Take the required alignment to be 128 bytes
        self.alignb = 128

        # Take the SoA size to be 32 elements
        self.soasz = 32
        self.csubsz = self.soasz

        # Some CUDA devices share L1 cache and shared memory; on these
        # devices CUDA allows us to specify a preference between L1
        # cache and shared memory.  For the sake of CUBLAS (which
        # benefits greatly from more shared memory but fails to
        # declare its preference) we set the global default to
        # PREFER_SHARED.
        self.cuda.set_cache_pref(prefer_shared=True)

        from gimmik.backends.cuda import (cublas, types)

        # Register our data types
        self.base_matrix_cls = types.CUDAMatrixBase
        self.const_matrix_cls = types.CUDAConstMatrix
        self.matrix_cls = types.CUDAMatrix
        self.matrix_bank_cls = types.CUDAMatrixBank
        self.matrix_slice_cls = types.CUDAMatrixSlice
        self.queue_cls = types.CUDAQueue
        self.view_cls = types.CUDAView
        self.xchg_matrix_cls = types.CUDAXchgMatrix
        self.xchg_view_cls = types.CUDAXchgView

        # Instantiate the base kernel providers
        kprovs = [cublas.CUDACUBLASKernels]
        self._providers = [k(self) for k in kprovs]

    def _malloc_impl(self, nbytes):
        # Allocate
        data = self.cuda.mem_alloc(nbytes)

        # Zero
        self.cuda.memset(data, 0, nbytes)

        return data
