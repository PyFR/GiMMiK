# -*- coding: utf-8 -*-

import numpy as np
from pyfr.backends import get_backend


class BaseTest(object):
    def __init__(self, platform, cfg):
        self.platform = platform
        self.cfg = cfg

        backend = get_backend(platform, cfg)
        self.backend = backend

    def mul_time_test(self):
        pass

    def test_malloc(self, mat, n=8192):
        backend = self.backend

        (n1, n2) = np.shape(mat)
        n0 = n*backend.soasz

        xin_0 = np.random.rand(n2, n0)

        self._xin = backend.matrix((n2, n0), initval=xin_0, tags={'align'})
        self._xout = backend.matrix((n1, n0), tags={'align'})

    def mul_time(self, src, mat, n_runs=100):
        pass

    def mul_validate(self, src, mat):
        pass
