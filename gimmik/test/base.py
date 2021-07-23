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

    def test_malloc(self, mat, n=1024):
        backend = self.backend

        (n1, n2) = np.shape(mat)
        n0 = n*backend.soasz

        print(f'{n0=}, {n1=}, {n2=}')

        self._xin = backend.matrix((n0, n2), tags={'align'})
        self._xout = backend.matrix((n0, n1), tags={'align'})
