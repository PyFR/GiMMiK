# -*- coding: utf-8 -*-

import numpy as np
from pyfr.backends import get_backend
from statistics import geometric_mean, stdev
import time


class BaseTest(object):
    def __init__(self, platform, cfg):
        self.platform = platform
        self.cfg = cfg

        backend = get_backend(platform, cfg)
        self.backend = backend

    def mul_time_test(self):
        pass

    def test_malloc(self, mat, n=4096):
        backend = self.backend

        (n1, n2) = np.shape(mat)
        n0 = n*backend.soasz

        xin_0 = np.random.rand(n2, n0)

        self._xin = backend.matrix((n2, n0), initval=xin_0, tags={'align'})
        self._xout = backend.matrix((n1, n0), tags={'align'})

    def mul_profile(self, src, mat, dtype, n_runs=100):
        pass

    def mul_validate(self, src, mat):
        pass

    def profile_kernel(self, kernel, mat):
        n_runs = self.cfg.getint('gimmik-profile', 'n_runs', 30)

        run_times = []
        for i in range(n_runs):
            start = time.time()
            kernel.run_sync()
            end = time.time()
            run_times.append(end - start)

        return self.profile_stats(run_times, mat)

    def profile_stats(self, run_time, mat):
        g_mean = geometric_mean(run_time)
        std_dev = stdev(run_time)

        memory_io = self._xin.nbytes + self._xout.nbytes
        bandwidth = memory_io/g_mean
        flops = self._xin.nrow*np.count_nonzero(mat)/g_mean

        stats = {'runtime': g_mean, 'stdev': std_dev, 'bandwidth': bandwidth,
                 'flops': flops,
                }
        return stats
