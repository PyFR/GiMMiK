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

        self._x = {}

    def mul_time_test(self):
        pass

    def prof_malloc(self, mat, in_name='in', out_name='out'):
        (n1, n2) = np.shape(mat)

        self.single_malloc(1, n2, name=in_name, rinit=True)
        self.single_malloc(1, n1, name=out_name)

    def single_malloc(self, n, m, name=None, x0=None, rinit=False):
        if name is None:
            raise ValueError("Invalid alloc name")
        backend = self.backend
        ne = n*self.cfg.getint('gimmik-profile', 'neles', 4096)*backend.soasz

        if x0 is None and not rinit:
            self._x[name] = backend.matrix((m, ne), tags={'align'})
        elif x0 is not None and not rinit:
            self._x[name] = backend.matrix((m, ne), initval=x0, tags={'align'})
        elif rinit:
            x_0 = np.random.rand(m, ne)
            self._x[name] = backend.matrix((m, ne), initval=x_0, tags={'align'})

    def malloc(self, n, m, x0=None, rinit=False):
        backend = self.backend

        if x0 is None and not rinit:
            x = backend.matrix((m, n), tags={'align'})
        elif x0 is not None and not rinit:
            x = backend.matrix((m, n), initval=x0, tags={'align'})
        elif rinit:
            x_0 = np.random.rand(m, n)
            x = backend.matrix((m, n), initval=x_0, tags={'align'})
        return x

    def mul_profile(self, src, mat):
        pass

    def mul_validate(self, src, mat):
        pass

    def profile_kernel(self, kernel, mat, b, out):
        n_runs = self.cfg.getint('gimmik-profile', 'n_runs', 30)

        run_times = []
        for i in range(n_runs):
            start = time.time()
            kernel.run_sync()
            end = time.time()
            run_times.append(end - start)

        return self.profile_stats(run_times, mat, b, out)

    def profile_stats(self, run_time, mat, b, out):
        g_mean = geometric_mean(run_time)
        std_dev = stdev(run_time)

        memory_io = b.nbytes + out.nbytes
        bandwidth = memory_io/g_mean
        flops = b.nrow*np.count_nonzero(mat)/g_mean

        stats = {'runtime': g_mean, 'stdev': std_dev, 'bandwidth': bandwidth,
                 'flops': flops,
                }
        return stats
