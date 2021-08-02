# -*- coding: utf-8 -*-

from gimmik.test.base import BaseTest
from gimmik.test.cuda import CUDATest
from gimmik.test.hip import HIPTest
from gimmik.test.opencl import OpenCLTest
from gimmik.test.openmp import OpenMPTest

import numpy as np
from pyfr.inifile import Inifile
from pyfr.util import subclass_where


def get_tester(name, cfg):
    return subclass_where(BaseTest, name=name.lower())(name, cfg)

def default_cfg(dtype, n_runs=30, block_dim=128, neles=4096):
    precision = {np.float32: 'single',
                 np.float64: 'double',
                }

    cfg_str = f'''
        [backend]
        precision = {precision[dtype]}
        [gimmik-profile]
        {n_runs=}
        {block_dim=}
        {neles=}
    '''
    return Inifile(cfg_str)
