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

def default_cfg(dtype):
    if dtype == np.float32:
        dtype = 'single'
    elif dtype == np.float64:
        dtype = 'double'
    else:
        raise ValueError('Invalid floating point data type')

    cfg_str = f'''
        [backend]
        precision = {dtype}
    '''
    return Inifile(cfg_str)
