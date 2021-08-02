# -*- coding: utf-8 -*-

import pkgutil
import re

from mako.template import Template
import numpy as np

from gimmik._version import __version__
from gimmik.test import default_cfg, get_tester

def generate_mm(mat, dtype, platform, alpha=1.0, beta=0.0, funcn='gimmik_mm'):
    # Data type
    dtype = np.dtype(dtype).type
    if dtype == np.float32:
        dtype = 'float'
    elif dtype == np.float64:
        dtype = 'double'
    else:
        raise ValueError('Invalid floating point data type')

    # Multiply the matrix through by alpha
    mat = alpha*mat

    # Template arguments
    tplargs = {'dtype': dtype, 'mat': mat, 'beta': beta, 'funcn': funcn}

    # Load and render the template
    tpl = pkgutil.get_data(__name__, 'kernels/{0}.mako'.format(platform))
    src = Template(tpl).render(**tplargs)

    # At single precision suffix all floating point constants by 'f'
    if dtype == 'float':
        src = re.sub(r'(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',
                     r'\g<0>f', src)

    # Return the source
    return src

def profile_generated(mat, dtype, src, platform):
    cfg = default_cfg(dtype)
    tester = get_tester(platform, cfg)

    return tester.mul_profile(src, mat)

def profile_cublas(mat, dtype, alpha=1., beta=0.):
    cfg = default_cfg(dtype)

    tester = get_tester('cuda', cfg)
    return tester.mul_cublas_profile(mat, alpha, beta)

def profile_rocblas(mat, dtype, alpha=1., beta=0.):
    cfg = default_cfg(dtype)

    tester = get_tester('hip', cfg)
    return tester.mul_rocblas_profile(mat, alpha, beta)
