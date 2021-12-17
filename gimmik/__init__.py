# -*- coding: utf-8 -*-

import pkgutil
import re

from mako.template import Template
import numpy as np

from gimmik._version import __version__


def generate_mm(mat, dtype, platform, alpha=1.0, beta=0.0, funcn='gimmik_mm',
                n=None, ldb=None, ldc=None):
    # Data type
    dtype = np.dtype(dtype).type
    if dtype == np.float32:
        dtype = 'float'
    elif dtype == np.float64:
        dtype = 'double'
    else:
        raise ValueError('Invalid floating point data type')

    if 0 < (n is None) + (ldb is None) + (ldc is None) < 3:
        raise ValueError('Must provide all of (n, ldb, ldc) or none')

    # Multiply the matrix through by alpha
    mat = alpha*mat

    # Template arguments
    tplargs = {
        'dtype': dtype,
        'mat': mat,
        'beta': beta,
        'funcn': funcn,
        'n': n,
        'ldb': ldb,
        'ldc': ldc
    }

    # Load and render the template
    tpl = pkgutil.get_data(__name__, f'kernels/{platform}.mako')
    src = Template(tpl).render(**tplargs)

    # At single precision suffix all floating point constants by 'f'
    if dtype == 'float':
        src = re.sub(r'(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',
                     r'\g<0>f', src)

    # Cleanup
    src = re.sub(r'\n\n+', r'\n\n', src.strip()) + '\n'
    src = re.sub(r'\w+$', '', src)

    # Return the source
    return src
