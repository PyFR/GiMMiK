# -*- coding: utf-8 -*-

import pkgutil
import re

from mako.template import Template
import numpy as np

from gimmik._version import __version__


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
