# -*- coding: utf-8 -*-

import pkgutil
import math
import re
import warnings

from mako.template import Template
import numpy as np

from gimmik._version import __version__


def generate_mm(mat, dtype, platform, alpha=1.0, beta=0.0, funcn='gimmik_mm', 
    maxlen=None):
    # Data type
    dtype = np.dtype(dtype).type

    # Platform language base
    pltfs = {'c'        : 'c',
             'c-omp'    : 'c',
             'cuda'     : 'c',
             'ispc'     : 'c',
             'opencl'   : 'c',
             'f90T'     : 'f90',
             'f90T-cuda': 'f90'}

    types = {'c'  : {np.float32: ( 'float','f'), 
                     np.float64: ('double', '')},
             'f90': {np.float32: ('real(kind=4)','_4'),
                     np.float64: ('real(kind=8)','_8')}}

    # Language continuation character
    cchar = {'c' : ' ',
             'f90': '&'}

    lang = pltfs[platform]

    # np type to language specific types
    try:
        (dtype,suffix) = types[lang][dtype]
    except KeyError:
        raise ValueError('Invalid floating point data type')

    # Multiply the matrix through by alpha
    mat = alpha*mat

    # Template arguments
    tplargs = {'dtype': dtype, 'mat': mat, 'beta': beta, 'funcn': funcn}

    # Load and render the template
    tpl = pkgutil.get_data(__name__, 'kernels/{0}.mako'.format(platform))
    src = Template(tpl).render(**tplargs)

    # Append suffix to handle typing
    src = re.sub(r'(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',
                 rf'\g<0>{suffix}', src)

    # Split lines to enforce line length max (needed for F90-F08 ISO)
    if maxlen:
        src = _line_split(maxlen,cchar[lang],src)
    elif lang == 'f90':
        warnings.warn('No maxlen given for F90 based kernel')

    # Return the source
    return src


def _line_split(maxlen,cchar,src):
    lines = src.splitlines()

    src = ''
    for line in lines:
        nidnt = len(line) - len(line.lstrip(' '))

        while math.ceil(len(line)/maxlen) > 1:
            ns = max(line[:maxlen].rfind('+ '),line[:maxlen].rfind('- '))
            src += line[:ns] + cchar + '\n'
            line = nidnt*' ' + line[ns:]

        src += line + '\n'

    return src
