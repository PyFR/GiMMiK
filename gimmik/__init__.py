# -*- coding: utf-8 -*-

import pkgutil
import re

from mako.template import Template
import numpy as np

from gimmik._version import __version__


def generate_mm(mat, dtype, platform, alpha=1.0, beta=0.0, tol=1e-10,
                funcn='gimmik_mm'):
    # Data type
    dtype = np.dtype(dtype).type
    if dtype == np.float32:
        dtype = 'float'
    elif dtype == np.float64:
        dtype = 'double'
    else:
        raise ValueError('Invalid floating point data type')

    # Multiply the matrix through by alpha and clean it up
    mat = _clean(alpha*mat, tol)

    # Template arguments
    tplargs = {'dtype': dtype, 'mat': mat, 'beta': beta, 'funcn': funcn}

    if platform == 'c-omp':
        tplargs['tilings'] = _tile(mat, sizes=[1])
    elif platform == 'opencl':
        tplargs['tiling'] = _partition(mat, 2)

    # Load and render the template
    tpl = pkgutil.get_data(__name__, 'kernels/{0}.mako'.format(platform))
    src = Template(tpl).render(**tplargs)

    # At single precision suffix all floating point constants by 'f'
    if dtype == 'float':
        src = re.sub(r'(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?',
                     r'\g<0>f', src)

    # Return the source
    return src


def _clean(mat, tol):
    # Copy the matrix
    mat = mat.copy()

    # Clamp small elements
    mat[abs(mat) < tol] = 0

    # Coalesce similar elements
    amfl = np.abs(mat.flat)
    amix = np.argsort(amfl)

    i, ix = 0, amix[0]
    for j, jx in enumerate(amix[1:], start=1):
        if amfl[jx] - amfl[ix] >= tol:
            if j - i > 1:
                amfl[amix[i:j]] = np.median(amfl[amix[i:j]])
            i, ix = j, jx

    if i != j:
        amfl[amix[i:]] = np.median(amfl[amix[i:]])

    # Fix up the signs and assign
    mat.flat = np.copysign(amfl, mat.flat)

    return mat


def _tile(mat, sizes=[1, 2, 3, 5, 7, 8], tol=0.1):
    tilings = {}

    for n in sizes:
        tiling = _partition(mat, n)

        minsz = min(end - start for start, end in tiling)
        maxsz = max(end - start for start, end in tiling)

        if len(tiling) == n and maxsz / minsz - 1 < tol:
            tilings[n] = tiling

    return tilings


def _partition(mat, n):
    idx = [0]
    num = np.count_nonzero(mat) // n
    cnt = 0

    for i, row in enumerate(mat):
        if cnt >= num:
            idx.append(i)
            cnt = 0

        cnt += np.count_nonzero(row)

    return list(zip(idx, idx[1:] + [i + 1]))
