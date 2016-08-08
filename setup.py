#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from setuptools import setup
import sys


# Python version
if sys.version_info[:2] < (3, 3):
    print('GiMMiK requires Python 3.3 or newer')
    sys.exit(-1)

# GiMMiK version
vfile = open('gimmik/_version.py').read()
vsrch = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", vfile, re.M)

if vsrch:
    version = vsrch.group(1)
else:
    print('Unable to find a version string in gimmik/_version.py')

# Data
package_data = {
    'gimmik': ['kernels/*.mako'],
}

# Hard dependencies
install_requires = [
    'mako',
    'numpy >= 1.7'
]

# Info
classifiers = [
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3.3',
    'Topic :: Scientific/Engineering'
]

# Long Description
long_description = '''GiMMiK is a Python based kernel generator for
matrix multiplication kernels for various accelerator platforms.  For
small operator matrices the generated kernels are capable of
outperfoming the state-of-the-art general matrix multiplication
routines such as cuBLAS GEMM or clBLAS GEMM.  GiMMiK was originally
developed as part of Bartosz Wozniak's master's thesis in the
Department of Computing at Imperial College London and is currently
maintained by Freddie Witherden.'''

setup(name='gimmik',
      version=version,

      # Packages
      packages=['gimmik'],
      package_data=package_data,
      install_requires=install_requires,

      # Metadata
      description='Generator of Matrix Multiplication Kernels',
      long_description=long_description,
      maintainer='Freddie Witherden',
      maintainer_email='freddie@witherden.org',
      url='https://github.com/vincentlab/GiMMiK',
      license='BSD',
      keywords=['Matrix Multiplication', 'GPU', 'CUDA', 'OpenCL'],
      classifiers=classifiers)
