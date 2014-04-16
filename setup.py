#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
from setuptools import setup


# Python version
if sys.version_info[:2] < (2, 7):
    print('PyFR requires Python 2.7 or newer. Python {}.{} detected'
          .format(*sys.version_info[:2]))
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
    'gimmik': ['kernels/*/*.mako'],
}

# Hard dependencies
install_requires = [
    'mako',
    'numpy >= 1.7'
]

# Tests dependencies
tests_require = [
    'pycuda >= 2011.2',
    'pyopencl >= 2013.2'
]

# Info
classifiers = [
    'License :: OSI Approved :: New BSD License',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering'
]

# Long Description
long_description = '''GiMMiK is a Python based kernel generator for matrix 
multiplication kernels for use on accelerator platforms. For small operator 
matrices the generated kernel are capable of outperfoming the state-of-the-art 
general matrix multiplication routines such as cuBLAS GEMM.
GiMMiK is currently being developed as part of Bartosz Wozniak's Master Thesis 
in the Department of Computing at Imperial College London under the supervision
of Prof. Paul Kelly and Dr. Peter Vincent.'''

setup(name = 'gimmik',
      version = version,

      # packages
      packages = ['gimmik'],
      package_data = package_data,
      install_requires = install_requires,

      # tests
      test_suite = 'gimmik.tests',
      tests_require = tests_require,

      # metadata
      description = 'Generator of Matrix Multiplication Kernels',
      long_description = long_description,
      author = 'Bartosz Wozniak',
      url = 'https://github.com/bartwozniak/GiMMiK',
      license = 'BSD',
      keywords = ['Matrix Multiplication', 'GPU', 'CUDA', 'OpenCL'],
      classifiers = classifiers)
