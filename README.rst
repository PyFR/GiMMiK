GiMMiK
======
GiMMiK (Generator of Matrix Multiplication Kernels) generates specialised,
high-performance matrix multiplication kernels for C, CUDA, HIP, ISPC, Metal,
OpenCL, and PTX.

What does GiMMiK do?
--------------------
Consider matrix multiplication of the form

C = α∙A×B + β∙C

GiMMiK generates kernels specialised to a given operator matrix.  The generated
code is fully unrolled, with each kernel computing a single column of the output
matrix.  GiMMiK is designed for block-by-panel matrix multiplication where the
operator matrix is small.  It removes sparsity from the operator matrix and
attempts to reduce common subexpressions.

How do I install GiMMiK?
------------------------
Install the latest release from PyPI with

::

    python -m pip install gimmik

To install from a source checkout, run

::

    python -m pip install .

GiMMiK requires Python 3.10 or newer, Mako 1.0 or newer, and NumPy 2.2 or
newer.

How do I use GiMMiK?
--------------------
Create a platform-specific matrix multiplication generator and request its
kernels.  Multiplying the operator matrix applies the corresponding ``alpha``
factor.

.. code:: python

    import numpy as np

    from gimmik import CUDAMatMul

    mat = np.array([[1.0, 0.0], [2.0, 3.0]])

    # Generate a CUDA kernel for C = 2*mat*B
    mm = CUDAMatMul(2.0*mat, beta=0.0)
    src, metadata = next(mm.kernels(np.float32))

The ``kernels`` method is a generator which yields a sequence of candidate
kernels for the operator matrix, each paired with a dictionary of metadata
describing how it should be launched.  Rather than taking the first
candidate, callers are expected to benchmark the sequence and keep whichever
kernel performs best on their hardware.

The available generators are ``CMatMul``, ``COpenMPMatMul``, ``CUDAMatMul``,
``HIPMatMul``, ``ISPCMatMul``, ``MetalMatMul``, ``OpenCLMatMul``, and
``PTXMatMul``.

Who uses GiMMiK?
----------------
GiMMiK was developed to improve the performance of the
`PyFR <https://www.pyfr.org/>`_ framework.  It was originally developed as part
of Bartosz Wozniak's master's thesis in the Department of Computing at Imperial
College London and is currently maintained by Freddie Witherden.
