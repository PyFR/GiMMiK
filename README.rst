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

Only kernels which suit the operator, the target and the arguments given are
yielded, so the sequence may be empty.  This is not an error: it means GiMMiK
has nothing to offer for the problem, and a vendor GEMM should be preferred.
Callers which benchmark the sequence therefore need to handle having found no
kernel at all.

The number of columns of B may either be baked into the kernel, by passing
``n`` together with the leading dimensions ``ldb`` and ``ldc``, or left as a
runtime argument by passing none of them.  A baked kernel takes just the two
pointers, whereas a runtime kernel is called as ``(n, b, ldb, c, ldc)``.

When ``n`` is baked the geometry is fixed, and the metadata carries it
directly.  When ``n`` is left to runtime the geometry is instead a function of
it, which the metadata describes under the ``launch`` key.  The key is present
only in this second case, and is empty on platforms which have no launch
geometry at all.  Each entry maps a geometry parameter onto a tuple of axes,
where an integer axis is a constant and a mapping axis is
``ceil(n / div) * mul``, with ``mul`` defaulting to one.

.. code:: python

    mm = CUDAMatMul(2.0*mat, beta=0.0)
    src, metadata = next(mm.kernels(np.float32))

    # A grid of ceil(n / 128) by 1 by 1
    metadata['launch'] == {'grid': ({'div': 128}, 1, 1)}

Being plain data, the description can be serialised alongside the kernel
source, allowing an application compiled ahead of time to work out its own
geometry with no access to the generator.  Callers which have a generator to
hand may instead evaluate the description with ``launch_config``.

.. code:: python

    # Grid needed to multiply a B with 40000 columns
    grid = mm.launch_config(metadata, 40000)['grid']

The available generators are ``CMatMul``, ``COpenMPMatMul``, ``CUDAMatMul``,
``HIPMatMul``, ``ISPCMatMul``, ``MetalMatMul``, ``OpenCLMatMul``, and
``PTXMatMul``.

Who uses GiMMiK?
----------------
GiMMiK was developed to improve the performance of the
`PyFR <https://www.pyfr.org/>`_ framework.  It was originally developed as part
of Bartosz Wozniak's master's thesis in the Department of Computing at Imperial
College London and is currently maintained by Freddie Witherden.
