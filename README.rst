GiMMiK
======
Generator of Matrix Multiplication Kernels - GiMMiK - is a tool for generation of high performance matrix multiplication kernel code for various accelerator platforms. Currently CUDA and OpenCL are the only supported platforms.

What does GiMMiK do?
--------------------
Consider matrix multiplication of the form

C = α ∙ A ⨉ B + β ∙ C

GiMMiK generates fully unrolled kernels, highly specialised to a given operator matrix. The generated code is fully unrolled - each kernel computes a single column of the output matrix. GiMMiK was designed to perform well in a Block by Panel type of matrix multiplication where the operator matrix is small. GiMMiK also removes any sparsity form the operator matrix as well as attempts to reduce common sub-expressions.

How do I install GiMMiK?
------------------------
Clone the git repository and use `setup.py` to install the GiMMiK package. You will need the following dependencies:

* `mako <http://www.makotemplates.org/>`_
* `numpy >= 1.7 <http://www.numpy.org/>`_

Once obtained, you can install GiMMiK by running

::

    python setup.py install

to perform a system-wide install. Alternatively, run
::

    python setup.py install --user

to install the package locally.

If you desire to test whether GiMMiK works correctly on your system, you can do so by running:
::

    python setup.py test

To execute the tests you will need these additional dependencies:

* `CUDA <https://developer.nvidia.com/cuda-downloads>`_ + `pycuda >= 2011.2 <http://mathema.tician.de/software/pycuda/>`_
* OpenCL + `pyopencl >= 2013.2 <http://mathema.tician.de/software/pyopencl/>`_

How do I use GiMMiK?
--------------------
Once installed, you are ready to use GiMMiK.

.. code:: python

    import gimmik.generator as gen
    from gimmik.platform import Platform

    ...

    # Generate kernel
    kernel = gen.generateKernel(data, alpha=2.0, beta=3.0, double=True, reduced=True,
                                platform=Platform.OPENCL)

    ...

Who uses GiMMiK?
----------------
GiMMiK was develop to improve performance of the `PyFR <http://www.pyfr.com>`_ framework for solving advection-diffusion type problems in the area of CFD.