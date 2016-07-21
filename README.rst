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

How do I use GiMMiK?
--------------------
Once installed, you are ready to use GiMMiK.

.. code:: python

    from gimmik import generate_mm

    ...

    # Generate a CUDA kernel for C = 2*mat*B
    src = generate_mm(mat, np.float32, platform='cuda', alpha=2.0, beta=0.0)

    ...

Who uses GiMMiK?
----------------
GiMMiK was develop to improve performance of the `PyFR <http://www.pyfr.org>`_ framework.
