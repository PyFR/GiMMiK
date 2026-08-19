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
runtime argument by passing none of them.  Passing them is a licence to
specialise rather than an instruction to, and an individual kernel may still
take its sizes at runtime; consult ``meta['args']`` to see which it did.

Call signatures
---------------
Kernels do not all take the same arguments.  Callers declare which call
signatures they are able to invoke, and GiMMiK only yields kernels which
match.

.. code:: python

    from gimmik import SIG_ABC, SIG_BC

    mm.kernels(np.float32, sigs={SIG_BC, SIG_ABC})

=============== ====================== ============================
Signature       Arguments, n baked     Arguments, n at runtime
=============== ====================== ============================
``bc``          ``(b, c)``             ``(n, b, ldb, c, ldc)``
``abc``         ``(a, b, c)``          ``(a, n, b, ldb, c, ldc)``
``bdesc-c``     ``(b_desc, c)``        ``(n, b_desc, c, ldc)``
``bdesc-cdesc`` ``(b_desc, c_desc)``   ``(n, b_desc, c_desc)``
=============== ====================== ============================

Arguments named ``_desc`` are TMA tensor map descriptors rather than
pointers, and arise only on PTX.  Every one of them is described by
``meta['operands']``, covered below.

The default is ``sigs={'bc'}``, so a caller which asks for nothing keeps the
classic two-pointer kernels and can never be handed one it cannot call.  Do
not infer the signature from what was passed to the constructor: consult
``meta['sig']`` for the label and ``meta['args']`` for the ordered tuple of
argument names, and build the call from those.

The signature of a kernel is not a property of the target alone.  The same
template can take C as a descriptor or as a plain pointer depending on
``beta``, since with ``beta=0`` there is nothing to read back and the result
is written straight out.  Two generators built from the same operator and
target, differing only in ``beta``, may therefore offer different
signatures.

``MatMul.sigs`` gives the signatures a platform is capable of emitting, and
``available_sigs`` reports which it would actually offer for a given operator
and target.  This distinguishes "GiMMiK has nothing for this problem" from
"you declined the only kernels which suited it".

.. code:: python

    mm.available_sigs(np.float32, gpu_family=9)

Passing A in a buffer
---------------------
Kernels with the ``abc`` signature read the operator matrix from a buffer
rather than from constants baked into the source.  GiMMiK owns the layout,
which is padded and swizzled to suit the target, and hands the caller the
bytes to upload.

.. code:: python

    src, meta = next(kgen)

    if meta['sig'] == 'abc':
        apack = mm.pack_a(meta)

``pack_a`` returns a one dimensional contiguous array which the caller copies
into a buffer of at least ``spec['nbytes']`` bytes, aligned to
``spec['align']``, where ``spec`` is ``mm.operands(meta)['a']``.  Its data type
is ``spec['dtype']``, which need not be the type the kernel was generated
for.  The buffer must stay valid and unmodified for as long as the kernel is
used; kernels only read it.

Packing is a pure function, and an operator may be named explicitly.  Because
A is no longer part of the source, one compiled kernel can serve many
operators of the same shape and sparsity pattern.

.. code:: python

    other = mm.pack_a(meta, a=2.0*mat2)

Note that with ``abc`` the generated source no longer identifies the
operator, so a cache of *bound* kernels must not be keyed on the source text
alone.  A supplied A must have the same shape as the one the generator was
built with, and must be zero wherever the kernel elides an all zero tile;
both are checked.  Metadata may only be used with the generator which
produced it.

When benchmarking a sequence which mixes signatures, bind a real A buffer to
the ``abc`` candidates.  They pay to read A on every launch where a ``bc``
kernel does not, so timings are otherwise not comparable.

Passing operands as tensor maps
-------------------------------
An argument named ``b_desc`` or ``c_desc`` is not a pointer but the address of
a TMA tensor map, which the caller encodes on the host before launch.  GiMMiK
states everything it requires of that map in ``meta['operands']``, keyed by
the argument name, so nothing has to be inferred from the kernel.  Ask for the
descriptions through ``operands``, whose sizes let a kernel which takes them at
runtime be described as precisely as one which baked them in.

.. code:: python

    for name, spec in mm.operands(meta, n, ldb, ldc).items():
        if spec['kind'] == 'tensormap':
            ...     # encode a map for spec['operand'] and pass its address

============================ =============================================
Field                        Meaning
============================ =============================================
``kind``                     ``'tensormap'``
``operand``                  which matrix to bind, ``'b'`` or ``'c'``
``rank``                     number of dimensions
``dtype``                    element type, as a NumPy dtype
``box``                      shape of the tile the kernel moves
``global_dim``               extent of the region the kernel addresses
``global_stride``            row stride in *bytes*, ``rank - 1`` entries
``elem_stride``              element stride within the box
``interleave``               interleave mode, ``'none'``
``swizzle``                  swizzle mode, ``'none'``
``l2_promotion``             L2 promotion mode, ``'none'``
``oob_fill``                 out of bounds fill mode, ``'none'``
============================ =============================================

Dimension zero is the contiguous one, matching the order
``cuTensorMapEncodeTiled`` expects: for ``b`` that is ``(n, k)`` and for
``c`` it is ``(n, m)``.  The modes are reported as names rather than as
driver constants so the description stays independent of any one API; treat
them as values to be translated, not as defaults to assume.  A caller may
give a larger ``global_dim`` than requested provided the strides still match
the ``ldb`` and ``ldc`` the kernel was generated for.

Metadata reference
------------------
Every kernel carries these four keys, and nothing else is guaranteed.

============== ==============================================================
Key            Meaning
============== ==============================================================
``sig``        call signature label, one of the table above
``args``       ordered tuple of argument names to build the call from
``operands``   operands the caller must prepare, possibly empty
``tplname``    name of the template the kernel came from
============== ==============================================================

The remaining keys depend on the platform and on the kernel.  Launch geometry
is named in the terms of the platform's own API.

======================== ================= ==================================
Key                      Platforms         Meaning
======================== ================= ==================================
``grid``                 CUDA, HIP, PTX,   grid to launch, in blocks except
                         Metal             on Metal where it is in threads;
                                           present only when ``n`` was baked
                                           in
``block``                CUDA, HIP, PTX    block dimensions
``threadgroup``          Metal             threadgroup dimensions
``shared``               CUDA, HIP, PTX    static shared memory, in bytes
``dynamic_shared``       CUDA              dynamic shared memory, in bytes
``threadgroup_mem_size`` Metal             threadgroup memory, in bytes
``global_work_size``     OpenCL            global work size, present only when
                                           ``n`` was baked in
``local_work_size``      OpenCL            work group size, or ``None``
``local_mem_size``       OpenCL            local memory, in bytes
``width``                CUDA, HIP, PTX,   elements per work item, for
                         Metal, OpenCL     vectorised kernels
``variant``              HIP, Metal, PTX   identifier for the tuning variant
======================== ================= ==================================

``operands`` maps an argument name to a description of what that argument
needs to be.  An argument absent from it is an ordinary pointer or scalar,
and for many kernels the mapping is empty.
Buffer operands carry ``kind``, ``dtype``, ``align``, ``nbytes`` and a private
``token``; tensor map operands carry the fields tabulated above.  ``variant``
is intended for logging and cache keys, and its form is not guaranteed.

Metadata is only meaningful to the generator which produced it, and is
consumed by ``pack_a``; do not construct it by hand or move it between
generators.

The ``launch`` key describes the geometry as a function of ``n``, and every
kernel carries it; it is empty only on platforms which have no launch geometry
at all.  Each entry maps a geometry parameter onto a tuple of axes, where an
integer axis is a constant and a mapping axis is ``ceil(n / div) * mul``, with
``mul`` defaulting to one.  A kernel with ``n`` baked in has a fixed geometry,
which the metadata additionally carries resolved.

.. code:: python

    mm = CUDAMatMul(2.0*mat, beta=0.0)
    src, metadata = next(mm.kernels(np.float32))

    # A grid of ceil(n / 128) by 1 by 1
    metadata['launch'] == {'grid': ({'div': 128}, 1, 1)}

Being plain data, the description can be serialised alongside the kernel
source, allowing an application compiled ahead of time to work out its own
geometry with no access to the generator.  Callers which have a generator to
hand may instead evaluate the description with ``launch_config``, which works
for every kernel and is the one way of asking which does not need to know
whether ``n`` was baked in.

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
