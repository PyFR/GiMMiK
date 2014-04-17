# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
from gimmik.types import Matrix
from gimmik.platform import Platform
from mako.template import Template

def generateKernel(data, platform, alpha=1.0, beta=0.0, double=True,
                   reduced=True, embedded=True):

    """Generate source code for a matrix multiplication kernel.

    Generated kernels are capable of computing a matrix product of the form
        C = alpha * A x B + beta * C.
    The source code for the generated kernel is specialised according to the 
    arguments given to this functions. The code can be then compiled and 
    executed on the platform of choice.

    Args:
        data (:class:`numpy.array`): The operator matrix *A*.

        platform (:class:`gimmik.Platform`): The platform to generate code for
        e.g. *Platform.CUDA*. See :class:`gimmik.Platform` for more detail.

    Kwargs:
        alpha (float): Operator matrix multiplier. See description.

        beta (float): Output matrix multiplier. See description.

        double (bool): If True double-precision mode is used.
        Single-precision otherwise.
        
        reduced (bool): If True common sub-expressions will be eliminated.

        embedded (bool): Currently unused.

    Returns:
        string. The source code of the kernel.
    """

    # Remember the precision we want to deal with
    if double:
        dtype = np.float64
        ctype = 'double'
    else:
        dtype = np.float32
        ctype = 'float'

    # Cast the data to double-precision to avoid unnecessary precision loss
    data = data.astype(np.float64)

    # Define the threshold for removing noise
    threshold = 10e-10

    # Remove too small values
    _removeSmall(data, threshold)
    
    # Close values can be aggregated together to reduce the number of constants
    _reduceSimilar(data, threshold)

    # Deal with the alpha constant for the kernel by pre-multiplying
    if alpha != 1.0:
        data *= alpha

    # Transform data into the internal representation, which has aggregated
    # information about positions of non-zero elements in the array
    matrix = Matrix(data, dtype)

    # Generate subterms for use in the kernel
    subterms = _generateSubterms(matrix, reduced)

    # Generate products for use in the kernel
    products = _generateProducts(matrix)

    # !!!
    # WARNING: The order of the subterms and products lists corresponds 
    # directly to the order in which they are output in the kernel code.
    # !!!

    # Find the appropriate kernel template to use
    templateFile = _lookupTemplate(matrix, platform, beta, embedded, reduced)

    # Output the generated kernel
    kernelTemplate = Template(filename=templateFile, disable_unicode=True)
    kernel = kernelTemplate.render(dtype=ctype,
                                subterms=subterms,
                                products=products,
                                beta=beta)

    return kernel

def _lookupTemplate(matrix, platform, beta, embedded, reduced):
    """
    Return the filename containing an appropriate template (according to
    the arguments given) for kernel generation.
    """
    from pkg_resources import Requirement, resource_filename
    
    if platform is Platform.CUDA:
        pl = 'cuda'
    elif platform is Platform.OPENCL:
        pl = 'opencl'

    if reduced:
        return resource_filename(__name__, 'kernels/' + pl + '/bpmm.mako')
    elif not reduced:
        return resource_filename(__name__, 'kernels/' + pl + '/bpmm_nr.mako')

def _generateSubterms(matrix, reduced):
    """
    Generates a list of subterms. Each subterm in the list is unique and
    itself is a tuple of column indices of the operator matrix
    with equal non-zero values in any row. If reduced is False, each subterm
    is a singleton. E.g. This matrix:
    . A . B . A
    . A A . . .
    . . . B . .
    will output three subterms (reduced==True):
    (1, 5) for the A entries in row 0
    (3)    for the B entry in row 0 (same subterm for row 2 will be ignored)
    (1, 2) for the A entries in row 1
    """
    subterms = set()
    for row in matrix.rowData:
        for subterm in row.values():
            if reduced:
                subterms.add(tuple(subterm))
            elif not reduced:
                subterms.update(map(lambda s: tuple([s]), subterm))
    return list(subterms)

def _generateProducts(matrix):
    """
    Generates a list of dictionaries, one for each row of the matrix,
    which map non-zero values in that row to a subterm. Subterms are tuples
    of column indices of the operator matrix with equal non-zero values.
    E.g. This matrix:
    . A . B . A
    . A A . . .
    . . . B . .
    will output three products:
    {A: (1, 5), B: (3)} for row 0
    {A: (1, 2)} for row 1
    {B: (3)} for row 2
    """
    products = []
    for row in matrix.rowData:
        products.append({nz: tuple(subterm) for nz, subterm in row.iteritems()})
    return products

def _removeSmall(data, threshold):
    """Set all values smaller than a threshhold to zero"""

    indices = np.abs(data) < threshold
    data[indices] = 0.0

def _reduceSimilar(data, threshold):
    """
    Find all non-zero values in the data and replace all values within
    a given threshold with their median in order to remove noise
    """

    # Gather all the non-zero values in the data array
    nonzeros = data[np.nonzero(data)].tolist()
    # Iterate until the list is exhaused
    while nonzeros:
        # Peek a non-zero value
        nz = nonzeros[0]
        # Find indices of all non-zero values within the threshold from nz
        nonzeros_arr = np.array(nonzeros)
        indices = np.abs(nonzeros_arr - nz) < threshold
        # Find the median of all the values within the threshold
        median = np.median(nonzeros_arr[indices])
        # For all non-zero values within the threshold
        for nz in nonzeros_arr[indices]:
            # Remove them from the nonzeros list
            nonzeros.remove(nz)
            # Replace them in the data array with the median
            data[data == nz] = median
