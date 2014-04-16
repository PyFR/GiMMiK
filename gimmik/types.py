# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict

class Matrix:
    """Wrapper for :class:`numpy.array` to provide more data to the generator.

    The Additional data stored in this wrapper class includes the information
    about the given array aggregated across rows and columns. There is little
    use for this class outside of this package.
    """
    
    def __init__(self, matrix, dtype):
        self.data = matrix
        self.dtype = dtype
        self.rowData = self._generateRowData()
        self.colData = self._generateColData()

    def _generateRowData(self):
        """
        Create a list, which maps each row in the matrix to a dictionary.
        The inner dictionary maps each non-zero value in the matrix to the 
        column index at which it occurs.
        """

        # Aggregate for the result
        rowData = [defaultdict(list) for i in range(self.data.shape[0])]

        # Create a 2 dimensional iterator
        it = np.nditer(self.data, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            value = it[0].item()
            # Cast the data in the matrix into the desired type
            if self.dtype(value) != 0.0:
                row, col = it.multi_index
                rowData[row][value].append(col)
            it.iternext()

        return rowData

    def _generateColData(self):
        """
        Create a list, which maps each column in the matrix to a list
        of indices of the rows containing non-zero values in that column.
        """

        # Aggregare for the result
        colData = [list() for i in range(self.data.shape[1])]

        # Create a 2 dimensional iterator
        it = np.nditer(self.data, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            value = it[0].item()
            # Cast the data in the matrix into the desired type
            if self.dtype(value) != 0.0:
                row, col = it.multi_index
                colData[col].append(row)
            it.iternext()

        return colData
