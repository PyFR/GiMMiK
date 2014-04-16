# -*- coding: utf-8 -*-

import unittest
import numpy as np
from gimmik.types import Matrix

class TypesTestCase(unittest.TestCase):

    def setUp(self):
        # Generate a random array of 1's and 0's
        x, y = np.random.randint(4, 10), np.random.randint(4, 10)
        matrix = np.random.randint(2, size= x * y).reshape(x, y)
        self.matrix = Matrix(matrix, np.float32)

    def tearDown(self):
        pass

    def test_dimensions(self):
        self.assertEqual(len(self.matrix.rowData), self.matrix.data.shape[0],
                "Dimensions of rowData do not match the number of rows")
        self.assertEqual(len(self.matrix.colData), self.matrix.data.shape[1],
                "Dimensions of colData do not match the number of columns")

    def test_rowData(self):
        rowData = self.matrix.rowData
        # Since the only non-zero value in the matrix is 1, we simply
        # look up the length of the list in rowData and compare
        # with the sum of elements in that row
        counts = [len(row[1]) for row in rowData]
        self.assertTrue(np.array_equal(counts, self.matrix.data.sum(axis=1)),
                'The number of elements in each row does not match')

    def test_colData(self):
        colData = self.matrix.colData
        # Since the only non-zero value in the matrix is 1, we simply
        # look up the length of the list in colData and compare
        # with the sum of elements in that col
        counts = [len(col) for col in colData]
        self.assertTrue(np.array_equal(counts, self.matrix.data.sum(axis=0)),
                'The number of elements in each column does not match')

if __name__ == '__main__':
    unittest.main()
