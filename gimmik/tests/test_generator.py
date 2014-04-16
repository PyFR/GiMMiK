# -*- coding: utf-8 -*-

import unittest
import numpy as np
import gimmik.generator as gen
import random

class TypesTestCase(unittest.TestCase):

    def setUp(self):
        # Generate a random array
        x, y = np.random.randint(5, 20), np.random.randint(5, 20)
        self.matrix = np.random.randint(5, 100, x * y).reshape(x, y)
        self.threshold = 5

    def tearDown(self):
        pass

    def test_removeSmall(self):
        # Add some values smaller than threshold
        smallValues = np.random.randint(-self.threshold + 1, self.threshold, 4)
        x, y = self.matrix.shape
        indx, indy = random.sample(range(x), 4), random.sample(range(y), 4)
        self.matrix[indx, indy] = smallValues

        self.assertEqual(
                self.matrix[np.abs(self.matrix) < self.threshold].size, 4,
                "The number of values below the threshold is incorrect")
        gen._removeSmall(self.matrix, self.threshold)
        self.assertEqual(self.matrix[self.matrix == 0].size, 4,
                "The number of 0 values should be 4")


    def test_reduceSimilar(self):
        # Alter the array so that there are no similar numbers
        self.matrix *= (self.threshold * 2)
        # Create some similar numers
        similar = np.arange(1, self.threshold + 1)
        ns = similar.size
        # Find ther median
        median = np.median(similar)
        # Find how many entries equal to median we have currently
        mn = self.matrix[self.matrix == median].size
        # Put the similar entries into the array
        x, y = self.matrix.shape
        indx, indy = random.sample(range(x), ns), random.sample(range(y), ns)
        self.matrix[indx, indy] = similar 
        # Find how many unique elements we have after including similar
        uniqnum = np.unique(self.matrix).size
        # Reduce the similar entries
        gen._reduceSimilar(self.matrix, self.threshold)
        self.assertEqual(self.matrix[self.matrix == median].size, ns + mn, 
                """ The number of median values was not change by the number
                    of similar values """)
        self.assertEqual(np.unique(self.matrix).size, 
                uniqnum - ns + 1 + (mn != 0), 
                """ The number of unique values was not change by the number
                    of similar values """)

if __name__ == '__main__':
    unittest.main()
