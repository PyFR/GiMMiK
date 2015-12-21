# -*- coding: utf-8 -*-

import unittest
import numpy as np
import gimmik.generator as gen
from pycuda import driver, compiler, gpuarray
from pycuda.tools import mark_cuda_test

class KernelGenerationTestCase(unittest.TestCase):

    def setUp(self):
        # Define matrix size
        self.size = 40
        # Define platform type
        self.platform = 'cuda'

    def tearDown(self):
        pass

    def _test_generateKernel(self, alpha, beta, double, reduced):
        # Translate parameters
        if double:
            dtype = np.float64
        else:
            dtype = np.float32
        # Generate three random matrices
        a_cpu = np.random.randn(self.size, self.size).astype(dtype)
        b_cpu = np.random.randn(self.size, self.size).astype(dtype)
        c_cpu = np.random.randn(self.size, self.size).astype(dtype)
        # Compute reference solution on the CPU to verify
        reference = c_cpu * beta + np.dot(a_cpu * alpha, b_cpu)
        # Allocate and copy the b and c array onto the Device
        b_gpu = gpuarray.to_gpu(b_cpu)
        c_gpu = gpuarray.to_gpu(c_cpu)
        # Re-assign reference to c_cpu
        c_cpu = reference
        # Generate the kernel code
        kernel = gen.generateKernel(a_cpu,
                                    alpha=alpha,
                                    beta=beta,
                                    double=double,
                                    reduced=reduced,
                                    platform=self.platform)
        # Get the strides of all matrices
        typeSize = dtype().itemsize
        bstride = b_gpu.strides[0] / typeSize
        cstride = c_gpu.strides[0] / typeSize

        # Compile and run the kernel
        module = compiler.SourceModule(kernel)
        function = module.get_function('gimmik_mm')
        block, grid = (self.size, 1, 1), (1, 1)
        function(b_gpu,
                 c_gpu,
                 np.int32(self.size),
                 np.int32(bstride),
                 np.int32(cstride),
                 block=block,
                 grid=grid)

        # Verify the result
        if double:
            rtol, atol = 1.e-4, 1.e-7
        else:
            rtol, atol = 1.e-3, 1.e-6
        self.assertTrue(np.allclose(c_cpu, c_gpu.get(), rtol, atol),
                "Reference solution differes from GPU's")

    @mark_cuda_test
    def test_generateReducedDouble(self):
        # Define test parameters
        alpha = 1.0
        beta = 0.0
        double=True
        reduced=True
        # Execute the test
        self._test_generateKernel(alpha, beta, double, reduced)

    @mark_cuda_test
    def test_generateReducedSingle(self):
        # Define test parameters
        alpha = 1.0
        beta = 0.0
        double=False
        reduced=True
        # Execute the test
        self._test_generateKernel(alpha, beta, double, reduced)

    @mark_cuda_test
    def test_generateNotReducedDouble(self):
        # Define test parameters
        alpha = 1.0
        beta = 0.0
        double=True
        reduced=False
        # Execute the test
        self._test_generateKernel(alpha, beta, double, reduced)

    @mark_cuda_test
    def test_generateNotReducedSingle(self):
        # Define test parameters
        alpha = 1.0
        beta = 0.0
        double=True
        reduced=False
        # Execute the test
        self._test_generateKernel(alpha, beta, double, reduced)

    @mark_cuda_test
    def test_generateAlphaBetaDouble(self):
        # Define test parameters
        alpha = 2.0
        beta = 3.0
        double=True
        reduced=True
        # Execute the test
        self._test_generateKernel(alpha, beta, double, reduced)

    @mark_cuda_test
    def test_generateAlphaBetaSingle(self):
        # Define test parameters
        alpha = 2.0
        beta = 3.0
        double=False
        reduced=True
        # Execute the test
        self._test_generateKernel(alpha, beta, double, reduced)

if __name__ == '__main__':
    unittest.main()
