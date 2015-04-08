# -*- coding: utf-8 -*-

import unittest
import numpy as np
import gimmik.generator as gen
import pyopencl as cl

class KernelGenerationTestCase(unittest.TestCase):

    def setUp(self):
        # Define matrix size
        self.size = 40
        # Define platform type
        self.platform = 'opencl'

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
        # Get the context and queue
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        # Shorthand for memroy flags
        mf = cl.mem_flags
        # Allocate and copy the b aray onto the Device
        b_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_cpu)
        # Allocate space for the output on the Device
        c_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=c_cpu)
        # Generate the kernel code
        kernel = gen.generateKernel(a_cpu,
                                    alpha=alpha,
                                    beta=beta,
                                    double=double,
                                    reduced=reduced,
                                    platform=self.platform)
        # Get the strides of all matrices
        typeSize = dtype().itemsize
        bstride = b_cpu.strides[0] / typeSize
        cstride = reference.strides[0] / typeSize

        # Compile and run the kernel
        program = cl.Program(ctx, kernel).build()
        gimmik_mm = program.gimmik_mm
        gimmik_mm(queue,
                  (reference.shape[0],),
                  None,
                  b_gpu,
                  c_gpu,
                  np.int32(self.size),
                  np.int32(bstride),
                  np.int32(cstride))

        # Get the product form device memory
        cl.enqueue_copy(queue, c_cpu, c_gpu)

        # Verify the result
        if double:
            rtol, atol = 1.e-4, 1.e-7
        else:
            rtol, atol = 1.e-3, 1.e-6
        self.assertTrue(np.allclose(reference, c_cpu, rtol, atol),
                "Reference solution differes from GPU's")

    def test_generateReducedDouble(self):
        # Define test parameters
        alpha = 1.0
        beta = 0.0
        double=True
        reduced=True
        # Execute the test
        self._test_generateKernel(alpha, beta, double, reduced)

    def test_generateReducedSingle(self):
        # Define test parameters
        alpha = 1.0
        beta = 0.0
        double=False
        reduced=True
        # Execute the test
        self._test_generateKernel(alpha, beta, double, reduced)

    def test_generateNotReducedDouble(self):
        # Define test parameters
        alpha = 1.0
        beta = 0.0
        double=True
        reduced=False
        # Execute the test
        self._test_generateKernel(alpha, beta, double, reduced)

    def test_generateNotReducedSingle(self):
        # Define test parameters
        alpha = 1.0
        beta = 0.0
        double=True
        reduced=False
        # Execute the test
        self._test_generateKernel(alpha, beta, double, reduced)

    def test_generateAlphaBetaDouble(self):
        # Define test parameters
        alpha = 2.0
        beta = 3.0
        double=True
        reduced=True
        # Execute the test
        self._test_generateKernel(alpha, beta, double, reduced)

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
