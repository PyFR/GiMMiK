// OpenCL host benchmark for GiMMiK-generated kernels on the Intel GPU.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

#include "build/dims.h"
#include "build/ocl_registry.h"

#define CK(x) do { cl_int _e = (x); if (_e != CL_SUCCESS) { \
    std::fprintf(stderr, "OpenCL error %d at %s:%d (%s)\n", _e, __FILE__, \
                 __LINE__, #x); std::exit(1); } } while (0)

static std::vector<double> read_bin(const char* path, size_t count) {
    std::vector<double> v(count);
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    if (std::fread(v.data(), sizeof(double), count, f) != count) {
        std::fprintf(stderr, "short read %s\n", path); std::exit(1);
    }
    std::fclose(f);
    return v;
}

static std::string read_text(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::string s(sz, '\0');
    if (std::fread(&s[0], 1, sz, f) != (size_t)sz) { std::exit(1); }
    std::fclose(f);
    return s;
}

static cl_device_id pick_gpu(cl_platform_id* out_plat) {
    cl_uint np = 0;
    CK(clGetPlatformIDs(0, nullptr, &np));
    std::vector<cl_platform_id> plats(np);
    CK(clGetPlatformIDs(np, plats.data(), nullptr));
    for (auto p : plats) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS
            || nd == 0)
            continue;
        std::vector<cl_device_id> devs(nd);
        CK(clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr));
        char vendor[256] = {0};
        clGetDeviceInfo(devs[0], CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
        if (std::strstr(vendor, "Intel") || std::strstr(vendor, "INTEL")) {
            *out_plat = p;
            return devs[0];
        }
    }
    std::fprintf(stderr, "No Intel GPU found\n");
    std::exit(1);
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 50;
    const size_t nB = (size_t)GMK_K * GMK_N;
    const size_t nC = (size_t)GMK_M * GMK_N;

    auto B = read_bin("bench/build/B.bin", nB);
    auto Cref = read_bin("bench/build/Cref.bin", nC);

    cl_platform_id plat;
    cl_device_id dev = pick_gpu(&plat);
    char name[256] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    std::printf("# OpenCL device: %s\n", name);

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CK(err);
    cl_command_queue_properties qp[] = {CL_QUEUE_PROPERTIES,
                                        CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err);
    CK(err);

    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, nB * sizeof(double),
                               nullptr, &err); CK(err);
    cl_mem dC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nC * sizeof(double),
                               nullptr, &err); CK(err);
    CK(clEnqueueWriteBuffer(q, dB, CL_TRUE, 0, nB * sizeof(double), B.data(),
                            0, nullptr, nullptr));

    std::vector<double> Cout(nC);
    const double gflop = 2.0 * GMK_NNZ * (double)GMK_N / 1e9;
    const double gbyte = (double)(GMK_NBIX + GMK_M) * GMK_N * 8.0 / 1e9;

    std::printf("# %-26s %10s %10s %10s %12s\n",
                "kernel", "ms", "GFLOP/s", "GB/s", "max_relerr");

    for (int ki = 0; ki < g_ocl_n; ki++) {
        const OclKernel& K = g_ocl[ki];
        std::string src = read_text(std::string("bench/build/") + K.file);
        const char* csrc = src.c_str();
        size_t slen = src.size();

        cl_program prog = clCreateProgramWithSource(ctx, 1, &csrc, &slen, &err);
        CK(err);
        cl_int be = clBuildProgram(prog, 1, &dev, "-cl-std=CL2.0", nullptr, nullptr);
        if (be != CL_SUCCESS) {
            size_t ls = 0;
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ls);
            std::string log(ls, '\0');
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, ls, &log[0], nullptr);
            std::fprintf(stderr, "build failed for %s:\n%s\n", K.entry, log.c_str());
            std::exit(1);
        }
        cl_kernel kern = clCreateKernel(prog, K.entry, &err); CK(err);
        CK(clSetKernelArg(kern, 0, sizeof(cl_mem), &dB));
        CK(clSetKernelArg(kern, 1, sizeof(cl_mem), &dC));

        // rounded global work size
        size_t gws[2] = {(size_t)K.g0, (size_t)K.g1};
        size_t lws[2] = {(size_t)K.l0, (size_t)K.l1};
        const size_t* lp = nullptr;
        if (K.l0 > 0) {
            lp = lws;
            gws[0] = ((gws[0] + lws[0] - 1) / lws[0]) * lws[0];
        }

        // correctness
        double zero = 0.0;
        CK(clEnqueueFillBuffer(q, dC, &zero, sizeof(double), 0,
                               nC * sizeof(double), 0, nullptr, nullptr));
        CK(clEnqueueNDRangeKernel(q, kern, K.gdim, nullptr, gws, lp, 0,
                                  nullptr, nullptr));
        CK(clFinish(q));
        CK(clEnqueueReadBuffer(q, dC, CL_TRUE, 0, nC * sizeof(double),
                               Cout.data(), 0, nullptr, nullptr));
        double maxrel = 0.0;
        for (size_t i = 0; i < nC; i++) {
            double d = std::fabs(Cout[i] - Cref[i]);
            double r = d / (std::fabs(Cref[i]) + 1e-30);
            maxrel = std::max(maxrel, r);
        }

        // warmup
        for (int w = 0; w < 5; w++)
            CK(clEnqueueNDRangeKernel(q, kern, K.gdim, nullptr, gws, lp, 0,
                                      nullptr, nullptr));
        CK(clFinish(q));

        double best_ms = 1e30;
        for (int r = 0; r < reps; r++) {
            cl_event ev;
            CK(clEnqueueNDRangeKernel(q, kern, K.gdim, nullptr, gws, lp, 0,
                                      nullptr, &ev));
            CK(clWaitForEvents(1, &ev));
            cl_ulong t0, t1;
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                                    sizeof(t0), &t0, nullptr);
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                                    sizeof(t1), &t1, nullptr);
            best_ms = std::min(best_ms, (t1 - t0) * 1e-6);
            clReleaseEvent(ev);
        }

        std::printf("%-28s %10.4f %10.1f %10.1f %12.2e  %s\n",
                    K.entry, best_ms, gflop / (best_ms * 1e-3),
                    gbyte / (best_ms * 1e-3), maxrel,
                    maxrel < 1e-9 ? "OK" : "FAIL");

        clReleaseKernel(kern);
        clReleaseProgram(prog);
    }

    clReleaseMemObject(dB);
    clReleaseMemObject(dC);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return 0;
}
