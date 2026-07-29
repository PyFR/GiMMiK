// OpenCL correctness validation across dtype/beta/mode cases.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vbuild/vcases.h"
#include "vbuild/vocl_registry.h"

#define CK(x) do { cl_int _e=(x); if(_e!=CL_SUCCESS){ \
    std::fprintf(stderr,"CL err %d @%d (%s)\n",_e,__LINE__,#x); std::exit(1);} } while(0)

static std::vector<char> read_bin(const std::string& p, size_t bytes) {
    std::vector<char> v(bytes);
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "open %s\n", p.c_str()); std::exit(1); }
    if (std::fread(v.data(), 1, bytes, f) != bytes) { std::exit(1); }
    std::fclose(f);
    return v;
}
static std::string read_text(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "open %s\n", p.c_str()); std::exit(1); }
    std::fseek(f, 0, SEEK_END); long s = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::string t(s, '\0');
    if (std::fread(&t[0], 1, s, f) != (size_t)s) std::exit(1);
    std::fclose(f); return t;
}

int main() {
    cl_uint np; CK(clGetPlatformIDs(0, nullptr, &np));
    std::vector<cl_platform_id> pl(np); CK(clGetPlatformIDs(np, pl.data(), nullptr));
    cl_device_id dev = nullptr;
    for (auto p : pl) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) == CL_SUCCESS && nd) {
            std::vector<cl_device_id> d(nd);
            clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, d.data(), nullptr);
            char v[128] = {0};
            clGetDeviceInfo(d[0], CL_DEVICE_VENDOR, sizeof(v), v, nullptr);
            if (std::strstr(v, "Intel")) { dev = d[0]; break; }
        }
    }
    if (!dev) { std::fprintf(stderr, "no Intel GPU\n"); return 1; }
    char name[128] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    std::printf("# OpenCL device: %s\n", name);

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); CK(err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, nullptr, &err); CK(err);

    std::printf("# %-34s %-9s %-6s %-7s %11s  %s\n",
                "kernel", "dtype", "beta", "mode", "norm_err", "status");

    int fails = 0;
    for (int i = 0; i < g_vocl_n; i++) {
        const VOcl& K = g_vocl[i];
        const VCase& c = g_cases[K.cas];
        const size_t nB = (size_t)c.k * c.n, nC = (size_t)c.m * c.n;
        const bool f32 = (c.code == 1);

        auto B = read_bin("bench/vbuild/case" + std::to_string(c.id) + "_B.bin", nB * c.dsize);
        auto Ci = read_bin("bench/vbuild/case" + std::to_string(c.id) + "_Ci.bin", nC * c.dsize);
        auto Cr = read_bin("bench/vbuild/case" + std::to_string(c.id) + "_Cref.bin", nC * sizeof(double));
        const double* Cref = reinterpret_cast<const double*>(Cr.data());

        cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, nB * c.dsize, nullptr, &err); CK(err);
        cl_mem dC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nC * c.dsize, nullptr, &err); CK(err);
        CK(clEnqueueWriteBuffer(q, dB, CL_TRUE, 0, nB * c.dsize, B.data(), 0, nullptr, nullptr));
        CK(clEnqueueWriteBuffer(q, dC, CL_TRUE, 0, nC * c.dsize, Ci.data(), 0, nullptr, nullptr));

        std::string src = read_text("bench/vbuild/" + std::string(K.file));
        const char* cs = src.c_str(); size_t sl = src.size();
        cl_program pr = clCreateProgramWithSource(ctx, 1, &cs, &sl, &err); CK(err);
        if (clBuildProgram(pr, 1, &dev, "-cl-std=CL2.0", nullptr, nullptr) != CL_SUCCESS) {
            size_t ls; clGetProgramBuildInfo(pr, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ls);
            std::string lg(ls, '\0');
            clGetProgramBuildInfo(pr, dev, CL_PROGRAM_BUILD_LOG, ls, &lg[0], nullptr);
            std::fprintf(stderr, "build %s:\n%s\n", K.entry, lg.c_str()); std::exit(1);
        }
        cl_kernel kern = clCreateKernel(pr, K.entry, &err); CK(err);

        // args + work sizes depend on static/dynamic
        size_t gws[2], lws[2] = {(size_t)K.l0, (size_t)K.l1};
        const size_t* lp = K.has_lws ? lws : nullptr;
        cl_int nn = c.n, ldb = c.ldb, ldc = c.ldc;
        size_t cols = (c.n + K.width - 1) / K.width;   // work-items along n
        if (K.is_dynamic) {
            CK(clSetKernelArg(kern, 0, sizeof(cl_int), &nn));
            CK(clSetKernelArg(kern, 1, sizeof(cl_mem), &dB));
            CK(clSetKernelArg(kern, 2, sizeof(cl_int), &ldb));
            CK(clSetKernelArg(kern, 3, sizeof(cl_mem), &dC));
            CK(clSetKernelArg(kern, 4, sizeof(cl_int), &ldc));
        } else {
            CK(clSetKernelArg(kern, 0, sizeof(cl_mem), &dB));
            CK(clSetKernelArg(kern, 1, sizeof(cl_mem), &dC));
        }
        gws[0] = cols; gws[1] = K.has_lws ? (size_t)K.l1 : 1;
        int gdim = K.has_lws ? 2 : 1;
        if (K.has_lws) gws[0] = ((cols + K.l0 - 1) / K.l0) * K.l0;

        CK(clEnqueueNDRangeKernel(q, kern, gdim, nullptr, gws, lp, 0, nullptr, nullptr));
        CK(clFinish(q));

        std::vector<char> out(nC * c.dsize);
        CK(clEnqueueReadBuffer(q, dC, CL_TRUE, 0, nC * c.dsize, out.data(), 0, nullptr, nullptr));

        double maxabs = 0.0, maxerr = 0.0;
        for (size_t e = 0; e < nC; e++) {
            double got = f32 ? (double)reinterpret_cast<float*>(out.data())[e]
                             : reinterpret_cast<double*>(out.data())[e];
            maxabs = std::max(maxabs, std::fabs(Cref[e]));
            maxerr = std::max(maxerr, std::fabs(got - Cref[e]));
        }
        double norm = maxerr / (maxabs + 1e-300);
        double tol = f32 ? 1e-5 : 1e-12;
        bool ok = norm < tol; fails += !ok;

        std::printf("%-36s %-9s %-6.2f %-7s %11.2e  %s\n",
                    K.entry, f32 ? "float32" : "float64", c.beta,
                    c.is_dynamic ? "dyn" : "static", norm, ok ? "OK" : "FAIL");

        clReleaseKernel(kern); clReleaseProgram(pr);
        clReleaseMemObject(dB); clReleaseMemObject(dC);
    }
    std::printf("# %d/%d passed\n", g_vocl_n - fails, g_vocl_n);
    clReleaseCommandQueue(q); clReleaseContext(ctx);
    return fails ? 1 : 0;
}
