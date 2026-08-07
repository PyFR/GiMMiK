// SYCL correctness validation across dtype/beta/mode cases.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>
#include "vsycl_common.hpp"
#include "vbuild/vcases.h"

static std::vector<char> read_bin(const std::string& p, size_t bytes) {
    std::vector<char> v(bytes);
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "open %s\n", p.c_str()); std::exit(1); }
    if (std::fread(v.data(), 1, bytes, f) != bytes) { std::exit(1); }
    std::fclose(f);
    return v;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("# SYCL device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("# %-34s %-9s %-6s %-7s %11s  %s\n",
                "kernel", "dtype", "beta", "mode", "norm_err", "status");

    int fails = 0;
    for (int i = 0; i < g_vsycl_n; i++) {
        const VSycl& K = g_vsycl[i];
        const VCase& c = g_cases[K.cas];
        const size_t nB = (size_t)c.k * c.n, nC = (size_t)c.m * c.n;
        const bool f32 = (c.code == 1);

        auto B = read_bin("bench/vbuild/case" + std::to_string(c.id) + "_B.bin",
                          nB * c.dsize);
        auto Ci = read_bin("bench/vbuild/case" + std::to_string(c.id) + "_Ci.bin",
                           nC * c.dsize);
        auto Cr = read_bin("bench/vbuild/case" + std::to_string(c.id) + "_Cref.bin",
                           nC * sizeof(double));
        const double* Cref = reinterpret_cast<const double*>(Cr.data());

        void* dB = sycl::malloc_device(nB * c.dsize, q);
        void* dC = sycl::malloc_device(nC * c.dsize, q);
        q.memcpy(dB, B.data(), nB * c.dsize).wait();
        q.memcpy(dC, Ci.data(), nC * c.dsize).wait();   // beta*C uses initial C

        K.fn(q, dB, dC, c.n, c.ldb, c.ldc).wait();

        std::vector<char> out(nC * c.dsize);
        q.memcpy(out.data(), dC, nC * c.dsize).wait();

        double maxabs = 0.0, maxerr = 0.0;
        for (size_t e = 0; e < nC; e++) {
            double got = f32 ? (double)reinterpret_cast<float*>(out.data())[e]
                             : reinterpret_cast<double*>(out.data())[e];
            maxabs = std::max(maxabs, std::fabs(Cref[e]));
            maxerr = std::max(maxerr, std::fabs(got - Cref[e]));
        }
        double norm = maxerr / (maxabs + 1e-300);
        double tol = f32 ? 1e-5 : 1e-12;
        bool ok = norm < tol;
        fails += !ok;

        std::printf("%-36s %-9s %-6.2f %-7s %11.2e  %s\n",
                    K.name, f32 ? "float32" : "float64", c.beta,
                    c.is_dynamic ? "dyn" : "static", norm, ok ? "OK" : "FAIL");

        sycl::free(dB, q);
        sycl::free(dC, q);
    }
    std::printf("# %d/%d passed\n", g_vsycl_n - fails, g_vsycl_n);
    return fails ? 1 : 0;
}
