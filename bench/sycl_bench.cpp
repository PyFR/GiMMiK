// SYCL host benchmark for GiMMiK-generated kernels on the Intel GPU.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <string>

#include <sycl/sycl.hpp>
#include "sycl_common.hpp"
#include "build/dims.h"

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

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 50;
    const size_t nB = (size_t)GMK_K * GMK_N;
    const size_t nC = (size_t)GMK_M * GMK_N;

    auto B = read_bin("bench/build/B.bin", nB);
    auto Cref = read_bin("bench/build/Cref.bin", nC);

    sycl::queue q{sycl::gpu_selector_v,
                  sycl::property::queue::enable_profiling()};
    std::printf("# SYCL device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    double* dB = sycl::malloc_device<double>(nB, q);
    double* dC = sycl::malloc_device<double>(nC, q);
    q.memcpy(dB, B.data(), nB * sizeof(double)).wait();

    std::vector<double> Cout(nC);

    // useful flops = 2*nnz*n ; bytes moved = (used B rows + m)*n*8
    const double gflop = 2.0 * GMK_NNZ * (double)GMK_N / 1e9;
    const double gbyte = (double)(GMK_NBIX + GMK_M) * GMK_N * 8.0 / 1e9;

    std::printf("# %-26s %10s %10s %10s %12s\n",
                "kernel", "ms", "GFLOP/s", "GB/s", "max_relerr");

    for (int ki = 0; ki < g_sycl_n; ki++) {
        const SyclKernel& K = g_sycl[ki];

        // correctness
        q.memset(dC, 0, nC * sizeof(double)).wait();
        K.fn(q, dB, dC).wait();
        q.memcpy(Cout.data(), dC, nC * sizeof(double)).wait();
        double maxrel = 0.0;
        for (size_t i = 0; i < nC; i++) {
            double d = std::fabs(Cout[i] - Cref[i]);
            double r = d / (std::fabs(Cref[i]) + 1e-30);
            maxrel = std::max(maxrel, r);
        }

        // warmup
        for (int w = 0; w < 5; w++) K.fn(q, dB, dC);
        q.wait();

        // timed: use device profiling, take the min
        double best_ms = 1e30;
        for (int r = 0; r < reps; r++) {
            sycl::event e = K.fn(q, dB, dC);
            e.wait();
            auto t0 = e.get_profiling_info<
                sycl::info::event_profiling::command_start>();
            auto t1 = e.get_profiling_info<
                sycl::info::event_profiling::command_end>();
            best_ms = std::min(best_ms, (t1 - t0) * 1e-6);
        }

        std::printf("%-28s %10.4f %10.1f %10.1f %12.2e  %s\n",
                    K.name, best_ms, gflop / (best_ms * 1e-3),
                    gbyte / (best_ms * 1e-3), maxrel,
                    maxrel < 1e-9 ? "OK" : "FAIL");
    }

    sycl::free(dB, q);
    sycl::free(dC, q);
    return 0;
}
