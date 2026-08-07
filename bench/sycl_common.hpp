#pragma once
#include <sycl/sycl.hpp>

struct SyclKernel {
    const char* name;
    const char* tpl;
    sycl::event (*fn)(sycl::queue&, const double*, double*);
};

extern const SyclKernel g_sycl[];
extern const int g_sycl_n;
