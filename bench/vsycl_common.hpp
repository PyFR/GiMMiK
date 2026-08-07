#pragma once
#include <sycl/sycl.hpp>

// uniform thunk: (queue, b, c, n, ldb, ldc) -> event
typedef sycl::event (*VRunFn)(sycl::queue&, void*, void*, int, int, int);

struct VSycl {
    const char* name;
    const char* tpl;
    int cas;
    VRunFn fn;
};

extern const VSycl g_vsycl[];
extern const int g_vsycl_n;
