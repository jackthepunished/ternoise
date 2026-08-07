#include "tensor.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>

Tensor load_tensor(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    int rank = 0;
    f >> rank;
    Tensor t;
    size_t n = 1;
    for (int i = 0; i < rank; ++i) {
        int d = 0;
        f >> d;
        t.dims.push_back(d);
        n *= (size_t)d;
    }
    t.data.resize(n);
    for (size_t i = 0; i < n; ++i) f >> t.data[i];
    if (!f) { std::fprintf(stderr, "truncated tensor %s\n", path.c_str()); std::exit(2); }
    return t;
}

void save_tensor(const std::string& path, const Tensor& t) {
    std::ofstream f(path);
    f << t.dims.size();
    for (int d : t.dims) f << " " << d;
    f << "\n";
    for (size_t i = 0; i < t.data.size(); ++i) f << (i ? " " : "") << t.data[i];
    f << "\n";
}
