#include "unpack.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>

std::vector<int64_t> unpack_weights(const std::string& bin_path, size_t count) {
    std::ifstream f(bin_path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", bin_path.c_str()); std::exit(2); }
    std::vector<unsigned char> raw((std::istreambuf_iterator<char>(f)),
                                   std::istreambuf_iterator<char>());
    if (raw.size() * 4 < count) {
        std::fprintf(stderr, "%s too small: %zu bytes for %zu weights\n",
                     bin_path.c_str(), raw.size(), count);
        std::exit(2);
    }
    std::vector<int64_t> w(count);
    for (size_t i = 0; i < count; ++i) {
        const int code = (raw[i / 4] >> (2 * (i % 4))) & 0b11;
        if (code == 0b11) {
            std::fprintf(stderr, "illegal weight encoding 11 at index %zu in %s\n",
                         i, bin_path.c_str());
            std::exit(2);
        }
        w[i] = (code == 0b10) ? -1 : code;  // 00 -> 0, 01 -> +1, 10 -> -1
    }
    return w;
}
