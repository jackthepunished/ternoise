// spec table for requant (QUANT_SPEC section 4)
#include "ops.h"
#include <cstdio>

int main() {
    const long long cases[][3] = {
        {0,0,0},{127,0,127},{200,0,127},{-200,0,-128},{5,1,3},{-5,1,-3},
        {3,1,2},{-3,1,-2},{2,2,1},{-2,2,-1},{36864,5,127},
    };
    for (auto& c : cases) {
        long long got = requant(c[0], (int)c[1]);
        if (got != c[2]) {
            std::printf("requant(%lld,%lld)=%lld want %lld\n", c[0], c[1], got, c[2]);
            return 1;
        }
    }
    std::puts("test_ops PASS");
    return 0;
}
