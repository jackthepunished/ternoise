#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Load a QUANT_SPEC 2-bit packed weight file. Errors (exit 2) on the illegal
// '11' encoding or a file too small for `count` weights.
std::vector<int64_t> unpack_weights(const std::string& bin_path, size_t count);
