#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Up to 4-D integer tensor, row-major, int64 storage (contract fits int32).
struct Tensor {
    std::vector<int> dims;
    std::vector<int64_t> data;
};

Tensor load_tensor(const std::string& path);   // io.cpp
void save_tensor(const std::string& path, const Tensor& t);
