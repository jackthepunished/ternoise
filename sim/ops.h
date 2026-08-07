#pragma once
#include "tensor.h"

int64_t requant(int64_t acc, int shift);
Tensor conv3x3(const Tensor& x, const Tensor& w, const Tensor& bias);  // raw accumulators
Tensor layer(const Tensor& x, const Tensor& w, const Tensor& bias, int shift, bool use_relu);
Tensor network(const Tensor& x, const std::vector<Tensor>& ws,
               const std::vector<Tensor>& bs, const std::vector<int>& shifts);
