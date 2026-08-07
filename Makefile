CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Werror

.PHONY: test pytest ctest
pytest:
	.venv/bin/pytest -q

golden: sim/main.cpp sim/ops.cpp sim/io.cpp sim/unpack.cpp sim/ops.h sim/tensor.h sim/unpack.h
	$(CXX) $(CXXFLAGS) sim/main.cpp sim/ops.cpp sim/io.cpp sim/unpack.cpp -o golden

test_ops: sim/test_ops.cpp sim/ops.cpp sim/io.cpp sim/ops.h sim/tensor.h
	$(CXX) $(CXXFLAGS) sim/test_ops.cpp sim/ops.cpp sim/io.cpp -o test_ops

HOSTSRC = host/main.cpp host/gl_loader.cpp host/shader_util.cpp
host_app: $(HOSTSRC) host/gl_loader.h host/shader_util.h
	$(CXX) $(CXXFLAGS) $(HOSTSRC) -o host_app -lglfw -lGL

test_quantize: host/test_quantize.cpp host/quantize.cpp host/quantize.h
	$(CXX) $(CXXFLAGS) host/test_quantize.cpp host/quantize.cpp -o test_quantize

ctest: golden test_ops test_quantize
	./test_ops
	./test_quantize
	@for d in vectors/case*/; do ./golden $$d || exit 1; done

test: pytest ctest
