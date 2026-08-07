CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Werror

.PHONY: test pytest ctest
pytest:
	.venv/bin/pytest -q

golden: sim/main.cpp sim/ops.cpp sim/io.cpp sim/ops.h sim/tensor.h
	$(CXX) $(CXXFLAGS) sim/main.cpp sim/ops.cpp sim/io.cpp -o golden

test_ops: sim/test_ops.cpp sim/ops.cpp sim/io.cpp sim/ops.h sim/tensor.h
	$(CXX) $(CXXFLAGS) sim/test_ops.cpp sim/ops.cpp sim/io.cpp -o test_ops

ctest: golden test_ops
	./test_ops
	@for d in vectors/case*/; do ./golden $$d || exit 1; done

test: pytest ctest
