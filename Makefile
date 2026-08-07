CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Werror

# python -m pytest (not the pytest script) so the cwd precedes the editable
# install on sys.path; from a git worktree, override PYTHON with the main
# checkout's interpreter: make test PYTHON=/path/to/repo/.venv/bin/python
PYTHON ?= .venv/bin/python

.PHONY: test pytest ctest
pytest:
	$(PYTHON) -m pytest -q

golden: sim/main.cpp sim/ops.cpp sim/io.cpp sim/unpack.cpp sim/ops.h sim/tensor.h sim/unpack.h
	$(CXX) $(CXXFLAGS) sim/main.cpp sim/ops.cpp sim/io.cpp sim/unpack.cpp -o golden

test_ops: sim/test_ops.cpp sim/ops.cpp sim/io.cpp sim/ops.h sim/tensor.h
	$(CXX) $(CXXFLAGS) sim/test_ops.cpp sim/ops.cpp sim/io.cpp -o test_ops

HOSTSRC = host/main.cpp host/gl_loader.cpp host/shader_util.cpp host/quantize.cpp
host_app: $(HOSTSRC) host/gl_loader.h host/shader_util.h host/quantize.h stb_impl.o
	$(CXX) $(CXXFLAGS) $(HOSTSRC) stb_impl.o -o host_app -lglfw -lGL

stb_impl.o: host/stb_impl.cpp host/stb_image_write.h
	$(CXX) -std=c++17 -O2 -c host/stb_impl.cpp -o stb_impl.o

test_quantize: host/test_quantize.cpp host/quantize.cpp host/quantize.h
	$(CXX) $(CXXFLAGS) host/test_quantize.cpp host/quantize.cpp -o test_quantize

ctest: golden test_ops test_quantize
	./test_ops
	./test_quantize
	@for d in vectors/case*/; do ./golden $$d || exit 1; done

test: pytest ctest

# ---- RTL (M4) ----
VERILATOR = verilator
VFLAGS = --cc --exe --build -j 0 -Wall
# verilator runs its generated sub-make with cwd = --Mdir, so every user C++
# source handed to it must be an absolute path ($(abspath ...)).

.PHONY: lint rtl_test

build/tb_linebuffer/tb_linebuffer: rtl/linebuffer.sv rtl/tb/tb_linebuffer.cpp
	@mkdir -p $(@D)
	$(VERILATOR) $(VFLAGS) --Mdir build/tb_linebuffer --top-module linebuffer \
	    rtl/linebuffer.sv $(abspath rtl/tb/tb_linebuffer.cpp) -o tb_linebuffer

lint:
	$(VERILATOR) --lint-only -Wall rtl/linebuffer.sv --top-module linebuffer

rtl_test: build/tb_linebuffer/tb_linebuffer
	./build/tb_linebuffer/tb_linebuffer
