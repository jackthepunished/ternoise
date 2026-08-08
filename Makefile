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

.PHONY: lint rtl_test mem_cases

# Golden case dirs converted to $readmemh inputs. Needs the python env, so
# from a git worktree run: make rtl_test PYTHON=/path/to/repo/.venv/bin/python
MEM_CASES = case01_conv_hand case02_conv_rand case03_conv_extreme \
            case04_network case05_network_packed
MEM_STAMPS = $(patsubst %,build/mem/%/meta.txt,$(MEM_CASES))

build/mem/%/meta.txt: vectors/%/params.json tools/case_to_mem.py tools/pack_weights.py
	@mkdir -p $(@D)
	$(PYTHON) -m tools.case_to_mem vectors/$* --out $(@D)

# A real-shape 64x64 network case built from a seeded random net through M3's
# calibration and export path. Bit-exactness needs no trained weights, so M4
# verification does not block on M3's training gate; the trained-weight run is
# M4 Task 7. This explicit rule takes precedence over the pattern rule above
# (there is no vectors/export64).
build/mem/export64/meta.txt: tools/export_case.py model/calibrate.py \
                             model/network.py tools/case_to_mem.py
	@mkdir -p $(@D)
	$(PYTHON) -c "import torch, numpy as np; \
	  from model.network import DenoiseNet; from model.calibrate import calibrate; \
	  from tools.export_case import export_case; \
	  torch.manual_seed(1058); net = DenoiseNet(); \
	  rng = np.random.default_rng(1058); \
	  f = rng.integers(-128, 128, size=(7, 64, 64)).astype(np.int64); \
	  calibrate(net, [f]); export_case(net, f, 'build/mem/export64_case')"
	$(PYTHON) -m tools.case_to_mem build/mem/export64_case --out $(@D)

mem_cases: $(MEM_STAMPS) build/mem/export64/meta.txt

build/tb_linebuffer/tb_linebuffer: rtl/linebuffer.sv rtl/tb/tb_linebuffer.cpp
	@mkdir -p $(@D)
	$(VERILATOR) $(VFLAGS) --Mdir build/tb_linebuffer --top-module linebuffer \
	    rtl/linebuffer.sv $(abspath rtl/tb/tb_linebuffer.cpp) -o tb_linebuffer

build/tb_requant/tb_requant: rtl/requant.sv rtl/tb/tb_requant.cpp sim/ops.cpp sim/io.cpp
	@mkdir -p $(@D)
	$(VERILATOR) $(VFLAGS) --Mdir build/tb_requant --top-module requant \
	    rtl/requant.sv $(abspath rtl/tb/tb_requant.cpp) \
	    $(abspath sim/ops.cpp) $(abspath sim/io.cpp) -o tb_requant

# One verilation per (C_IN, C_OUT, RELU) shape - the conv layer's channel
# counts are elaboration-time parameters, so each golden conv case needs its
# own model. TB_* mirror the -G values into the harness, which re-checks them
# against the case's meta.txt.
CONV_SRC = rtl/pe.sv rtl/requant.sv rtl/conv3x3.sv
CONV_TB  = $(abspath rtl/tb/tb_conv3x3.cpp) $(abspath sim/io.cpp)

define CONV_BUILD
build/tb_conv3x3_$(1)/tb_conv3x3: $$(CONV_SRC) rtl/tb/tb_conv3x3.cpp sim/io.cpp
	@mkdir -p $$(@D)
	$$(VERILATOR) $$(VFLAGS) --Mdir $$(@D) --top-module conv3x3 \
	    -GC_IN=$(2) -GC_OUT=$(3) -GRELU="1'b$(4)" \
	    -CFLAGS "-DTB_C_IN=$(2) -DTB_C_OUT=$(3) -DTB_RELU=$(4)" \
	    $$(CONV_SRC) $$(CONV_TB) -o tb_conv3x3
endef

$(eval $(call CONV_BUILD,case01_conv_hand,1,1,1))
$(eval $(call CONV_BUILD,case02_conv_rand,8,8,1))
$(eval $(call CONV_BUILD,case03_conv_extreme,32,4,0))

CONV_BINS = build/tb_conv3x3_case01_conv_hand/tb_conv3x3 \
            build/tb_conv3x3_case02_conv_rand/tb_conv3x3 \
            build/tb_conv3x3_case03_conv_extreme/tb_conv3x3

# +case= is the golden vector dir (input/expected/bias), +meta= the converted
# meta.txt, +w1=/+b1= the .mem files the DUT $readmemh's.
CONV_RUN = $(1) +case=vectors/$(2) +meta=build/mem/$(2)/meta.txt \
           +w1=build/mem/$(2)/w1.mem +b1=build/mem/$(2)/b1.mem

# The whole v1 network. One verilation covers every network case: channels are
# fixed by the topology, dims and shifts are runtime ports.
TOP_SRC = rtl/pe.sv rtl/requant.sv rtl/linebuffer.sv rtl/conv3x3.sv \
          rtl/delay_fifo.sv rtl/denoiser_top.sv
TOP_TB  = $(abspath rtl/tb/tb_denoiser.cpp) $(abspath sim/io.cpp)

build/tb_denoiser/tb_denoiser: $(TOP_SRC) rtl/tb/tb_denoiser.cpp sim/io.cpp
	@mkdir -p $(@D)
	$(VERILATOR) $(VFLAGS) --Mdir $(@D) --top-module denoiser_top \
	    $(TOP_SRC) $(TOP_TB) -o tb_denoiser

# $(1) = golden vector dir, $(2) = converted .mem dir, $(3) = report tag.
TOP_RUN = ./build/tb_denoiser/tb_denoiser +case=$(1) +meta=$(2)/meta.txt +tag=$(3) \
          +w1=$(2)/w1.mem +b1=$(2)/b1.mem +w2=$(2)/w2.mem +b2=$(2)/b2.mem \
          +w3=$(2)/w3.mem +b3=$(2)/b3.mem +w4=$(2)/w4.mem +b4=$(2)/b4.mem \
          +w5=$(2)/w5.mem +b5=$(2)/b5.mem

lint:
	$(VERILATOR) --lint-only -Wall rtl/linebuffer.sv --top-module linebuffer
	$(VERILATOR) --lint-only -Wall rtl/pe.sv --top-module pe
	$(VERILATOR) --lint-only -Wall rtl/requant.sv --top-module requant
	$(VERILATOR) --lint-only -Wall $(CONV_SRC) --top-module conv3x3
	$(VERILATOR) --lint-only -Wall rtl/delay_fifo.sv --top-module delay_fifo
	$(VERILATOR) --lint-only -Wall $(TOP_SRC) --top-module denoiser_top

rtl_test: build/tb_linebuffer/tb_linebuffer build/tb_requant/tb_requant \
          $(CONV_BINS) build/tb_denoiser/tb_denoiser mem_cases
	./build/tb_linebuffer/tb_linebuffer
	./build/tb_requant/tb_requant
	$(call CONV_RUN,./build/tb_conv3x3_case01_conv_hand/tb_conv3x3,case01_conv_hand)
	$(call CONV_RUN,./build/tb_conv3x3_case02_conv_rand/tb_conv3x3,case02_conv_rand)
	$(call CONV_RUN,./build/tb_conv3x3_case03_conv_extreme/tb_conv3x3,case03_conv_extreme)
	$(call TOP_RUN,vectors/case04_network,build/mem/case04_network,case04_network)
	$(call TOP_RUN,vectors/case05_network_packed,build/mem/case05_network_packed,case05_network_packed)
	$(call TOP_RUN,build/mem/export64_case,build/mem/export64,export64)
