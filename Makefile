UNAME_S   := $(shell uname -s)
CXX       ?= clang++
ifeq ($(UNAME_S),Linux)
CXX       := g++
endif
CXXFLAGS  := -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter
OBJCXX    := clang++ -O3 -std=c++17 -fobjc-arc -framework Foundation -framework Metal
METAL     := xcrun -sdk macosx metal
METALLIB  := xcrun -sdk macosx metallib
NVCC      ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CUDA_ARCH ?= native
NVCCFLAGS := -O3 -std=c++17 -arch=$(CUDA_ARCH) --use_fast_math
NVCC_ENC_FLAGS := -O3 -std=c++17 -arch=$(CUDA_ARCH) --fmad=false --prec-div=true --prec-sqrt=true
METAL_ENC_FLAGS := -fno-fast-math -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise -ffp-contract=off -I src/converters
HIPCC     ?= hipcc
HIP_ARCH  ?= gfx1100
HIPFLAGS  := -O3 -std=c++17 --offload-arch=$(HIP_ARCH)
BUILD     := build
PREFIX    ?= $(HOME)/.local
BINDIR    := $(PREFIX)/bin
LIBDIR    := $(PREFIX)/lib/tfq3

.PHONY: all metal cuda hip generic qwen27 flashnext glm inkling2 tfq4 tools test test-metal test-tfq4-metal test-cuda test-hip install uninstall clean

ifeq ($(UNAME_S),Darwin)
all: tools metal
else
all: tools cuda
endif

$(BUILD):
	@mkdir -p $(BUILD)

tools: $(BUILD)/tfq3-inspect $(BUILD)/tfq3-qwen27-package $(BUILD)/tfq3-train-tables

metal: generic qwen27 flashnext glm inkling2 tfq4 $(BUILD)/tfq3-metal-test $(BUILD)/tfq3-metal-test.metallib $(BUILD)/tfq4-metal-test $(BUILD)/tfq4-metal-test.metallib

cuda: tools $(BUILD)/tfq3-generic-convert-cuda $(BUILD)/tfq3-qwen27-convert-cuda $(BUILD)/tfq3-flashnext-convert-cuda $(BUILD)/tfq3-glm-convert-cuda $(BUILD)/tfq3-inkling2-convert-cuda $(BUILD)/tfq3-ernie-convert-cuda $(BUILD)/tfq4-convert-cuda $(BUILD)/tfq3-cuda-test

generic: $(BUILD)/tfq3-generic-convert $(BUILD)/tfq3-generic-convert.metallib

tfq4: $(BUILD)/tfq4-convert $(BUILD)/tfq4-convert.metallib

qwen27: $(BUILD)/tfq3-qwen27-convert $(BUILD)/tfq3-qwen27-convert.metallib $(BUILD)/tfq3-qwen27-package

flashnext: $(BUILD)/tfq3-flashnext-convert $(BUILD)/tfq3-flashnext-convert.metallib

glm: $(BUILD)/tfq3-glm-convert $(BUILD)/tfq3-glm-convert.metallib

inkling2: $(BUILD)/tfq3-inkling2-convert $(BUILD)/tfq3-inkling2-convert.metallib

$(BUILD)/tfq3-inspect: src/tools/tfq3_inspect.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/tfq3-qwen27-package: src/converters/qwen27/package.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/tfq3-train-tables: src/tools/tfq3_train_tables.cpp src/converters/tfq_constants.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

# --- Apple Silicon Metal Targets (encoders use -fno-fast-math for cross-platform bit identity) ---
METAL_ENC_HDRS := src/converters/tfq_constants.hpp src/converters/canonical_tables.hpp src/converters/tfq_tables_io.hpp

$(BUILD)/tfq3-generic-convert: src/converters/generic/convert.mm src/converters/generic/planner.hpp $(METAL_ENC_HDRS) | $(BUILD)
	$(OBJCXX) $< -o $@

$(BUILD)/tfq3-generic-convert.metallib: src/converters/generic/convert.metal src/converters/metal_encode_common.metal | $(BUILD)
	$(METAL) $(METAL_ENC_FLAGS) -c $< -o $(BUILD)/generic.air
	$(METALLIB) $(BUILD)/generic.air -o $@
	@rm -f $(BUILD)/generic.air

$(BUILD)/tfq3-qwen27-convert: src/converters/qwen27/convert.mm $(METAL_ENC_HDRS) | $(BUILD)
	$(OBJCXX) $< -o $@

$(BUILD)/tfq3-qwen27-convert.metallib: src/converters/qwen27/convert.metal src/converters/metal_encode_common.metal | $(BUILD)
	$(METAL) $(METAL_ENC_FLAGS) -c $< -o $(BUILD)/qwen27.air
	$(METALLIB) $(BUILD)/qwen27.air -o $@
	@rm -f $(BUILD)/qwen27.air

$(BUILD)/tfq3-flashnext-convert: src/converters/flashnext/convert.mm src/converters/flashnext/flashnext_common.hpp src/converters/moe_heat_rank.hpp $(METAL_ENC_HDRS) | $(BUILD)
	$(OBJCXX) $< -o $@

$(BUILD)/tfq3-flashnext-convert.metallib: src/converters/flashnext/convert.metal src/converters/metal_encode_common.metal | $(BUILD)
	$(METAL) $(METAL_ENC_FLAGS) -c $< -o $(BUILD)/flashnext.air
	$(METALLIB) $(BUILD)/flashnext.air -o $@
	@rm -f $(BUILD)/flashnext.air

$(BUILD)/tfq3-glm-convert: src/converters/glm/convert.mm src/converters/moe_heat_rank.hpp $(METAL_ENC_HDRS) | $(BUILD)
	$(OBJCXX) $< -o $@

$(BUILD)/tfq3-glm-convert.metallib: src/converters/glm/convert.metal src/converters/metal_encode_common.metal | $(BUILD)
	$(METAL) $(METAL_ENC_FLAGS) -c $< -o $(BUILD)/glm.air
	$(METALLIB) $(BUILD)/glm.air -o $@
	@rm -f $(BUILD)/glm.air

$(BUILD)/tfq3-inkling2-convert: src/converters/inkling2/convert.mm src/converters/inkling2/ink2_common.hpp $(METAL_ENC_HDRS) | $(BUILD)
	$(OBJCXX) $< -o $@

$(BUILD)/tfq3-inkling2-convert.metallib: src/converters/inkling2/convert.metal src/converters/metal_encode_common.metal | $(BUILD)
	$(METAL) $(METAL_ENC_FLAGS) -c $< -o $(BUILD)/inkling2.air
	$(METALLIB) $(BUILD)/inkling2.air -o $@
	@rm -f $(BUILD)/inkling2.air

$(BUILD)/tfq3-metal-test: src/tools/tfq3_metal_bench.mm | $(BUILD)
	$(OBJCXX) $< -o $@

$(BUILD)/tfq3-metal-test.metallib: src/kernels/metal/tfq3_metal.metal | $(BUILD)
	$(METAL) -c $< -o $(BUILD)/tfq3_metal.air
	$(METALLIB) $(BUILD)/tfq3_metal.air -o $@
	@rm -f $(BUILD)/tfq3_metal.air

$(BUILD)/tfq4-convert: src/converters/tfq4/convert.mm src/converters/tfq4/tfq4_format.hpp src/converters/generic/planner.hpp $(METAL_ENC_HDRS) | $(BUILD)
	$(OBJCXX) $< -o $@

$(BUILD)/tfq4-convert.metallib: src/converters/tfq4/convert.metal | $(BUILD)
	$(METAL) $(METAL_ENC_FLAGS) -c $< -o $(BUILD)/tfq4_convert.air
	$(METALLIB) $(BUILD)/tfq4_convert.air -o $@
	@rm -f $(BUILD)/tfq4_convert.air

$(BUILD)/tfq4-metal-test: src/tools/tfq4_metal_bench.mm src/converters/tfq4/tfq4_format.hpp | $(BUILD)
	$(OBJCXX) $< -o $@

$(BUILD)/tfq4-metal-test.metallib: src/kernels/metal/tfq4_metal.metal | $(BUILD)
	$(METAL) -fno-fast-math -c $< -o $(BUILD)/tfq4_metal.air
	$(METALLIB) $(BUILD)/tfq4_metal.air -o $@
	@rm -f $(BUILD)/tfq4_metal.air

# --- NVIDIA CUDA Targets (sm_80+: Ampere, Ada, Hopper, Blackwell / GB10) ---
# Encoders compile with --fmad=false (no --use_fast_math) for cross-platform bit identity with Metal.
$(BUILD)/tfq3-generic-convert-cuda: src/converters/generic/convert.cu src/converters/generic/planner.hpp src/converters/cuda_encode_common.cuh $(METAL_ENC_HDRS) | $(BUILD)
	$(NVCC) $(NVCC_ENC_FLAGS) $< -o $@

$(BUILD)/tfq4-convert-cuda: src/converters/tfq4/convert.cu src/converters/tfq4/tfq4_format.hpp src/converters/generic/planner.hpp $(METAL_ENC_HDRS) | $(BUILD)
	$(NVCC) $(NVCC_ENC_FLAGS) $< -o $@

$(BUILD)/tfq3-qwen27-convert-cuda: src/converters/qwen27/convert.cu src/converters/cuda_encode_common.cuh $(METAL_ENC_HDRS) | $(BUILD)
	$(NVCC) $(NVCC_ENC_FLAGS) $< -o $@

$(BUILD)/tfq3-flashnext-convert-cuda: src/converters/flashnext/convert.cu src/converters/flashnext/flashnext_common.hpp src/converters/cuda_encode_common.cuh src/converters/moe_heat_rank.hpp $(METAL_ENC_HDRS) | $(BUILD)
	$(NVCC) $(NVCC_ENC_FLAGS) $< -o $@

$(BUILD)/tfq3-glm-convert-cuda: src/converters/glm/convert.cu src/converters/cuda_encode_common.cuh src/converters/moe_heat_rank.hpp $(METAL_ENC_HDRS) | $(BUILD)
	$(NVCC) $(NVCC_ENC_FLAGS) $< -o $@

$(BUILD)/tfq3-inkling2-convert-cuda: src/converters/inkling2/convert.cu src/converters/inkling2/ink2_common.hpp src/converters/cuda_encode_common.cuh $(METAL_ENC_HDRS) | $(BUILD)
	$(NVCC) $(NVCC_ENC_FLAGS) $< -o $@

$(BUILD)/tfq3-ernie-convert-cuda: src/converters/ernie/convert.cu src/converters/cuda_encode_common.cuh $(METAL_ENC_HDRS) | $(BUILD)
	$(NVCC) $(NVCC_ENC_FLAGS) $< -o $@

$(BUILD)/tfq3-cuda-test: src/tools/tfq3_cuda_bench.cu src/kernels/cuda/tfq3_cuda.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD)/tfq4-e8-bench: src/tools/tfq4_e8_bench.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@

# --- AMD ROCm / HIP Targets (RDNA3/4 gfx1100/gfx1200, CDNA2/3 gfx90a/gfx942) ---
hip: $(BUILD)/tfq3-hip-test

$(BUILD)/tfq3-hip-test: src/tools/tfq3_hip_bench.hip src/kernels/hip/tfq3_hip.h | $(BUILD)
	$(HIPCC) $(HIPFLAGS) $< -o $@

test-hip: hip
	@$(BUILD)/tfq3-hip-test

test: all
	@python3 reference/tfq3_reference.py --self-test
	@python3 reference/tfq4_lattice.py --self-test
	@python3 tests/vectors_test.py
	@python3 tests/e2e_test.py
	@python3 tests/inkling2_test.py
	@if [ "$(UNAME_S)" = "Darwin" ] && [ -x "$(BUILD)/tfq3-metal-test" ]; then $(BUILD)/tfq3-metal-test; fi
	@if [ "$(UNAME_S)" = "Darwin" ] && [ -x "$(BUILD)/tfq4-metal-test" ]; then $(BUILD)/tfq4-metal-test; fi
	@if [ "$(UNAME_S)" != "Darwin" ] && [ -x "$(BUILD)/tfq3-cuda-test" ]; then $(BUILD)/tfq3-cuda-test; fi
	@echo "All TFQ3 & TFQ4 targets built and verified in $(BUILD)/"

test-metal: $(BUILD)/tfq3-metal-test $(BUILD)/tfq3-metal-test.metallib
	@$(BUILD)/tfq3-metal-test --bench

test-tfq4-metal: $(BUILD)/tfq4-metal-test $(BUILD)/tfq4-metal-test.metallib
	@$(BUILD)/tfq4-metal-test --bench

test-cuda: cuda
	@$(BUILD)/tfq3-cuda-test --bench

install: all
	@mkdir -p "$(BINDIR)" "$(LIBDIR)"
	@install -m 0755 bin/tfq3 "$(BINDIR)/tfq3"
	@install -m 0755 $(BUILD)/tfq3-inspect $(BUILD)/tfq3-qwen27-package "$(LIBDIR)/"
	@install -m 0644 src/converters/flashnext/package.py "$(LIBDIR)/flashnext_package.py"
ifeq ($(UNAME_S),Darwin)
	@install -m 0755 $(BUILD)/tfq3-generic-convert $(BUILD)/tfq3-qwen27-convert $(BUILD)/tfq3-flashnext-convert $(BUILD)/tfq3-glm-convert $(BUILD)/tfq3-inkling2-convert $(BUILD)/tfq4-convert "$(LIBDIR)/"
	@install -m 0644 $(BUILD)/tfq3-generic-convert.metallib $(BUILD)/tfq3-qwen27-convert.metallib $(BUILD)/tfq3-flashnext-convert.metallib $(BUILD)/tfq3-glm-convert.metallib $(BUILD)/tfq3-inkling2-convert.metallib $(BUILD)/tfq4-convert.metallib "$(LIBDIR)/"
else
	@install -m 0755 $(BUILD)/tfq3-generic-convert-cuda $(BUILD)/tfq3-qwen27-convert-cuda $(BUILD)/tfq3-flashnext-convert-cuda $(BUILD)/tfq3-glm-convert-cuda $(BUILD)/tfq3-inkling2-convert-cuda $(BUILD)/tfq3-ernie-convert-cuda $(BUILD)/tfq4-convert-cuda $(BUILD)/tfq3-cuda-test "$(LIBDIR)/"
endif
	@echo "Installed tfq3 -> $(BINDIR)/tfq3 (runtime binaries in $(LIBDIR))"

uninstall:
	rm -f "$(BINDIR)/tfq3"
	rm -rf "$(LIBDIR)"

clean:
	rm -rf $(BUILD)
