CUDA_HOME ?= /usr/local/cuda

NVCC       ?= $(CUDA_HOME)/bin/nvcc
ARCH       ?= -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
NVCC_FLAGS ?= -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include $(ARCH) --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS   ?= -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib

TARGETS := pagerank pagerank-parV1 pagerank-parV2 pagerank-tiempos
COMMON_DEPS := utils.cu params.h

.PHONY: all clean

all: $(TARGETS)

pagerank: pagerank.cu $(COMMON_DEPS)
pagerank-parV1: pagerank-parV1.cu $(COMMON_DEPS)
pagerank-parV2: pagerank-parV2.cu $(COMMON_DEPS)
pagerank-tiempos: pagerank-tiempos.cu $(COMMON_DEPS)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LD_FLAGS)

clean:
	rm -rf *.o $(TARGETS)
