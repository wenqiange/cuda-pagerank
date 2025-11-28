CUDA_HOME   = /Soft/cuda/12.2.2
NVCC        = $(CUDA_HOME)/bin/nvcc
ARCH        = -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include $(ARCH) --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib

SRC    = pagerank.cu
EXE	    = pagerank.exe
OBJ	    = pagerank.o

default: $(EXE)

pagerank.o: $(SRC)
	$(NVCC) -c -o $@ $(SRC) $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
