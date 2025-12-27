CUDA_HOME   = /Soft/cuda/12.2.2
NVCC        = $(CUDA_HOME)/bin/nvcc
ARCH        = -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include $(ARCH) --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64

# -------- FUENTES --------
SRC_TIEMPOS = pagerank-tiempos.cu
SRC_V1 = pagerank-parV1.cu
SRC_V2 = pagerank-parV2.cu
SRC_V3 = pagerank-parV3.cu

# -------- OBJETOS Y EJECUTABLES --------
OBJ_TIEMPOS = pagerank-tiempos.o
OBJ_V1 = pagerank-parV1.o
OBJ_V2 = pagerank-parV2.o
OBJ_V3 = pagerank-parV3.o

# -------- EJECUTABLES --------
EXE_TIEMPOS = pagerank-tiempos.exe
EXE_V1 = pagerank-parV1.exe
EXE_V2 = pagerank-parV2.exe
EXE_V3 = pagerank-parV3.exe

# -------- TARGET POR DEFECTO --------
default: $(EXE_V1) $(EXE_V2) $(EXE_V3)

# -------- COMPILACIÓN OBJETOS --------
$(OBJ_V1): $(SRC_V1)
	$(NVCC) -c -o $@ $< $(NVCC_FLAGS)

$(OBJ_V2): $(SRC_V2)
	$(NVCC) -c -o $@ $< $(NVCC_FLAGS)
	
$(OBJ_V3): $(SRC_V3)
	$(NVCC) -c -o $@ $< $(NVCC_FLAGS)

# -------- LINKADO EJECUTABLES --------
$(EXE_V1): $(OBJ_V1)
	$(NVCC) $< -o $@ $(LD_FLAGS)

$(EXE_V2): $(OBJ_V2)
	$(NVCC) $< -o $@ $(LD_FLAGS)

$(EXE_V3): $(OBJ_V3)
	$(NVCC) $< -o $@ $(LD_FLAGS)

# -------- LIMPIEZA --------
clean:
	rm -rf *.o *.exe
