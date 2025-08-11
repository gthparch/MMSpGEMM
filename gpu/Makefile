#NVCC_FLAGS = -O3 -w -arch=compute_61 -code=sm_86 -gencode=arch=compute_61,code=sm_86
NVCC_FLAGS = -O3 -w -arch=compute_80 -code=sm_80 -gencode=arch=compute_80,code=sm_80 -Xptxas -v
INCLUDES=-I/nethome/elorimer3/sparse-matrix-libs/moderngpu/src -I..

load_balance_clean: load_balance_clean.cu
	nvcc --extended-lambda $(NVCC_FLAGS) -o load_balance_clean load_balance_clean.cu ${INCLUDES} ../MatrixMarket.cpp DeviceMatrix.cu

split: split.cu
	nvcc --extended-lambda $(NVCC_FLAGS) -o split split.cu ${INCLUDES} ../MatrixMarket.cpp DeviceMatrix.cu
