#NVCC_FLAGS = --extended-lambda -O3 -w -arch=compute_86 -code=sm_86 -gencode=arch=compute_86,code=sm_86
NVCC_FLAGS = --extended-lambda -O3 -w -arch=compute_80 -code=sm_80 -gencode=arch=compute_80,code=sm_80
INCLUDES=-Imoderngpu/src -I..

all: compute split

compute: compute.cu
	nvcc $(NVCC_FLAGS) -o compute compute.cu ${INCLUDES} ../MatrixMarket.cpp DeviceMatrix.cu

split: split.cu
	nvcc $(NVCC_FLAGS) -o split split.cu ${INCLUDES} ../MatrixMarket.cpp DeviceMatrix.cu

split_heap: split_heap.cu
	nvcc $(NVCC_FLAGS) -o split_heap split_heap.cu ${INCLUDES} ../MatrixMarket.cpp DeviceMatrix.cu
