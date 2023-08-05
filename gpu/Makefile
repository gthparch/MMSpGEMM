#NVCC_FLAGS = -O3 -w -arch=compute_61 -code=sm_86 -gencode=arch=compute_61,code=sm_86
NVCC_FLAGS = -g -w -arch=compute_86 -code=sm_86 -gencode=arch=compute_86,code=sm_86
#NVCC_FLAGS = -O3 -w -arch=sm_86

mytest: main.cu
	nvcc --extended-lambda $(NVCC_FLAGS) -o mytest main.cu -I/home/elorimer3/sparse-matrix-libs/moderngpu/src -I.. ../MatrixMarket.cpp DeviceMatrix.cu

mytest2: main2.cu
	nvcc --extended-lambda $(NVCC_FLAGS) -o mytest2 main2.cu -I/home/elorimer3/sparse-matrix-libs/moderngpu/src -I.. ../MatrixMarket.cpp DeviceMatrix.cu

load_balance: load_balance.cu
	nvcc --extended-lambda $(NVCC_FLAGS) -o load_balance load_balance.cu -I/home/elorimer3/sparse-matrix-libs/moderngpu/src -I.. ../MatrixMarket.cpp DeviceMatrix.cu
