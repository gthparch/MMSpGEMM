# MMSpGEMM

## Organization

The top-level root directory contains some python test scripts and a reference partitioning implementation (partitioner.py). The CUDA code is in 'gpu'. The partitioner programs (CUDA and Python) take the two input matrix file names in Matrix Market format and output the metadata files for second stage execution. A few extra files and programs used in experiments that didn't make it in the paper are provided for curiosity.

## Building

Ensure nvcc is in your path. Change directory to 'gpu' directory. Run 'make'. This should build two binaries: split and compute from the two respective source .cu files. The split program performs the partitioning and writes the metadata to files (lb_data.bin, lb_block_ptrs.bin). The compute program reads these two files as well as the input matrices and performs the SpGEMM computation using the partitioning information.
