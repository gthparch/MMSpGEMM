# MMSpGEMM

## Organization

The top-level root directory contains some python test scripts and a reference partitioning implementation (test.py). The CUDA code is in 'gpu'.

## Building

Ensure nvcc is in your path. Change directory to 'gpu' directory. Run 'make'. This should build two binaries: split and load_balance_clean from the two respective source .cu files. The split program performs the partitioning and writes the metadata to files (lb_data.bin, lb_block_ptrs.bin). The load_balance_clean program reads these two files as well as the input matrices and performs the SpGEMM computation using the partitioning information.



