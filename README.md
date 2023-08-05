# load_balance_spgemm
WIP load-balancing SpGEMM for GPU using k-way merge path

load-balance-test.cpp builds the load balancing structures on the CPU and writes them to files. It also has some test functions for applying them on blocks. Big mess

gpu directory has the GPU code using ModernGPU and CUB for applying the pre-written LB partitioning files to blocks. TODO: reduction, compaction, global shuffle

Mostly just checking this in as backup
