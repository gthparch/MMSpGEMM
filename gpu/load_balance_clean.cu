#include <moderngpu/cta_load_balance.hxx>
#include <moderngpu/cta_scan.hxx>
#include <moderngpu/cta_mergesort.hxx>

#include <cub/cub.cuh>

#include <iostream>
#include <fstream>
#include <random>
#include <cstring>
#include <chrono>
#include <map>
#include <unordered_map>

#include "MatrixMarket.h"
#include "DeviceMatrix.h"

#include "../MyHash.h"


// must match BLOCK_SIZE used to generate blocks in ../load-balance-test.cpp
#define BLOCK_SIZE          2048
#define ITEMS_PER_THREAD    16
#define NUM_THREADS         128

// TODO: Put this in a proper header file to share
class TSplit
{
public:
    int a_row;
    int bp;
    int b_col;
};


struct CustomLess
{
    template <typename DataType>
    __device__ bool operator()(const DataType& lhs, const DataType& rhs) { return lhs < rhs; }
};


__global__ void cuda_load_block_coop(const int *AmRowPtrs, const int *AmColIdx, const float *AmCSRVals,
                                     const int *BmRowPtrs, const int *BmColIdx, const float *BmCSRVals,
                                     const int *lb_data, const int *lb_block_ptrs, const TSplit* lb_thread_splits, float *output,
                                     int *out_keys, float *out_vals, int *atomic_p)
{
    int block = blockIdx.x;
    int cur_block_ptr = lb_block_ptrs[block];
    int next_block_ptr = lb_block_ptrs[block+1];
    int end_row = lb_data[next_block_ptr];
    int start_row = lb_data[cur_block_ptr];

    typedef cub::BlockRadixSort<int, NUM_THREADS, ITEMS_PER_THREAD, float, 4, true> BlockRadixSort;
    typedef cub::BlockReduce<float, NUM_THREADS> BlockReduce;
    __shared__ union {
        typename BlockRadixSort::TempStorage block_radix_storage;
        typename BlockReduce::TempStorage block_reduce_storage;
        float all_vals[BLOCK_SIZE];
    } smem;
    int thread_keys[ITEMS_PER_THREAD];
    float thread_vals[ITEMS_PER_THREAD];

    TSplit split = lb_thread_splits[block * NUM_THREADS + threadIdx.x];
    int row_end = AmRowPtrs[split.a_row + 1];
    int brow = AmColIdx[split.bp] - 1;
    int seg_end = BmRowPtrs[brow+1];

    int coeff_start = AmRowPtrs[split.a_row];
    if (split.a_row == end_row)
        seg_end = BmRowPtrs[brow] + lb_data[next_block_ptr + 2 + (split.bp - coeff_start)];
    float Acoeff = AmCSRVals[split.bp];

    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        if (split.b_col >= seg_end)
        {
            split.bp++;
            if (split.bp >= row_end) {
                split.a_row++;
                split.bp = AmRowPtrs[split.a_row];
                row_end = AmRowPtrs[split.a_row + 1];
            }
            Acoeff = AmCSRVals[split.bp];
            brow = AmColIdx[split.bp] - 1;
            split.b_col = BmRowPtrs[brow];
            seg_end = BmRowPtrs[brow+1];
            if (split.a_row == end_row) {
                int coeff_start = AmRowPtrs[split.a_row];
                seg_end = BmRowPtrs[brow] + lb_data[next_block_ptr + 2 + (split.bp - coeff_start)];
            }
        }

        thread_keys[i] = ((split.a_row - start_row) << 25) | BmColIdx[split.b_col];
        thread_vals[i] = BmCSRVals[split.b_col] * Acoeff;
        split.b_col++;
    }

    __syncthreads();

    BlockRadixSort(smem.block_radix_storage).Sort(thread_keys, thread_vals); //, 0, 24);

    __syncthreads();




/************************************************
 * Reduction phase
 */
    #define FULL_MASK   0xffffffff
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    typedef cub::BlockLoad<int, NUM_THREADS, ITEMS_PER_THREAD> BlockLoadInt;
    typedef cub::BlockLoad<float, NUM_THREADS, ITEMS_PER_THREAD> BlockLoadFloat;
    typedef cub::BlockStore<int, NUM_THREADS, ITEMS_PER_THREAD> BlockStoreInt;
    typedef cub::BlockStore<float, NUM_THREADS, ITEMS_PER_THREAD> BlockStoreFloat;

    // XXX: Hard-coded 4 warps
    // Could combine these in the union below ...?
	__shared__ int delta_shared[4 + NUM_THREADS];	
    __shared__ float carry_out[NUM_THREADS];
    __shared__ int block_keys[4 + 1];       // reuse delta_shared?

    if (lane == 0)
        block_keys[warp] = thread_keys[0];

    int p = 0;
    float pcarry_out = 0;
    for (int i=1; i < ITEMS_PER_THREAD; i++)
    {
        if (thread_keys[p] == thread_keys[i]) {
            thread_vals[p] += thread_vals[i];
            pcarry_out = thread_vals[p];
        }
        else {
            p++;
            thread_keys[p] = thread_keys[i];
            thread_vals[p] = thread_vals[i];
            pcarry_out = 0;
        }
    }
    carry_out[threadIdx.x] = pcarry_out;

    int neighbor_key = __shfl_down_sync(FULL_MASK, thread_keys[0], 1);
    if (lane == 31) {
        neighbor_key = block_keys[warp+1];
    }
    // what happens to the very last warp carry_out?
    bool rseg_end = (p > 0);
    if (thread_keys[ITEMS_PER_THREAD-1] != neighbor_key) {
        p++;
        rseg_end = true;
        carry_out[threadIdx.x] = 0.0;
    }

    uint warp_mask = FULL_MASK >> (31 - lane);
    uint cta_mask = 0x7fffffff >> (31 - lane);

    uint warp_bits = __ballot_sync(FULL_MASK, rseg_end);
    delta_shared[warp] = warp_bits;
    __syncthreads();

    const int NumWarps = NUM_THREADS / 32;
    if (threadIdx.x < NumWarps) {
        uint cta_bits = __ballot_sync(0xf, 0 != delta_shared[threadIdx.x]);
        int warpSegment = 31 - __clz(cta_mask & cta_bits);
        int start = (-1 != warpSegment) ? 
            (31 - __clz(delta_shared[warpSegment]) + 32 * warpSegment) : 0;
        delta_shared[NumWarps + threadIdx.x] = start;
    }
    __syncthreads();
 
    // Find the closest flag to the left of this thread within the warp.
    // Include the flag for this thread.
    int start = 31 - __clz(warp_mask & warp_bits);
    if(-1 != start) start += ~31 & threadIdx.x;
    else start = delta_shared[NumWarps + warp];
    __syncthreads();
 
    uint tid_delta = threadIdx.x - start;

    for (int offset = 1; offset < NUM_THREADS; offset += offset)
    {
        if (tid_delta >= offset)
            carry_out[threadIdx.x] += carry_out[threadIdx.x - offset];
        __syncthreads();
    }

    // apply carry_out ...
    if (p > 0 && threadIdx.x > 0)
        thread_vals[0] += carry_out[threadIdx.x - 1];

    // do need inclusive sum 
    typedef cub::BlockScan<int, NUM_THREADS> BlockScanWriteOut;
    __shared__ typename BlockScanWriteOut::TempStorage bswo_storage;

    int start_p, total_p;
    BlockScanWriteOut(bswo_storage).ExclusiveSum(p, start_p, total_p);

    // TODO: coop write worth it?
    __shared__ int shared_p;
    if (threadIdx.x == 0)
        shared_p = atomicAdd(atomic_p, total_p);
    __syncthreads();

    start_p += shared_p;
    for (int i=0; i < p; i++) {
        out_keys[start_p + i] = thread_keys[i];
        out_vals[start_p + i] = thread_vals[i];
    }
//    out_counts[threadIdx.x] = p;


/*
 * End reduction phase
 **************************************************************/
}


int main(int argc, char **argv)
{
    mgpu::standard_context_t context;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mtx file>" << std::endl;
        exit(1);
    }

    MatrixMarket matA { argv[1] };
    MatrixMarket& matB = matA;
    DeviceMatrix dmA(matA, context);
    DeviceMatrix& dmB = dmA;

    std::ifstream if_data("../lb_data.bin", std::ifstream::binary);
    std::ifstream if_block_ptr("../lb_block_ptrs.bin", std::ifstream::binary);
    std::ifstream if_thread_splits("../lb_thread_splits.bin", std::ifstream::binary);

    if_data.seekg(0, if_data.end);
    if_block_ptr.seekg(0, if_block_ptr.end);
    if_thread_splits.seekg(0, if_thread_splits.end);
    int lb_data_size = if_data.tellg();
    int lb_block_ptrs_size = if_block_ptr.tellg();
    int lb_thread_splits_size = if_thread_splits.tellg();
    if_data.seekg(0, if_data.beg);
    if_block_ptr.seekg(0, if_block_ptr.beg);
    if_thread_splits.seekg(0, if_thread_splits.beg);

    std::cout << "lb_data size = " << lb_data_size << std::endl;
    std::cout << "lb_block_ptr size = " << lb_block_ptrs_size << std::endl;
    std::cout << "lb_thread_splits size = " << lb_thread_splits_size << std::endl;

    std::vector<int> lb_data, lb_block_ptrs;
    std::vector<std::vector<TSplit>> lb_thread_splits;
    lb_data.resize(lb_data_size >> 2);
    lb_block_ptrs.resize(lb_block_ptrs_size >> 2);
    lb_thread_splits.resize(lb_block_ptrs_size >> 2);
    for (int i=0; i < lb_thread_splits.size()-1; i++) {
        lb_thread_splits[i].resize(128);
        if_thread_splits.read((char *)lb_thread_splits[i].data(), sizeof(TSplit) * 128);
    }

    if_data.read((char *)lb_data.data(), lb_data_size);
    if_block_ptr.read((char *)lb_block_ptrs.data(), lb_block_ptrs_size);

    std::cout << "read lb_data = " << lb_data.size() << std::endl;
    std::cout << "read lb_block_ptrs = " << lb_block_ptrs.size() << std::endl;
    std::cout << "read lb_thread_splits" << std::endl;

    /*
    for (int i=1; i < lb_block_ptrs.size(); i++) {
        if (lb_data[lb_block_ptrs[i]] == lb_data[lb_block_ptrs[i-1]]) {
            std::cout << "single row: " << i << std::endl;
            break;
        }
    }
    */

    std::vector<TSplit> lb_flat_splits;
    for (int i = 0; i < lb_thread_splits.size()-1; i++)
    {
        for (int j = 0; j < NUM_THREADS; j++)
            lb_flat_splits.push_back(lb_thread_splits[i][j]);
    }
    mgpu::mem_t<int> d_lb_data = mgpu::to_mem(lb_data, context);
    mgpu::mem_t<int> d_lb_block_ptrs = mgpu::to_mem(lb_block_ptrs, context);
    mgpu::mem_t<TSplit> d_flat_tsplits = mgpu::to_mem(lb_flat_splits, context);

    std::vector<float> out_buffer;
    std::vector<int> out_keys;
    std::vector<float> out_vals;
    out_keys.resize(100000000);
    out_vals.resize(100000000);
    std::vector<int> atomic_p;
    atomic_p.resize(2);

    out_buffer.resize(lb_block_ptrs.size());
    mgpu::mem_t<float> d_output = mgpu::to_mem(out_buffer, context);
    mgpu::mem_t<int> d_out_keys = mgpu::to_mem(out_keys, context);
    mgpu::mem_t<float> d_out_vals = mgpu::to_mem(out_vals, context);
    mgpu::mem_t<int> d_atomic_p = mgpu::to_mem(atomic_p, context);

    /*
    int numBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, cuda_load_block_coop, NUM_THREADS, 2048 * 4);
    std::cout << "max occupancy(?) = " << numBlocks << std::endl;
    */

    auto start = std::chrono::system_clock::now();
    cuda_load_block_coop<<<lb_block_ptrs.size()-1, NUM_THREADS>>>(dmA.raw.d_row_ptrs, dmA.raw.d_col_idx, dmA.raw.d_values,
                                             dmB.raw.d_row_ptrs, dmB.raw.d_col_idx, dmB.raw.d_values,
                                             d_lb_data.data(), d_lb_block_ptrs.data(), d_flat_tsplits.data(), d_output.data(), d_out_keys.data(), d_out_vals.data(), d_atomic_p.data());
    cudaDeviceSynchronize();
    auto finish = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(finish - start).count();
    std::cout << "elapsed: " << elapsed << std::endl;

    std::vector<float> read_back = mgpu::from_mem(d_output);
    std::cout << "Result 0: " << read_back[0] << std::endl;

    return 0;
}
