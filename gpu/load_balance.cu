#include <moderngpu/cta_load_balance.hxx>
#include <moderngpu/cta_scan.hxx>
#include <moderngpu/cta_mergesort.hxx>
//#include <moderngpu/kernel_load_balance.hxx>

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
#define BLOCK_SIZE      2048
#define ITEMS_PER_THREAD    16
#define NUM_THREADS     128

// TODO: Put this in a proper header file to share
class TSplit
{
public:
    int a_row;
    int bp;
    int b_col;
};


void sort_placeholder(int *p, int *r)
{
    std::stable_sort(p, r);
}


int load_multi_block(const MatrixMarket& A, const MatrixMarket& B, const std::vector<int>& lb_data, int prev_block_ptr, int block_ptr, int *out_idx, float *out_vals)
{
    int start_row = lb_data[prev_block_ptr];
    int end_row = lb_data[block_ptr];
    int total_count = 0;

    struct s {
        int idx;
        float data;
    } shared[BLOCK_SIZE];
//    int shared_idx[BLOCK_SIZE];
//    float shared_data[BLOCK_SIZE];
//    int ranger[BLOCK_SIZE];
//    int ranger_p = 0;
//    MyHash reducer;
//    std::unordered_map<int, float> reducer;

    // TODO: multiply by A coefficient
    for (int row=start_row; row <= end_row; row++)
    {
        int coeff_start = A.mRowPtrs[row];
        int coeff_end = A.mRowPtrs[row+1];
        for (int bp = coeff_start; bp < coeff_end; bp++)
        {
            int brow = A.mColIdx[bp]-1;
            int seg_start = B.mRowPtrs[brow];
            int seg_end = B.mRowPtrs[brow+1];
            if (row == start_row)
                seg_start += lb_data[prev_block_ptr + 2 + (bp - coeff_start)];
            if (row == end_row)
                seg_end = B.mRowPtrs[brow] + lb_data[block_ptr + 2 + (bp - coeff_start)];

            int count = seg_end - seg_start;
//            std::memcpy(shared_idx + total_count, B.mColIdx.data() + seg_start, count * sizeof(int));
//            std::memcpy(shared_data + total_count, B.mCSRVals.data() + seg_start, count * sizeof(float));
            for (int i=0; i < count; i++) {
//                shared_idx[total_count + i] = B.mColIdx[seg_start + i] | (row << 24);
//                shared_data[total_count + i] = B.mCSRVals[seg_start + i] * A.mCSRVals[bp];
                shared[total_count + i].idx = B.mColIdx[seg_start + i] | (row << 24);
                shared[total_count + i].data = B.mCSRVals[seg_start + i] * A.mCSRVals[bp];
//                ranger[ranger_p++] = ranger_p;
//                int key = B.mColIdx[seg_start + i] | (row << 24);
//                reducer[key] += B.mCSRVals[seg_start + i];
//                reducer.update(key, B.mCSRVals[seg_start + i]);
            }

            total_count += count;
//            std::cout << "memcpy B[" << brow << "] from " << seg_start << " - " << seg_end << " = " << count << std::endl;
        }
    }

    // TODO: Sort keys and values so that below compaction is correct
//    std::stable_sort((int*)shared_idx, (int*)(shared_idx + BLOCK_SIZE));
//    std::stable_sort(ranger, ranger + BLOCK_SIZE, [&shared_idx](const int& a, const int& b) { return shared_idx[a] < shared_idx[b]; });
//    sort_placeholder(shared_idx, shared_idx + BLOCK_SIZE);
    std::stable_sort(shared, shared + BLOCK_SIZE, [](const struct s& a, const struct s& b) { return a.idx < b.idx; });

    // Also need to split off the row from the indices (use 64-bit instead?)
    int p = 0;
    /*
    float accum = shared_data[ranger[0]];
    for (int i=1; i < BLOCK_SIZE; i++)
    {
        if (shared_idx[ranger[i]] == shared_idx[ranger[i-1]])
            accum += shared_data[ranger[i]];
            continue;
        out_idx[p] = shared_idx[ranger[i-1]];
        out_vals[p] = accum;
        accum = 0.0;
        p++;
    }
    */
    float accum = shared[0].data;
    for (int i=1; i < BLOCK_SIZE; i++)
    {
        if (shared[i].idx == shared[i-1].idx)
            accum += shared[i].data;
            continue;
        out_idx[p] = shared[i-1].idx;
        out_vals[p] = accum;
        accum = 0.0;
        p++;
    }

    return p;
}


__global__ void test_block_sort(const int *d_keys, const float *d_vals)
{
    enum { nt = 128, vt = 16 };
    mgpu::kv_array_t<int, float, vt> unsorted;

    for (int i=0; i < vt; i++) {
        unsorted.keys[i] = d_keys[threadIdx.x * vt + i];
        unsorted.vals[i] = d_vals[threadIdx.x * vt + i];
        printf("tid: %d, vt: %d, unsorted: %d, %f\n", threadIdx.x, i, unsorted.keys[i], unsorted.vals[i]);
    }
    __syncthreads();

    mgpu::cta_sort_t<nt, vt, int, float> reducer;
    __shared__ mgpu::cta_sort_t<nt, vt, int, float>::storage_t reducer_storage;

    mgpu::kv_array_t<int, float, vt> sorted = reducer.block_sort(unsorted, threadIdx.x, nt * vt, mgpu::less_t<int>(), reducer_storage);

    for (int i=0; i < vt; i++) {
        printf("tid: %d, vt: %d, key = %d, val = %f\n", threadIdx.x, i, sorted.keys[i], sorted.vals[i]);
    }
}


__global__ void cuda_load_multi_block(const int *A_row_ptrs, const int *A_col_idx, const float *A_vals, 
                                const int *B_row_ptrs, const int *B_col_idx, const float *B_vals,
                                const const int *lb_data, int prev_block_ptr, int block_ptr) //, int *segments, int *partitions)
{
    int start_row = lb_data[prev_block_ptr];
    int end_row = lb_data[block_ptr];

    int Asz = A_row_ptrs[end_row + 1] - A_row_ptrs[start_row];

    __shared__ struct {
        int row_col;
        float v;
    } block_data[BLOCK_SIZE];

    // LBS load into shared memory

    // radix sort in shared memory

    // reduce
    
    // streaming store
}

struct CustomLess
{
    template <typename DataType>
    __device__ bool operator()(const DataType& lhs, const DataType& rhs) { return lhs < rhs; }
};

__global__ void cuda_load_block_coop(const int *AmRowPtrs, const int *AmColIdx, const float *AmCSRVals,
                                     const int *BmRowPtrs, const int *BmColIdx, const float *BmCSRVals,
                                     const int *lb_data, const int *lb_block_ptrs, const TSplit* lb_thread_splits, float *output)
{
    int block = blockIdx.x;
    int cur_block_ptr = lb_block_ptrs[block];
    int next_block_ptr = lb_block_ptrs[block+1];
    int end_row = lb_data[next_block_ptr];
    int start_row = lb_data[cur_block_ptr];

    /*
    int c = A.mRowPtrs[1] - A.mRowPtrs[0];
    std::cout << "c = " << c << std::endl;
    std::cout << B.mRowPtrs[0] << ", " << B.mRowPtrs[1] << std::endl;
//    for (int i=0; i < c; i++)
//    {
//        A.mRowPtrs[0]


    const std::vector<TSplit>& tsplits = lb_thread_splits[block];
    for (int i = 0; i < tsplits.size(); i++) {
        std::cout << "thread split " << i << " = " << tsplits[i].a_row << ", " << tsplits[i].bp << ", " << tsplits[i].b_col << std::endl;
    }
    */

    /*
    mgpu::cta_sort_t<NUM_THREADS, ITEMS_PER_THREAD, int, float> reducer;
    __shared__ mgpu::cta_sort_t<NUM_THREADS, ITEMS_PER_THREAD, int, float>::storage_t reducer_storage;
    mgpu::kv_array_t<int, float, ITEMS_PER_THREAD> unsorted;
    */

    typedef cub::BlockRadixSort<int, NUM_THREADS, ITEMS_PER_THREAD, float, 4, true> BlockRadixSort;
    typedef cub::BlockReduce<float, NUM_THREADS> BlockReduce;
    __shared__ union {
        typename BlockRadixSort::TempStorage block_radix_storage;
        typename BlockReduce::TempStorage block_reduce_storage;
        float all_vals[BLOCK_SIZE];
    } smem;
    int thread_keys[ITEMS_PER_THREAD];
    float thread_vals[ITEMS_PER_THREAD];

//    __shared__ float values[BLOCK_SIZE];
//    __shared__ int keys[BLOCK_SIZE];

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

//        vals[threadIdx.x * ITEMS_PER_THREAD + i] = BmCSRVals[split.b_col] * Acoeff;
//        keys[threadIdx.x * ITEMS_PER_THREAD + i] = ((splt.a_row - start_row) << 25) | BmColIdx[split.b_col];

//        unsorted.keys[i] = ((split.a_row - start_row) << 25) | BmColIdx[split.b_col];
//        unsorted.vals[i] = BmCSRVals[split.b_col] * Acoeff;

        thread_keys[i] = ((split.a_row - start_row) << 25) | BmColIdx[split.b_col];
//        thread_keys[i] = ((split.a_row - start_row) << 16) | BmColIdx[split.b_col];
        thread_vals[i] = BmCSRVals[split.b_col] * Acoeff;
        split.b_col++;
    }

    __syncthreads();

//    mgpu::kv_array_t<int, float, ITEMS_PER_THREAD> sorted = reducer.block_sort(unsorted, threadIdx.x, BLOCK_SIZE, mgpu::less_t<int>(), reducer_storage);
//    BlockMergeSort(block_merge_storage).Sort(thread_keys, CustomLess());
//    BlockMergeSort(block_merge_storage).Sort(thread_keys, thread_vals, CustomLess());
    BlockRadixSort(smem.block_radix_storage).Sort(thread_keys, thread_vals); //, 0, 24);

    /*
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
        smem.all_vals[threadIdx.x * ITEMS_PER_THREAD + i] = thread_vals[i];
    __syncthreads();

    if (threadIdx.x == 0) {
        float accum(0.0);
        for (int i = 0; i < BLOCK_SIZE; i++)
            accum += smem.all_vals[i];
        output[block] = accum;
    }
    */

    __syncthreads();

    // reduce to test
    float agg = BlockReduce(smem.block_reduce_storage).Sum(thread_vals);

    if (threadIdx.x == 0)
        output[block] = agg;

    /*
    if (threadIdx.x == 0)
    {
        float accum(0.0);
        for (int i=0; i < ITEMS_PER_THREAD; i++)
//            accum += sorted.vals[i];
            accum += thread_vals[i];
        output[block] = accum;
    }
    */
}



// Try again to pass whole DeviceMatrix structure
__global__ void cuda_load_block(const int *A_row_ptrs, const int *A_col_idx, const float *A_vals, 
                                const int *B_row_ptrs, const int *B_col_idx, const float *B_vals,
                                const const int *lb_data, int prev_block_ptr, int block_ptr) //, int *segments, int *partitions)
{
//    enum { nt = 128, vt = 7 };
    enum { nt = 128, vt = 17 };     // This has to multiply to BLOCK_SIZE?
    typedef mgpu::cta_load_balance_t<nt, vt> load_balance_t;
    typedef mgpu::cta_scan_t<nt, int> prefix_sum_scan_t;

    mgpu::cta_sort_t<nt, 16, int, float> reducer;
    __shared__ mgpu::cta_sort_t<nt, 16, int, float>::storage_t reducer_storage;

    mgpu::kv_array_t<int, float, 16> unsorted;

    // union ...
    __shared__ int segment_offsets[128];        // Again, handle > nt segment offsets

    __shared__ load_balance_t::storage_t lbs_storage;
    __shared__ prefix_sum_scan_t::storage_t ps_storage;


    int start_row = lb_data[prev_block_ptr];
    int end_row = lb_data[block_ptr];

    // don't necessarily need size in lb_data ...
    int start_sz = lb_data[prev_block_ptr + 1];
    int end_sz = lb_data[block_ptr + 1];

    if (start_row == end_row)
    {
        // assert start_sz == end_sz
        // build segments (end_offset - start_offset)

        // TODO: Handle case where start_sz > nt
        int segsize = 0;

        if (threadIdx.x < start_sz) {
            segsize = lb_data[block_ptr + threadIdx.x + 2] - lb_data[prev_block_ptr + threadIdx.x + 2];
//            printf("tid: %d, segsize: %d, start_sz: %d, %d - %d\n", threadIdx.x, segsize, start_sz, lb_data[block_ptr + threadIdx.x + 2], lb_data[prev_block_ptr + threadIdx.x + 2]);
        }
        __syncthreads();

        // exclusive scan to get segments
        auto sum_scan = prefix_sum_scan_t().scan(threadIdx.x, segsize, ps_storage);
//        printf("tid: %d, scan: %d, reduction: %d\n", threadIdx.x, sum_scan.scan, sum_scan.reduction);

        // everyone write their sum_scan.scan into shared memory for next LBS step
        if (threadIdx.x <= start_sz) {
            segment_offsets[threadIdx.x] = sum_scan.scan;
        }
        __syncthreads();

        // load B values into shared memory
//        printf("segment_offsets[%d] = %d\n", threadIdx.x, segment_offsets[threadIdx.x]);
        int partitions[] = { 0, segment_offsets[start_sz] };
        auto lbs = load_balance_t().load_balance(segment_offsets[start_sz], segment_offsets, start_sz+1, threadIdx.x, blockIdx.x, partitions, lbs_storage);

        // vt == 16 is always rank:-1 ...
        int Arow = A_row_ptrs[start_row];
//        for (int i=0; i < vt - 1; i++) {
        for (int i=0; i < 16; i++) {
//            printf("tid %d, vt: %d loading segment: %d, rank: %d\n", threadIdx.x, i, lbs.segments[i], lbs.ranks[i]);
            // grab A coeff from segments[i] index, multiply by B[segments[i]][ranks[i] + start[i]], store in B_data

            // grab B colidx from B[segments[i]][ranks[i] + start[i]], store in B_idx

            int Brow = A_col_idx[Arow + lbs.segments[i]];
            // float Acoeff = A_vals[Arow + lbs.segments[i]];
            int Brp = B_row_ptrs[Brow];
            int start_offset = lb_data[prev_block_ptr + 2 + lbs.segments[i]];
//            B_idx[threadIdx.x * 16 + i] = B_col_idx[Brp + lbs.ranks[i] + start_offset];
//            B_data[threadIdx.x * 16 + i] = B_vals[Brp + lbs.ranks[i] + start_offset]; // * Acoeff;
            unsorted.keys[i] = B_col_idx[Brp + lbs.ranks[i] + start_offset];
            unsorted.vals[i] = B_vals[Brp + lbs.ranks[i] + start_offset]; // * Acoeff;
            printf("tid %d, vt: %d, unsorted kv: %d = %f\n", threadIdx.x, i, unsorted.keys[i], unsorted.vals[i]);
        }
//        __syncthreads();
    }
    else
    {



    }

    // radix sort B_idx / B_data (CUB)

    // moderngpu only has a merge sort
    mgpu::kv_array_t<int, float, 16> sorted = reducer.block_sort(unsorted, threadIdx.x, BLOCK_SIZE, mgpu::less_t<int>(), reducer_storage);

    for (int i=0; i < 16; i++)
    {
        printf("tid: %d, vt: %d, sorted kv: %d = %f\n", threadIdx.x, i, sorted.keys[i], sorted.vals[i]);
//            printf("tid: %d, vt: %d, sorted kv: %d = %f\n", threadIdx.x, i, unsorted.keys[i], unsorted.vals[i]);
    }

    // compact

    // store
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

    for (int i=1; i < lb_block_ptrs.size(); i++) {
        if (lb_data[lb_block_ptrs[i]] == lb_data[lb_block_ptrs[i-1]]) {
            std::cout << "single row: " << i << std::endl;
            break;
        }
    }


//    std::vector<int> segments = { 0, 5, 7, 13, 3, 18, 9, 14, 25 };
    /*
    std::vector<int> segments = { 0, 5, 12, 25, 28, 46, 55, 69, 94 };
    std::vector<int> partitions = { 0, 94 };
    mgpu::mem_t<int> dsegments = mgpu::to_mem(segments, context);
    mgpu::mem_t<int> dpartitions = mgpu::to_mem(partitions, context);

    mgpu::mem_t<int> mp_data = load_balance_partitions(94, dsegments.data(), 8, 128, context);
    std::vector<int> mp_data_vec = mgpu::from_mem(mp_data);

    for (int i=0; i < mp_data_vec.size(); i++)
        std::cout << "mp_data_vec: " << mp_data_vec[i] << std::endl;
        */

    std::vector<TSplit> lb_flat_splits;
    for (int i = 0; i < lb_thread_splits.size()-1; i++)
    {
        for (int j = 0; j < 128; j++)
            lb_flat_splits.push_back(lb_thread_splits[i][j]);
    }
    mgpu::mem_t<int> d_lb_data = mgpu::to_mem(lb_data, context);
    mgpu::mem_t<int> d_lb_block_ptrs = mgpu::to_mem(lb_block_ptrs, context);
    mgpu::mem_t<TSplit> d_flat_tsplits = mgpu::to_mem(lb_flat_splits, context);

    std::vector<float> out_buffer;
    out_buffer.resize(lb_block_ptrs.size());
    mgpu::mem_t<float> d_output = mgpu::to_mem(out_buffer, context);

    /*
    cuda_load_block<<<1, 128>>>(dmA.raw.d_row_ptrs, dmA.raw.d_col_idx, dmA.raw.d_values,
                                dmB.raw.d_row_ptrs, dmB.raw.d_col_idx, dmB.raw.d_values,
                                d_lb_data.data(), lb_block_ptrs[455], lb_block_ptrs[456]); //, dsegments.data(), dpartitions.data());
                                */

    int numBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, cuda_load_block_coop, NUM_THREADS, 2048 * 4);
    std::cout << "max occupancy(?) = " << numBlocks << std::endl;

    auto start = std::chrono::system_clock::now();
    cuda_load_block_coop<<<lb_block_ptrs.size()-1, NUM_THREADS>>>(dmA.raw.d_row_ptrs, dmA.raw.d_col_idx, dmA.raw.d_values,
                                             dmB.raw.d_row_ptrs, dmB.raw.d_col_idx, dmB.raw.d_values,
                                             d_lb_data.data(), d_lb_block_ptrs.data(), d_flat_tsplits.data(), d_output.data());
    cudaDeviceSynchronize();
    auto finish = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(finish - start).count();
    std::cout << "elapsed: " << elapsed << std::endl;

    std::vector<float> read_back = mgpu::from_mem(d_output);
    std::cout << "Result 0: " << read_back[0] << std::endl;

    return 0;





    start = std::chrono::system_clock::now();
    int *out_idx = new int[4000000];
    float *out_vals = new float[4000000];
    int p =0;
    for (int i=1; i < lb_block_ptrs.size(); i++) {
        int count = load_multi_block(matA, matB, lb_data, lb_block_ptrs[i-1], lb_block_ptrs[i], out_idx + p, out_vals + p);
        p += count;
    }
    
    // need to do a fix-up pass and re-sort the output rows

    finish = std::chrono::system_clock::now();
    elapsed = std::chrono::duration<double, std::milli>(finish - start).count();
    std::cout << "elapsed: " << elapsed << std::endl;


    /*
    std::vector<int> keys(2048);
    std::vector<float> vals(2048);
    std::random_device rd;
    std::uniform_int_distribution<int> dist(0, 100);
    for (int i=0; i < 2048; i++) {
        keys[i] = dist(rd);
        vals[i] = i;
    }
    mgpu::mem_t<int> dkeys = mgpu::to_mem(keys, context);
    mgpu::mem_t<float> dvals = mgpu::to_mem(vals, context);
    test_block_sort<<<1, 128>>>(dkeys.data(), dvals.data());
    */

    cudaError_t res = cudaGetLastError();
    if (res != cudaSuccess) {
        std::cerr << "ERROR executing kernel: " << cudaGetErrorName(res) << " : " << cudaGetErrorString(res) << std::endl;
        exit(1);
    }
    else {
        std::cout << "cudaSuccess" << std::endl;
    }
    cudaDeviceSynchronize();

    /*
    mgpu::transform_lbs([=]MGPU_DEVICE(int i, int s, int r) {
        printf("i:%d, s:%d, r:%d\n", i, s, r);
    }, 94, dsegments.data(), 9, context);
    */
}
