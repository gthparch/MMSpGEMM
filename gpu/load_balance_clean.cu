#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/kernel_load_balance.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_reduce.hxx>
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

//constexpr int COLUMN_BITS   = 19;
//int COLUMN_BITS;
//constexpr int TOTAL_BITS    = 32;

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


// XXX: TODO: Fix using 32-bit ints for pointers in kernel
struct block_output_row
{
    uint64_t start;
    uint64_t key_in;
    uint64_t key_out;
    uint32_t size;
    float carry_out;
};


void CheckCuda(cudaError_t e)
{
    if (e != cudaSuccess) {
        std::cerr << "CUDA ERROR! " << cudaGetErrorName(e) << " : " << cudaGetErrorString(e) << std::endl;
        exit(1);
    }
}


__global__ void cuda_build_thread_splits(const int *lb_data, const int *lb_block_ptrs, const int *BmRowPtrs, const int *BmColIdx, TSplit *thread_splits, int nblocks)
{
    int block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block > nblocks)
        return;

    int cur_bp = lb_block_ptrs[block];
    int next_bp = lb_block_ptrs[block+1];
    int start_row = lb_data[cur_bp];
    int end_row = lb_data[next_bp];
    int copy_count(0);
    int ti(0);

    for (int row = start_row; row <= end_row; row++)
    {
        int coeff_start = BmRowPtrs[row];
        int coeff_end = BmRowPtrs[row+1];
        for (int bp = coeff_start; bp < coeff_end; bp++)
        {
            int brow = BmColIdx[bp]-1;
            int seg_start = BmRowPtrs[brow];
            int seg_end = BmRowPtrs[brow+1];
            if (row == start_row)
                seg_start += lb_data[cur_bp + 3 + (bp - coeff_start)];
            if (row == end_row)
                seg_end = BmRowPtrs[brow] + lb_data[next_bp + 3 + (bp - coeff_start)];

            int count = seg_end - seg_start;
            for (int i = 0; i < count; i++)
            {
                if (copy_count % ITEMS_PER_THREAD == 0) {
                    thread_splits[block * NUM_THREADS + ti] = {row, bp, seg_start + i};
                    ti++;
                }
                copy_count++;
            }
            /* doesn't work (yet)
            int next = ITEMS_PER_THREAD - (copy_count % ITEMS_PER_THREAD);
            count -= next;
            while (count > 0) {
                copy_count += next;
                thread_splits[block * NUM_THREADS + ti++] = {row, bp, seg_start + next};
                next = ITEMS_PER_THREAD - (copy_count % ITEMS_PER_THREAD);
                count -= next;
            }
            copy_count += (count + next);
            */
        }
    }
}


//template <int COLUMN_BITS=COLUMN_BITS, int TOTAL_BITS=TOTAL_BITS>
__global__ void cuda_load_block_coop(const int *AmRowPtrs, const int *AmColIdx, const float *AmCSRVals,
                                     const int *BmRowPtrs, const int *BmColIdx, const float *BmCSRVals,
                                     const int *lb_data, const int *lb_block_ptrs, const TSplit* lb_thread_splits, /* float *output_vals, int *output_keys, */
                                     uint32_t *out_keys, float *out_vals, int *atomic_p, block_output_row *out_meta, int *out_sizes, int column_bits, int total_bits)
{
    int block = blockIdx.x;
    int cur_block_ptr = lb_block_ptrs[block];
    int next_block_ptr = lb_block_ptrs[block+1];
    int end_row = lb_data[next_block_ptr];
    int start_row = lb_data[cur_block_ptr];

    typedef cub::BlockRadixSort<uint32_t, NUM_THREADS, ITEMS_PER_THREAD, float, 4, true> BlockRadixSort;
    __shared__ union {
        typename BlockRadixSort::TempStorage block_radix_storage;
    } smem;
    uint32_t thread_keys[ITEMS_PER_THREAD];
    float thread_vals[ITEMS_PER_THREAD];

    TSplit split = lb_thread_splits[block * NUM_THREADS + threadIdx.x];
    int row_end = AmRowPtrs[split.a_row + 1];
    int brow = AmColIdx[split.bp] - 1;
    int seg_end = BmRowPtrs[brow+1];

    int coeff_start = AmRowPtrs[split.a_row];
    if (split.a_row == end_row)
        seg_end = BmRowPtrs[brow] + lb_data[next_block_ptr + 3 + (split.bp - coeff_start)];
    float Acoeff = AmCSRVals[split.bp];
//    printf("split.a_row = %d, split.bp = %d, split.b_col = %d, col_idx = %d, seg_end = %d\n", split.a_row, split.bp, split.b_col, BmColIdx[split.b_col], seg_end);

    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        while (split.b_col >= seg_end)
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
            if (split.a_row == start_row) {
                int coeff_start = AmRowPtrs[split.a_row];
                split.b_col += lb_data[cur_block_ptr + 3 + (split.bp - coeff_start)];
            }
            if (split.a_row == end_row) {
                int coeff_start = AmRowPtrs[split.a_row];
                seg_end = BmRowPtrs[brow] + lb_data[next_block_ptr + 3 + (split.bp - coeff_start)];
            }
        }

        thread_keys[i] = ((split.a_row - start_row) << column_bits) | BmColIdx[split.b_col];
        thread_vals[i] = BmCSRVals[split.b_col] * Acoeff;

//        printf("tid = %d, thread_keys[%d] = %d, %d, %d, b_col = %d, vals = %f, %f, %f\n", threadIdx.x, i, split.a_row, BmColIdx[split.b_col], thread_keys[i], split.b_col, thread_vals[i], BmCSRVals[split.b_col], Acoeff);
        split.b_col++;
    }

    __syncthreads();

    BlockRadixSort(smem.block_radix_storage).Sort(thread_keys, thread_vals, 0, total_bits);

    __syncthreads();


    // debugging post-sort
    /*
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        output_vals[threadIdx.x * ITEMS_PER_THREAD + i] = thread_vals[i];
        output_keys[threadIdx.x * ITEMS_PER_THREAD + i] = thread_keys[i];
    }
    */



/************************************************
 * Reduction phase
 */
    #define FULL_MASK   0xffffffff
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    // XXX: Hard-coded 4 warps
    // Could combine these in the union below ...?
	__shared__ int delta_shared[4 + NUM_THREADS];	
    __shared__ float carry_out[NUM_THREADS];
    __shared__ uint32_t block_keys[4 + 1];       // reuse delta_shared?
    block_keys[4] = 0xffffffff;

    if (lane == 0)
        block_keys[warp] = thread_keys[0];

    int p = 0;
//    float pcarry_out = 0;
    for (int i=1; i < ITEMS_PER_THREAD; i++)
    {
        if (thread_keys[p] == thread_keys[i]) {
            thread_vals[p] += thread_vals[i];
//            pcarry_out = thread_vals[p];
        }
        else {
            p++;
            thread_keys[p] = thread_keys[i];
            thread_vals[p] = thread_vals[i];
//            pcarry_out = 0;
        }
    }
//    carry_out[threadIdx.x] = pcarry_out;
    carry_out[threadIdx.x] = thread_vals[p];

    __syncthreads();
    uint32_t neighbor_key = __shfl_down_sync(FULL_MASK, thread_keys[0], 1);
    if (lane == 31) {
        neighbor_key = block_keys[warp+1];
    }
//    __syncthreads();
    // what happens to the very last warp carry_out?
    bool rseg_end = (p > 0);
    if (thread_keys[ITEMS_PER_THREAD-1] != neighbor_key) {
        p++;
        rseg_end = true;
        carry_out[threadIdx.x] = 0.0;
    }
//    __syncthreads();

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
    if (threadIdx.x == 0) {
        shared_p = atomicAdd(atomic_p, total_p);
    }
    __syncthreads();

    if (threadIdx.x == NUM_THREADS-1) {
        out_meta[block].start = shared_p;
        if (lb_data[cur_block_ptr + 2])
            total_p--;      // may want to do this before block above so that we only write total_p - 1
        out_sizes[block] = total_p;
        out_meta[block].carry_out = carry_out[threadIdx.x];
    }

    start_p += shared_p;
    for (int i=0; i < p; i++) {
        out_keys[start_p + i] = thread_keys[i];
        out_vals[start_p + i] = thread_vals[i];
    }


/*
 * End reduction phase
 **************************************************************/
}


int main(int argc, char **argv)
{
    mgpu::standard_context_t context;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <mtx file> <row bits>" << std::endl;
        exit(1);
    }

    int row_bits = atoi(argv[2]);

    MatrixMarket matA { argv[1] };
    MatrixMarket& matB = matA;
    DeviceMatrix dmA(matA, context);
    DeviceMatrix& dmB = dmA;

    int column_bits = int(logf((float)matA.mRows) / logf(2.0)) + 1;
    std::cout << "column_bits = " << column_bits << std::endl;

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
    std::vector<int> out_buffer_keys;

    std::vector<uint32_t> out_keys, final_keys_rows, final_keys_cols;
    std::vector<int> segments;
    std::vector<float> out_vals, final_vals;
    std::vector<block_output_row> out_meta;
    std::vector<int> out_sizes;
    out_meta.resize(lb_block_ptrs.size());
    out_sizes.resize(lb_block_ptrs.size());
    out_keys.resize(200000000);
    out_vals.resize(200000000);
    final_keys_rows.resize(200000000);
    final_keys_cols.resize(200000000);
    final_vals.resize(200000000);
    segments.resize(lb_block_ptrs.size());
    std::vector<int> atomic_p;
    atomic_p.resize(2);

    out_buffer.resize(2048);
    out_buffer_keys.resize(2048);
    mgpu::mem_t<float> d_output = mgpu::to_mem(out_buffer, context);
    mgpu::mem_t<int> d_output_keys = mgpu::to_mem(out_buffer_keys, context);

    mgpu::mem_t<uint32_t> d_out_keys = mgpu::to_mem(out_keys, context);
    mgpu::mem_t<float> d_out_vals = mgpu::to_mem(out_vals, context);
    mgpu::mem_t<int> d_atomic_p = mgpu::to_mem(atomic_p, context);
    mgpu::mem_t<block_output_row> d_out_meta = mgpu::to_mem(out_meta, context);
    mgpu::mem_t<int> d_out_sizes = mgpu::to_mem(out_sizes, context);

    mgpu::mem_t<uint32_t> d_final_keys_rows = mgpu::to_mem(final_keys_rows, context);
    mgpu::mem_t<uint32_t> d_final_keys_cols = mgpu::to_mem(final_keys_cols, context);
    mgpu::mem_t<float> d_final_vals = mgpu::to_mem(final_vals, context);
    mgpu::mem_t<int> d_segments = mgpu::to_mem(segments, context);

    mgpu::mem_t<int> total_count(1, context);

    // row counts; need prefix sum to finalize
    uint32_t *d_final_row_counts_raw;
    cudaMalloc(&d_final_row_counts_raw, sizeof(uint32_t) * matA.mRows);

    int *d_unique_out, *d_counts_out, *d_num_runs_out;
    cudaMalloc(&d_unique_out, sizeof(int) * matA.mRows);
    cudaMalloc(&d_counts_out, sizeof(int) * matA.mRows);
    cudaMemset(d_counts_out, 0x0, sizeof(int) * matA.mRows);
    cudaMalloc(&d_num_runs_out, sizeof(int) * 1);

    int *d_counts_out2;
    int *h_row_ptrs = new int[matA.mRows];
    cudaMalloc(&d_counts_out2, matA.mRows * sizeof(int));

#if 0
    TSplit *junk, *h_junk;
    h_junk = new TSplit[lb_block_ptrs.size() * 128];
    cudaMalloc(&junk, sizeof(TSplit) * lb_block_ptrs.size() * 128);
    cudaDeviceSynchronize();
    int nblocks = ((lb_block_ptrs.size()-1) / 128) + 1;
    std::cout << "nblocks = " << nblocks << ", nthreads = 128, " << lb_block_ptrs.size()-1 << std::endl;
    auto start = std::chrono::system_clock::now();
//    cuda_build_thread_splits<<<lb_block_ptrs.size()-1, 1>>>(d_lb_data.data(), d_lb_block_ptrs.data(), dmB.raw.d_row_ptrs, dmB.raw.d_col_idx, junk);
    cuda_build_thread_splits<<<nblocks, 128>>>(d_lb_data.data(), d_lb_block_ptrs.data(), dmB.raw.d_row_ptrs, dmB.raw.d_col_idx, junk, lb_block_ptrs.size()-1);
    cudaDeviceSynchronize();
    auto finish = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(finish - start).count();
    std::cout << "elapsed: " << elapsed << std::endl;

    cudaMemcpy(h_junk, junk, sizeof(TSplit) * lb_block_ptrs.size() * 128, cudaMemcpyDeviceToHost);
    std::ofstream ofs("test_splits.bin");
    ofs.write((const char *)h_junk, sizeof(TSplit) * (lb_block_ptrs.size()-1) * 128);

#else


    /*
    int numBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, cuda_load_block_coop, NUM_THREADS, 2048 * 4);
    std::cout << "max occupancy(?) = " << numBlocks << std::endl;
    */

    auto start = std::chrono::system_clock::now();
    // Block multiplies
    int total_bits = column_bits + row_bits;         // this can be per block
    cuda_load_block_coop<<<lb_block_ptrs.size()-1, NUM_THREADS>>>(dmA.raw.d_row_ptrs, dmA.raw.d_col_idx, dmA.raw.d_values,
                                             dmB.raw.d_row_ptrs, dmB.raw.d_col_idx, dmB.raw.d_values,
                                             d_lb_data.data(), d_lb_block_ptrs.data(), d_flat_tsplits.data(), d_out_keys.data(),
                                             d_out_vals.data(), d_atomic_p.data(),
                                             d_out_meta.data(), d_out_sizes.data(), column_bits, total_bits);


    // (exclusive) prefix sum scan segments; needed for the shuffle gather and applying the carries
    mgpu::scan(d_out_sizes.data(), lb_block_ptrs.size()-1, d_segments.data(), mgpu::plus_t<int>(), total_count.data(), context);
    int total_count_h = mgpu::from_mem(total_count)[0];

    block_output_row *p = d_out_meta.data();
    float *d_out_vals_raw = d_out_vals.data();
    uint32_t *d_out_keys_raw = d_out_keys.data();
    float *d_final_vals_raw = d_final_vals.data();
    uint32_t *d_final_keys_rows_raw = d_final_keys_rows.data();
    uint32_t *d_final_keys_cols_raw = d_final_keys_cols.data();
    const int *d_lb_data_data = d_lb_data.data();
    const int *d_lb_block_ptrs_data = d_lb_block_ptrs.data();
    // Shuffle gather and split out rows and columns into two arrays
    // (TODO: See if computing the row indices in the kernel is faster and/or doing a symbolic pass for this first)
    mgpu::transform_lbs([=]MGPU_DEVICE(int index, int seg, int rank) {
        int pp = p[seg].start + rank;
        int start_row = d_lb_data_data[d_lb_block_ptrs_data[seg]];
        uint32_t key = d_out_keys_raw[pp];
        d_final_keys_cols_raw[index] = key & ((1 << column_bits) - 1);
        d_final_keys_rows_raw[index] = (key >> column_bits) + start_row; // & ((1<<25)-1);        // only need to keep the column
        d_final_vals_raw[index] = d_out_vals_raw[pp];
    }, total_count_h, d_segments.data(), lb_block_ptrs.size()-1, context);

    // Apply carries
    int *d_segments_data = d_segments.data();
    mgpu::transform([=]MGPU_DEVICE(int i) {
        int curblock = d_lb_block_ptrs_data[i];
        int overflow = d_lb_data_data[curblock + 2];
        if (overflow)
            atomicAdd(d_final_vals_raw + d_segments_data[i+1], p[i].carry_out);
    }, lb_block_ptrs.size()-1, context);

    void *temp_storage = NULL;
    size_t temp_storage_bytes;
    cub::DeviceRunLengthEncode::Encode(temp_storage, temp_storage_bytes, d_final_keys_rows_raw, (int*)0, (int*)0, (int*)0, total_count_h);
    std::cout << "temp_storage_bytes = " << temp_storage_bytes << std::endl;
    CheckCuda(cudaMalloc(&temp_storage, temp_storage_bytes));

    CheckCuda(cub::DeviceRunLengthEncode::Encode(temp_storage, temp_storage_bytes, d_final_keys_rows_raw, d_unique_out, d_counts_out, d_num_runs_out, total_count_h));

    mgpu::scan(d_counts_out, matA.mRows, d_counts_out2, context);

    CheckCuda(cudaDeviceSynchronize());
    auto finish = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(finish - start).count();
    std::cout << "elapsed: " << elapsed << std::endl;

    int h_num_runs_out;
    cudaMemcpy(&h_num_runs_out, d_num_runs_out, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "h_num_runs_out = " << h_num_runs_out << std::endl;
    std::cout << "Readback and writing row pointers ..." << std::endl;
    cudaMemcpy(h_row_ptrs, d_counts_out2, sizeof(int) * matA.mRows, cudaMemcpyDeviceToHost);
    std::ofstream ofs("row_ptrs.bin");
    ofs.write((const char*)h_row_ptrs, sizeof(int) * h_num_runs_out);

    std::cout << "Readback and writing column indices ..." << std::endl;
    std::vector<uint32_t> res_columns = mgpu::from_mem(d_final_keys_cols);
    ofs.close();
    ofs.open("col_indices.bin");
    ofs.write((const char*)res_columns.data(), sizeof(uint32_t) * total_count_h);

    std::cout << "Readback and writing values ..." << std::endl;
    std::vector<float> res_values = mgpu::from_mem(d_final_vals);
    ofs.close();
    ofs.open("values.bin");
    ofs.write((const char*)res_values.data(), sizeof(float) * total_count_h);
#endif


#if 0


    /*
    float *d_out_vals_raw = d_out_vals.data();
    uint32_t *d_out_keys_raw = d_out_keys.data();
    mgpu::transform_scan<int>([=]MGPU_DEVICE(int i) {
        if (p[i].key_out == p[i+1].key_in) {
            p[i].size--;
            d_out_vals_raw[p[i+1].start] += p[i].carry_out;
        }
        return p[i].size;
    }, lb_block_ptrs.size() - 1, d_segments.data(), mgpu::plus_t<int>(), total_count.data(), context);
    */



    // TODO: XXX: Dynamic memory allocation (also initial memory sizing)

    /*
    std::vector<int> res_segments = mgpu::from_mem(d_segments);
    std::ofstream ofs1("res_segments.bin");
    ofs1.write((const char *)res_segments.data(), sizeof(int) * (lb_block_ptrs.size() - 1));
    */
//    for (int i=0; i < 100; i++)
//        std::cout << "res_segments: " << res_segments[i] << std::endl;

    // interval scatter to move rows into place
    float *d_final_vals_raw = d_final_vals.data();
    uint32_t *d_final_keys_raw = d_final_keys.data();
//    uint32_t *d_final_row_counts_raw = d_final_row_counts.data();
    int num_segments = lb_block_ptrs.size() - 1;

    // can cache p[seg].start ...
    int total_count_h = mgpu::from_mem(total_count)[0];
    std::cout << "total_count_h = " << total_count_h << std::endl;

    std::vector<uint32_t> result_out_keys = mgpu::from_mem(d_out_keys);
    std::vector<int> res_segments = mgpu::from_mem(d_segments);
    std::vector<block_output_row> res_out_meta = mgpu::from_mem(d_out_meta);        // p array

    // Do the shuffle in CPU for debugging
    int outp = 0;
    uint32_t *output = new uint32_t[5000000];
    for (int i = 0; i < lb_block_ptrs.size()-1; i++)
    {
        const block_output_row& meta_row = res_out_meta[i];
        for (int j = meta_row.start; j < meta_row.start + meta_row.size; j++)
        {
            output[outp++] = (result_out_keys[j] >> 25) + lb_data[lb_block_ptrs[i]];
        }
    }

    std::cout << "CPU copied " << outp << " elements";
    std::ofstream ofs("xyz.bin");
    ofs.write((const char *)output, sizeof(uint32_t) * outp);






    const int *d_lb_data_data = d_lb_data.data();
    const int *d_lb_block_ptrs_data = d_lb_block_ptrs.data();
    mgpu::transform_lbs([=]MGPU_DEVICE(int index, int seg, int rank) {
//        d_final_keys_raw[index] = d_out_keys_raw[p[seg].start + rank]; // & ((1<<25)-1);        // only need to keep the column
        int start_row = d_lb_data_data[d_lb_block_ptrs_data[seg]];
        d_final_keys_raw[index] = (d_out_keys_raw[p[seg].start + rank] >> 25); // + start_row; // & ((1<<25)-1);        // only need to keep the column
        d_final_vals_raw[index] = d_out_vals_raw[p[seg].start + rank];
    }, total_count_h, d_segments.data(), num_segments, context);

    /*
    std::vector<float> xyz = mgpu::from_mem(d_final_vals);
    std::ofstream ofs("xyz.bin");
    ofs.write((const char *)xyz.data(), sizeof(float) * total_count_h);
    */


    /*
    for (int i=0; i < 500; i++)
        std::cout << xyz[i] << " ";
    std::cout << std::endl;
    */

    // scan count for row ptrs
//    cub::DeviceRunLengthEncode::Encode + prefix sum
    /*
    void *temp_storage = NULL;
    size_t temp_storage_bytes;
    cub::DeviceRunLengthEncode::Encode(temp_storage, temp_storage_bytes, d_final_keys_raw, (int*)0, (int*)0, (int*)0, total_count_h);
    std::cout << "temp_storage_bytes = " << temp_storage_bytes << std::endl;
    cudaMalloc(&temp_storage, temp_storage_bytes);
    */

    cub::DeviceRunLengthEncode::Encode(temp_storage, temp_storage_bytes, d_final_keys_raw, d_unique_out, d_counts_out, d_num_runs_out, total_count_h);
    int h_num_runs_out;
    cudaMemcpy(&h_num_runs_out, d_num_runs_out, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "h_num_runs_out = " << h_num_runs_out << std::endl;

    /* This gets the wrong answer!! Too many rows *
    auto compact = mgpu::transform_compact(total_count_h-1, context);
    int stream_count  =compact.upsweep([=]MGPU_DEVICE(int i) {
        return (d_final_keys_raw[i] != d_final_keys_raw[i+1]);
    });
    std::cout << "stream_count = " << stream_count << std::endl;
    mgpu::mem_t<int> row_ptrs(stream_count, context);
    int *row_ptrs_data = row_ptrs.data();
    compact.downsweep([=]MGPU_DEVICE(int dst, int src) {
        row_ptrs_data[dst] = src;
    });
    */


    cudaDeviceSynchronize();
    auto finish = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(finish - start).count();
    std::cout << "elapsed: " << elapsed << std::endl;

    /*
    std::vector<float> read_back = mgpu::from_mem(d_output);
    std::vector<int> read_back_keys = mgpu::from_mem(d_output_keys);
    std::cout << "size: " << read_back.size() << ", " << read_back_keys.size() << std::endl;
    for (int i = 0; i < 2048; i++) {
        std::cout << i << ": " << read_back_keys[i] << " = " << read_back[i] << std::endl;
    }
    */


    // print out the block output table
    std::vector<block_output_row> block_table = mgpu::from_mem(d_out_meta);
    out_keys = mgpu::from_mem(d_out_keys);
    out_vals = mgpu::from_mem(d_out_vals);

//    for (auto it = block_table.begin(); it != block_table.end(); ++it)
    unsigned total = 0;
    for (unsigned int i = 0; i < block_table.size()-1; i++)
    {
        std::cout << "Block " << i << ": " << block_table[i].start << "," << block_table[i].size << "," << block_table[i].carry_out << "," << block_table[i].key_in << "," << block_table[i].key_out << std::endl;
        /*
        if (block_table[i].size < 2)
            std::cout << "Small block." << std::endl;
        if (block_table[i].key_out == block_table[i+1].key_in) {
            block_table[i].size--;
            out_vals[block_table[i+1].start] += carry_out;
        }
        */
        total += block_table[i].size;
    }
    total += block_table[block_table.size()-1].size;
    std::cout << "total: " << total << std::endl;

    // output block bp
    /*
    int bp = 0;
    for (int i = block_table[bp].start; i < block_table[bp].start + block_table[bp].size; i++)
    {
        int row = (out_keys[i] >> 25) + lb_data[lb_block_ptrs[bp]];
        int col = out_keys[i] & ((1 << 25) - 1);
        std::cout << "Block 0: " << (i - block_table[bp].start) << " : (" << row << ", " << col << "): " << out_vals[i] << std::endl;
    }
    */

    return 0;

#endif
}
