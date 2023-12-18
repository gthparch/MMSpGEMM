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


// XXX: TODO: Fix using 32-bit ints for pointers in kernel
struct block_output_row
{
    uint64_t start;
    /*
    uint64_t key_in;
    uint64_t key_out;
    uint32_t size;
    */
    float carry_out;
};


void CheckCuda(cudaError_t e)
{
    if (e != cudaSuccess) {
        std::cerr << "CUDA ERROR! " << cudaGetErrorName(e) << " : " << cudaGetErrorString(e) << std::endl;
        exit(1);
    }
}


__device__ int lower_bound_search(int *arr, int q, int N)
{
    int mid;

    int low = 0;
    int high = N;

    while (low < high)
    {
        mid = low + (high - low) / 2;

        if (q <= arr[mid])
            high = mid;
        else
            low = mid + 1;
    }

    if (low < N && arr[low] < q)
        low++;

    return low;
}

__device__ int upper_bound_search(int *arr, int q, int N)
{
    int mid;

    int low = 0;
    int high = N;

    while (low < high)
    {
        mid = low + (high - low) / 2;

        if (q >= arr[mid])
            low = mid + 1;
        else
            high = mid;
    }

    if (low < N && arr[low] <= q)
        low++;

    return low;
}


__global__ void cuda_build_thread_splits_opt(const int *lb_data, const int *lb_block_ptrs, const int *ARowPtrs, const int *AColIdx,
                                             const float *AmCSRVals, const int *BmRowPtrs, const int *BmColIdx, const float *BmCSRVals,
                                             int nblocks, int column_bits, float *out_test)
{
    int block = blockIdx.x;

    int cur_bp = lb_block_ptrs[block];
    int next_bp = lb_block_ptrs[block+1];
    int start_row = lb_data[cur_bp];
    int end_row = lb_data[next_bp];

    // these two can be a union to save space, but check that the cumsum dependencies work
    __shared__ int row_counts[NUM_THREADS];
    __shared__ int cum_row_counts[NUM_THREADS];     // should be one more than NUM_THREADS since this is an exclusive scan
    __shared__ int shared_cum_sizes[BLOCK_SIZE];

    uint32_t thread_keys[ITEMS_PER_THREAD];
    float thread_vals[ITEMS_PER_THREAD];

    int segment_sizes[ITEMS_PER_THREAD];            // this can be higher; this limits the number of segments we allow

    int nrows = end_row - start_row + 1;

    // XXX: Can't handle blocks with more than NUM_THREADS rows ...
    if (threadIdx.x < nrows)
    {
        int row = start_row + threadIdx.x;
        row_counts[threadIdx.x] = ARowPtrs[row+1] - ARowPtrs[row];
    }
    __syncthreads();

    // This sucks but the overhead of load balancing this seems like it will outweight the benefit
    if (threadIdx.x == 0)
    {
        shared_cum_sizes[0] = 0;
        cum_row_counts[0] = 0;
        for (int i=0; i < nrows; i++)
            cum_row_counts[i+1] = row_counts[i] + cum_row_counts[i];
    }
    __syncthreads();

    // XXX: assert total count is less than some reasonable number
    // segments_per_thread < ITEMS_PER_THREAD

    int segments_per_thread = (cum_row_counts[nrows] / NUM_THREADS) + 1;
    int s = threadIdx.x * segments_per_thread;
    int last_ss = 0;
    int seg = 0;
    if (s < cum_row_counts[nrows])
    {
        // XXX: Slow linear scan; not expecting many rows (on the order of 5-10)
        int p = 1;
        // XXX: Not checking boundary condition since we checked s in the condition above
        while (cum_row_counts[p] <= s) p++;
        int row = p - 1;
        int rank = s - cum_row_counts[row];       // rank is now the column in that row that we're computing seg size for

        int col_start = ARowPtrs[row + start_row];
        for (seg = 0; seg < segments_per_thread; seg++)
        {
            // every row has to have one non-zero (else this should be a while)
            if (seg + s == cum_row_counts[row+1]) {
                row++;
                // check row is in range
                if (row == nrows)
                    break;

                // if start_row or end_row, take the sizes from splits
                col_start = ARowPtrs[row + start_row];
                rank = 0;
            }

            int brow = AColIdx[col_start + rank] - 1;
            int bstart = BmRowPtrs[brow];
            int bend = BmRowPtrs[brow+1];

            if (row == 0)
                bstart += lb_data[cur_bp + 3 + rank];
            if (row == nrows-1)
                bend = BmRowPtrs[brow] + lb_data[next_bp + 3 + rank];

            segment_sizes[seg] = bend - bstart + last_ss;
            last_ss = segment_sizes[seg];
            rank++;
        }
    }
    __syncthreads();

    // May want to set up temp storage explicitly to reuse it
    // Can we do an exclusive sum with only the first s < cumrow threads?
    cub::BlockScan<int, NUM_THREADS>().ExclusiveSum(last_ss, last_ss);

    // Now write out shared_cum_sizes
    if (s < cum_row_counts[nrows])
    {
        for (int i = 0; i < seg; i++) {
            shared_cum_sizes[s + i + 1] = segment_sizes[i] + last_ss;
        }
    }
    __syncthreads();

    /*
    if (shared_cum_sizes[cum_row_counts[nrows]] != BLOCK_SIZE)
    {
        printf("ERROR in block %d not equal BLOCK_SIZE\n", block);
    }
    */

    // each thread finds itself in the scan above and copies ITEMS_PER_THREAD starting from that location
    int copy_start = upper_bound_search(shared_cum_sizes, threadIdx.x * ITEMS_PER_THREAD, cum_row_counts[nrows]) - 1;
    int row_start = upper_bound_search(cum_row_counts, copy_start, nrows) - 1;
    int row_rank = copy_start - cum_row_counts[row_start];
    int col_rank = threadIdx.x * ITEMS_PER_THREAD - shared_cum_sizes[copy_start];

    int ap = ARowPtrs[row_start + start_row] + row_rank;
    int row_end = ARowPtrs[row_start + start_row + 1];
    float acoeff = AmCSRVals[ap];
    int acol = AColIdx[ap] - 1;         // also == brow
    int initial_ap = ARowPtrs[row_start + start_row];      // not adding row_rank
    int bp = BmRowPtrs[acol] + col_rank;
    if (row_start == 0)
        bp += lb_data[cur_bp + 3 + (ap - initial_ap)];

    for (int i=0; i < ITEMS_PER_THREAD; i++)
    {
        while (threadIdx.x * ITEMS_PER_THREAD + i == shared_cum_sizes[copy_start + 1])
        {
            // bump the next ap
            ap++;
            if (ap == row_end) {
                row_start++;
                row_end = ARowPtrs[row_start + start_row + 1];
            }
            acoeff = AmCSRVals[ap];
            bp = BmRowPtrs[AColIdx[ap] - 1];
            copy_start++;
            if (row_start == 0) {
                bp += lb_data[cur_bp + 3 + (ap - initial_ap)];
            }
        }
        thread_keys[i] = (row_start << column_bits) | BmColIdx[bp];
//        thread_vals[i] = BmCSRVals[bp] * acoeff;
        out_test[(blockIdx.x * NUM_THREADS + threadIdx.x) * ITEMS_PER_THREAD + i] = BmCSRVals[bp];
        bp++;
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
            /* Original */
            for (int i = 0; i < count; i++)
            {
                if (copy_count % ITEMS_PER_THREAD == 0) {
                    thread_splits[block * NUM_THREADS + ti] = {row, bp, seg_start + i};
                    ti++;
                }
                copy_count++;
            }

            // This is really no faster than the original code :-(
            // Also unverified
            /*
            if (copy_count % ITEMS_PER_THREAD == 0)
            {
                thread_splits[block * NUM_THREADS + ti] = {row, bp, seg_start};
                ti++;
            }
            int i = ITEMS_PER_THREAD - (copy_count % ITEMS_PER_THREAD);
            while (i < count) {
                thread_splits[block * NUM_THREADS + ti] = {row, bp, seg_start + i};
                ti++;
                i += ITEMS_PER_THREAD;
            }
            copy_count += count;
            */
        }
    }
}


//template <int COLUMN_BITS=COLUMN_BITS, int TOTAL_BITS=TOTAL_BITS>
__global__ void cuda_load_block_coop(const int *AmRowPtrs, const int *AmColIdx, const float *AmCSRVals,
                                     const int *BmRowPtrs, const int *BmColIdx, const float *BmCSRVals,
                                     const int *lb_data, const int *lb_block_ptrs, const TSplit* lb_thread_splits,
                                     uint32_t *out_keys, float *out_vals, int *atomic_p, block_output_row *out_meta, int *out_sizes,
                                     int column_bits, float *out_test)
{
    int block = blockIdx.x;
    int cur_block_ptr = lb_block_ptrs[block];
    int next_block_ptr = lb_block_ptrs[block+1];
    int end_row = lb_data[next_block_ptr];
    int start_row = lb_data[cur_block_ptr];
    int total_bits = column_bits + (32 - __clz(end_row-start_row));

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
//        thread_vals[i] = BmCSRVals[split.b_col] * Acoeff;
        out_test[(blockIdx.x * NUM_THREADS + threadIdx.x) * ITEMS_PER_THREAD + i] = BmCSRVals[split.b_col];

        split.b_col++;
    }

    return;

    __syncthreads();

    BlockRadixSort(smem.block_radix_storage).Sort(thread_keys, thread_vals, 0, total_bits);

    __syncthreads();


    // debugging post-sort
    /*
    bool debug = false;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if ((thread_keys[i] >> column_bits) + start_row == 358 && (thread_keys[i] & ((1 << column_bits) - 1)) == 534) {
            debug = true;
            printf("block %d (post-sort), tid: %d, i = %d, val = %f\n", block, threadIdx.x, i, thread_vals[i]);
        }
//        output_vals[threadIdx.x * ITEMS_PER_THREAD + i] = thread_vals[i];
//        output_keys[threadIdx.x * ITEMS_PER_THREAD + i] = thread_keys[i];
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
    for (int i=1; i < ITEMS_PER_THREAD; i++)
    {
        if (thread_keys[p] == thread_keys[i]) {
            thread_vals[p] += thread_vals[i];
        }
        else {
            p++;
            thread_keys[p] = thread_keys[i];
            thread_vals[p] = thread_vals[i];
        }
    }
    carry_out[threadIdx.x] = thread_vals[p];
    __syncthreads();        // XXX: Leave me

    uint32_t neighbor_key = __shfl_down_sync(FULL_MASK, thread_keys[0], 1);
    if (lane == 31) {
        neighbor_key = block_keys[warp+1];
    }
//    __syncthreads();      // Commenting seems OK
    // what happens to the very last warp carry_out?
    bool rseg_end = (p > 0);
    if (thread_keys[ITEMS_PER_THREAD-1] != neighbor_key) {
        p++;
        if (threadIdx.x < (NUM_THREADS-1)) {
            rseg_end = true;
            carry_out[threadIdx.x] = 0.0;
        }
    }
    __syncthreads();

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
//    if (p > 0 && threadIdx.x > 0)
    if (threadIdx.x > 0) {
        thread_vals[0] += carry_out[threadIdx.x - 1];
    }

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
        if (!rseg_end)
            out_meta[block].carry_out = thread_vals[0];
        else
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

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mtx file>" << std::endl;
        exit(1);
    }

    MatrixMarket matA { argv[1] };
    MatrixMarket& matB = matA;
    DeviceMatrix dmA(matA, context);
    DeviceMatrix& dmB = dmA;

    int column_bits = int(logf((float)matA.mRows) / logf(2.0)) + 1;
    std::cout << "column_bits = " << column_bits << std::endl;

    std::ifstream if_data("../lb_data.bin", std::ifstream::binary);
    std::ifstream if_block_ptr("../lb_block_ptrs.bin", std::ifstream::binary);
//    std::ifstream if_thread_splits("../lb_thread_splits.bin", std::ifstream::binary);

    if_data.seekg(0, if_data.end);
    if_block_ptr.seekg(0, if_block_ptr.end);
//    if_thread_splits.seekg(0, if_thread_splits.end);
    int lb_data_size = if_data.tellg();
    int lb_block_ptrs_size = if_block_ptr.tellg();
//    int lb_thread_splits_size = if_thread_splits.tellg();
    if_data.seekg(0, if_data.beg);
    if_block_ptr.seekg(0, if_block_ptr.beg);
//    if_thread_splits.seekg(0, if_thread_splits.beg);

    std::cout << "lb_data size = " << lb_data_size << std::endl;
    std::cout << "lb_block_ptr size = " << lb_block_ptrs_size << std::endl;
//    std::cout << "lb_thread_splits size = " << lb_thread_splits_size << std::endl;

    std::vector<int> lb_data, lb_block_ptrs;
//    std::vector<std::vector<TSplit>> lb_thread_splits;
    lb_data.resize(lb_data_size >> 2);
    lb_block_ptrs.resize(lb_block_ptrs_size >> 2);
    /*
    lb_thread_splits.resize(lb_block_ptrs_size >> 2);
    for (int i=0; i < lb_thread_splits.size()-1; i++) {
        lb_thread_splits[i].resize(128);
        if_thread_splits.read((char *)lb_thread_splits[i].data(), sizeof(TSplit) * 128);
    }
    */

    if_data.read((char *)lb_data.data(), lb_data_size);
    if_block_ptr.read((char *)lb_block_ptrs.data(), lb_block_ptrs_size);

    std::cout << "read lb_data = " << lb_data.size() << std::endl;
    std::cout << "read lb_block_ptrs = " << lb_block_ptrs.size() << std::endl;
//    std::cout << "read lb_thread_splits" << std::endl;

    /*
    for (int i=1; i < lb_block_ptrs.size(); i++) {
        if (lb_data[lb_block_ptrs[i]] == lb_data[lb_block_ptrs[i-1]]) {
            std::cout << "single row: " << i << std::endl;
            break;
        }
    }
    */

    /*
    std::vector<TSplit> lb_flat_splits;
    for (int i = 0; i < lb_thread_splits.size()-1; i++)
    {
        for (int j = 0; j < NUM_THREADS; j++)
            lb_flat_splits.push_back(lb_thread_splits[i][j]);
    }
    */
    mgpu::mem_t<int> d_lb_data = mgpu::to_mem(lb_data, context);
    mgpu::mem_t<int> d_lb_block_ptrs = mgpu::to_mem(lb_block_ptrs, context);
//    mgpu::mem_t<TSplit> d_flat_tsplits = mgpu::to_mem(lb_flat_splits, context);

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

    out_buffer.resize(BLOCK_SIZE);
    out_buffer_keys.resize(BLOCK_SIZE);
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

    cudaEvent_t start_evt, stop_tsplit_evt, stop_compute_evt;
    cudaEvent_t stop_shuffle_evt, stop_carries_evt, stop_rle_evt;
    float elapsed;

    CheckCuda(cudaEventCreate(&start_evt));
    CheckCuda(cudaEventCreate(&stop_tsplit_evt));
    CheckCuda(cudaEventCreate(&stop_compute_evt));
    CheckCuda(cudaEventCreate(&stop_shuffle_evt));
    CheckCuda(cudaEventCreate(&stop_carries_evt));
    CheckCuda(cudaEventCreate(&stop_rle_evt));

    mgpu::mem_t<float> out_test(lb_block_ptrs.size() * NUM_THREADS * ITEMS_PER_THREAD, context);
#if 1
    TSplit *d_flat_tsplits;
    cudaMalloc(&d_flat_tsplits, sizeof(TSplit) * lb_block_ptrs.size() * NUM_THREADS);
    cudaDeviceSynchronize();
    int nblocks = ((lb_block_ptrs.size()-1) / NUM_THREADS) + 1;
    std::cout << "nblocks = " << nblocks << ", nthreads = " << NUM_THREADS << ", " << lb_block_ptrs.size()-1 << std::endl;
    cudaEventRecord(start_evt, 0);
//    cuda_build_thread_splits<<<nblocks, NUM_THREADS>>>(d_lb_data.data(), d_lb_block_ptrs.data(), dmB.raw.d_row_ptrs, dmB.raw.d_col_idx, d_flat_tsplits, lb_block_ptrs.size()-1);
    cuda_build_thread_splits_opt<<<lb_block_ptrs.size()-1, NUM_THREADS>>>(d_lb_data.data(), d_lb_block_ptrs.data(), dmA.raw.d_row_ptrs, dmA.raw.d_col_idx, dmA.raw.d_values, dmB.raw.d_row_ptrs, dmB.raw.d_col_idx, dmB.raw.d_values, lb_block_ptrs.size()-1, column_bits, out_test.data());
    cudaEventRecord(stop_tsplit_evt, 0);
    cudaEventSynchronize(stop_tsplit_evt);
    cudaEventElapsedTime(&elapsed, start_evt, stop_tsplit_evt);
    std::cout << "Thread splits elapsed: " << elapsed << std::endl;

    std::vector<float> read_out_test = mgpu::from_mem(out_test);
    std::ofstream ofsx("out_test.bin");
    ofsx.write((const char *)read_out_test.data(), sizeof(float) * read_out_test.size());
    exit(0);
#else
    cudaEventRecord(start_evt, 0);
#endif


    /*
    int numBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, cuda_load_block_coop, NUM_THREADS, 2048 * 4);
    std::cout << "max occupancy(?) = " << numBlocks << std::endl;
    */

//    auto start = std::chrono::system_clock::now();
    // Block multiplies
    cuda_load_block_coop<<<lb_block_ptrs.size()-1, NUM_THREADS>>>(dmA.raw.d_row_ptrs, dmA.raw.d_col_idx, dmA.raw.d_values,
                                             dmB.raw.d_row_ptrs, dmB.raw.d_col_idx, dmB.raw.d_values,
                                             d_lb_data.data(), d_lb_block_ptrs.data(), d_flat_tsplits, d_out_keys.data(),
                                             d_out_vals.data(), d_atomic_p.data(),
                                             d_out_meta.data(), d_out_sizes.data(), column_bits, out_test.data());

    cudaEventRecord(stop_compute_evt, 0);
    cudaEventSynchronize(stop_compute_evt);
//    CheckCuda(cudaDeviceSynchronize());
//    auto finish = std::chrono::system_clock::now();
//    double elapsed = std::chrono::duration<double, std::milli>(finish - start).count();
    cudaEventElapsedTime(&elapsed, start_evt, stop_compute_evt);
    std::cout << "Compute elapsed: " << elapsed << std::endl;

    /*
    std::vector<float> read_out_test = mgpu::from_mem(out_test);
    std::ofstream ofsx("out_test.ref.bin");
    ofsx.write((const char *)read_out_test.data(), sizeof(float) * read_out_test.size());
    exit(0);
    */


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
        d_final_keys_rows_raw[index] = (key >> column_bits) + start_row;
        d_final_vals_raw[index] = d_out_vals_raw[pp];
    }, total_count_h, d_segments.data(), lb_block_ptrs.size()-1, context);

    cudaEventRecord(stop_shuffle_evt, 0);

    // Apply carries
    int *d_segments_data = d_segments.data();
    mgpu::transform([=]MGPU_DEVICE(int i) {
        int curblock = d_lb_block_ptrs_data[i];
        int overflow = d_lb_data_data[curblock + 2];
        if (overflow)
            atomicAdd(d_final_vals_raw + d_segments_data[i+1], p[i].carry_out);
    }, lb_block_ptrs.size()-2, context);

    cudaEventRecord(stop_carries_evt, 0);

    void *temp_storage = NULL;
    size_t temp_storage_bytes;
    cub::DeviceRunLengthEncode::Encode(temp_storage, temp_storage_bytes, d_final_keys_rows_raw, (int*)0, (int*)0, (int*)0, total_count_h);
    std::cout << "temp_storage_bytes = " << temp_storage_bytes << std::endl;
    CheckCuda(cudaMalloc(&temp_storage, temp_storage_bytes));

    CheckCuda(cub::DeviceRunLengthEncode::Encode(temp_storage, temp_storage_bytes, d_final_keys_rows_raw, d_unique_out, d_counts_out, d_num_runs_out, total_count_h));

    mgpu::scan(d_counts_out, matA.mRows, d_counts_out2, context);

    cudaEventRecord(stop_rle_evt, 0);
    cudaEventSynchronize(stop_rle_evt);

    cudaEventElapsedTime(&elapsed, start_evt, stop_shuffle_evt);
    std::cout << "Shuffle elapsed: " << elapsed << std::endl;
    cudaEventElapsedTime(&elapsed, start_evt, stop_carries_evt);
    std::cout << "Carries elapsed: " << elapsed << std::endl;
    cudaEventElapsedTime(&elapsed, start_evt, stop_rle_evt);
    std::cout << "RLE/Total elapsed: " << elapsed << std::endl;

    /*
    CheckCuda(cudaDeviceSynchronize());
    auto finish = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(finish - start).count();
    std::cout << "elapsed: " << elapsed << std::endl;
    */

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
}
