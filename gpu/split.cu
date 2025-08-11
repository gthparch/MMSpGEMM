#include <iostream>
#include <fstream>
#include "MatrixMarket.h"
#include "DeviceMatrix.h"

#include <moderngpu/kernel_segreduce.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_compact.hxx>

#include <cub/cub.cuh>


constexpr int BLOCK_SIZE = 2048;

constexpr int NUM_ITERS = 1;
constexpr int NUM_THREADS_SPLIT = 6;
constexpr int NUM_THREADS_SCAN_GEN = 128;

void CheckCuda(cudaError_t success)
{
    if (success != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(success) << std::endl;
        exit(1);
    }
}


constexpr int MAX_ELEMENT = 999999999;

struct node_t {
    int value;
    int list;
};


__device__ void tournament_tree_kth_largest(int **A, int *b, int m, int w_k, int k, node_t *T, int SIZE)
{
    int POT = (32 - __clz(SIZE)) - 1;
    for (int i=0; i < m; i++) {
        if (b[i] == 0) {
            T[SIZE + i].value = -MAX_ELEMENT;
            T[SIZE + i].list = -1;
            continue;
        }
        T[SIZE + i].value = A[i][b[i]-1] - 1;
        T[SIZE + i].list = i;
    }

    // First iteration, just propagate up the tree
    for (int l = POT-1; l >= 0; l--)
    {
        int l0 = 1 << l;
        int l1 = 1 << (l+1);
        for (int j = 0; j < l0; j++)
        {
            if (T[l1 + j*2].value > T[l1 + j*2 + 1].value)
                T[l0 + j] = T[l1 + j*2];
            else
                T[l0 + j] = T[l1 + j*2+1];
        }
    }

    int winner = T[1].list;
    b[winner] -= w_k;

    T[SIZE + winner].list = winner;
    if (b[winner] == 0)
        T[SIZE + winner].value = -MAX_ELEMENT;
    else
        T[SIZE + winner].value = A[winner][b[winner]-1] - 1;

    // Now just propagate the winning list
    for (int i = 0; i < k-1; i++)
    {
        int j = winner;
        for (int l = POT-1; l >= 0; l--)
        {
            int l0 = 1 << l;
            int l1 = 1 << (l+1);
            int j_floor = (j >> 1) * 2;
            if (T[l1 + j_floor].value > T[l1 + j_floor + 1].value)
                T[l0 + (j >> 1)] = T[l1 + j_floor];
            else
                T[l0 + (j >> 1)] = T[l1 + j_floor + 1];
            j = j >> 1;
        }
        winner = T[1].list;
        b[winner] -= w_k;
        
        T[SIZE + winner].list = winner;
        if (b[winner] == 0)
            T[SIZE + winner].value = -MAX_ELEMENT;
        else
            T[SIZE + winner].value = A[winner][b[winner]-1] - 1;
   }
}


__device__ void tournament_tree_kth_largest_reverse(int **A, int *alen, int *b, int m, int w_k, int k, node_t *T, int SIZE)
{
    int POT = (32 - __clz(SIZE)) - 1;
    for (int i=0; i < m; i++) {
        if (b[i] == alen[i]) {
            T[SIZE + i].value = -MAX_ELEMENT;
            T[SIZE + i].list = -1;
            continue;
        }
        T[SIZE + i].value = -A[i][b[i]] + 1;
        T[SIZE + i].list = i;
    }

    for (int i=m; i < SIZE; i++) {
        T[SIZE + i].value = -MAX_ELEMENT;
        T[SIZE + i].list = -1;
    }

    // First iteration, just propagate up the tree
    for (int l = POT-1; l >= 0; l--)
    {
        int l0 = 1 << l;
        int l1 = 1 << (l+1);
        for (int j = 0; j < l0; j++)
        {
            if (T[l1 + j*2].value > T[l1 + j*2 + 1].value)
                T[l0 + j] = T[l1 + j*2];
            else
                T[l0 + j] = T[l1 + j*2+1];
        }
    }

    int winner = T[1].list;
    b[winner] += w_k;

    T[SIZE + winner].list = winner;
    if (b[winner] >= alen[winner])
        T[SIZE + winner].value = -MAX_ELEMENT;
    else
        T[SIZE + winner].value = -A[winner][b[winner]] + 1;

    // Now just propagate the winning list
    for (int i = 0; i < k-1; i++)
    {
        int j = winner;
        for (int l = POT-1; l >= 0; l--)
        {
            int l0 = 1 << l;
            int l1 = 1 << (l+1);
            int j_floor = (j >> 1) * 2;
            if (T[l1 + j_floor].value > T[l1 + j_floor + 1].value)
                T[l0 + (j >> 1)] = T[l1 + j_floor];
            else
                T[l0 + (j >> 1)] = T[l1 + j_floor + 1];
            j = j >> 1;
        }
        winner = T[1].list;
        b[winner] += w_k;

        T[SIZE + winner].list = winner;
        if (b[winner] >= alen[winner])
            T[SIZE + winner].value = -MAX_ELEMENT;
        else
            T[SIZE + winner].value = -A[winner][b[winner]] + 1;
   }
}


__device__ void tournament_tree_kth_smallest(int **A, int *alen, int *b, int m, int w_k, int k, node_t *T, int SIZE)
{
    int POT = (32 - __clz(SIZE)) - 1;
    for (int i=0; i < m; i++) {
        if (b[i] + w_k > alen[i]) {
            T[SIZE + i].value = MAX_ELEMENT;
            T[SIZE + i].list = -1;
            continue;
        }
        T[SIZE + i].value = A[i][b[i] + w_k - 1] - 1;
        T[SIZE + i].list = i;
    }

    // First iteration, just propagate up the tree
    for (int l = POT-1; l >= 0; l--)
    {
        int l0 = (1 << l);
        int l1 = (1 << (l+1));
        for (int j = 0; j < (1 << l); j++)
        {
            if (T[l1 + j*2].value < T[l1 + j*2 + 1].value)
                T[l0 + j] = T[l1 + j*2];
            else
                T[l0 + j] = T[l1 + j*2+1];
        }
    }

    int winner = T[1].list;
    b[winner] += w_k;

    T[SIZE + winner].list = winner;
    if (b[winner] + w_k > alen[winner])
        T[SIZE + winner].value = MAX_ELEMENT;
    else
        T[SIZE + winner].value = A[winner][b[winner]+w_k-1] - 1;

    // Now just propagate the winning list
    for (int i = 0; i < k-1; i++)
    {
        int j = winner;
        for (int l = POT-1; l >= 0; l--)
        {
            int l0 = 1 << l;
            int l1 = 1 << (l+1);
            int j_floor = (j >> 1) * 2;
            if (T[l1 + j_floor].value < T[l1 + j_floor + 1].value)
                T[l0 + (j >> 1)] = T[l1 + j_floor];
            else
                T[l0 + (j >> 1)] = T[l1 + j_floor + 1];
            j = j >> 1;
        }
        winner = T[1].list;
        b[winner] += w_k;

        T[SIZE + winner].list = winner;
        if (b[winner] + w_k > alen[winner])
            T[SIZE + winner].value = MAX_ELEMENT;
        else
            T[SIZE + winner].value = A[winner][b[winner]+w_k-1] - 1;
   }
}


__device__ void tournament_tree_kth_smallest_reverse(int **A, int *alen, int *b, int m, int w_k, int k, node_t *T, int SIZE)
{
    int POT = (32 - __clz(SIZE)) - 1;
    for (int i=0; i < m; i++) {
        if (b[i] - w_k < 0) {
            T[SIZE + i].value = MAX_ELEMENT;
            T[SIZE + i].list = -1;
            continue;
        }
        T[SIZE + i].value = -A[i][b[i] - w_k] + 1;
        T[SIZE + i].list = i;
    }

    for (int i=m; i < SIZE; i++) {
        T[SIZE + i].value = MAX_ELEMENT;
        T[SIZE + i].list = -1;
    }

    // First iteration, just propagate up the tree
    for (int l = POT-1; l >= 0; l--)
    {
        int l0 = (1 << l);
        int l1 = (1 << (l+1));
        for (int j = 0; j < (1 << l); j++)
        {
            if (T[l1 + j*2].value < T[l1 + j*2 + 1].value)
                T[l0 + j] = T[l1 + j*2];
            else
                T[l0 + j] = T[l1 + j*2+1];
        }
    }

    int winner = T[1].list;
    b[winner] -= w_k;

    T[SIZE + winner].list = winner;
    if (b[winner] - w_k < 0)
        T[SIZE + winner].value = MAX_ELEMENT;
    else
        T[SIZE + winner].value = -A[winner][b[winner]-w_k] + 1;

    // Now just propagate the winning list
    for (int i = 0; i < k-1; i++)
    {
        int j = winner;
        for (int l = POT-1; l >= 0; l--)
        {
            int l0 = 1 << l;
            int l1 = 1 << (l+1);
            int j_floor = (j >> 1) * 2;
            if (T[l1 + j_floor].value < T[l1 + j_floor + 1].value)
                T[l0 + (j >> 1)] = T[l1 + j_floor];
            else
                T[l0 + (j >> 1)] = T[l1 + j_floor + 1];
            j = j >> 1;
        }
        winner = T[1].list;
        b[winner] -= w_k;

        T[SIZE + winner].list = winner;
        if (b[winner] - w_k < 0)
            T[SIZE + winner].value = MAX_ELEMENT;
        else
            T[SIZE + winner].value = -A[winner][b[winner]-w_k] + 1;
   }
}


// +/- 1 is because we are zero-based while the Python reference code is 1-based using scipy
__device__ inline int compute_lmax(int **A, int *b, int blen)
{
    int lmax = -MAX_ELEMENT;
    for (int i=0; i < blen; i++) {
        if (b[i] > 0 && (A[i][b[i]-1]-1) > lmax)
            lmax = A[i][b[i]-1] - 1;
    }
    return lmax;
}

__device__ inline int compute_lmax_reverse(int **A, int *alen, int *b, int blen)
{
    int lmax = -MAX_ELEMENT;
    for (int i=0; i < blen; i++) {
        if (b[i] < alen[i] && (-A[i][b[i]]+1) > lmax)
            lmax = -A[i][b[i]]+1;
    }
    return lmax;
}

__device__ inline bool compute_carry(int **A, int *alen, int *b, int blen, int lmax)
{
    for (int i=0; i < blen; i++)
    {
        if (b[i] < alen[i] && A[i][b[i]]-1 == lmax)
            return true;
    }
    return false;
}

__device__ inline bool compute_carry_reverse(int **A, int *alen, int *b, int blen, int lmax)
{
    for (int i=0; i < blen; i++)
    {
        if (b[i] > 0 && -A[i][b[i]-1]+1 == lmax)
            return true;
    }
    return false;
}


__device__ bool row_splitter(int **A, int *alen, int *b, int m, int p, node_t *T, int MAXSIZE)
{
    // assert m < MAXSIZE
    int n_max = -1;
    for (int i=0; i < m; i++) {
        b[i] = 0;
        n_max = (alen[i] > n_max) ? alen[i] : n_max;
    }

    if (p == 0)
        return false;

    // Do this fill in once instead of every call to TT
    // kth_smallest
    for (int i = 0; i < MAXSIZE; i++)
    {
        T[MAXSIZE + i].value = MAX_ELEMENT;
        T[MAXSIZE + i].list = -1;
    }

    // kth_largest
    for (int i = 0; i < MAXSIZE; i++)
    {
        T[MAXSIZE*3 + i].value = -MAX_ELEMENT;
        T[MAXSIZE*3 + i].list = -1;
    }

    // Handle short splits
    if (p < m) {
        tournament_tree_kth_smallest(A, alen, b, m, 1, p, T, MAXSIZE);
        int lmax = compute_lmax(A, b, m);
        return compute_carry(A, alen, b, m, lmax);
    }

    int r = ceilf(logf((float)p / m) / logf(2.0));
    int two_r = 1 << r;
    int alpha = n_max / two_r;              // implicit floor
    int n = two_r * (alpha + 1) - 1;
    int k = ceilf((float)p / n * alpha);

    // Initial partition for the recursion
    tournament_tree_kth_smallest(A, alen, b, m, two_r, k, T, MAXSIZE);
    int lmax = compute_lmax(A, b, m);

    if (lmax == MAX_ELEMENT)
    {
        // XXX: Need to handle this case
        return;
    }

    // r iterative steps
    for (int k=0; k < r; k++)
    {
        int Lsize = 0;
        int w_k = 1 << (r - k - 1);
        int target_size = ceilf((float)p * (n / w_k) / n);

        for (int i=0; i < m; i++)
            Lsize += b[i] / w_k;

        for (int i=0; i < m; i++)
        {
            if (b[i] + w_k > alen[i])
                continue;
            int undecided = A[i][b[i] + w_k - 1] - 1;
            if (undecided < lmax) {
                b[i] += w_k;
                Lsize++;
            }
        }
        if (Lsize > target_size) {
            tournament_tree_kth_largest(A, b, m, w_k, Lsize - target_size, T + (MAXSIZE*2), MAXSIZE);
        }
        if (Lsize < target_size) {
            tournament_tree_kth_smallest(A, alen, b, m, w_k, target_size - Lsize, T, MAXSIZE);
        }

        lmax = compute_lmax(A, b, m);
    }

    return compute_carry(A, alen, b, m, lmax);
}


__device__ bool row_splitter_reverse(int **A, int *alen, int *b, int m, int p, node_t *T, int MAXSIZE)
{
    int n_max = -1;
    for (int i=0; i < m; i++) {
        b[i] = alen[i];
        n_max = (alen[i] > n_max) ? alen[i] : n_max;
    }

    if (p == 0)
        return false;

    // Handle short splits
    if (p < m) {
        tournament_tree_kth_smallest_reverse(A, alen, b, m, 1, p, T, MAXSIZE);
        int lmax = compute_lmax_reverse(A, alen, b, m);
        return compute_carry_reverse(A, alen, b, m, lmax);
    }

    int r = ceilf(logf((float)p / m) / logf(2.0));
    int two_r = 1 << r;
    int alpha = n_max / two_r;              // implicit floor
    int n = two_r * (alpha + 1) - 1;
    int k = ceilf((float)p / n * alpha);
    
    // Initial partition for the recursion
    tournament_tree_kth_smallest_reverse(A, alen, b, m, two_r, k, T, MAXSIZE);
    int lmax = compute_lmax_reverse(A, alen, b, m);

    // r iterative steps
    for (int k=0; k < r; k++)
    {
        int Lsize = 0;
        int w_k = 1 << (r - k - 1);
        int target_size = ceilf((float)p * (n / w_k) / n);

        for (int i=0; i < m; i++)
            Lsize += (alen[i] - b[i]) / w_k;

        for (int i=0; i < m; i++)
        {
            if (b[i] - w_k < 0)
                continue;
            int undecided = -A[i][b[i] - w_k] + 1;
            if (undecided < lmax) {
                b[i] -= w_k;
                Lsize++;
            }
        }
        if (Lsize > target_size) {
            tournament_tree_kth_largest_reverse(A, alen, b, m, w_k, Lsize - target_size, T + (MAXSIZE * 2), MAXSIZE);
        }
        if (Lsize < target_size) {
            tournament_tree_kth_smallest_reverse(A, alen, b, m, w_k, target_size - Lsize, T, MAXSIZE);
        }

        lmax = compute_lmax_reverse(A, alen, b, m);
    }

    return compute_carry_reverse(A, alen, b, m, lmax);
}


struct block_data_t {
    int row;
    int p;
    bool reverse;
};

__global__ void split_matrix_fwd(int *row_ptrs, int *col_idx, int *Brow_ptrs, int *Bcol_idx, int *base, bool *carry_out, block_data_t *splits, int *out_ptrs, int *work_ptrs, int *workspace, int nsplits, int *indices)
{
    int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (threadId >= nsplits)
        return;

    threadId = indices[threadId];
    int row = splits[threadId].row;
    int m = row_ptrs[row+1] - row_ptrs[row];
    int npot = 1 << (32 - __clz(m-1));
    int p = splits[threadId].p;
    int *b = base + out_ptrs[threadId];

    uintptr_t Aaddr = (uintptr_t)(workspace + work_ptrs[threadId]);
    if (Aaddr % 8 != 0)
        Aaddr += 8 - (Aaddr % 8);
    int **A = (int **)Aaddr;
    Aaddr += m * 8;
    int *alen = (int *)Aaddr;
    Aaddr += m * 4;
    node_t *T = (node_t *)Aaddr;

    // assert m < MAXSIZE or handle exception
    for (int i=0; i < m; i++) {
        int brow = col_idx[row_ptrs[row] + i] - 1;
        A[i] = Bcol_idx + Brow_ptrs[brow];
        alen[i] = Brow_ptrs[brow+1] - Brow_ptrs[brow];
    }

    carry_out[threadId] = row_splitter(A, alen, b, m, p, T, npot);
}


__global__ void split_matrix_reverse(int *row_ptrs, int *col_idx, int *Brow_ptrs, int *Bcol_idx, int *base, bool *carry_out, block_data_t *splits, int *out_ptrs, int *work_ptrs, int *workspace, int nsplits, int *indices)
{
    int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (threadId >= nsplits)
        return;

    threadId = indices[threadId];
    int row = splits[threadId].row;
    int m = row_ptrs[row+1] - row_ptrs[row];
    int npot = 1 << (32 - __clz(m-1));
    int p = splits[threadId].p;
    int *b = base + out_ptrs[threadId];


    uintptr_t Aaddr = (uintptr_t)(workspace + work_ptrs[threadId]);
    if (Aaddr % 8 != 0)
        Aaddr += 8 - (Aaddr % 8);
    int **A = (int **)Aaddr;
    Aaddr += m * 8;
    int *alen = (int *)Aaddr;
    Aaddr += m * 4;
    node_t *T = (node_t *)Aaddr;

    // assert m < MAXSIZE or handle exception
    for (int i=0; i < m; i++) {
        int brow = col_idx[row_ptrs[row] + i] - 1;
        A[i] = Bcol_idx + Brow_ptrs[brow];
        alen[i] = Brow_ptrs[brow+1] - Brow_ptrs[brow];
    }

    carry_out[threadId] = row_splitter_reverse(A, alen, b, m, p, T, npot);
}


constexpr int ROWS_PER_THREAD = 14;
__global__ void scan_gen_blocks(int *cum_row_sizes, int *row_sizes, block_data_t *out, int nblocks, int nrows,
                                int *fwd_indices, int *reverse_indices, unsigned int *block_indices_cnt)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int start = tid * ROWS_PER_THREAD;
    if (start > nrows)
        return;

    int last_rsize = cum_row_sizes[start];
    int last_block = last_rsize / BLOCK_SIZE;
    for (int i=start+1; i <= start + ROWS_PER_THREAD; i++) {
        if (i > nrows)
            return;
        int row_size = cum_row_sizes[i];
        int block = row_size / BLOCK_SIZE;
        while (block > last_block) {
            out[last_block].row = i;
            out[last_block].p = ((last_block+1) * BLOCK_SIZE) - last_rsize;
            if (out[last_block].p / (float)row_sizes[i] > 0.5) {
                out[last_block].p = row_sizes[i] - out[last_block].p;
                out[last_block].reverse = true;
                reverse_indices[atomicAdd(block_indices_cnt + 1, 1)] = last_block;
            }
            else {
                out[last_block].reverse = false;
                fwd_indices[atomicAdd(block_indices_cnt, 1)] = last_block;
            }
            last_block++;
        }
        last_rsize = row_size;
    }
}


__global__ void compute_partial_row_sizes(int *Arowptr, int *Acolidx, int *Browlens, int *out_sizes, int nrows)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid > nrows)
        return;

    int size = 0;
    // tid is the row we're computing partial sizes for
    for (int i=Arowptr[tid]; i < Arowptr[tid+1]; i++)
    {
        int brow = Acolidx[i] - 1;
        size += Browlens[brow];
    }

    out_sizes[tid] = size;
}


struct SelectReverse
{
    __host__ __device__ __forceinline__
    SelectReverse() {}

    __host__ __device__ __forceinline__
    bool operator()(const block_data_t& a) const {
        return a.reverse;
    }
};
struct SelectForward
{
    __host__ __device__ __forceinline__
    SelectForward() {}

    __host__ __device__ __forceinline__
    bool operator()(const block_data_t& a) const {
        return !a.reverse;
    }
};


int main(int argc, char **argv)
{
    mgpu::standard_context_t context;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <mtx file> <mtx file>" << std::endl;
        exit(1);
    }

    // Initial setup stuff
    MatrixMarket matA { argv[1] };
    MatrixMarket matB { argv[2] };
    DeviceMatrix dmA(matA, context);
    DeviceMatrix dmB(matB, context);

    float time;
    cudaEvent_t start, finish_psizes, finish_gen_splits, finish_fwd, stop;
    CheckCuda(cudaEventCreate(&start));
    CheckCuda(cudaEventCreate(&finish_psizes));
    CheckCuda(cudaEventCreate(&finish_gen_splits));
    CheckCuda(cudaEventCreate(&finish_fwd));
    CheckCuda(cudaEventCreate(&stop));

    mgpu::mem_t<int> d_row_sizes(matA.mRows, context);
    mgpu::mem_t<int> d_cumrow_sizes(matA.mRows, context);
    mgpu::mem_t<int> d_total_partials(1, context);

    int scan_blocks = (matA.mRows / (ROWS_PER_THREAD * NUM_THREADS_SCAN_GEN)) + 1;
    mgpu::mem_t<int> d_out_final(1, context);
    mgpu::mem_t<int> d_work_final(1, context);

    mgpu::mem_t<int> Browlens(matA.mRows, context);     // should be matB

    cudaDeviceSynchronize();

    // Find the rows and where to split them (on GPU)
    cudaEventRecord(start, 0);
    int *Acolidx = dmA.raw.d_col_idx;
    int *Arowptrs = dmA.raw.d_row_ptrs;
    int *Browptrs = dmB.raw.d_row_ptrs;
    int *d_Browlens = Browlens.data(); 
    mgpu::transform([=]MGPU_DEVICE(int index)
        {
            d_Browlens[index] = Browptrs[index+1] - Browptrs[index];
        }, matA.mRows, context);        // should be matB.mRows

    int n_partials_blocks = (matA.mRows / 128) + 1;
    std::cout << "n_partials_blocks = " << n_partials_blocks << std::endl;
    compute_partial_row_sizes<<<n_partials_blocks, 128>>>(Arowptrs, Acolidx, d_Browlens, d_row_sizes.data(), matA.mRows);

    mgpu::scan<mgpu::scan_type_inc>(d_row_sizes.data(), matA.mRows, d_cumrow_sizes.data(), mgpu::plus_t<int>(),
                                    d_total_partials.data(), context);
    cudaEventRecord(finish_psizes, 0);

    int total_partials = mgpu::from_mem(d_total_partials)[0];
    int total_blocks = total_partials / BLOCK_SIZE;
    std::cout << "Num blocks = " << total_blocks << std::endl;

    // have scan_gen_blocks split the indices into forward and reverse arrays using atomics
    mgpu::mem_t<int> fwd_block_indices(total_blocks, context);
    mgpu::mem_t<int> reverse_block_indices(total_blocks, context);
    std::vector<unsigned int> init_block_indices_cnt = { 0, 0 };
    mgpu::mem_t<unsigned int> block_indices_cnt = mgpu::to_mem(init_block_indices_cnt, context);

    mgpu::mem_t<bool> d_carry_out(total_blocks+1, context);
    mgpu::mem_t<int> d_out_ptrs(total_blocks, context);
    mgpu::mem_t<int> d_work_ptrs(total_blocks, context);
    mgpu::mem_t<block_data_t> block_data(total_blocks, context);
    cudaMemset(block_data.data(), 0, sizeof(block_data_t) * total_blocks);
    std::cout << "scan_blocks = " << scan_blocks << ", total_blocks = " << total_blocks << std::endl;
    scan_gen_blocks<<<scan_blocks, NUM_THREADS_SCAN_GEN>>>(d_cumrow_sizes.data(), d_row_sizes.data(), block_data.data(), total_blocks, matA.mRows,
                                                           fwd_block_indices.data(), reverse_block_indices.data(),
                                                            block_indices_cnt.data());

    // prefix sum to get block 'out' value
    block_data_t *d_block_data = block_data.data();
    auto out_scan = [=]MGPU_DEVICE(int index)
        {
            int r = d_block_data[index].row;
            int m = Arowptrs[r + 1] - Arowptrs[r];
            return m;
        };
    mgpu::transform_scan<int>(out_scan, total_blocks, d_out_ptrs.data(), mgpu::plus_t<int>(), d_out_final.data(), context);

    // duplication with kernel above, but hopefully this isn't too slow
    mgpu::transform_scan<int>([=]MGPU_DEVICE(int index) {
        int r = d_block_data[index].row;
        int m = Arowptrs[r + 1] - Arowptrs[r];
        // XXX: Make sure m >= 1
        int npot = 1 << (32 - __clz(m - 1));
        // Fix this mess, A is a pointer so 2 words, alen is an array of words, so another m words + 16 for alignment
        // 2 * npot for node_t but node_ts are 8 bytes
        return 3 * m + 8 * npot + 16;
    }, total_blocks, d_work_ptrs.data(), mgpu::plus_t<int>(), d_work_final.data(), context);

    // Allocate working memory for the variable sizes
    int total_work_size = mgpu::from_mem(d_work_final)[0];
    std::cout << "Working memory size = " << total_work_size * sizeof(int) << std::endl;
    mgpu::mem_t<int> d_work(total_work_size * 2, context);

    cudaEventRecord(finish_gen_splits, 0);

    std::vector<unsigned int> h_block_counters = mgpu::from_mem(block_indices_cnt);
    std::cout << "Num forward blocks: " << h_block_counters[0] << std::endl;
    std::cout << "Num reverse blocks: " << h_block_counters[1] << std::endl;

    int total_split_size = mgpu::from_mem(d_out_final)[0];
    mgpu::mem_t<int> d_output(total_split_size, context);

    // total_blocks is number of split blocks, while n_blocks is CUDA blocks to compute that many splits
    int n_blocks = (h_block_counters[0] / NUM_THREADS_SPLIT) + 1;
    std::cout << "Using " << n_blocks << " CUDA blocks for forward splits." << std::endl;
    split_matrix_fwd<<<n_blocks, NUM_THREADS_SPLIT>>>(dmA.raw.d_row_ptrs, dmA.raw.d_col_idx, dmB.raw.d_row_ptrs, dmB.raw.d_col_idx, d_output.data(),
                                            d_carry_out.data(), block_data.data(), d_out_ptrs.data(), d_work_ptrs.data(), d_work.data(), h_block_counters[0], fwd_block_indices.data());
    cudaEventRecord(finish_fwd, 0);

    n_blocks = (h_block_counters[1] / NUM_THREADS_SPLIT) + 1;
    std::cout << "Using " << n_blocks << " CUDA blocks for reverse splits." << std::endl;
    split_matrix_reverse<<<n_blocks, NUM_THREADS_SPLIT>>>(dmA.raw.d_row_ptrs, dmA.raw.d_col_idx, dmB.raw.d_row_ptrs, dmB.raw.d_col_idx, d_output.data(),
                                            d_carry_out.data(), block_data.data(), d_out_ptrs.data(), d_work_ptrs.data(), d_work.data(), h_block_counters[1], reverse_block_indices.data());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, finish_psizes);
    std::cout << "Finished partial sizes: " << time << " ms" << std::endl;
    cudaEventElapsedTime(&time, start, finish_gen_splits);
    std::cout << "Finished finding split points: " << time << " ms" << std::endl;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Finished computing splits: " << time << " ms" << std::endl;

    cudaDeviceSynchronize();
    CheckCuda(cudaGetLastError());


    // Finally, read the data back and write it into two files: lb_block_ptrs.bin and lb_data.bin
    // in the format that load_balance_clean expects. Specifically:
    // lb_block_ptrs.bin is simply the indices of the blocks in lb_data.bin
    // lb_data.bin is an array of split data:
    //   word 0 = row
    //   word 1 = number of non-zeros in row == number of split points
    //            (not entirely necessary, as this can be easily computed from the row)
    //   word 2 = carry flag (whether we need to apply carries between this block and the next)
    //   word 3..k+3 = k split points for the row
    std::vector<int> h_out_ptrs = mgpu::from_mem(d_out_ptrs);
    std::ofstream lb_block_ptrs_file("lb_block_ptrs.bin");
    int d = 0;
    int first_row_size = matA.mRowPtrs[1] - matA.mRowPtrs[0];
    lb_block_ptrs_file.write((const char *)&d, sizeof(d));
    for (int i=0; i < total_blocks; i++)
    {
        d = (i * 3) + h_out_ptrs[i] + first_row_size + 3;
        lb_block_ptrs_file.write((const char *)&d, sizeof(d));
    }

    // Copy back block data from GPU (from_mem with bool array doesn't work with mgpu, so using cudaMemcpy)
    bool *h_carry_out = new bool[total_blocks+1];
    CheckCuda(cudaMemcpy(h_carry_out, d_carry_out.data(), sizeof(bool) * (total_blocks+1), cudaMemcpyDeviceToHost));
    std::vector<int> h_output = mgpu::from_mem(d_output);
    std::vector<block_data_t> h_block_data = mgpu::from_mem(block_data);

    std::ofstream lb_data_file("lb_data.bin");
    /* first block is all 0 */
    d = 0;
    lb_data_file.write((const char *)&d, sizeof(d));     // first is the row ( == 0 )
    lb_data_file.write((const char *)&first_row_size, sizeof(first_row_size));      // next number of splits
    int bcarry = h_carry_out[0];
    lb_data_file.write((const char *)&bcarry, sizeof(int));        // next is the carry which is 0 for the first block
    // next is the split points. For the first row, these are all zero (and d is zero)
    for (int i=0; i < first_row_size; i++)
        lb_data_file.write((const char *)&d, sizeof(d));

    for (int block_idx=0; block_idx < total_blocks; block_idx++)
    {
        int row = h_block_data[block_idx].row;
        lb_data_file.write((const char *)&row, sizeof(row));
        int split_row_size = matA.mRowPtrs[row + 1] - matA.mRowPtrs[row];
        lb_data_file.write((const char *)&split_row_size, sizeof(split_row_size));
        bcarry = h_carry_out[block_idx+1];      // convert bool carry to int
        // last block is always 0 carry?
        if (block_idx == total_blocks - 1)
            lb_data_file.write((const char *)&d, sizeof(int));
        else
            lb_data_file.write((const char *)&bcarry, sizeof(int));
        lb_data_file.write((const char *)(h_output.data() + h_out_ptrs[block_idx]), sizeof(int) * split_row_size);
    }
}
