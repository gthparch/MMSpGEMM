#include <iostream>
#include <fstream>
#include "MatrixMarket.h"
#include "DeviceMatrix.h"

#include <moderngpu/kernel_segreduce.hxx>
//#include <moderngpu/kernel_reduce.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_compact.hxx>

constexpr int BLOCK_SIZE = 2048;

constexpr int NUM_ITERS = 10;
constexpr int NUM_THREADS = 64;

void CheckCuda(cudaError_t success)
{
    if (success != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(success) << std::endl;
        exit(1);
    }
}


constexpr int MAX_ELEMENT = 999999999;

template <int POT=5>
__device__ void tournament_tree_kth_largest(int **A, int *b, int m, int w_k, int k)
{
    constexpr int SIZE = 1 << POT;
    struct node_t {
        int value;
        int list;
    } T[SIZE * 2];

    for (int i=0; i < m; i++) {
        if (b[i] == 0) {
            T[SIZE + i].value = -MAX_ELEMENT;
            T[SIZE + i].list = -1;
            continue;
        }
        T[SIZE + i].value = A[i][b[i]-1] - 1;
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


template <int POT=5>
__device__ void tournament_tree_kth_largest_reverse(int **A, int *alen, int *b, int m, int w_k, int k)
{
    constexpr int SIZE = 1 << POT;
    struct node_t {
        int value;
        int list;
    } T[SIZE * 2];

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


template <int POT=5>
__device__ void tournament_tree_kth_smallest(int **A, int *alen, int *b, int m, int w_k, int k)
{
    constexpr int SIZE = 1 << POT;
    struct node_t {
        int value;
        int list;
    } T[SIZE * 2];

    for (int i=0; i < m; i++) {
        if (b[i] + w_k > alen[i]) {
            T[SIZE + i].value = MAX_ELEMENT;
            T[SIZE + i].list = -1;
            continue;
        }
        T[SIZE + i].value = A[i][b[i] + w_k - 1] - 1;
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


template <int POT=5>
__device__ void tournament_tree_kth_smallest_reverse(int **A, int *alen, int *b, int m, int w_k, int k)
{
    constexpr int SIZE = 1 << POT;
    struct node_t {
        int value;
        int list;
    } T[SIZE * 2];

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


template <int MAXSIZE=32>
__device__ bool row_splitter(int **A, int *alen, int *b, int m, int p)
{
    // assert m < MAXSIZE
    int n_max = -1;
    for (int i=0; i < m; i++) {
        b[i] = 0;
        n_max = (alen[i] > n_max) ? alen[i] : n_max;
    }

    if (p == 0)
        return false;

    // Handle short splits
    if (p < m) {
        tournament_tree_kth_smallest(A, alen, b, m, 1, p);
        int lmax = compute_lmax(A, b, m);
        return compute_carry(A, alen, b, m, lmax);
    }

    int r = ceilf(logf((float)p / m) / logf(2.0));
    int two_r = 1 << r;
    int alpha = n_max / two_r;              // implicit floor
    int n = two_r * (alpha + 1) - 1;
    int k = ceilf((float)p / n * alpha);

    // Initial partition for the recursion
    tournament_tree_kth_smallest(A, alen, b, m, two_r, k);
    int lmax = compute_lmax(A, b, m);

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
            tournament_tree_kth_largest(A, b, m, w_k, Lsize - target_size);
        }
        if (Lsize < target_size) {
            tournament_tree_kth_smallest(A, alen, b, m, w_k, target_size - Lsize);
        }

        lmax = compute_lmax(A, b, m);
    }

    return compute_carry(A, alen, b, m, lmax);
}


template <int MAXSIZE=32>
__device__ bool row_splitter_reverse(int **A, int *alen, int *b, int m, int p)
{
    // assert m < MAXSIZE
    int n_max = -1;
    for (int i=0; i < m; i++) {
        b[i] = alen[i];
        n_max = (alen[i] > n_max) ? alen[i] : n_max;
    }

    if (p == 0)
        return false;

    // Handle short splits
    if (p < m) {
        tournament_tree_kth_smallest_reverse(A, alen, b, m, 1, p);
        int lmax = compute_lmax_reverse(A, alen, b, m);
        return compute_carry_reverse(A, alen, b, m, lmax);
    }

    int r = ceilf(logf((float)p / m) / logf(2.0));
    int two_r = 1 << r;
    int alpha = n_max / two_r;              // implicit floor
    int n = two_r * (alpha + 1) - 1;
    int k = ceilf((float)p / n * alpha);
    
//    printf("m = %d, r = %d, two_r = %d, n_max = %d, alpha = %d, n = %d\n", m, r, two_r, n_max, alpha, n);
//    printf("k = %d\n", k);

    // Initial partition for the recursion
    tournament_tree_kth_smallest_reverse(A, alen, b, m, two_r, k);
//    for (int i=0; i < m; i++)
//        printf("%d ", b[i]);
    /*
    if (threadIdx.x == 0)
    {
    printf("b: [");
    for (int i=0; i < m; i++)
        printf("%d ", b[i]);
    printf("]\n");
    }
    */
    int lmax = compute_lmax_reverse(A, alen, b, m);
//    printf("r = %d, alpha = %d, n = %d, k = %d, n_max = %d, lmax = %d\n", r, alpha, n, k, n_max, lmax);
//    if (threadIdx.x == 0) printf("first lmax = %d\n", lmax);

    // r iterative steps
    for (int k=0; k < r; k++)
    {
        int Lsize = 0;
        int w_k = 1 << (r - k - 1);
        int target_size = ceilf((float)p * (n / w_k) / n);

        for (int i=0; i < m; i++)
            Lsize += (alen[i] - b[i]) / w_k;
//        if (threadIdx.x == 0) printf("Lsize (after decided) = %d\n", Lsize);

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
        /*
        printf("Lsize = %d\n", Lsize);
        printf("pre-boundary: ");
        for (int i=0; i < m; i++)
            printf("%d ", A[i][b[i]-1]-1);
        printf("\n");

        printf("Lsize = %d, target_size = %d\n", Lsize, target_size);
        */
//        if (threadIdx.x == 0) printf("Lsize = %d, target_size = %d\n", Lsize, target_size);
        if (Lsize > target_size) {
//            if (threadIdx.x == 0) printf("moving %d largest from L to H\n", Lsize - target_size);
            tournament_tree_kth_largest_reverse(A, alen, b, m, w_k, Lsize - target_size);
        }
        if (Lsize < target_size) {
//            if (threadIdx.x == 0) printf("moving %d smallest from H to L\n", target_size - Lsize);
            tournament_tree_kth_smallest_reverse(A, alen, b, m, w_k, target_size - Lsize);
        }

        lmax = compute_lmax_reverse(A, alen, b, m);
//        if (threadIdx.x == 0) printf("new lmax = %d\n", lmax);

        /*
        printf("post-boundary: ");
        for (int i=0; i < m; i++)
            printf("%d ", A[i][b[i]-1]-1);
        printf("\n");
        */
    }

    return compute_carry_reverse(A, alen, b, m, lmax);
}


struct split_t {
    int row;
    int p;
    int out;
    bool reverse;
};


// TODO: compute carry flag
template <int MAXSIZE=32>
__global__ void split_matrix(int *row_ptrs, int *col_idx, int *base, bool *carry_out, split_t *splits, int nsplits)
{
    int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (threadId >= nsplits)
        return;

    int row = splits[threadId].row;
    int p = splits[threadId].p;
    int *out = base + splits[threadId].out;

    int *A[MAXSIZE];
    int b[MAXSIZE];
    int alen[MAXSIZE];
    int m = row_ptrs[row+1] - row_ptrs[row];

    // assert m < MAXSIZE or handle exception
    for (int i=0; i < m; i++) {
        int brow = col_idx[row_ptrs[row] + i] - 1;
        A[i] = col_idx + row_ptrs[brow];
        alen[i] = row_ptrs[brow+1] - row_ptrs[brow];
    }

    bool carry;
    if (splits[threadId].reverse)
        // p is precomputed for reversed rows row_size - split_pt
        carry = row_splitter_reverse(A, alen, b, m, p);
    else
        carry = row_splitter(A, alen, b, m, p);

    carry_out[threadId] = carry;
    for (int i=0; i < m; i++)
        out[i] = b[i];
}


struct block_data_t {
    int row;
    int p;
    bool reverse;
};
constexpr int ROWS_PER_THREAD = 14;
__global__ void scan_gen_blocks(int *cum_row_sizes, int *row_sizes, block_data_t *out)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int start = tid * ROWS_PER_THREAD;

    int last_rsize = cum_row_sizes[start];
    int last_block = last_rsize / BLOCK_SIZE;
    for (int i=start+1; i <= start + ROWS_PER_THREAD; i++) {
        int row_size = cum_row_sizes[i];
        int block = row_size / BLOCK_SIZE;
        if (last_block + 1 == block) {
            out[last_block].row = i;
            out[last_block].p = (block * BLOCK_SIZE) - last_rsize;
            if (out[last_block].p / (float)row_sizes[i] > 0.5) {
                out[last_block].p = row_sizes[i] - out[last_block].p;
                out[last_block].reverse = true;
            }
            else
                out[last_block].reverse = false;
        }
        last_block = block;
        last_rsize = row_size;
    }
}


int main(int argc, char **argv)
{
    mgpu::standard_context_t context;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <mtx file> <splits>" << std::endl;
        exit(1);
    }

    MatrixMarket matA { argv[1] };
    DeviceMatrix dmA(matA, context);
    DeviceMatrix& dmB = dmA;

    std::ifstream ifs(argv[2]);
    std::vector<split_t> splits;
    while (ifs.good()) {
        int row, p, out;
        bool reverse;
        ifs >> row >> p >> out >> reverse;
        if (!ifs.good())
            break;
        splits.push_back({row, p, out, reverse});
    }

    int last_row = splits[splits.size()-1].row;
    int last_row_size = matA.mRowPtrs[last_row+1] - matA.mRowPtrs[last_row];
    int out_size = splits[splits.size()-1].out + last_row_size;
    std::cout << splits.size() << " splits loaded, size = " << out_size << std::endl;

    mgpu::mem_t<split_t> d_splits = mgpu::to_mem(splits, context);
    // XXX: Size this from the outs above
    mgpu::mem_t<int> d_output(out_size, context);
    mgpu::mem_t<bool> d_carry_out(splits.size()+1, context);

    float time;
    cudaEvent_t start, stop;
    CheckCuda(cudaEventCreate(&start));
    CheckCuda(cudaEventCreate(&stop));

    // First find the splits
    mgpu::mem_t<int> d_row_sizes(matA.mRows, context);
    mgpu::mem_t<int> d_cumrow_sizes(matA.mRows, context);
    mgpu::mem_t<int> d_total_partials(1, context);

    int scan_blocks = matA.mRows / (ROWS_PER_THREAD * 128);
    mgpu::mem_t<int> d_out_ptrs(scan_blocks, context);
    mgpu::mem_t<int> d_out_final(1, context);

    cudaDeviceSynchronize();
#if 0
    cudaEventRecord(start, 0);
    int *Acolidx = dmA.raw.d_col_idx;
    int *Arowptrs = dmA.raw.d_row_ptrs;
    int *Browptrs = dmB.raw.d_row_ptrs;
    mgpu::lbs_segreduce([=]MGPU_DEVICE(int index, int seg, int rank)
        {
            int brow = Acolidx[Arowptrs[seg] + rank] - 1;
            return Browptrs[brow+1] - Browptrs[brow];
        }, matA.mCSRVals.size(), dmA.raw.d_row_ptrs, matA.mRows, d_row_sizes.data(), mgpu::plus_t<int>(), 0, context);
    mgpu::scan<mgpu::scan_type_inc>(d_row_sizes.data(), matA.mRows, d_cumrow_sizes.data(), mgpu::plus_t<int>(), d_total_partials.data(), context);

    std::vector<int> total_partials = mgpu::from_mem(d_total_partials);
    int total_blocks = total_partials[0] / BLOCK_SIZE;
    std::cout << "total_partials = " << total_partials[0] << ", blocks = " << total_blocks << std::endl;

    mgpu::mem_t<block_data_t> block_data(total_blocks, context);
    scan_gen_blocks<<<scan_blocks, 128>>>(d_cumrow_sizes.data(), d_row_sizes.data(), block_data.data());

    // prefix sum to get block 'out' value
    block_data_t *d_block_data = block_data.data();
    mgpu::transform_scan<int>([=]MGPU_DEVICE(int index)
        {
            int r = d_block_data[index].row;
            return Arowptrs[r + 1] - Arowptrs[r];
        }, scan_blocks, d_out_ptrs.data(), mgpu::plus_t<int>(), d_out_final.data(), context);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Elapsed: " << time << " ms" << std::endl;

//    std::cout << "stream_count = " << stream_count << std::endl;
/*
    std::vector<int> out_row_sizes = mgpu::from_mem(d_cumrow_sizes);
    std::cout << "out_row_sizes.size = " << out_row_sizes.size() << std::endl;
    for (int i = 0; i < 15; i++)
        std::cout << "out_row_sizes[" << i << "] = " << out_row_sizes[i] << std::endl;
        */

    std::vector<int> h_out_ptrs = mgpu::from_mem(d_out_ptrs);
    for (int i=0; i < 10; i++)
        std::cout << "out_ptr " << i << " = " << h_out_ptrs[i] << std::endl;

    std::vector<block_data_t> h_block_data = mgpu::from_mem(block_data);
    for (int i=0; i < 10; i++)
        std::cout << "split at " << h_block_data[i].row << ", " << h_block_data[i].p << ", " << h_block_data[i].reverse << std::endl;

    exit(1);

#else

    /*
    cudaThreadSetLimit(cudaLimitStackSize, 8192);
    size_t stack_size = 0;
    cudaThreadGetLimit(&stack_size, cudaLimitStackSize);
    std::cout << "Stack size: " << stack_size << std::endl;
    CheckCuda(cudaGetLastError());
    */
    cudaEventRecord(start, 0);
    int n_blocks = (splits.size() / NUM_THREADS) + 1;
    std::cout << "num blocks = " << n_blocks << std::endl;
    for (int i=0; i < NUM_ITERS; i++)
        split_matrix<<<n_blocks, NUM_THREADS>>>(dmA.raw.d_row_ptrs, dmA.raw.d_col_idx, d_output.data(),
                                                d_carry_out.data(), d_splits.data(), splits.size());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Elapsed: " << (time / NUM_ITERS) << " ms" << std::endl;
    cudaDeviceSynchronize();
    CheckCuda(cudaGetLastError());

    std::ofstream lb_block_ptrs_file("lb_block_ptrs.bin");
    int d = 0;
    int first_row_size = matA.mRowPtrs[1] - matA.mRowPtrs[0];
    lb_block_ptrs_file.write((const char *)&d, sizeof(d));
    // Check correctness
    for (int i=0; i < splits.size(); i++)
    {
        d = (i * 3) + splits[i].out + first_row_size + 3;
        lb_block_ptrs_file.write((const char *)&d, sizeof(d));
    }

    // Copy back data from GPU (from_mem with bool array doesn't work with mgpu, so using cudaMemcpy)
//    std::vector<bool> h_carry_out = mgpu::from_mem<bool>(d_carry_out);
    bool *h_carry_out = new bool[splits.size()+1];
    CheckCuda(cudaMemcpy(h_carry_out, d_carry_out.data(), sizeof(bool) * (splits.size()+1), cudaMemcpyDeviceToHost));
    std::vector<int> h_output = mgpu::from_mem(d_output);

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

    // XXX: switch splits to block_data
    for (int block_idx=0; block_idx < splits.size(); block_idx++)
    {
        int row = splits[block_idx].row;
        lb_data_file.write((const char *)&row, sizeof(row));
        int split_row_size = matA.mRowPtrs[row + 1] - matA.mRowPtrs[row];
        lb_data_file.write((const char *)&split_row_size, sizeof(split_row_size));
        bcarry = h_carry_out[block_idx+1];      // convert bool carry to int
        lb_data_file.write((const char *)&bcarry, sizeof(int));
        lb_data_file.write((const char *)(h_output.data() + splits[block_idx].out), sizeof(int) * split_row_size);
    }
#endif
}
