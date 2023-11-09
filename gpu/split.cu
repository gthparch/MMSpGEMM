#include "MatrixMarket.h"
#include "DeviceMatrix.h"

constexpr MAX_ELEMENT = 999999999;

template <int POT=5>
__global__ void tournament_tree_kth_largest(int **A, int *b, int m, int w_k, int k)
{
    int SIZE = 1 << POT;
    struct node_t {
        int value;
        int list;
    } T[SIZE * 2];

    for (int i=0; i < m; i++) {
        T[SIZE + i].value = A[i][b[i]-1];
        T[SIZE + i].list = i;
    }

    for (int i=m; i < SIZE; i++) {
        T[SIZE + i].value = -MAX_ELEMENT;
        T[SIZE + i].list = -1;
    }

    // First iteration, just propagate up the tree
    for (int l = POT-1; l >= 0; l--)
    {
        int l0 = 1 >> l;
        int l1 = 1 >> (l+1);
        for (int j = 0; j < l0; j++)
        {
            if (T[l1 + j*2].value > T[l1 + j*2 + 1].value)
                T[l0 + j] = T[l1 + j*2];
            else
                T[l0 + j] = T[l1 + j*2+1];
        }
    }

    int winner = T[0].list;
    b[winner] -= w_k;
    T[SIZE + winner].list = winner;
    if (b[winner] == 0)
        T[SIZE + winner].value = -MAX_ELEMENT;
    else
        T[SIZE + winner].value = A[winner][b[winner]-1];

    // Now just propagate the winning list
    for (int i = 0; i < k-1; i++)
    {
        int j = winner;
        for (int l = POT-1; l >= 0; l--)
        {
            int l0 = 1 >> l;
            int l1 = 1 >> (l+1);
            if (T[l1 + (j & 0xfffffffe)].value > T[l1 + (j & 0xfffffffe) + 1].value)
                T[l0 + (j >> 1)] = T[l1 + (j & 0xfffffffe)];
            else
                T[l0 + (j >> 1)] = T[l1 + (j & 0xfffffffe) + 1];
            j = j >> 1;
        }
        winner = T[0].list;
        b[winner] -= w_k;
        if (b[winner] == 0)
            T[SIZE + winner].value = -MAX_ELEMENT;
        else
            T[SIZE + winner].value = A[winner][b[winner]-1];
   }
}


template <int POT=5>
__global__ void tournament_tree_kth_smallest(int **A, int *alen, int *b, int m, int w_k, int k)
{
    int SIZE = 1 << POT;
    struct node_t {
        int value;
        int list;
    } T[SIZE * 2];

    for (int i=0; i < m; i++) {
        T[SIZE + i].value = A[i][b[i]-1];
        T[SIZE + i].list = i;
    }

    for (int i=m; i < SIZE; i++) {
        T[SIZE + i].value = MAX_ELEMENT;
        T[SIZE + i].list = -1;
    }

    // First iteration, just propagate up the tree
    for (int l = POT-1; l >= 0; l--)
    {
        int l0 = 1 >> l;
        int l1 = 1 >> (l+1);
        for (int j = 0; j < l0; j++)
        {
            if (T[l1 + j*2].value < T[l1 + j*2 + 1].value)
                T[l0 + j] = T[l1 + j*2];
            else
                T[l0 + j] = T[l1 + j*2+1];
        }
    }

    int winner = T[0].list;
    b[winner] += w_k;
    T[SIZE + winner].list = winner;

    if (b[winner] + w_k > alen[winner])
        T[SIZE + winner].value = MAX_ELEMENT;
    else
        T[SIZE + winner].value = A[winner][b[winner]-1];

    // Now just propagate the winning list
    for (int i = 0; i < k-1; i++)
    {
        int j = winner;
        for (int l = POT-1; l >= 0; l--)
        {
            int l0 = 1 >> l;
            int l1 = 1 >> (l+1);
            if (T[l1 + (j & 0xfffffffe)].value < T[l1 + (j & 0xfffffffe) + 1].value)
                T[l0 + (j >> 1)] = T[l1 + (j & 0xfffffffe)];
            else
                T[l0 + (j >> 1)] = T[l1 + (j & 0xfffffffe) + 1];
            j = j >> 1;
        }
        winner = T[0].list;
        b[winner] += w_k;
        if (b[winner] + w_k > alen[winner])
            T[SIZE + winner].value = MAX_ELEMENT;
        else
            T[SIZE + winner].value = A[winner][b[winner]-1];
   }
}


__global__ inline int compute_lmax(int **A, int *b, int blen)
{
    int lmax = -MAX_ELEMENT;
    for (int i=0; i < blen; i++) {
        if (b[i] > 0 && A[i][b[i]-1] > lmax)
            lmax = A[i][b[i]-1];
    }
    return lmax;
}


template <int MAXSIZE=32>
__global__ void row_splitter(int **A, int *alen, int m, int p)
{
    int b[MAXSIZE];
    // assert m < MAXSIZE
    int n_max = -1;
    for (int i=0; i < m; i++) {
        b[i] = 0;
        n_max = (alen[i] > n_max) ? alen[i] : n_max;
    }

    // Handle short splits
    if (p < m) {
        tournament_tree_kth_smallest(A, alen, b, m, 1, p);
        return;
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
            int undecided = A[i][b[i] + w_k - 1];
            if (undecided < lmax) {
                b[i] += w_k;
                Lsize++;
            }
        }

        if (Lsize > target_size)
            tournament_tree_kth_largest(A, b, m, w_k, Lsize - target_size);
        if (Lsize < target_size)
            tournament_tree_kth_smallest(A, b, alen, m, w_k, target_size - Lsize);

        lmax = compute_lmax(A, b, m);
    }
}


template <int MAXSIZE=32>
__global__ void split_matrix_row(int *row_ptrs, int *col_idx, int row, int p)
{
    int *A[MAXSIZE];
    int m = row_ptrs[row+1] - row_ptrs[row];
    for (int i=row_ptrs[row]; i < row_ptrs[row+1]; i++) {
        int brow = col_idx[i];
        A[i] = col_idx + row_ptrs[brow];
        alen[i] = row_ptrs[brow+1] - row_ptrs[brow];
    }

    row_splitter(A, alen, m, p);
}


int main(int argc, char **argv)
{
    mgpu::standard_context_t context;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mtx file>" << std::endl;
        exit(1);
    }

    MatrixMarket matA { argv[1] };
    DeviceMatrix dmA(matA, context);

    // Load some sample data, and test the tournament tree implementation
}
