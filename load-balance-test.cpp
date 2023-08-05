#include "MatrixMarket.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>

constexpr int BLOCK_SIZE = 2048;

// These need to multiply to exactly BLOCK_SIZE above
constexpr int NUM_THREADS = 128;
constexpr int ITEMS_PER_THREAD = 16;
/*
constexpr int NUM_THREADS = 1;
constexpr int ITEMS_PER_THREAD = 2048;
*/

class TSplit {
public:
    int a_row;
    int bp;
    int b_col;
};

int lower_bound_row_bs(const int *row, int s, int p, int len)
{
    int low = p;
    int high = len;

    int midpt = (low + high) / 2;
    while ((high - low) > 1)
    {
        if (row[midpt] > s)
            high = midpt;
        else
            low = midpt;
        midpt = (low + high) / 2;
    }

    return midpt;
}


int comps = 0;
static inline int lower_bound_row(const int *row, int s, int p, int len)
{
    // binary search lower bound to find s in row; can also pass a min/max 

    // Galloping search
    int i = 2;
//    comps = 0;
    while ((p + i) < len && row[p+i] < s) {
//        comps++;
        p += (i);
        i = i << 1;
    }

    while (p < len)
    {
        if (row[p] >= s)
            break;
//        comps++;
        p++;
    }

    return p;
}


void print_block_data(const int* data)
{
    std::cout << "row: " << data[0] << std::endl;
    std::cout << "n_splits: " << data[1] << std::endl;
    for (int i=0; i < data[1]; i++) {
        std::cout << "... " << data[2+i] << std::endl;
    }
}


int search_block_path(const MatrixMarket &M, int row, int split, std::vector<int>& row_splits)
{
    // collect all the b rows
    std::vector<const int*> rows;
    std::vector<int> row_size;
    for (int rp = M.mRowPtrs[row]; rp < M.mRowPtrs[row+1]; rp++)
    {
        rows.push_back(M.mColIdx.data() + M.mRowPtrs[M.mColIdx[rp]-1]);
        row_size.push_back(M.mRowPtrs[M.mColIdx[rp]] - M.mRowPtrs[M.mColIdx[rp]-1]);
    }

    int search_low = 0;
    int search_high = M.mRows;
    row_splits.resize(rows.size());
    std::vector<int> row_bounds(rows.size());
    std::vector<int> row_bounds_high(rows.size());

    for (int i=0; i < rows.size(); i++)
    {
        row_bounds_high[i] = row_size[i];
    }

//    std::cout << "search_block_path start" << std::endl;
    int last_total = -1;
    int total_size = 0;
    while (true)
    {
        if (search_high - search_low < 2)
            break;

        int midpt = (search_low + search_high) / 2;

        total_size = 0;
        for (int i = 0; i < rows.size(); i++) {
//            row_splits[i] = lower_bound_row(rows[i], midpt, row_bounds[i], row_bounds_high[i]);
            row_splits[i] = lower_bound_row(rows[i], midpt, row_bounds[i], row_size[i]);
//            std::cout << "search_block_path comps = " << comps << std::endl;
//            row_splits[i] = lower_bound_row_bs(rows[i], midpt, row_bounds[i], row_size[i]);
            total_size += row_splits[i];
            if (total_size > split)
                break;
        }

        if (total_size > split) {
            search_high = midpt;
            row_bounds_high = row_splits;
        } else {
            search_low = midpt;
            row_bounds = row_splits;
        }

//        if (total_size == last_total)
//            break;

        last_total = total_size;
    }
//    std::cout << "search_block_path end" << std::endl;


    /*
    int q = 183;
    int count = 0;
    for (int i=0; i < rows.size(); i++)
    {
        // count less q
        for (int j = 0; j < row_size[i]; j++)
            if (rows[i][j] < q)
                count++;
    }
    std::cout << "q = 183: " << count << std::endl;
    q = 184;
    count = 0;
    for (int i=0; i < rows.size(); i++)
    {
        // count less q
        for (int j = 0; j < row_size[i]; j++)
            if (rows[i][j] < q)
                count++;
    }
    std::cout << "q = 184: " << count << std::endl;
    q = 185;
    count = 0;
    for (int i=0; i < rows.size(); i++)
    {
        // count less q
        for (int j = 0; j < row_size[i]; j++)
            if (rows[i][j] < q)
                count++;
    }
    std::cout << "q = 185: " << count << std::endl;
    */




//    std::cout << "search_block_path(row=" << row << ",split=" << split << ",low=" << search_low << ",high=" << search_high << ") = ";
    total_size = 0;
    for (int i = 0; i < row_splits.size(); i++) {
//        std::cout << row_splits[i] << " (" << rows[i][row_splits[i]-1] << "," << rows[i][row_splits[i]] << ") ";
        total_size += row_splits[i];
    }
//    std::cout << std::endl;
//    std::cout << "total_size = " << total_size << std::endl;

    if (total_size > split) {
        int excess = total_size - split;
        for (int i = 0; i < row_splits.size(); i++) {
//            if (rows[i][row_splits[i]] == search_high) {
            if (row_splits[i] > 0 && rows[i][row_splits[i]-1] == search_low) {
                row_splits[i]--;
                excess--;
                if (excess == 0)
                    break;
            }
        }
    } else if (total_size < split) {
        int deficit = split - total_size;
        for (int i=0; i < row_splits.size(); i++) {
            if (rows[i][row_splits[i]] == search_low) {         // ???
                row_splits[i]++;
                deficit--;
                if (deficit == 0)
                    break;
            }
        }
    }

    /*
    std::cout << "adjusted:" << std::endl;
    total_size = 0;
    for (int i = 0; i < row_splits.size(); i++) {
        std::cout << row_splits[i] << " (" << rows[i][row_splits[i]-1] << "," << rows[i][row_splits[i]] << ") ";
        total_size += row_splits[i];
    }
    std::cout << std::endl;
    std::cout << "total_size = " << total_size << std::endl;
    */

    return search_low;
}


void load_block(const MatrixMarket& A, const std::vector<int>& block_data, const std::vector<int>& block_ptrs, int block_id)
{
    int bp = block_ptrs[block_id];
    int bprev = block_ptrs[block_id - 1];

    int start_row = block_data[bprev + 0];
    int start_sz = block_data[bprev + 1];

    int end_row = block_data[bp + 0];
    int end_sz = block_data[bp + 1];

    if (start_row == end_row)
    {
        std::cout << "A overhead: " << start_sz << " == " << end_sz << std::endl;

        if (start_sz > 2000) {
            int total = 0;
            int count = 0;
            for (int i=0; i < start_sz; i++) {
                int start_offset = block_data[bprev + 2 + i];
                int end_offset = block_data[bp + 2 + i];
                total += end_offset - start_offset;
//                std::cout << " ... A overhead: " << start_offset << " - " << end_offset << ", diff: " << (end_offset - start_offset) << ", total: " << total << std::endl;
                if (end_offset - start_offset == 0)
                    count++;
            }
            std::cout << " ... A overhead: " << count << std::endl;
        }

        // read A coefficients into shared mem

        // read segments of B 
        for (int i = 0; i < start_sz; i++)
        {
            // brow = A.mColIdx[A.mRowPtrs[start_row] + i]
            // start_offset = block_data[bprev + 2 + i]
            // end_offset = block_data[bp + 2 + i]
            // copy B[brow][start_offset:end_offset] into shared mem at p
            // increment p by sizeof memcpy
        }
    }
    else {
        int overhead = start_sz + end_sz;

        // load start_row

        // load middle rows
        for (int r = start_row + 1; r < end_row; r++)
        {
            overhead += A.mRowPtrs[r+1] - A.mRowPtrs[r];
        }

        // load end_row

        std::cout << "A overhead: " << overhead << std::endl;
    }
}


float load_coop_seq(const MatrixMarket& A, const MatrixMarket& B, const std::vector<int>& lb_data, const std::vector<int>& lb_block_ptrs, int block, std::vector<int>& output_cols, std::vector<float>& output_vals)
{
    struct elementT {
        int key;
        float val;
    } elements[BLOCK_SIZE];
//    float accum(0.0);
    int count(0);

    int cur_block_ptr = lb_block_ptrs[block];
    int next_block_ptr = lb_block_ptrs[block+1];
    int start_row = lb_data[cur_block_ptr];
    int end_row = lb_data[next_block_ptr];
    std::cout << "load_coop_seq; start_row = " << start_row << ", end_row = " << end_row << std::endl;

    for (int row = start_row; row <= end_row; row++)
    {
        int a_start = A.mRowPtrs[row];
        int a_end = A.mRowPtrs[row+1];

        for (int bp = a_start; bp < a_end; bp++)
        {
            int brow = A.mColIdx[bp] - 1;
            float Acoeff = A.mCSRVals[bp];

            int b_start = B.mRowPtrs[brow];
            int b_end = B.mRowPtrs[brow+1];

            if (row == start_row)
                b_start += lb_data[cur_block_ptr + 2 + (bp - a_start)];
            if (row == end_row)
                b_end = B.mRowPtrs[brow] + lb_data[next_block_ptr + 2 + (bp - a_start)];

            for (int j = b_start; j < b_end; j++)
            {
//                std::cout << "load_coop_seq accumulating " << row << ", brow = " << brow << ", b_col = " << j << ", bp = " << Acoeff << std::endl;
//                accum += Acoeff * B.mCSRVals[j];
                elements[count].key = ((row - start_row) << 25) | B.mColIdx[j];
                elements[count].val = Acoeff * B.mCSRVals[j];
                count++;
            }
        }
    }

    std::cout << "load_coop_seq added " << count << " values." << std::endl;

    // sort, reduce, write to global memory
    std::sort(elements, elements + BLOCK_SIZE, [](const elementT& a, const elementT& b) { return a.key < b.key; });
    int last_key = elements[0].key;
    float accum = 0; // elements[0].val;
    for (int i=0; i < BLOCK_SIZE; i++)
    {
        if (elements[i].key != last_key) {
            output_cols.push_back(last_key);
            output_vals.push_back(accum);
            accum = 0.0;
        }
        accum += elements[i].val;
        last_key = elements[i].key;
    }

    std::cout << "carry-out key = " << last_key << " (" << (last_key >> 25) << "," << (last_key & ((1 << 25) - 1)) << ") = " << accum << std::endl;

    return accum;
}


float load_coop_simt(const MatrixMarket& A, const MatrixMarket& B, const std::vector<int>& lb_data, const std::vector<int>& lb_block_ptrs, const std::vector<std::vector<TSplit>>& lb_thread_splits, int block)
{
    int cur_block_ptr = lb_block_ptrs[block];
    int next_block_ptr = lb_block_ptrs[block+1];
    int end_row = lb_data[next_block_ptr];

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


    float *shared = new float[BLOCK_SIZE];
    for (int tid = 0; tid < NUM_THREADS; tid++)
    {
        TSplit split = lb_thread_splits[block][tid];
        int row_end = A.mRowPtrs[split.a_row + 1];
        int brow = A.mColIdx[split.bp] - 1;
        int seg_end = B.mRowPtrs[brow+1];

        int coeff_start = A.mRowPtrs[split.a_row];
        if (split.a_row == end_row)
            seg_end = B.mRowPtrs[brow] + lb_data[next_block_ptr + 2 + (split.bp - coeff_start)];
        float Acoeff = A.mCSRVals[split.bp];

        std::cout << "begin row_end = " << row_end << ", brow = " << brow << ", seg_end = " << seg_end << std::endl;
        for (int i = 0; i < ITEMS_PER_THREAD; i++)
        {
            if (split.b_col >= seg_end)
            {
                split.bp++;
                if (split.bp >= row_end) {
                    split.a_row++;
                    split.bp = A.mRowPtrs[split.a_row];
                    row_end = A.mRowPtrs[split.a_row + 1];
                }
                Acoeff = A.mCSRVals[split.bp];
                brow = A.mColIdx[split.bp] - 1;
                split.b_col = B.mRowPtrs[brow];
                seg_end = B.mRowPtrs[brow+1];
                if (split.a_row == end_row) {
                    int coeff_start = A.mRowPtrs[split.a_row];
                    seg_end = B.mRowPtrs[brow] + lb_data[next_block_ptr + 2 + (split.bp - coeff_start)];
                }
            }

            std::cout << "tid " << tid << " copying a_row: " << split.a_row << ", a_col: " << split.bp << ", b_col: " << split.b_col << ", brow: " << brow << ", acoeff = " << Acoeff << std::endl;
//            split.b_col++;
            shared[tid * ITEMS_PER_THREAD + i] = B.mCSRVals[split.b_col++] * Acoeff;
        }
    }

    // reduce to test
    float accum(0.0);
    for (int i=0; i < BLOCK_SIZE; i++)
        accum += shared[i];
    return accum;
}




int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <mtx file>" << std::endl;
        exit(1);
    }

    MatrixMarket mat(argv[1]);

    // output row temp space size
    std::vector<int> row_size(mat.mRows);
    for (int row=0; row < mat.mRows; row++)
    {
        int total(0);
        for (int rp = mat.mRowPtrs[row]; rp < mat.mRowPtrs[row+1]; rp++)
            total += mat.mRowPtrs[mat.mColIdx[rp]] - mat.mRowPtrs[mat.mColIdx[rp]-1];
        row_size[row] = total;
    }

    std::vector<int> row_size_scan(mat.mRows);
    // prefix sum
    int accum(0);
    for (int i=0; i < row_size.size(); i++)
    {
        accum += row_size[i];
        row_size_scan[i] = accum;
    }

    int n_blocks = row_size_scan[mat.mRows-1] / BLOCK_SIZE;
    std::cout << "n_blocks = " << n_blocks << std::endl;
//    for (int i = 0; i < 50; i++)
//        std::cout << row_size_scan[i] << std::endl;

    // vectorized sorted search to find the search points, data = (row, offset)
    int i = 0;
    int j = BLOCK_SIZE;
    int count = 0;

    std::vector<int> lb_data;
    std::vector<int> lb_block_ptrs;
    std::vector<std::vector<TSplit>> lb_thread_splits;

    // Initialize first block data to 0 for start of matrix
    lb_data.push_back(0);
    lb_data.push_back(mat.mRowPtrs[1] - mat.mRowPtrs[0]);
    for (int i = 0; i < lb_data[1]; i++)
        lb_data.push_back(0);
    lb_block_ptrs.push_back(0);

    auto start = std::chrono::system_clock::now();
    while (i < row_size_scan.size()) {
        if (row_size_scan[i] < j) {
            i++;
            continue;
        }

        while (j <= row_size_scan[i])
        {
            std::vector<int> row_splits;
            int split = j - row_size_scan[i-1];
//            std::cout << "(" << i << "," << j << "," << split << "," << row_size_scan[i] << ")" << std::endl;
            lb_block_ptrs.push_back(lb_data.size());
            search_block_path(mat, i, split, row_splits);
            lb_data.push_back(i);
            lb_data.push_back(row_splits.size());
            std::copy(row_splits.begin(), row_splits.end(), std::back_inserter(lb_data));
            j += BLOCK_SIZE;
        }
    }

    std::cout << "lb_block_ptrs[0] = " << lb_block_ptrs[0] << std::endl;
    std::cout << "lb_data[0] = " << lb_data[0] << ", " << lb_data[1] << ", " << lb_data[2] << std::endl;

    // Build the per-thread cooperative load splits
    // much faster way to do this ...
    lb_thread_splits.resize(lb_block_ptrs.size()-1);
    for (int block = 0; block < lb_block_ptrs.size()-1; block++)
    {
        int cur_bp = lb_block_ptrs[block];
        int next_bp = lb_block_ptrs[block+1];
        int start_row = lb_data[cur_bp];
        int end_row = lb_data[next_bp];
        int copy_count(0);

        for (int row = start_row; row <= end_row; row++)
        {
            int coeff_start = mat.mRowPtrs[row];
            int coeff_end = mat.mRowPtrs[row+1];
            for (int bp = coeff_start; bp < coeff_end; bp++)
            {
                int brow = mat.mColIdx[bp]-1;
                int seg_start = mat.mRowPtrs[brow];
                int seg_end = mat.mRowPtrs[brow+1];
                if (row == start_row)
                    seg_start += lb_data[cur_bp + 2 + (bp - coeff_start)];
                if (row == end_row)
                    seg_end = mat.mRowPtrs[brow] + lb_data[next_bp + 2 + (bp - coeff_start)];

                int count = seg_end - seg_start;
                for (int i = 0; i < count; i++)
                {
                    if (copy_count % ITEMS_PER_THREAD == 0) {
                        lb_thread_splits[block].push_back({row, bp, seg_start + i});
                    }
                    copy_count++;
                }
            }
        }
    }


    auto finish = std::chrono::system_clock::now();
    float ms_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();

    std::cout << "Elasped ms: " << ms_elapsed << std::endl;

    std::cout << "writing lb_data.bin" << std::endl;
    std::ofstream of("lb_data.bin");
    of.write((const char *)lb_data.data(), sizeof(int) * lb_data.size());
    of.close();
    std::ofstream of2("lb_block_ptrs.bin");
    of2.write((const char *)lb_block_ptrs.data(), sizeof(int) * lb_block_ptrs.size());
    of2.close();
    std::cout << "sizeof(TSplit) = " << sizeof(TSplit) << std::endl;
    std::ofstream of3("lb_thread_splits.bin");
    for (int i=0; i < lb_thread_splits.size(); i++)
        of3.write((const char *)lb_thread_splits[i].data(), sizeof(TSplit) * lb_thread_splits[i].size());
    of3.close();

    std::cout << "lb_data size = " << lb_data.size() << std::endl;
    std::cout << "lb_block_ptrs size = " << lb_block_ptrs.size() << std::endl;
    std::cout << "lb_thread_splits size = " << lb_thread_splits.size() << std::endl;

    print_block_data(lb_data.data() + lb_block_ptrs[0]);
    std::cout << "----" << std::endl;
    print_block_data(lb_data.data() + lb_block_ptrs[1]);
    float simt_val = load_coop_simt(mat, mat, lb_data, lb_block_ptrs, lb_thread_splits, 0);
    std::vector<int> output_cols;
    std::vector<float> output_vals;
    float seq_val = load_coop_seq(mat, mat, lb_data, lb_block_ptrs, 0, output_cols, output_vals);
    std::cout << "load_coop_simt: " << simt_val << " == " << seq_val << std::endl;

    load_coop_seq(mat, mat, lb_data, lb_block_ptrs, 1, output_cols, output_vals);
    std::cout << "output_cols.size = " << output_cols.size() << ", output_vals.size = " << output_vals.size() << std::endl;
    for (int i=0; i < output_cols.size(); i++)
    {
        int row = output_cols[i] >> 25;
        int col = output_cols[i] & ((1 << 25) - 1);
        std::cout << "(" << row << "," << col << "): " << output_vals[i] << std::endl;
    }




    // TODO: Need to add a 0 block data/ptrs to avoid starting at 1
    /*
    for (int b=1; b < n_blocks; b++)
        load_block(mat, lb_data, lb_block_ptrs, b);
        */


    // parallel k-dim merge path for each block
}
