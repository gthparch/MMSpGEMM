#include <vector>
#include <algorithm>
//#include "dynamic_min.h"
#include "dynamic_min_heap.h"
#include <iostream>
#include <chrono>

#include "MatrixMarket.h"

//#define TEST_SPLITS     1


#define BITS        13
class Element
{
public:
//    Element() : mValue(-1), mId(-1) { }
//    Element(int _value, int _id) : mValue(_value), mId(_id) { }

    Element() : _m(0) { }
    Element(unsigned _value, unsigned _id) : _m(((uint64_t)_value << BITS) | _id) { }
    unsigned key() { return _m & ((1 << BITS) - 1); }
    unsigned value() { return _m >> BITS; }

    uint64_t _m;

    /*
    int key() { return mId; }
    int value() { return mValue; }
    int mValue, mId;
    */
};

struct CompareLess
{
    bool operator()(const Element& a, const Element& b) const
    {
        return a._m < b._m;
        /*
        if (a.mValue < b.mValue)
            return true;
        if (a.mValue > b.mValue)
            return false;
        return a.mId < b.mId;
        */
    }
};

struct CompareGreater
{
    bool operator()(const Element& a, const Element& b) const
    {
        return a._m > b._m;
        /*
        if (a.mValue > b.mValue)
            return true;
        if (a.mValue < b.mValue)
            return false;
        return a.mId > b.mId;
        */
    }
};


#if 0
std::vector<int> split_matrix_one_heap(const std::vector<int *>& A, const std::vector<int>& alen, int p, int N)
{
    int m = A.size();

    std::vector<int> S(m);
    std::vector<int> L(m);          // starts with 0
    std::vector<int> R(m);
    std::set<int> non_empty;
    int remaining = p;
    for (int i = 0; i < m; i++) {
        L[i] = 0;           // not necessary; should be initialized to 0
        R[i] = std::min(alen[i], p);
        S[i] = std::min(remaining, int(alen[i] * p / N + 1.0));
        std::cout << "alen[i] = " << alen[i] << ", S[i] = " << S[i] << std::endl;
        remaining -= S[i];
        if (S[i] > L[i])
            non_empty.insert(i);
    }

    std::vector<Element> initial_list;
    for (int i = 0; i < S.size(); i++)
        if (S[i] > 0) initial_list.push_back(Element(A[i][S[i]-1], i));
        else initial_list.push_back(Element(-9999999, i));

    std::cout << "non_empty: ";
    for (auto it = non_empty.begin(); it != non_empty.end(); ++it)
        std::cout << *it << " ";
    std::cout << std::endl;

    std::cout << "S[i]: ";
    for (int i = 0; i < S.size(); i++)
        std::cout << S[i] << " ";
    std::cout << std::endl;

    DynamicMinMax<Element, CompareLess> Heap(initial_list, S.size());
    while (!non_empty.empty())
    {
        Element m = Heap.first();
        // pick the first non empty list that is not this one...
        int min_list = m.key();
        auto it = non_empty.begin();
        if (*it == min_list) {
            it++;
            if (it == non_empty.end())
                break;
        }
        int swap_list = *it;
        std::cout << "picked min list: " << min_list << ", value = " << m.value() << std::endl;
        std::cout << "picked swap list: " << swap_list << std::endl;

        int delta_max = (S[swap_list] - L[swap_list]) / 2;
        int delta_min = (R[min_list] - S[min_list]) / 2;

        int delta = std::min(delta_min, delta_max);
        std::cout << "R[swap_list] = " << R[swap_list] << ", L[swap_list] = " << L[swap_list] << std::endl;
        std::cout << "delta_max = " << delta_max << ", delta_min = " << delta_min << ", delta = " << delta << std::endl;
        S[swap_list] -= delta;
        S[min_list] += delta;

        L[min_list] = S[min_list];
        R[swap_list] = S[swap_list];

    std::cout << "S[i]: ";
    for (int i = 0; i < S.size(); i++)
        std::cout << S[i] << " ";
    std::cout << std::endl;
    std::cout << "L[i]: ";
    for (int i = 0; i < L.size(); i++)
        std::cout << L[i] << " ";
    std::cout << std::endl;
    std::cout << "R[i]: ";
    for (int i = 0; i < R.size(); i++)
        std::cout << R[i] << " ";
    std::cout << std::endl;

        // update the heap
        Heap.update(min_list, A[min_list][S[min_list]-1]);
        Heap.update(swap_list, A[swap_list][S[swap_list]-1]);

        if (S[min_list] == R[min_list])
            Heap.update(min_list, -1);
        if (S[swap_list] == R[swap_list])
            Heap.update(swap_list, -1);

        // update non_empty list
        non_empty.erase(min_list);
        non_empty.erase(swap_list);
        if (S[min_list] > L[min_list])
            non_empty.insert(min_list);
        if (S[swap_list] > L[swap_list])
            non_empty.insert(swap_list);
    }
}
#endif


// N only used to compute initial splits
/*std::vector<int>*/ void split_matrix(const std::vector<int *>& A, const std::vector<int>& alen, int p, int N, std::vector<int>& L)
{
    int m = A.size();

    std::vector<int> S(m);
//    std::vector<int> L(m);          // starts with 0
    std::vector<int> R(m);
    int remaining = p;
    for (int i = 0; i < m; i++) {
//        L[i] = 0;           // not necessary; should be initialized to 0
        R[i] = std::min(alen[i], p);
        S[i] = std::min(remaining, int(alen[i] * p / N + 1.0));
        remaining -= S[i];
    }

    // allocating space for buffers
//    std::vector<Element> initial_left(S.size()), initial_right(S.size());
//    std::vector<int> keymap_space_left(S.size()), keymap_space_right(S.size());
    Element *initial_left = new Element[S.size()];
    Element *initial_right = new Element[S.size()];
    int *keymap_space_left = new int[S.size()];
    int *keymap_space_right = new int[S.size()];

    int left_count = 0, right_count = 0;
    for (int i=0; i < S.size(); i++)
    {
        if (S[i] > L[i]) initial_left[left_count++] = Element(A[i][S[i]-1], i);
        if (S[i] < R[i]) initial_right[right_count++] = Element(A[i][S[i]], i);
    }

    DynamicMinMax<Element, CompareGreater> Left(initial_left, left_count, keymap_space_left, S.size());
    DynamicMinMax<Element, CompareLess> Right(initial_right, right_count, keymap_space_right, S.size());


    while (!Left.empty() && !Right.empty())
    {
        Element lmax = Left.first();
        Element rmin = Right.first();

        // check if we converged
        if (lmax.value() < rmin.value())
            break;

        R[lmax.key()] = S[lmax.key()] - 1;
        L[rmin.key()] = S[rmin.key()] + 1;

        int delta_max = (R[lmax.key()] - L[lmax.key()]) / 2;
        int delta_min = (R[rmin.key()] - R[rmin.key()]) / 2;

        int delta = std::min(delta_min, delta_max);
        S[lmax.key()] = R[lmax.key()] - delta;
        S[rmin.key()] = L[rmin.key()] + delta;

        // update dynamic data structures
        std::vector<int> key_updates = { lmax.key(), rmin.key() };
        for (auto k : key_updates)
        {
            if (S[k] > L[k]) Left.update(k, A[k][S[k]-1]);
            else Left.update(k, -1);        // delete

            if (S[k] < R[k]) Right.update(k, A[k][S[k]]);
            else Right.update(k, -1);       // delete
        }
    }

    delete[] initial_left;
    delete[] initial_right;
    delete[] keymap_space_left;
    delete[] keymap_space_right;

    L = S;
//    return S;
}

bool test_split(const std::vector<int *>& A, const std::vector<int> alen, int p, const std::vector<int>& b)
{
    // flatten, sort A
    std::vector<int> flatA;
    std::vector<int> test;
    for (int i = 0; i < A.size(); i++) {
        for (int j = 0; j < alen[i]; j++)
            flatA.push_back(A[i][j]);
        for (int j = 0; j < b[i]; j++)
            test.push_back(A[i][j]);
    }
    std::sort(flatA.begin(), flatA.end());
    std::sort(test.begin(), test.end());

    // print out the sorted ref
    /*
    std::cout << "Ref:" << std::endl;
    for (int i = 0; i < p; i++)
        std::cout << flatA[i] << " ";
    std::cout << std::endl;

    std::cout << "Test:" << std::endl;
    for (int i = 0; i < test.size(); i++)
        std::cout << test[i] << " ";
    std::cout << std::endl;
    */

    if (test.size() != p) {
//        std::cout << "test_size = " << test.size() << ", p = " << p << std::endl;
        return false;
    }

    for (int i = 0; i < p; i++)
        if (test[i] != flatA[i]) {
//            std::cout << "mismatch at " << i << ": " << test[i] << " != " << flatA[i] << std::endl;
            return false;
        }

    return true;
}


int main(int argc, char **argv)
{
    /*
    std::vector<Element> v;
    v.push_back(Element(25, 0));
    v.push_back(Element(31, 1));
    v.push_back(Element(18, 2));
    v.push_back(Element(44, 3));
    v.push_back(Element(55, 4));
    v.push_back(Element(9, 5));
    v.push_back(Element(90, 6));
    v.push_back(Element(19, 7));
    DynamicMinMax<Element, CompareGreater> d(v, v.size());

    Element e = d.first();
    std::cout << "first: " << e.mValue << ", " << e.mId << std::endl;

    d.update(e.key(), 22);
    e = d.first();
    std::cout << "first: " << e.mValue << ", " << e.mId << std::endl;
    */



    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <mtx file>" << std::endl;
        return -1;
    }

    MatrixMarket mat(argv[1]);

    /*
    std::vector<std::vector<int>> Av = {
        {5, 8, 11, 15, 17, 20, 25, 29, 33, 34, 36, 45, 49, 55, 58, 60, 65 },
        {3, 21, 25, 29, 30, 33, 38, 41, 44, 48, 52, 55, 60, 75, 81, 90 },
        {2, 5, 7, 10, 13, 18, 22, 29, 30, 35, 40},
        {1, 2, 3, 8, 9, 12, 15, 22, 31, 32, 36, 44, 48, 49, 52, 56, 62, 66, 68, 71, 79, 80, 88, 90, 100},
        {1, 9, 22, 23, 24, 29, 39, 45, 48}
    };
    std::vector<int *> A = {
        Av[0].data(), Av[1].data(), Av[2].data(), Av[3].data(), Av[4].data() };
    std::vector<int> alen = { Av[0].size(), Av[1].size(), Av[2].size(), Av[3].size(), Av[4].size() };

    int N = 0;
    for (int i = 0; i < Av.size(); i++)
        N += Av[i].size();
    std::cout << "N = " << N << std::endl;

    int split_point = 20;
//    std::vector<int> b = split_matrix(A, alen, split_point, N);
    std::vector<int> b = split_matrix_one_heap(A, alen, split_point, N);
    bool success = test_split(A, alen, split_point, b);
    if (!success)
        std::cout << "FAILED" << std::endl;
    else
        std::cout << "PASSED" << std::endl;
        */


#if 1
    constexpr int BLOCK_SIZE = 2048;
    int next_block = BLOCK_SIZE;
    int total = 0;
    int splits = 0;
    std::vector<int> b;
    int last_row = -1;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < mat.mRows; i++)
    {
        int row_size = 0;
        std::vector<int> alen;
        std::vector<int *> A;
        for (int j = mat.mRowPtrs[i]; j < mat.mRowPtrs[i+1]; j++) {
            int b = mat.mColIdx[j] - 1;
            int s = mat.mRowPtrs[b+1] - mat.mRowPtrs[b];
            alen.push_back(s);
            A.push_back(mat.mColIdx.data() + mat.mRowPtrs[b]);
            row_size += s;
            total += s;
        }
//        std::cout << "row_size = " << row_size << ", total = " << total << ", next_block = " << next_block << std::endl;

        while (next_block <= total)
        {
            // reset b
            if (last_row != i) {
//                std::cout << "resetting/resizing b" << std::endl;
//                b.resize(A.size());
                b.assign(A.size(), 0);
            }
            int split_pt = next_block - (total - row_size);
//            std::cout << "split at " << i << ", " << split_pt << " / " << row_size << ", m = " << A.size() << std::endl;
            if (split_pt == row_size)
                b = alen;
            else
                split_matrix(A, alen, split_pt, row_size, b);
            #ifdef TEST_SPLITS
            bool success = test_split(A, alen, split_pt, b);
            if (!success) {
                std::cerr << "FAILED";
                return -1;
            }
            #endif
            next_block += BLOCK_SIZE;
            splits++;
            last_row = i;
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(finish - start).count(); 
    std::cout << "Splits: " << splits << std::endl;
    std::cout << "Elapsed: " << elapsed << " ms" << std::endl;
#endif
}
