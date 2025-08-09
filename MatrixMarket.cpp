
#include "MatrixMarket.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>
#include <algorithm>


MatrixMarket::MatrixMarket(const std::string& fname)
{
    std::ifstream ifs(fname);

    // check ifs good or throw exception
    if (!ifs.good()) {
        std::cerr << "failed to open: " << fname << std::endl;
        exit(1);
    }

    // assume that only comments are at the beginning of the file

    // read the first line to see if symmetric
    // TODO: check if this is the '%%MatrixMarket' line
    std::string header, matrix, coordinate, real, symmetric;
    ifs >> header >> matrix >> coordinate >> real >> symmetric;

    bool is_symmetric(false);
    if (symmetric.compare("symmetric") == 0)
        is_symmetric = true;

    bool is_pattern(false);
    if (real.compare("pattern") == 0)
        is_pattern = true;

    std::string line;
    std::getline(ifs, line);        // read the trailing newline on the header line above
    while (std::getline(ifs, line) && line[0] == '%')
        ;

    // first data line is the rows, cols, nnz (or nnz / 2, if symmetric)
    std::istringstream iss(line);
    int nLines, row, col;
    float value;
    iss >> mRows >> mCols >> nLines;

    if (!is_pattern)
    {
        while (ifs >> row >> col >> value)
        {
            mRowCoords.push_back(row);
            mColCoords.push_back(col);
            mValues.push_back(value);

            // assuming symmetric
            if (is_symmetric && row != col) {
                mRowCoords.push_back(col);
                mColCoords.push_back(row);
                mValues.push_back(value);
            }

            nLines--;
        }
    }
    else 
    {
        while (ifs >> row >> col)
        {
            mRowCoords.push_back(row);
            mColCoords.push_back(col);
            // make up a value
            mValues.push_back(0.1234);

            // assuming symmetric
            if (is_symmetric && row != col) {
                mRowCoords.push_back(col);
                mColCoords.push_back(row);
                mValues.push_back(0.1234);
            }

            nLines--;
        }
    }

    std::cout << "Matrix dimensions: " << mRows << " x " << mCols << ", nnz = " << mValues.size() << std::endl;

    // Convert to CSR

    std::vector<size_t> idxs = sort_indexes();
    int lastRow = 0;
    for (size_t i = 0; i < idxs.size(); i++)
    {
        mCSRVals.push_back(mValues[idxs[i]]);
        mColIdx.push_back(mColCoords[idxs[i]]);
        if (lastRow != mRowCoords[idxs[i]]) {
            // Handling empty rows
            if (mRowCoords[idxs[i]] != lastRow + 1) {
//                std::cout << "empty row. lastRow: " << lastRow << ", mRowCoords: " << mRowCoords[idxs[i]] << std::endl;
                for (int j=lastRow; j < mRowCoords[idxs[i]]-1; j++)
                    mRowPtrs.push_back(i);
            }
            mRowPtrs.push_back(i);
            lastRow = mRowCoords[idxs[i]];
        }
    }
    mRowPtrs.push_back(mValues.size());

    // Also store CSC for inner-product and outer-product traversal
}


std::vector<size_t> MatrixMarket::sort_indexes()
{
    std::vector<size_t> idx(mValues.size());
    std::iota(idx.begin(), idx.end(), 0);

    // also sort by column index secondarily
    std::stable_sort(idx.begin(), idx.end(), [this](size_t a, size_t b) {return (((uint64_t)mRowCoords[a] << 32) + (uint64_t)mColCoords[a]) < (((uint64_t)mRowCoords[b] << 32) + (uint64_t)mColCoords[b]); });

    return idx;
}
