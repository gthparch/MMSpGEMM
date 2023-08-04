#ifndef MATRIX_MARKET_H_
#define MATRIX_MARKET_H_

#include <vector>
#include <string>

class MatrixMarket
{
public:
    MatrixMarket(const std::string& fname);

//protected:
    std::vector<size_t> sort_indexes();

    int mRows, mCols; //, mNNZ;

    // COO basically
    std::vector<int> mRowCoords, mColCoords;
    std::vector<float> mValues;

    // CSR
    std::vector<int> mRowPtrs;
    std::vector<int> mColIdx;
    std::vector<float> mCSRVals;
};


#endif
