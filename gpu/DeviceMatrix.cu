#include "DeviceMatrix.h"
#include <iostream>

DeviceMatrix::DeviceMatrix(const MatrixMarket& m, mgpu::standard_context_t& ctx)
{
    std::cout << "Transferring matrix to device ..." << std::endl;
    row_ptrs = mgpu::to_mem(m.mRowPtrs, ctx);
    values = mgpu::to_mem(m.mCSRVals, ctx);
    col_idx = mgpu::to_mem(m.mColIdx, ctx);

    raw.d_row_ptrs = row_ptrs.data();
    raw.d_values = values.data();
    raw.d_col_idx = col_idx.data();
}
