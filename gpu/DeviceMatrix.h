#ifndef DEVICE_MATRIX_H
#define DEVICE_MATRIX_H

#include <moderngpu/transform.hxx>
#include <moderngpu/memory.hxx>
#include "MatrixMarket.h"


class RawDeviceMatrix
{
public:
    int *d_row_ptrs;
    float *d_values;
    int *d_col_idx;
};

class DeviceMatrix
{
public:
    explicit DeviceMatrix(const MatrixMarket& m, mgpu::standard_context_t& ctx);

    mgpu::mem_t<int> row_ptrs;
    mgpu::mem_t<float> values;
    mgpu::mem_t<int> col_idx;

    // These are device pointers
    RawDeviceMatrix raw;
};



#endif
