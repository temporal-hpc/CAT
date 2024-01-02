#pragma once

#include "Memory/Allocators/Allocator.cuh"

template <typename T>
class DataDomain {
   public:
    virtual void allocate() = 0;
    virtual void free() = 0;

    virtual T* getData() = 0;
    virtual int getStride() = 0;
    virtual size_t getTotalSize() = 0;

    virtual T getElementAt(size_t index) = 0;
    virtual void setElementAt(size_t index, T value) = 0;
};