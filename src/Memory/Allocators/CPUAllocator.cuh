#pragma once

#include "Memory/Allocators/Allocator.cuh"

template <typename T>
class CPUAllocator : Allocator<T> {
   public:
    T* allocate(size_t size) override;
    void deallocate(void* ptr) override;
};

#include "Memory/Allocators/CPUAllocator.tpp"