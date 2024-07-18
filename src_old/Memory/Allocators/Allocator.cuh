#pragma once
#include <cstddef>

template <typename T>
class Allocator {
   public:
    virtual T* allocate(size_t size) = 0;
    virtual void deallocate(void* ptr) = 0;
};