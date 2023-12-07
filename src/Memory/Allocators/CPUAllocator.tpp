#include "Memory/Allocators/CPUAllocator.cuh"

template <typename T>
T* CPUAllocator<T>::allocate(size_t size) {
    T* ptr = nullptr;
    ptr = (T*)malloc(size * sizeof(T));
    return ptr;
}

template <typename T>
void CPUAllocator<T>::deallocate(void* ptr) {
    delete (ptr);
}
