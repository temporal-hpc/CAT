#include "Memory/Allocators/GPUAllocator.cuh"

template <typename T>
T* GPUAllocator<T>::allocate(size_t size) {
    T* ptr = nullptr;
    cudaMalloc(&ptr, size * sizeof(T));
    return ptr;
}

template <typename T>
void GPUAllocator<T>::deallocate(void* ptr) {
    cudaFree(ptr);
}
