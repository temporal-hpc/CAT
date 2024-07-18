#include "Memory/Allocators/CPUAllocator.cuh"

template <typename T>
T* CPUAllocator<T>::allocate(size_t size) {
    lDebug(1, "Allocating %llu bytes on CPU\n", size * sizeof(T));
    T* ptr = nullptr;
    ptr = (T*)malloc(size * sizeof(T));
    return ptr;
}

template <typename T>
void CPUAllocator<T>::deallocate(void* ptr) {
    delete (ptr);
}
