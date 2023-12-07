#include <stdio.h>
#include "Memory/CADataDomain.cuh"
template <typename T>
CADataDomain<T>::CADataDomain(Allocator<T>* pAllocator, int pSideLength, int pHaloWidth) {
    this->haloWidth = pHaloWidth;
    this->sideLength = pSideLength + 2 * pHaloWidth;
    this->totalSize = (this->sideLength) * (this->sideLength);
    this->allocator = pAllocator;
}

template <typename T>
int CADataDomain<T>::getHaloWidth() {
    return haloWidth;
}

template <typename T>
void CADataDomain<T>::allocate() {
    data = allocator->allocate(totalSize);
}

template <typename T>
void CADataDomain<T>::free() {
    allocator->deallocate(data);
}

template <typename T>
T* CADataDomain<T>::getData() {
    return data;
}

template <typename T>
int CADataDomain<T>::getSideLength() {
    return sideLength;
};

template <typename T>
size_t CADataDomain<T>::getTotalSize() {
    return totalSize;
};

template <typename T>
T CADataDomain<T>::getElementAt(size_t index) {
    return data[index];
};

template <typename T>
void CADataDomain<T>::setElementAt(size_t index, T value) {
    data[index] = value;
};

template <typename T>
int CADataDomain<T>::getSideLengthWithoutHalo() {
    return sideLength - 2 * haloWidth;
};

template <typename T>
T CADataDomain<T>::getInnerElementAt(int i, int j) {
    return data[(i + haloWidth) * sideLength + (j + haloWidth)];
}

template <typename T>
void CADataDomain<T>::setInnerElementAt(int i, int j, T value) {
    data[(i + haloWidth) * sideLength + (j + haloWidth)] = value;
}
