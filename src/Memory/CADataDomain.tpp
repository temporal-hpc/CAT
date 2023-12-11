#include <stdio.h>
#include "Memory/CADataDomain.cuh"
template <typename T>
CADataDomain<T>::CADataDomain(Allocator<T>* pAllocator, int pSideLengthWithoutHalo, int pHaloWidth) {
    this->haloWidth = pHaloWidth;
    this->sideLength = pSideLengthWithoutHalo + 2 * pHaloWidth;
    this->totalSize = (this->sideLength) * (this->sideLength);
    this->allocator = pAllocator;
}

template <typename T>
int CADataDomain<T>::getHaloWidth() {
    return this->haloWidth;
}

template <typename T>
void CADataDomain<T>::allocate() {
    this->data = this->allocator->allocate(this->totalSize);
}

template <typename T>
void CADataDomain<T>::free() {
    this->allocator->deallocate(this->data);
}

template <typename T>
T* CADataDomain<T>::getData() {
    return this->data;
}

template <typename T>
int CADataDomain<T>::getSideLength() {
    return this->sideLength;
};

template <typename T>
size_t CADataDomain<T>::getTotalSize() {
    return this->totalSize;
};

template <typename T>
T CADataDomain<T>::getElementAt(size_t index) {
    return this->data[index];
};

template <typename T>
void CADataDomain<T>::setElementAt(size_t index, T value) {
    this->data[index] = value;
};

template <typename T>
int CADataDomain<T>::getSideLengthWithoutHalo() {
    return this->sideLength - 2 * this->haloWidth;
};

template <typename T>
T CADataDomain<T>::getInnerElementAt(int i, int j) {
    return this->data[(i + this->haloWidth) * this->sideLength + (j + this->haloWidth)];
}

template <typename T>
void CADataDomain<T>::setInnerElementAt(int i, int j, T value) {
    this->data[(i + this->haloWidth) * this->sideLength + (j + this->haloWidth)] = value;
}
