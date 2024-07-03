#include <stdio.h>
#include "Memory/CADataDomain.cuh"
template <typename T>
CADataDomain<T>::CADataDomain(Allocator<T>* pAllocator, int pHorizontalSideLengthWithoutHalo, int pHaloWidth) {
    this->horizontalHaloSize = pHaloWidth;
    this->verticalHaloSize = pHaloWidth;

    this->fullHorizontalSize = pHorizontalSideLengthWithoutHalo + 2 * pHaloWidth;
    this->fullVerticalSize = pHorizontalSideLengthWithoutHalo + 2 * pHaloWidth;

    this->totalSize = (this->fullHorizontalSize) * (size_t)(this->fullVerticalSize);
    this->allocator = pAllocator;
}
template <typename T>
CADataDomain<T>::CADataDomain(Allocator<T>* pAllocator, int pHorizontalSideLengthWithoutHalo, int pVerticalSideLengthWithoutHalo, int pHorizontalHaloWidth, int pVerticalHaloWidth) {
    this->horizontalHaloSize = pHorizontalHaloWidth;
    this->verticalHaloSize = pVerticalHaloWidth;

    this->fullHorizontalSize = pHorizontalSideLengthWithoutHalo + 2 * pHorizontalHaloWidth;
    this->fullVerticalSize = pVerticalSideLengthWithoutHalo + 2 * pVerticalHaloWidth;

    this->totalSize = (this->fullHorizontalSize) * (size_t)(this->fullVerticalSize);
    this->allocator = pAllocator;
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
size_t CADataDomain<T>::getStride() {
    return (size_t)this->fullHorizontalSize;
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
int CADataDomain<T>::getHorizontalHaloSize() {
    return this->horizontalHaloSize;
}
template <typename T>
int CADataDomain<T>::getVerticalHaloSize() {
    return this->verticalHaloSize;
}

template <typename T>
int CADataDomain<T>::getFullHorizontalSize() {
    return this->fullHorizontalSize;
};

template <typename T>
int CADataDomain<T>::getFullVerticalSize() {
    return this->fullVerticalSize;
};

template <typename T>
int CADataDomain<T>::getInnerHorizontalSize() {
    return this->fullHorizontalSize - 2 * this->horizontalHaloSize;
};

template <typename T>
int CADataDomain<T>::getInnerVerticalSize() {
    return this->fullVerticalSize - 2 * this->verticalHaloSize;
};

template <typename T>
T CADataDomain<T>::getInnerElementAt(int i, int j) {
    return this->data[(i + this->verticalHaloSize) * getStride() + (j + this->horizontalHaloSize)];
}

template <typename T>
void CADataDomain<T>::setInnerElementAt(int i, int j, T value) {
    this->data[(i + this->verticalHaloSize) * getStride() + (j + this->horizontalHaloSize)] = value;
}
