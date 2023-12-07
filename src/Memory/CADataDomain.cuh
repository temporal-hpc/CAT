#pragma once

#include "Memory/Allocators/Allocator.cuh"
#include "Memory/DataDomain.cuh"

template <typename T>
class CADataDomain : public DataDomain<T> {
   private:
    int sideLength;
    int haloWidth;
    size_t totalSize;

    T* data;
    Allocator<T>* allocator;

   public:
    CADataDomain(Allocator<T>* pAllocator, int pSideLength, int pHaloWidth);

    void allocate() override;
    void free() override;

    T* getData() override;
    int getSideLength() override;
    size_t getTotalSize() override;

    // this functions below could be in a CADataAccesor class, but this also should not
    // be mutable, so it is not necessary
    T getElementAt(size_t index) override;
    void setElementAt(size_t index, T value) override;

    int getHaloWidth();
    int getSideLengthWithoutHalo();
    T getInnerElementAt(int i, int j);
    void setInnerElementAt(int i, int j, T value);
};

#include "Memory/CADataDomain.tpp"
