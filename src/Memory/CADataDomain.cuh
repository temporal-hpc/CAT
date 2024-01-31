#pragma once

#include "Memory/Allocators/Allocator.cuh"
#include "Memory/DataDomain.cuh"

template <typename T>
class CADataDomain : public DataDomain<T> {
   protected:
    int fullHorizontalSize;
    int fullVerticalSize;

    int horizontalHaloSize;
    int verticalHaloSize;

    size_t totalSize;

    T* data;
    Allocator<T>* allocator;

   public:
    CADataDomain(Allocator<T>* pAllocator, int pSideLengthWithoutHalo, int pHaloWidth);
    CADataDomain(Allocator<T>* pAllocator, int pHorizontalSideLengthWithoutHalo, int pVerticalSideLengthWithoutHalo, int pHorizontalHaloWidth, int pVerticalHaloWidth);

    void allocate() override;
    void free() override;

    T* getData() override;
    size_t getStride() override;
    size_t getTotalSize() override;

    // this functions below could be in a CADataAccesor class, but this also should not
    // be mutable, so it is not necessary
    T getElementAt(size_t index) override;
    void setElementAt(size_t index, T value) override;

    int getHorizontalHaloSize();
    int getVerticalHaloSize();

    int getFullHorizontalSize();
    int getFullVerticalSize();

    int getInnerHorizontalSize();
    int getInnerVerticalSize();

    T getInnerElementAt(int i, int j);
    void setInnerElementAt(int i, int j, T value);
};

#include "Memory/CADataDomain.tpp"
