#include "CellularAutomata/Solvers/SHARED.cuh"
#include "Debug.h"

template <typename T> void SHARED<T>::setupBlockSize()
{
    this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
    this->boundaryBlock = dim3(256);
    lDebug(0, "Block size: %d, %d\n", this->GPUBlock.x, this->GPUBlock.y);
    lDebug(0, "Boundary block size: %d\n", this->boundaryBlock.x);
}
template <typename T> void SHARED<T>::setupGridSize()
{
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    this->GPUGrid = dim3(ceil((float)n / (float)(this->GPUBlock.x)), ceil((float)n / (float)(this->GPUBlock.y)));
    this->horizontalBoundaryGrid = dim3((int)ceil(n / (float)this->boundaryBlock.x));
    this->verticalBoundaryGrid =
        dim3((int)ceil((this->dataDomainDevice->getFullHorizontalSize()) / (float)this->boundaryBlock.x));

    lDebug(0, "Grid size: %d, %d", this->GPUGrid.x, this->GPUGrid.y);
    lDebug(0, "Horizontal boundary grid size: %d", this->horizontalBoundaryGrid.x);
    lDebug(0, "Vertical boundary grid size: %d", this->verticalBoundaryGrid.x);
}

template <typename T> void SHARED<T>::moveCurrentDeviceStateToGPUBuffer()
{
    copyFromMTYPEAndCast<<<this->GPUGrid, this->GPUBlock>>>(this->dataDomainDevice->getData(),
                                                            this->visibleDataDevice->getData(),
                                                            this->dataDomainDevice->getFullHorizontalSize());
}

template <typename T> void SHARED<T>::moveGPUBufferToCurrentDeviceState()
{
    copyToMTYPEAndCast<<<this->GPUGrid, this->GPUBlock>>>(this->visibleDataDevice->getData(),
                                                          this->dataDomainDevice->getData(),
                                                          this->dataDomainDevice->getFullHorizontalSize());
}

template <typename T> void SHARED<T>::fillHorizontalBoundaryConditions()
{
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    copy_Rows<<<this->horizontalBoundaryGrid, this->boundaryBlock>>>(
        n, this->dataDomainDevice->getData(), RADIUS, 2 * this->dataDomainDevice->getHorizontalHaloSize());
}

template <typename T> void SHARED<T>::fillVerticalBoundaryConditions()
{
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    copy_Cols<<<this->verticalBoundaryGrid, this->boundaryBlock>>>(n, this->dataDomainDevice->getData(), RADIUS,
                                                                   2 * this->dataDomainDevice->getHorizontalHaloSize());
}

template <typename T> void SHARED<T>::CAStepAlgorithm()
{
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    SHARED_KERNEL<<<this->GPUGrid, this->GPUBlock, sharedMemorySize>>>(
        this->dataDomainDevice->getData(), this->dataDomainBufferDevice->getData(), n, n, RADIUS, 2 * RADIUS);

    (cudaDeviceSynchronize());
}
