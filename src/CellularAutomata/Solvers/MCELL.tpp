#include "CellularAutomata/Solvers/MCELL.cuh"
#include "Debug.h"

template <typename T> void MCELL<T>::setupBlockSize()
{
    this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
    this->boundaryBlock = dim3(256);
    lDebug(0, "Block size: %d, %d\n", this->GPUBlock.x, this->GPUBlock.y);
    lDebug(0, "Boundary block size: %d\n", this->boundaryBlock.x);
}

template <typename T> void MCELL<T>::setupGridSize()
{
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    int block_x = (this->GPUBlock.x * this->cellsPerThread);
    int block_y = this->GPUBlock.y;

    lDebug(0, "Effective block: %d, %d", block_x, block_y);

    this->GPUGrid = dim3((n + block_x - 1) / block_x, (n + block_y - 1) / block_y);
    this->horizontalBoundaryGrid = dim3((int)ceil(n / (float)this->boundaryBlock.x));
    this->verticalBoundaryGrid =
        dim3((int)ceil((this->dataDomainDevice->getFullHorizontalSize()) / (float)this->boundaryBlock.x));

    lDebug(0, "Grid size: %d, %d", this->GPUGrid.x, this->GPUGrid.y);
    lDebug(0, "Horizontal boundary grid size: %d", this->horizontalBoundaryGrid.x);
    lDebug(0, "Vertical boundary grid size: %d", this->verticalBoundaryGrid.x);
}

template <typename T> void MCELL<T>::moveCurrentDeviceStateToGPUBuffer()
{
    copyFromMTYPEAndCast<<<this->GPUGrid, this->GPUBlock>>>(this->dataDomainDevice->getData(),
                                                            this->visibleDataDevice->getData(),
                                                            this->dataDomainDevice->getFullHorizontalSize());
}

template <typename T> void MCELL<T>::moveGPUBufferToCurrentDeviceState()
{
    copyToMTYPEAndCast<<<this->GPUGrid, this->GPUBlock>>>(this->visibleDataDevice->getData(),
                                                          this->dataDomainDevice->getData(),
                                                          this->dataDomainDevice->getFullHorizontalSize());
}

template <typename T> void MCELL<T>::fillHorizontalBoundaryConditions()
{
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    copy_Rows<<<this->horizontalBoundaryGrid, this->boundaryBlock>>>(
        n, this->dataDomainDevice->getData(), RADIUS, 2 * this->dataDomainDevice->getHorizontalHaloSize());
}

template <typename T> void MCELL<T>::fillVerticalBoundaryConditions()
{
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    copy_Cols<<<this->verticalBoundaryGrid, this->boundaryBlock>>>(n, this->dataDomainDevice->getData(), RADIUS,
                                                                   2 * this->dataDomainDevice->getHorizontalHaloSize());
}

template <typename T> void MCELL<T>::CAStepAlgorithm()
{
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    // moveKernel2<<<this->GPUGrid, this->GPUBlock, sharedMemorySize>>>(this->dataDomainDevice->getData(),
    // this->dataDomainBufferDevice->getData(), n, n, this->cellsPerThread, RADIUS, 2 *
    // this->dataDomainDevice->getHorizontalHaloSize());
    MCELL_KERNEL<<<this->GPUGrid, this->GPUBlock, sharedMemorySize>>>(
        this->dataDomainDevice->getData(), this->dataDomainBufferDevice->getData(), n, n, this->cellsPerThread, RADIUS,
        2 * this->dataDomainDevice->getHorizontalHaloSize());

    (cudaDeviceSynchronize());
}
