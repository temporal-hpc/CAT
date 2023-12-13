#include "CellularAutomata/Solvers/MillanCellsGPUSolver.cuh"
#include "Debug.h"

template <typename T>
void MillanCellsGPUSolver<T>::setupBlockSize() {
    this->GPUBlock = dim3(BSIZE3DX / this->cellsPerThread, BSIZE3DY);
    this->boundaryBlock = dim3(256);
    lDebug(0, "Block size: %d, %d\n", this->GPUBlock.x, this->GPUBlock.y);
    lDebug(0, "Boundary block size: %d\n", this->boundaryBlock.x);
}
template <typename T>
void MillanCellsGPUSolver<T>::setupGridSize() {
    int n = this->dataDomainDevice->getSideLengthWithoutHalo();
    this->GPUGrid = dim3(ceil((float)n / (float)(this->GPUBlock.x - 2 * this->dataDomainDevice->getHaloWidth())), ceil((float)n / (float)(this->GPUBlock.y - 2 * this->dataDomainDevice->getHaloWidth())));
    this->horizontalBoundaryGrid = dim3((int)ceil(n / (float)this->boundaryBlock.x));
    this->verticalBoundaryGrid = dim3((int)ceil((this->dataDomainDevice->getSideLength()) / (float)this->boundaryBlock.x));

    lDebug(0, "Grid size: %d, %d", this->GPUGrid.x, this->GPUGrid.y);
    lDebug(0, "Horizontal boundary grid size: %d", this->horizontalBoundaryGrid.x);
    lDebug(0, "Vertical boundary grid size: %d", this->verticalBoundaryGrid.x);
}

template <typename T>
void MillanCellsGPUSolver<T>::moveCurrentDeviceStateToGPUBuffer() {
    copyFromMTYPEAndCast<<<this->GPUGrid, this->GPUBlock>>>(this->dataDomainDevice->getData(), this->visibleDataDevice->getData(), this->dataDomainDevice->getSideLength());
}

template <typename T>
void MillanCellsGPUSolver<T>::moveGPUBufferToCurrentDeviceState() {
    copyToMTYPEAndCast<<<this->GPUGrid, this->GPUBlock>>>(this->visibleDataDevice->getData(), this->dataDomainDevice->getData(), this->dataDomainDevice->getSideLength());
}

template <typename T>
void MillanCellsGPUSolver<T>::fillHorizontalBoundaryConditions() {
    int n = this->dataDomainDevice->getSideLengthWithoutHalo();
    copy_Rows<<<this->horizontalBoundaryGrid, this->boundaryBlock>>>(n, this->dataDomainDevice->getData(), RADIUS, 2 * this->dataDomainDevice->getHaloWidth());
}

template <typename T>
void MillanCellsGPUSolver<T>::fillVerticalBoundaryConditions() {
    int n = this->dataDomainDevice->getSideLengthWithoutHalo();
    copy_Cols<<<this->verticalBoundaryGrid, this->boundaryBlock>>>(n, this->dataDomainDevice->getData(), RADIUS, 2 * this->dataDomainDevice->getHaloWidth());
}

template <typename T>
void MillanCellsGPUSolver<T>::CAStepAlgorithm() {
    int n = this->dataDomainDevice->getSideLengthWithoutHalo();
    moveKernel<<<this->GPUGrid, this->GPUBlock, sharedMemorySize>>>(this->dataDomainDevice->getData(), this->dataDomainBufferDevice->getData(), n, n, this->cellsPerThread, RADIUS, 2 * this->dataDomainDevice->getHaloWidth());

    (cudaDeviceSynchronize());
}