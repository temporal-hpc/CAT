#include "CellularAutomata/Solvers/SharedMemoryGPUSolver.cuh"

template <typename T>
void SharedMemoryGPUSolver<T>::setupBlockSize() {
    this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
    this->boundaryBlock = dim3(256);
}
template <typename T>
void SharedMemoryGPUSolver<T>::setupGridSize() {
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    this->GPUGrid = dim3((n + 80 - 1) / 80, (n + 80 - 1) / 80);
    this->horizontalBoundaryGrid = dim3((int)ceil(n / (float)this->boundaryBlock.x));
    this->verticalBoundaryGrid = dim3((int)ceil((this->dataDomainDevice->getFullHorizontalSize()) / (float)this->boundaryBlock.x));
}

template <typename T>
void SharedMemoryGPUSolver<T>::moveCurrentDeviceStateToGPUBuffer() {
    copyFromMTYPEAndCast<<<this->GPUGrid, this->GPUBlock>>>(this->dataDomainDevice->getData(), this->visibleDataDevice->getData(), this->dataDomainDevice->getFullHorizontalSize());
}

template <typename T>
void SharedMemoryGPUSolver<T>::moveGPUBufferToCurrentDeviceState() {
    copyToMTYPEAndCast<<<this->GPUGrid, this->GPUBlock>>>(this->visibleDataDevice->getData(), this->dataDomainDevice->getData(), this->dataDomainDevice->getFullHorizontalSize());
}

template <typename T>
void SharedMemoryGPUSolver<T>::fillHorizontalBoundaryConditions() {
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    copyHorizontalHalo<<<this->horizontalBoundaryGrid, this->boundaryBlock>>>(this->dataDomainDevice->getData(), n, this->dataDomainDevice->getFullHorizontalSize());
}

template <typename T>
void SharedMemoryGPUSolver<T>::fillVerticalBoundaryConditions() {
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    copyVerticalHalo<<<this->verticalBoundaryGrid, this->boundaryBlock>>>(this->dataDomainDevice->getData(), n, this->dataDomainDevice->getFullHorizontalSize());
}

template <typename T>
void SharedMemoryGPUSolver<T>::CAStepAlgorithm() {
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    ClassicV1GoLStep<<<this->GPUGrid, this->GPUBlock>>>(this->dataDomainDevice->getData(), this->dataDomainBufferDevice->getData(), n, this->dataDomainDevice->getFullHorizontalSize());
    (cudaDeviceSynchronize());
}