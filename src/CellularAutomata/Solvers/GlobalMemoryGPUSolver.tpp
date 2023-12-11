#include "CellularAutomata/Solvers/GlobalMemoryGPUSolver.cuh"

template <typename T>
void GlobalMemoryGPUSolver<T>::setupBlockSize() {
    this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
    this->boundaryBlock = dim3(256);
}
template <typename T>
void GlobalMemoryGPUSolver<T>::setupGridSize() {
    int n = this->dataDomainDevice->getSideLengthWithoutHalo();
    this->GPUGrid = dim3((n + this->GPUBlock.x - 1) / this->GPUBlock.x, (n + this->GPUBlock.y - 1) / this->GPUBlock.y);
    this->horizontalBoundaryGrid = dim3((int)ceil(n / (float)this->boundaryBlock.x));
    this->verticalBoundaryGrid = dim3((int)ceil((this->dataDomainDevice->getSideLength()) / (float)this->boundaryBlock.x));
}

template <typename T>
void GlobalMemoryGPUSolver<T>::moveCurrentDeviceStateToGPUBuffer() {
    copyFromMTYPEAndCast<<<this->GPUGrid, this->GPUBlock>>>(this->dataDomainDevice->getData(), this->visibleDataDevice->getData(), this->dataDomainDevice->getSideLength());
}

template <typename T>
void GlobalMemoryGPUSolver<T>::moveGPUBufferToCurrentDeviceState() {
    copyToMTYPEAndCast<<<this->GPUGrid, this->GPUBlock>>>(this->visibleDataDevice->getData(), this->dataDomainDevice->getData(), this->dataDomainDevice->getSideLength());
}

template <typename T>
void GlobalMemoryGPUSolver<T>::fillHorizontalBoundaryConditions() {
    int n = this->dataDomainDevice->getSideLengthWithoutHalo();
    copyHorizontalHalo<<<this->horizontalBoundaryGrid, this->boundaryBlock>>>(this->dataDomainDevice->getData(), n, this->dataDomainDevice->getSideLength());
}

template <typename T>
void GlobalMemoryGPUSolver<T>::fillVerticalBoundaryConditions() {
    int n = this->dataDomainDevice->getSideLengthWithoutHalo();
    copyVerticalHalo<<<this->verticalBoundaryGrid, this->boundaryBlock>>>(this->dataDomainDevice->getData(), n, this->dataDomainDevice->getSideLength());
}

template <typename T>
void GlobalMemoryGPUSolver<T>::CAStepAlgorithm() {
    int n = this->dataDomainDevice->getSideLengthWithoutHalo();
    ClassicGlobalMemGoLStep<<<this->GPUGrid, this->GPUBlock>>>(this->dataDomainDevice->getData(), this->dataDomainBufferDevice->getData(), n, this->dataDomainDevice->getSideLength());
    (cudaDeviceSynchronize());
}