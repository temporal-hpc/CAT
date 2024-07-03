#include "CellularAutomata/Solvers/TensorCoreGPUSolver.cuh"

template <typename T>
void TensorCoreGPUSolver<T>::setupBlockSize() {
    this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
    this->boundaryBlock = dim3(256);
    this->castingKernelsBlockSize = dim3(16, 16);
}
template <typename T>
void TensorCoreGPUSolver<T>::setupGridSize() {
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    this->GPUGrid = dim3((n + (NREGIONS_H * 16) - 1) / (NREGIONS_H * 16), (n + (NREGIONS_V * 16) - 1) / (NREGIONS_V * 16));
    this->castingKernelsGridSize = dim3((this->dataDomainDevice->getFullHorizontalSize() + this->castingKernelsBlockSize.x - 1) / this->castingKernelsBlockSize.x, (this->dataDomainDevice->getFullHorizontalSize() + this->castingKernelsBlockSize.y - 1) / this->castingKernelsBlockSize.y);
    this->horizontalBoundaryGrid = dim3((int)ceil(n / (float)this->boundaryBlock.x));
    this->verticalBoundaryGrid = dim3((int)ceil((this->dataDomainDevice->getFullHorizontalSize()) / (float)this->boundaryBlock.x));
}

template <typename T>
void TensorCoreGPUSolver<T>::setupSharedMemoryCarveout() {
    sharedMemoryBytes = ((NREGIONS_H + 2) * (NREGIONS_V + 2) * 16 * 16 * 2 + 256 * 2) * sizeof(FTYPE);
    cudaFuncSetAttribute(TensorV1GoLStep, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemoryBytes);
}

template <typename T>
void TensorCoreGPUSolver<T>::createStream() {
    cudaStreamCreate(&mainStream);
}

template <typename T>
void TensorCoreGPUSolver<T>::moveCurrentDeviceStateToGPUBuffer() {
    convertFp16ToFp32<<<this->castingKernelsGridSize, this->castingKernelsBlockSize>>>(this->visibleDataDevice->getData(), this->dataDomainDevice->getData(), this->dataDomainDevice->getFullHorizontalSize());
}

template <typename T>
void TensorCoreGPUSolver<T>::moveGPUBufferToCurrentDeviceState() {
    convertFp32ToFp16<<<this->castingKernelsGridSize, this->castingKernelsBlockSize>>>(this->dataDomainDevice->getData(), this->visibleDataDevice->getData(), this->dataDomainDevice->getFullHorizontalSize());
}

template <typename T>
void TensorCoreGPUSolver<T>::fillHorizontalBoundaryConditions() {
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    copyHorizontalHaloTensor<<<this->horizontalBoundaryGrid, this->boundaryBlock>>>(this->dataDomainDevice->getData(), n, this->dataDomainDevice->getFullHorizontalSize());
}

template <typename T>
void TensorCoreGPUSolver<T>::fillVerticalBoundaryConditions() {
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    copyVerticalHaloTensor<<<this->verticalBoundaryGrid, this->boundaryBlock>>>(this->dataDomainDevice->getData(), n, this->dataDomainDevice->getFullHorizontalSize());
}

template <typename T>
void TensorCoreGPUSolver<T>::CAStepAlgorithm() {
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    TensorV1GoLStep<<<this->GPUGrid, this->GPUBlock, sharedMemoryBytes, mainStream>>>(this->dataDomainDevice->getData(), this->dataDomainBufferDevice->getData(), n, this->dataDomainDevice->getFullHorizontalSize());
    (cudaDeviceSynchronize());
}