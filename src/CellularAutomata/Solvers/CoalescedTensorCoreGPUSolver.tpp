#include "CellularAutomata/Solvers/CoalescedTensorCoreGPUSolver.cuh"

template <typename T>
void CoalescedTensorCoreGPUSolver<T>::setupBlockSize() {
    this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
    this->boundaryBlock = dim3(16, 16);
    this->castingKernelsBlockSize = dim3(16, 16);
}
template <typename T>
void CoalescedTensorCoreGPUSolver<T>::setupGridSize() {
    int n = this->dataDomainDevice->getSideLengthWithoutHalo();
    this->GPUGrid = dim3((n + (NREGIONS_H * 16) - 1) / (NREGIONS_H * 16), (n + (NREGIONS_V * 16) - 1) / (NREGIONS_V * 16));
    this->castingKernelsGridSize = dim3((this->dataDomainDevice->getSideLength() + this->castingKernelsBlockSize.x - 1) / this->castingKernelsBlockSize.x, (this->dataDomainDevice->getSideLength() + this->castingKernelsBlockSize.y - 1) / this->castingKernelsBlockSize.y);
    this->horizontalBoundaryGrid = dim3(2 * (int)ceil(n / (float)this->boundaryBlock.x));
    this->verticalBoundaryGrid = dim3(2 * (int)ceil((this->dataDomainDevice->getSideLength()) / (float)this->boundaryBlock.x));
}

template <typename T>
void CoalescedTensorCoreGPUSolver<T>::createStream() {
    cudaStreamCreate(&mainStream);
}

template <typename T>
void CoalescedTensorCoreGPUSolver<T>::setupSharedMemoryCarveout() {
    sharedMemoryBytes = ((NREGIONS_H + 2) * (NREGIONS_V + 2) * 16 * 16 * 2 + 256 * 2) * sizeof(FTYPE);
    cudaFuncSetAttribute(TensorCoalescedV1GoLStep, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemoryBytes);
}

template <typename T>
void CoalescedTensorCoreGPUSolver<T>::moveCurrentDeviceStateToGPUBuffer() {
    convertFp16ToFp32AndUndoChangeLayout<<<this->castingKernelsGridSize, this->castingKernelsBlockSize>>>(this->visibleDataDevice->getData(), this->dataDomainDevice->getData(), this->dataDomainDevice->getSideLength());
}

template <typename T>
void CoalescedTensorCoreGPUSolver<T>::moveGPUBufferToCurrentDeviceState() {
    convertFp32ToFp16AndDoChangeLayout<<<this->castingKernelsGridSize, this->castingKernelsBlockSize>>>(this->dataDomainDevice->getData(), this->visibleDataDevice->getData(), this->dataDomainDevice->getSideLength());
}

template <typename T>
void CoalescedTensorCoreGPUSolver<T>::fillHorizontalBoundaryConditions() {
    int n = this->dataDomainDevice->getSideLengthWithoutHalo();
    copyHorizontalHaloCoalescedVersion<<<this->horizontalBoundaryGrid, this->boundaryBlock>>>(this->dataDomainDevice->getData(), n, this->dataDomainDevice->getSideLength());
}

template <typename T>
void CoalescedTensorCoreGPUSolver<T>::fillVerticalBoundaryConditions() {
    int n = this->dataDomainDevice->getSideLengthWithoutHalo();
    copyVerticalHaloCoalescedVersion<<<this->verticalBoundaryGrid, this->boundaryBlock>>>(this->dataDomainDevice->getData(), n, this->dataDomainDevice->getSideLength());
}

template <typename T>
void CoalescedTensorCoreGPUSolver<T>::CAStepAlgorithm() {
    int n = this->dataDomainDevice->getSideLengthWithoutHalo();
    TensorCoalescedV1GoLStep<<<this->GPUGrid, this->GPUBlock, sharedMemoryBytes, mainStream>>>(this->dataDomainDevice->getData(), this->dataDomainBufferDevice->getData(), n, this->dataDomainDevice->getSideLength());
    (cudaDeviceSynchronize());
}