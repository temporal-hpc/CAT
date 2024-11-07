#include "CellularAutomata/Solvers/CATWithoutCAT.cuh"

template <typename T> void CATWithoutCAT<T>::setupBlockSize()
{
    this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
    this->boundaryBlock = dim3(16, 16);
    this->castingKernelsBlockSize = dim3(16, 16);
}
template <typename T> void CATWithoutCAT<T>::setupGridSize()
{
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    this->GPUGrid =
        dim3((n + (NREGIONS_H * 16) - 1) / (NREGIONS_H * 16), (n + (NREGIONS_V * 16) - 1) / (NREGIONS_V * 16));
    this->castingKernelsGridSize =
        dim3((this->dataDomainDevice->getFullHorizontalSize() + this->castingKernelsBlockSize.x - 1) /
                 this->castingKernelsBlockSize.x,
             (this->dataDomainDevice->getFullHorizontalSize() + this->castingKernelsBlockSize.y - 1) /
                 this->castingKernelsBlockSize.y);
    this->horizontalBoundaryGrid = dim3(2 * (int)ceil(n / (float)this->boundaryBlock.x));
    this->verticalBoundaryGrid =
        dim3(2 * (int)ceil((this->dataDomainDevice->getFullHorizontalSize()) / (float)this->boundaryBlock.x));
    lDebug(1, "Grid size: %d %d\n", this->GPUGrid.x, this->GPUGrid.y);
    lDebug(1, "horizontalBoundaryGrid size: %d %d\n", this->horizontalBoundaryGrid.x, this->horizontalBoundaryGrid.y);
    lDebug(1, "verticalBoundaryGrid size: %d %d\n", this->verticalBoundaryGrid.x, this->verticalBoundaryGrid.y);
    lDebug(1, "castingKernelsGridSize size: %d %d\n", this->castingKernelsGridSize.x, this->castingKernelsGridSize.y);
}

template <typename T> void CATWithoutCAT<T>::setupSharedMemoryCarveout()
{
    sharedMemoryBytes = ((NREGIONS_H + 2) * (NREGIONS_V + 2) * 16 * 16 + 256 * 2) * sizeof(FTYPE);
    cudaFuncSetAttribute(CAT_KERNEL, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemoryBytes);
    if (sharedMemoryBytes > 100000)
    {
        int carveout = int(60 + ((sharedMemoryBytes - 100000) / 64000.0) * 40.0);
        carveout = carveout > 100 ? 100 : carveout;
        cudaFuncSetAttribute(CAT_KERNEL, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    }
}

template <typename T> void CATWithoutCAT<T>::createStream()
{
    cudaStreamCreate(&mainStream);
}

template <typename T> void CATWithoutCAT<T>::moveCurrentDeviceStateToGPUBuffer()
{
    convertFp16ToFp32AndUndoChangeLayout<<<this->castingKernelsGridSize, this->castingKernelsBlockSize>>>(
        this->visibleDataDevice->getData(), this->dataDomainDevice->getData(),
        this->dataDomainDevice->getFullHorizontalSize());
}

template <typename T> void CATWithoutCAT<T>::moveGPUBufferToCurrentDeviceState()
{
    convertFp32ToFp16AndDoChangeLayout<<<this->castingKernelsGridSize, this->castingKernelsBlockSize>>>(
        this->dataDomainDevice->getData(), this->visibleDataDevice->getData(),
        this->dataDomainDevice->getFullHorizontalSize());
}

template <typename T> void CATWithoutCAT<T>::fillHorizontalBoundaryConditions()
{
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    copyHorizontalHaloCoalescedVersion<<<this->horizontalBoundaryGrid, this->boundaryBlock>>>(
        this->dataDomainDevice->getData(), n, this->dataDomainDevice->getFullHorizontalSize());
}

template <typename T> void CATWithoutCAT<T>::fillVerticalBoundaryConditions()
{
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    copyVerticalHaloCoalescedVersion<<<this->verticalBoundaryGrid, this->boundaryBlock>>>(
        this->dataDomainDevice->getData(), n, this->dataDomainDevice->getFullHorizontalSize());
}

template <typename T> void CATWithoutCAT<T>::CAStepAlgorithm()
{
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    CATWithoutCAT_KERNEL<<<this->GPUGrid, this->GPUBlock, sharedMemoryBytes, mainStream>>>(
        this->dataDomainDevice->getData(), this->dataDomainBufferDevice->getData(), n,
        this->dataDomainDevice->getFullHorizontalSize());
    gpuErrchk(cudaDeviceSynchronize());
}