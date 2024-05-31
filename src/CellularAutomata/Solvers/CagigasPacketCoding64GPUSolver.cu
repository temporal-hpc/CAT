#include "CellularAutomata/Solvers/CagigasPacketCoding64GPUSolver.cuh"

void CagigasPacketCoding64GPUSolver::createStream() {
    cudaStreamCreate(&mainStream);
}

void CagigasPacketCoding64GPUSolver::setupSharedMemoryCarveout() {
    sharedMemoryBytes = (BSIZE3DX + 2 * dataDomainDevice->getHorizontalHaloSize()) * (BSIZE3DY + 2 * dataDomainDevice->getVerticalHaloSize()) * sizeof(uint64_t);
    lDebug(1, "Shared memory carveout set to: %d bytes", sharedMemoryBytes);
}

void CagigasPacketCoding64GPUSolver::createVisibleDataBuffer() {
    CPUAllocator<int>* cpuAllocator = new CPUAllocator<int>();
    Allocator<int>* cAllocator = reinterpret_cast<Allocator<int>*>(cpuAllocator);
    hostVisibleData = new CADataDomain<int>(cAllocator, dataDomainDevice->getInnerVerticalSize(), dataDomainDevice->getVerticalHaloSize());
    hostVisibleData->allocate();
    lDebug(1, "Visible data buffer created");
}

void CagigasPacketCoding64GPUSolver::createVisibleDataDeviceBuffer() {
    GPUAllocator<int>* gpuAllocator = new GPUAllocator<int>();
    Allocator<int>* gAllocator = reinterpret_cast<Allocator<int>*>(gpuAllocator);
    visibleDataDevice = new CADataDomain<int>(gAllocator, dataDomainDevice->getInnerVerticalSize(), dataDomainDevice->getVerticalHaloSize());
    visibleDataDevice->allocate();
    lDebug(1, "Visible data device buffer created");
}

void CagigasPacketCoding64GPUSolver::setupBlockSize() {
    this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
    this->boundaryBlock = dim3(256);
}

void CagigasPacketCoding64GPUSolver::setupGridSize() {
    int innerHorizontalSize = this->dataDomainDevice->getInnerHorizontalSize();
    int innerVerticalSize = this->dataDomainDevice->getInnerVerticalSize();
    int fullVerticalSize = this->dataDomainDevice->getFullVerticalSize();
    this->GPUGrid = dim3((int)ceil(innerHorizontalSize / (float)this->GPUBlock.x), (int)ceil(innerVerticalSize / (float)this->GPUBlock.y));

    this->horizontalBoundaryGrid = dim3((int)ceil(innerHorizontalSize / (float)this->boundaryBlock.x));
    this->verticalBoundaryGrid = dim3((int)ceil(fullVerticalSize / (float)this->boundaryBlock.x));

    lDebug(1, "Grid size set to: (%d, %d, %d)", this->GPUGrid.x, this->GPUGrid.y, this->GPUGrid.z);
    lDebug(1, "Block size set to: (%d, %d, %d)", this->GPUBlock.x, this->GPUBlock.y, this->GPUBlock.z);
    lDebug(1, "Horizontal boundary grid size set to: (%d, %d, %d)", this->horizontalBoundaryGrid.x, this->horizontalBoundaryGrid.y, this->horizontalBoundaryGrid.z);
    lDebug(1, "Vertical boundary grid size set to: (%d, %d, %d)", this->verticalBoundaryGrid.x, this->verticalBoundaryGrid.y, this->verticalBoundaryGrid.z);
}

void CagigasPacketCoding64GPUSolver::moveCurrentDeviceStateToGPUBuffer() {
    int innerHorizontalSize = this->dataDomainDevice->getInnerHorizontalSize();
    int innerVerticalSize = this->dataDomainDevice->getInnerVerticalSize();
    int horizontalHaloSize = this->dataDomainDevice->getHorizontalHaloSize();
    int verticalHaloSize = this->dataDomainDevice->getVerticalHaloSize();

    unpackState<<<this->GPUGrid, this->GPUBlock>>>(this->dataDomainDevice->getData(), this->visibleDataDevice->getData(), innerHorizontalSize, innerVerticalSize, horizontalHaloSize, verticalHaloSize);
    gpuErrchk(cudaDeviceSynchronize());
}

void CagigasPacketCoding64GPUSolver::moveGPUBufferToCurrentDeviceState() {
    int innerHorizontalSize = this->dataDomainDevice->getInnerHorizontalSize();
    int innerVerticalSize = this->dataDomainDevice->getInnerVerticalSize();
    int horizontalHaloSize = this->dataDomainDevice->getHorizontalHaloSize();
    int verticalHaloSize = this->dataDomainDevice->getVerticalHaloSize();

    printf("OUTSIDE: GRID_SIZE: %i, verticalHalo: %i\n", innerVerticalSize, verticalHaloSize);
    packState<<<this->GPUGrid, this->GPUBlock>>>(this->visibleDataDevice->getData(), this->dataDomainDevice->getData(), innerHorizontalSize, innerVerticalSize, horizontalHaloSize, verticalHaloSize);
    gpuErrchk(cudaDeviceSynchronize());
}

void CagigasPacketCoding64GPUSolver::fillHorizontalBoundaryConditions() {
    int innerHorizontalSize = this->dataDomainDevice->getInnerHorizontalSize();
    int innerVerticalSize = this->dataDomainDevice->getInnerVerticalSize();
    int horizontalHaloSize = this->dataDomainDevice->getHorizontalHaloSize();
    int verticalHaloSize = this->dataDomainDevice->getVerticalHaloSize();

    ghostRows<<<this->horizontalBoundaryGrid, this->boundaryBlock>>>(this->dataDomainDevice->getData(), innerHorizontalSize, innerVerticalSize, horizontalHaloSize, verticalHaloSize);
    gpuErrchk(cudaDeviceSynchronize());
}

void CagigasPacketCoding64GPUSolver::fillVerticalBoundaryConditions() {
    int innerHorizontalSize = this->dataDomainDevice->getInnerHorizontalSize();
    int innerVerticalSize = this->dataDomainDevice->getInnerVerticalSize();
    int horizontalHaloSize = this->dataDomainDevice->getHorizontalHaloSize();
    int verticalHaloSize = this->dataDomainDevice->getVerticalHaloSize();

    ghostCols<<<this->verticalBoundaryGrid, this->boundaryBlock>>>(this->dataDomainDevice->getData(), innerHorizontalSize, innerVerticalSize, horizontalHaloSize, verticalHaloSize);
    gpuErrchk(cudaDeviceSynchronize());
}

void CagigasPacketCoding64GPUSolver::CAStepAlgorithm() {
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    int horizontalHaloSize = this->dataDomainDevice->getHorizontalHaloSize();
    int verticalHaloSize = this->dataDomainDevice->getVerticalHaloSize();

    GOL33<<<this->GPUGrid, this->GPUBlock, sharedMemoryBytes, mainStream>>>(this->dataDomainDevice->getData(), this->dataDomainBufferDevice->getData(), this->CALookUpTable, n, dataDomainDevice->getInnerVerticalSize(), horizontalHaloSize, verticalHaloSize);
    // GOL<<<this->GPUGrid, this->GPUBlock>>>(this->dataDomainDevice->getData(), this->dataDomainBufferDevice->getData(), this->CALookUpTable, n, dataDomainDevice->getInnerVerticalSize(), horizontalHaloSize, verticalHaloSize);
    gpuErrchk(cudaDeviceSynchronize());
}
