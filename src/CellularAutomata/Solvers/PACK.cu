#include "CellularAutomata/Solvers/PACK.cuh"

void PACK::createStream()
{
    cudaStreamCreate(&mainStream);
}

void PACK::setupSharedMemoryCarveout()
{
    sharedMemoryBytes = (BSIZE3DX + 2 * dataDomainDevice->getHorizontalHaloSize()) *
                        (BSIZE3DY + 2 * dataDomainDevice->getVerticalHaloSize()) * sizeof(uint64_t);
    lDebug(1, "Shared memory carveout set to: %d bytes", sharedMemoryBytes);
}

void PACK::createVisibleDataBuffer()
{
    CPUAllocator<int> *cpuAllocator = new CPUAllocator<int>();
    Allocator<int> *cAllocator = reinterpret_cast<Allocator<int> *>(cpuAllocator);
    hostVisibleData = new CADataDomain<int>(cAllocator, dataDomainDevice->getInnerVerticalSize(),
                                            dataDomainDevice->getVerticalHaloSize());
    hostVisibleData->allocate();
    lDebug(1, "Visible data buffer created");
}

void PACK::createVisibleDataDeviceBuffer()
{
    GPUAllocator<int> *gpuAllocator = new GPUAllocator<int>();
    Allocator<int> *gAllocator = reinterpret_cast<Allocator<int> *>(gpuAllocator);
    visibleDataDevice = new CADataDomain<int>(gAllocator, dataDomainDevice->getInnerVerticalSize(),
                                              dataDomainDevice->getVerticalHaloSize());
    visibleDataDevice->allocate();
    lDebug(1, "Visible data device buffer created");
}

void PACK::setupBlockSize()
{
    this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
    this->boundaryBlock = dim3(256);
}

void PACK::setupGridSize()
{
    int innerHorizontalSize = this->dataDomainDevice->getInnerHorizontalSize();
    int innerVerticalSize = this->dataDomainDevice->getInnerVerticalSize();
    int fullVerticalSize = this->dataDomainDevice->getFullVerticalSize();
    this->GPUGrid = dim3((int)ceil(innerHorizontalSize / (float)this->GPUBlock.x),
                         (int)ceil(innerVerticalSize / (float)this->GPUBlock.y));

    this->horizontalBoundaryGrid = dim3((int)ceil(innerHorizontalSize / (float)this->boundaryBlock.x));
    this->verticalBoundaryGrid = dim3((int)ceil(fullVerticalSize / (float)this->boundaryBlock.x));

    lDebug(1, "Grid size set to: (%d, %d, %d)", this->GPUGrid.x, this->GPUGrid.y, this->GPUGrid.z);
    lDebug(1, "Block size set to: (%d, %d, %d)", this->GPUBlock.x, this->GPUBlock.y, this->GPUBlock.z);
    lDebug(1, "Horizontal boundary grid size set to: (%d, %d, %d)", this->horizontalBoundaryGrid.x,
           this->horizontalBoundaryGrid.y, this->horizontalBoundaryGrid.z);
    lDebug(1, "Vertical boundary grid size set to: (%d, %d, %d)", this->verticalBoundaryGrid.x,
           this->verticalBoundaryGrid.y, this->verticalBoundaryGrid.z);
}

void PACK::moveCurrentDeviceStateToGPUBuffer()
{
    int innerHorizontalSize = this->dataDomainDevice->getInnerHorizontalSize();
    int innerVerticalSize = this->dataDomainDevice->getInnerVerticalSize();
    int horizontalHaloSize = this->dataDomainDevice->getHorizontalHaloSize();
    int verticalHaloSize = this->dataDomainDevice->getVerticalHaloSize();

    unpackState<<<this->GPUGrid, this->GPUBlock>>>(this->dataDomainDevice->getData(),
                                                   this->visibleDataDevice->getData(), innerHorizontalSize,
                                                   innerVerticalSize, horizontalHaloSize, verticalHaloSize);
    gpuErrchk(cudaDeviceSynchronize());
}

void PACK::moveGPUBufferToCurrentDeviceState()
{
    int innerHorizontalSize = this->dataDomainDevice->getInnerHorizontalSize();
    int innerVerticalSize = this->dataDomainDevice->getInnerVerticalSize();
    int horizontalHaloSize = this->dataDomainDevice->getHorizontalHaloSize();
    int verticalHaloSize = this->dataDomainDevice->getVerticalHaloSize();

    // printf("OUTSIDE: GRID_SIZE: %i, verticalHalo: %i\n", innerVerticalSize, verticalHaloSize);
    packState<<<this->GPUGrid, this->GPUBlock>>>(this->visibleDataDevice->getData(), this->dataDomainDevice->getData(),
                                                 innerHorizontalSize, innerVerticalSize, horizontalHaloSize,
                                                 verticalHaloSize);
    gpuErrchk(cudaDeviceSynchronize());
}

void PACK::fillHorizontalBoundaryConditions()
{
    int innerHorizontalSize = this->dataDomainDevice->getInnerHorizontalSize();
    int innerVerticalSize = this->dataDomainDevice->getInnerVerticalSize();
    int horizontalHaloSize = this->dataDomainDevice->getHorizontalHaloSize();
    int verticalHaloSize = this->dataDomainDevice->getVerticalHaloSize();

    ghostRows<<<this->horizontalBoundaryGrid, this->boundaryBlock>>>(this->dataDomainDevice->getData(),
                                                                     innerHorizontalSize, innerVerticalSize,
                                                                     horizontalHaloSize, verticalHaloSize);
    gpuErrchk(cudaDeviceSynchronize());
}

void PACK::fillVerticalBoundaryConditions()
{
    int innerHorizontalSize = this->dataDomainDevice->getInnerHorizontalSize();
    int innerVerticalSize = this->dataDomainDevice->getInnerVerticalSize();
    int horizontalHaloSize = this->dataDomainDevice->getHorizontalHaloSize();
    int verticalHaloSize = this->dataDomainDevice->getVerticalHaloSize();

    ghostCols<<<this->verticalBoundaryGrid, this->boundaryBlock>>>(this->dataDomainDevice->getData(),
                                                                   innerHorizontalSize, innerVerticalSize,
                                                                   horizontalHaloSize, verticalHaloSize);
    gpuErrchk(cudaDeviceSynchronize());
}

void PACK::CAStepAlgorithm()
{
    int n = this->dataDomainDevice->getInnerHorizontalSize();
    int horizontalHaloSize = this->dataDomainDevice->getHorizontalHaloSize();
    int verticalHaloSize = this->dataDomainDevice->getVerticalHaloSize();
#if RADIUS == 1
    // i) Cagigas original code, optimized for r=1
    CAGIGAS_KERNEL<<<this->GPUGrid, this->GPUBlock>>>(this->dataDomainDevice->getData(),
                                                      this->dataDomainBufferDevice->getData(), this->CALookUpTable, n,
                                                      dataDomainDevice->getInnerVerticalSize());
#else
    // iii) Shared memory generalization of Cagigas for r in [1..15] with optimal number of memory accesses.
    PACK_KERNEL<<<this->GPUGrid, this->GPUBlock, sharedMemoryBytes, mainStream>>>(
        this->dataDomainDevice->getData(), this->dataDomainBufferDevice->getData(), this->CALookUpTable, n,
        dataDomainDevice->getInnerVerticalSize(), horizontalHaloSize, verticalHaloSize);
#endif
    gpuErrchk(cudaDeviceSynchronize());
}
