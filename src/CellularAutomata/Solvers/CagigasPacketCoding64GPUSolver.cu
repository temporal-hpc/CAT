#include "CellularAutomata/Solvers/CagigasPacketCoding64GPUSolver.cuh"

void CagigasPacketCoding64GPUSolver::createVisibleDataBuffer() {
    CPUAllocator<int>* cpuAllocator = new CPUAllocator<int>();
    Allocator<int>* cAllocator = reinterpret_cast<Allocator<int>*>(cpuAllocator);
    hostVisibleData = new CADataDomain<int>(cAllocator, dataDomainDevice->getSideLengthWithoutHalo() * CagigasPacketCoding64GPUSolver::elementsPerCel, dataDomainDevice->getHaloWidth());
    hostVisibleData->allocate();
    lDebug(1, "Visible data buffer created");
}

void CagigasPacketCoding64GPUSolver::createVisibleDataDeviceBuffer() {
    GPUAllocator<int>* gpuAllocator = new GPUAllocator<int>();
    Allocator<int>* gAllocator = reinterpret_cast<Allocator<int>*>(gpuAllocator);
    visibleDataDevice = new CADataDomain<int>(gAllocator, dataDomainDevice->getSideLengthWithoutHalo() * CagigasPacketCoding64GPUSolver::elementsPerCel, dataDomainDevice->getHaloWidth());
    visibleDataDevice->allocate();
    lDebug(1, "Visible data device buffer created");
}

void CagigasPacketCoding64GPUSolver::setupBlockSize() {
    this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
    this->boundaryBlock = dim3(256);
}

void CagigasPacketCoding64GPUSolver::setupGridSize() {
    int n = this->dataDomainDevice->getSideLengthWithoutHalo();
    this->GPUGrid = dim3((int)ceil(n / (float)this->GPUBlock.x), (int)ceil((n * 8) / (float)this->GPUBlock.y));

    this->horizontalBoundaryGrid = dim3((int)ceil(n / (float)this->boundaryBlock.x));
    this->verticalBoundaryGrid = dim3((int)ceil(((n + 2) * 8) / (float)this->boundaryBlock.x));

    lDebug(1, "Grid size set to: (%d, %d, %d)", this->GPUGrid.x, this->GPUGrid.y, this->GPUGrid.z);
    lDebug(1, "Block size set to: (%d, %d, %d)", this->GPUBlock.x, this->GPUBlock.y, this->GPUBlock.z);
    lDebug(1, "Horizontal boundary grid size set to: (%d, %d, %d)", this->horizontalBoundaryGrid.x, this->horizontalBoundaryGrid.y, this->horizontalBoundaryGrid.z);
    lDebug(1, "Vertical boundary grid size set to: (%d, %d, %d)", this->verticalBoundaryGrid.x, this->verticalBoundaryGrid.y, this->verticalBoundaryGrid.z);
}

void CagigasPacketCoding64GPUSolver::moveCurrentDeviceStateToGPUBuffer() {
    int n = this->dataDomainDevice->getSideLengthWithoutHalo();
    unpackState<<<this->GPUGrid, this->GPUBlock>>>(this->dataDomainDevice->getData(), this->visibleDataDevice->getData(), n, n * 8);
}

void CagigasPacketCoding64GPUSolver::moveGPUBufferToCurrentDeviceState() {
    int n = this->dataDomainDevice->getSideLengthWithoutHalo();
    packState<<<this->GPUGrid, this->GPUBlock>>>(this->visibleDataDevice->getData(), this->dataDomainDevice->getData(), n, n * 8);
}

void CagigasPacketCoding64GPUSolver::fillHorizontalBoundaryConditions() {
    int n = this->dataDomainDevice->getSideLengthWithoutHalo();
    ghostRows<<<this->horizontalBoundaryGrid, this->boundaryBlock>>>(this->dataDomainDevice->getData(), n, n * 8);
}

void CagigasPacketCoding64GPUSolver::fillVerticalBoundaryConditions() {
    int n = this->dataDomainDevice->getSideLengthWithoutHalo();
    ghostCols<<<this->verticalBoundaryGrid, this->boundaryBlock>>>(this->dataDomainDevice->getData(), n, n * 8);
}

void CagigasPacketCoding64GPUSolver::CAStepAlgorithm() {
    int n = this->dataDomainDevice->getSideLengthWithoutHalo();
    GOL<<<this->GPUGrid, this->GPUBlock>>>(this->dataDomainDevice->getData(), this->dataDomainBufferDevice->getData(), this->CALookUpTable, n, n * 8);
    (cudaDeviceSynchronize());
}