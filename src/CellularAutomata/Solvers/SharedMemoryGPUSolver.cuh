#pragma once

#include "CellularAutomata/GPUSolver.cuh"
#include "Memory/HostMemoryCA.cuh"

class SharedMemoryGPUSolver : public GPUSolver<MTYPE> {
   private:
    HostMemoryCA* hostMemory;
    GPUMemoryCA<MTYPE>* deviceMemory;

    dim3 GPUBlock;
    dim3 GPUGrid;

    void setupBlockSize();
    void setupGridSize(size_t size);

    void adjustSharedMemory();
    /// copyhsot to device
   public:
    void doStep() override;
    void doSteps(int stepNumber) override;
};