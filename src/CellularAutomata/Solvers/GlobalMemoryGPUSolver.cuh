#pragma once

#include "CellularAutomata/CASolver.cuh"
#include "Memory/GPUMemoryCA.cuh"
#include "Memory/HostMemoryCA.cuh"

class GlobalMemoryGPUSolver : public CASolver {
   private:
    HostMemoryCA* hostMemory;
    GPUMemoryCA<MTYPE>* deviceMemory;

    dim3 GPUBlock;
    dim3 GPUGrid;

    /// copyhsot to device
   public:
    void doStep() override;
    void doSteps(int stepNumber) override;
};