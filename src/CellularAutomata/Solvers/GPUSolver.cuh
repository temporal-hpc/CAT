#pragma once

#include "CellularAutomata/CASolver.cuh"
#include "Memory/HostMemoryCA.cuh"

template <typename T>
class GPUSolver : public CASolver {
   private:
    HostMemoryCA* hostMemory;
    GPUMemoryCA<MTYPE>* deviceMemory;
    int deviceId;

    dim3 GPUBlock;
    dim3 GPUGrid;

    void setupBlockSize();
    void setupGridSize(size_t size);

    void adjustSharedMemory();

    void transferDeviceToHost();
    void transferHostToDevice();

   public:
    GPUSolver(int deviceId, HostMemoryCA* mem, GPUMemoryCA<T>* gpuMem);

    void doStep() override;
    void doSteps(int stepNumber) override;
};