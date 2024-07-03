#pragma once

#include "GPUKernels.cuh"

#include "CellularAutomata/Solvers/GPUSolver.cuh"

template <typename T>
class SharedMemoryGPUSolver : public GPUSolver<T> {
   private:
    virtual void setupBlockSize() override;
    virtual void setupGridSize() override;

    virtual void moveCurrentDeviceStateToGPUBuffer() override;
    virtual void moveGPUBufferToCurrentDeviceState() override;
    virtual void fillHorizontalBoundaryConditions() override;
    virtual void fillVerticalBoundaryConditions() override;

    virtual void CAStepAlgorithm() override;

   public:
    SharedMemoryGPUSolver(int deviceId, CADataDomain<T>* deviceData, CADataDomain<T>* deviceDataBuffer) : GPUSolver<T>(deviceId, deviceData, deviceDataBuffer) {
        this->setupBlockSize();
        this->setupGridSize();
    };
};

#include "CellularAutomata/Solvers/SharedMemoryGPUSolver.tpp"