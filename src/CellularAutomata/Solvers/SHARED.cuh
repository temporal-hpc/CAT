#pragma once

#include "GPUKernels.cuh"

#include "CellularAutomata/Solvers/GPUSolver.cuh"
#include "Debug.h"

template <typename T> class SHARED : public GPUSolver<T>
{
  private:
    int sharedMemorySize;
    int cellsPerThread;

    virtual void setupBlockSize() override;
    virtual void setupGridSize() override;

    virtual void moveCurrentDeviceStateToGPUBuffer() override;
    virtual void moveGPUBufferToCurrentDeviceState() override;
    virtual void fillHorizontalBoundaryConditions() override;
    virtual void fillVerticalBoundaryConditions() override;

    virtual void CAStepAlgorithm() override;

  public:
    SHARED(int deviceId, CADataDomain<T> *deviceData, CADataDomain<T> *deviceDataBuffer)
        : GPUSolver<T>(deviceId, deviceData, deviceDataBuffer)
    {
        this->setupBlockSize();
        this->setupGridSize();

        sharedMemorySize = sizeof(T) * (this->GPUBlock.x + 2 * RADIUS) * (this->GPUBlock.y + 2 * RADIUS);
        lDebug(1, "SHARED: sharedMemorySize = %d", sharedMemorySize);
    };
};

#include "CellularAutomata/Solvers/SHARED.tpp"
