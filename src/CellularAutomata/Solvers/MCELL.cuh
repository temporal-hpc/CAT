#pragma once

#include "GPUKernels.cuh"

#include "CellularAutomata/Solvers/GPUSolver.cuh"
#include "Debug.h"

template <typename T> class MCELL : public GPUSolver<T>
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
    MCELL(int deviceId, CADataDomain<T> *deviceData, CADataDomain<T> *deviceDataBuffer, int pCellsPerThread)
        : GPUSolver<T>(deviceId, deviceData, deviceDataBuffer)
    {
        this->cellsPerThread = pCellsPerThread;
        this->setupBlockSize();
        this->setupGridSize();
        sharedMemorySize = sizeof(T) * (BSIZE3DX * 2 + 2 * RADIUS) * (BSIZE3DY + 2 * RADIUS);
        lDebug(1, "MCELL: sharedMemorySize = %d", sharedMemorySize);
        lDebug(1, "MCELL: cellsPerThread = %d", this->cellsPerThread);
    };
};

#include "CellularAutomata/Solvers/MCELL.tpp"
