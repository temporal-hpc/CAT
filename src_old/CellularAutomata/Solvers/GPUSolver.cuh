#pragma once

#include "CellularAutomata/Solvers/CASolver.cuh"
#include "Memory/CADataDomain.cuh"
#include "Memory/CAStateGenerator.cuh"
#include "randCA.cuh"

template <typename T> class GPUSolver : public CASolver
{
  protected:
    GPUSolver()
    {
    }

    CADataDomain<int> *visibleDataDevice;

    CADataDomain<T> *dataDomainDevice;
    CADataDomain<T> *dataDomainBufferDevice;

    int deviceId;

    dim3 GPUBlock;
    dim3 GPUGrid;

    dim3 boundaryBlock;
    dim3 horizontalBoundaryGrid;
    dim3 verticalBoundaryGrid;

    // void adjustSharedMemory();
    virtual void resetState(int seed, float density) override;

    virtual void createVisibleDataBuffer();
    virtual void createVisibleDataDeviceBuffer();

    virtual void copyCurrentStateToHostVisibleData() override;
    virtual void copyHostVisibleDataToCurrentState() override;

    void swapPointers() override;

    virtual void setBlockSize() = 0;
    virtual void setGridSize() = 0;

    virtual void moveCurrentDeviceStateToGPUBuffer() = 0;
    virtual void moveGPUBufferToCurrentDeviceState() = 0;

    virtual void fillHorizontalBoundaryConditions() = 0;
    virtual void fillVerticalBoundaryConditions() = 0;

    virtual void CAStepAlgorithm() = 0;

  public:
    GPUSolver(int deviceId, CADataDomain<T> *deviceData, CADataDomain<T> *deviceDataBuffer);
};

#include "CellularAutomata/Solvers/GPUSolver.tpp"