#pragma once

#include "GPUKernels.cuh"

#include "CellularAutomata/Solvers/GPUSolver.cuh"
#include "Debug.h"
#include "Defines.h"

class PACK : public GPUSolver<uint64_t>
{
  private:
    int *CALookUpTable;

    size_t sharedMemoryBytes;
    cudaStream_t mainStream;

    virtual void setupSharedMemoryCarveout();
    virtual void createStream();

    virtual void createVisibleDataBuffer() override;
    virtual void createVisibleDataDeviceBuffer() override;

    virtual void setupBlockSize() override;
    virtual void setupGridSize() override;

    virtual void moveCurrentDeviceStateToGPUBuffer() override;
    virtual void moveGPUBufferToCurrentDeviceState() override;
    virtual void fillHorizontalBoundaryConditions() override;
    virtual void fillVerticalBoundaryConditions() override;

    virtual void CAStepAlgorithm() override;

  public:
    static constexpr float elementsPerCel = 8.0f;

    PACK(int deviceId, CADataDomain<uint64_t> *deviceData, CADataDomain<uint64_t> *deviceDataBuffer)
    {
        dataDomainDevice = deviceData;
        dataDomainBufferDevice = deviceDataBuffer;

        cudaSetDevice(deviceId);

        createVisibleDataBuffer();
        createVisibleDataDeviceBuffer();

        this->setupBlockSize();
        this->setupGridSize();
        this->setupSharedMemoryCarveout();
        this->createStream();
        cudaMalloc(&CALookUpTable, sizeof(int) * 2 * (CAGIGAS_CELL_NEIGHBOURS + 1));
        dim3 block = dim3(1, 2, 1);
        dim3 grid = dim3((CAGIGAS_CELL_NEIGHBOURS + 1), 1, 1);
        kernel_init_lookup_table<<<grid, block>>>(CALookUpTable);
        cudaDeviceSynchronize();
    };
};
