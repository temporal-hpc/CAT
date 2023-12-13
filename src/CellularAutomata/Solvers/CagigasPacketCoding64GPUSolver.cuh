#pragma once

#include "GPUKernels.cuh"

#include "CellularAutomata/Solvers/GPUSolver.cuh"
#include "Debug.h"

#define CELL_NEIGHBOURS 8

class CagigasPacketCoding64GPUSolver : public GPUSolver<uint64_t> {
   private:
    int* CALookUpTable;

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
    static const int elementsPerCel = 8;

    CagigasPacketCoding64GPUSolver(int deviceId, CADataDomain<uint64_t>* deviceData, CADataDomain<uint64_t>* deviceDataBuffer) {
        dataDomainDevice = deviceData;
        dataDomainBufferDevice = deviceDataBuffer;

        cudaSetDevice(deviceId);

        createVisibleDataBuffer();
        createVisibleDataDeviceBuffer();

        this->setupBlockSize();
        this->setupGridSize();
        cudaMalloc(&CALookUpTable, sizeof(int) * 2 * (CELL_NEIGHBOURS + 1));
        kernel_init_lookup_table<<<1, this->GPUBlock>>>(CALookUpTable);
        cudaDeviceSynchronize();
    };
};
