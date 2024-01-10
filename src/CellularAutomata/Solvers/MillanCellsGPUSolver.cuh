#pragma once

#include "GPUKernels.cuh"

#include "CellularAutomata/Solvers/GPUSolver.cuh"
#include "Debug.h"

template <typename T>
class MillanCellsGPUSolver : public GPUSolver<T> {
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
    MillanCellsGPUSolver(int deviceId, CADataDomain<T>* deviceData, CADataDomain<T>* deviceDataBuffer, int pCellsPerThread) : GPUSolver<T>(deviceId, deviceData, deviceDataBuffer) {
        sharedMemorySize = sizeof(T) * this->GPUBlock.x * this->GPUBlock.y;
        this->cellsPerThread = pCellsPerThread;
        this->setupBlockSize();
        this->setupGridSize();
        lDebug(1, "MillanCellsGPUSolver: sharedMemorySize = %d", sharedMemorySize);
        lDebug(1, "MillanCellsGPUSolver: cellsPerThread = %d", this->cellsPerThread);

        if (BSIZE3DX != 32 || BSIZE3DY != 32) {
            lDebug(1, "ERROR!: MillanCellsGPUSolver: BSIZE3DX != 32 || BSIZE3DY != 32");
            throw std::runtime_error("ERROR!: MillanCellsGPUSolver: BSIZE3DX != 32 || BSIZE3DY != 32");
        }
    };
};

#include "CellularAutomata/Solvers/MillanCellsGPUSolver.tpp"