#pragma once

#include "GPUKernels.cuh"

#include "CellularAutomata/Solvers/TensorCoreGPUSolver.cuh"

template <typename T>
class CoalescedTensorCoreGPUSolver : public GPUSolver<T> {
   private:
    dim3 castingKernelsBlockSize;
    dim3 castingKernelsGridSize;

    size_t sharedMemoryBytes;
    cudaStream_t mainStream;

    virtual void setupBlockSize() override;
    virtual void setupGridSize() override;

    virtual void moveCurrentDeviceStateToGPUBuffer() override;
    virtual void moveGPUBufferToCurrentDeviceState() override;
    virtual void fillHorizontalBoundaryConditions() override;
    virtual void fillVerticalBoundaryConditions() override;

    virtual void CAStepAlgorithm() override;

    virtual void setupSharedMemoryCarveout();
    virtual void createStream();

   public:
    CoalescedTensorCoreGPUSolver(int deviceId, CADataDomain<T>* deviceData, CADataDomain<T>* deviceDataBuffer) : GPUSolver<T>(deviceId, deviceData, deviceDataBuffer) {
        this->setupBlockSize();
        this->setupGridSize();
        this->setupSharedMemoryCarveout();
        this->createStream();
        cudaMemset(this->dataDomainBufferDevice->getData(), 0, sizeof(half) * this->dataDomainBufferDevice->getTotalSize());
    };
};
#include "CellularAutomata/Solvers/CoalescedTensorCoreGPUSolver.tpp"