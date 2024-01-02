
#include "CellularAutomata/CADataPrinter.cuh"
#include "CellularAutomata/Solvers/CASolver.cuh"
#include "CellularAutomata/Solvers/GPUSolver.cuh"
#include "Memory/Allocators/CPUAllocator.cuh"
#include "Memory/Allocators/GPUAllocator.cuh"

#include "Memory/CADataDomain.cuh"
#include "Memory/CAStateGenerator.cuh"

template <typename T>
GPUSolver<T>::GPUSolver(int deviceId, CADataDomain<T>* deviceData, CADataDomain<T>* deviceDataBuffer) {
    dataDomainDevice = deviceData;
    dataDomainBufferDevice = deviceDataBuffer;

    cudaSetDevice(deviceId);

    createVisibleDataBuffer();
    createVisibleDataDeviceBuffer();
}
template <typename T>
void GPUSolver<T>::createVisibleDataBuffer() {
    CPUAllocator<int>* cpuAllocator = new CPUAllocator<int>();
    Allocator<int>* cAllocator = reinterpret_cast<Allocator<int>*>(cpuAllocator);
    hostVisibleData = new CADataDomain<int>(cAllocator, dataDomainDevice->getInnerHorizontalSize(), dataDomainDevice->getHorizontalHaloSize());
    hostVisibleData->allocate();
}

template <typename T>
void GPUSolver<T>::createVisibleDataDeviceBuffer() {
    GPUAllocator<int>* gpuAllocator = new GPUAllocator<int>();
    Allocator<int>* gAllocator = reinterpret_cast<Allocator<int>*>(gpuAllocator);
    visibleDataDevice = new CADataDomain<int>(gAllocator, dataDomainDevice->getInnerHorizontalSize(), dataDomainDevice->getHorizontalHaloSize());
    visibleDataDevice->allocate();
}

template <typename T>
void GPUSolver<T>::copyCurrentStateToHostVisibleData() {
    moveCurrentDeviceStateToGPUBuffer();
    cudaMemcpy(hostVisibleData->getData(), visibleDataDevice->getData(), sizeof(int) * hostVisibleData->getTotalSize(), cudaMemcpyDeviceToHost);
}

template <typename T>
void GPUSolver<T>::copyHostVisibleDataToCurrentState() {
    cudaMemcpy(visibleDataDevice->getData(), hostVisibleData->getData(), sizeof(int) * visibleDataDevice->getTotalSize(), cudaMemcpyHostToDevice);
    moveGPUBufferToCurrentDeviceState();
}

template <typename T>
void GPUSolver<T>::swapPointers() {
    CADataDomain<T>* temp = dataDomainDevice;
    dataDomainDevice = dataDomainBufferDevice;
    dataDomainBufferDevice = temp;
}