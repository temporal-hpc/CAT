#include "CellularAutomata/CADataPrinter.cuh"
#include "CellularAutomata/Solvers/HostSolver.cuh"
#include "Memory/Allocators/CPUAllocator.cuh"

template <typename T>
HostSolver<T>::HostSolver(CADataDomain<T>* domain, CADataDomain<T>* domainBuffer) {
    dataDomain = domain;
    dataDomainBuffer = domainBuffer;

    CPUAllocator<int>* cpuAllocator = new CPUAllocator<int>();
    Allocator<int>* allocator = reinterpret_cast<Allocator<int>*>(cpuAllocator);
    hostVisibleData = new CADataDomain<int>(allocator, dataDomain->getInnerHorizontalSize(), dataDomain->getHorizontalHaloSize());
    hostVisibleData->allocate();
}

template <typename T>
void HostSolver<T>::copyCurrentStateToHostVisibleData() {
    for (int i = 0; i < dataDomain->getTotalSize(); ++i) {
        T value = dataDomain->getElementAt(i);
        hostVisibleData->setElementAt(i, (int)value);
    }
}
template <typename T>
void HostSolver<T>::copyHostVisibleDataToCurrentState() {
    for (int i = 0; i < hostVisibleData->getTotalSize(); ++i) {
        int value = hostVisibleData->getElementAt(i);
        dataDomain->setElementAt(i, value);
    }
}

template <typename T>
void HostSolver<T>::swapPointers() {
    CADataDomain<T>* temp = dataDomain;
    dataDomain = dataDomainBuffer;
    dataDomainBuffer = temp;
}

template <typename T>
T HostSolver<T>::transitionFunction(int k, int a, int b) {
    return (1 - (((k - a) >> 31) & 0x1)) * (1 - (((b - k) >> 31) & 0x1));
}

template <typename T>
void HostSolver<T>::CAStepAlgorithm() {
    for (int i = 0; i < dataDomain->getInnerHorizontalSize(); ++i) {
        for (int j = 0; j < dataDomain->getInnerHorizontalSize(); ++j) {
            int liveNeighbors = countAliveNeighbors(i, j);
            T cellValue = dataDomain->getInnerElementAt(i, j);
            T result = cellValue * transitionFunction(liveNeighbors, SMIN, SMAX) + (1 - cellValue) * transitionFunction(liveNeighbors, BMIN, BMAX);

            dataDomainBuffer->setInnerElementAt(i, j, result);
        }
    }
}

template <typename T>
int HostSolver<T>::countAliveNeighbors(int y, int x) {
    int aliveNeighbors = 0;

    for (int i = -RADIUS; i <= RADIUS; ++i) {
        for (int j = -RADIUS; j <= RADIUS; ++j) {
            if (i == 0 && j == 0)
                continue;
            aliveNeighbors += dataDomain->getInnerElementAt(y + i, x + j);
        }
    }

    return aliveNeighbors;
}

template <typename T>
void HostSolver<T>::fillHorizontalBoundaryConditions() {
    for (int h = 0; h < dataDomain->getHorizontalHaloSize(); ++h) {
        for (int j = 0; j < dataDomain->getInnerHorizontalSize(); ++j) {
            size_t topIndex = (dataDomain->getHorizontalHaloSize() + h) * dataDomain->getFullHorizontalSize() + dataDomain->getHorizontalHaloSize() + j;
            size_t bottomIndex = topIndex + (dataDomain->getInnerHorizontalSize()) * dataDomain->getFullHorizontalSize();
            T value = dataDomain->getElementAt(topIndex);
            dataDomain->setElementAt(bottomIndex, value);
        }

        for (int j = 0; j < dataDomain->getInnerHorizontalSize(); ++j) {
            size_t topIndex = (h)*dataDomain->getFullHorizontalSize() + dataDomain->getHorizontalHaloSize() + j;
            size_t bottomIndex = topIndex + (dataDomain->getInnerHorizontalSize()) * dataDomain->getFullHorizontalSize();

            T value = dataDomain->getElementAt(bottomIndex);
            dataDomain->setElementAt(topIndex, value);
        }
    }
}

template <typename T>
void HostSolver<T>::fillVerticalBoundaryConditions() {
    for (int h = 0; h < dataDomain->getHorizontalHaloSize(); ++h) {
        for (int i = 0; i < dataDomain->getFullHorizontalSize(); ++i) {
            size_t leftIndex = i * dataDomain->getFullHorizontalSize() + h;
            size_t rightIndex = leftIndex + dataDomain->getInnerHorizontalSize();
            T value = dataDomain->getElementAt(rightIndex);
            dataDomain->setElementAt(leftIndex, value);
        }

        for (int i = 0; i < dataDomain->getFullHorizontalSize(); ++i) {
            size_t leftIndex = i * dataDomain->getFullHorizontalSize() + dataDomain->getHorizontalHaloSize() + h;
            size_t rightIndex = leftIndex + dataDomain->getInnerHorizontalSize();
            T value = dataDomain->getElementAt(leftIndex);
            dataDomain->setElementAt(rightIndex, value);
        }
    }
}
