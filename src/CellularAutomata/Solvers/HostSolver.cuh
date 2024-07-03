
#pragma once

#include "CellularAutomata/Solvers/CASolver.cuh"
#include "Memory/CADataDomain.cuh"
#include "Memory/CAStateGenerator.cuh"

template <typename T>
class HostSolver : public CASolver {
   private:
    CADataDomain<T>* dataDomain;
    CADataDomain<T>* dataDomainBuffer;

    void CAStepAlgorithm() override;
    void fillHorizontalBoundaryConditions() override;
    void fillVerticalBoundaryConditions() override;

    void swapPointers() override;
    void copyCurrentStateToHostVisibleData() override;
    void copyHostVisibleDataToCurrentState() override;

    T transitionFunction(int k, int a, int b);
    int countAliveNeighbors(int i, int j);

   public:
    HostSolver(CADataDomain<T>* domain, CADataDomain<T>* domainBuffer);
};

#include "CellularAutomata/Solvers/HostSolver.tpp"