
#pragma once

#include "CellularAutomata/Solvers/CASolver.cuh"
#include "Memory/CADataDomain.cuh"
#include "Memory/CAStateGenerator.cuh"

template <typename T>
class HostSolver : public CASolver {
   private:
    CADataDomain<T>* dataDomain;
    CADataDomain<T>* dataDomainBuffer;

    void CAStepAlgorithm();
    int countAliveNeighbors(int i, int j);
    void fillHorizontalBoundaryConditions();
    void fillVerticalBoundaryConditions();

    void swapPointers() override;
    void fillBoundaryConditions() override;
    T transitionFunction(int k, int a, int b);

   public:
    HostSolver(CADataDomain<T>* domain, CADataDomain<T>* domainBuffer);

    void resetState(int seed, float density) override;
    void* getCurrentState() override;

    void doStep() override;
    void doSteps(int stepNumber) override;
    void printCurrentState() override;
};

#include "CellularAutomata/Solvers/HostSolver.tpp"