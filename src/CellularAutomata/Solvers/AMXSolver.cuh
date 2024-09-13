
#pragma once

#include "CellularAutomata/Solvers/CASolver.cuh"
#include "Memory/CADataDomain.cuh"
#include "Memory/CAStateGenerator.cuh"
#include <immintrin.h>


class AMXSolver : public CASolver {
   private:

    uint8_t pi_1[16*64];
    uint8_t pi_2[16*64];
    uint8_t pi_3[16*64];


    CADataDomain<uint8_t>* dataDomain;
    CADataDomain<uint8_t>* dataDomainBuffer;

    void CAStepAlgorithm() override;
    void fillHorizontalBoundaryConditions() override;
    void fillVerticalBoundaryConditions() override;

    void swapPointers() override;
    void copyCurrentStateToHostVisibleData() override;
    void copyHostVisibleDataToCurrentState() override;

    uint8_t transitionFunction(int k, int a, int b);
    int countAliveNeighbors(int i, int j);

    void fillTridiag();
    void setupAMX();

   public:
    AMXSolver(CADataDomain<uint8_t>* domain, CADataDomain<uint8_t>* domainBuffer);
};
