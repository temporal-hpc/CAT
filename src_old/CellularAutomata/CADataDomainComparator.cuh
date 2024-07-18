#pragma once

#include "CellularAutomata/Solvers/CASolver.cuh"
#include "Memory/CADataDomain.cuh"

class CADataDomainComparator {
   private:
    CASolver* solver;
    CASolver* referenceSolver;

    bool areDifferentSize();

   public:
    CADataDomainComparator(CASolver* pSolver, CASolver* pReferenceSolver);

    bool compareCurrentStates();
};

#include "CellularAutomata/CADataDomainComparator.tpp"