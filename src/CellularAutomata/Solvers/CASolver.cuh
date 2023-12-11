#ifndef CASOLVER_H
#define CASOLVER_H

#include "CellularAutomata/CADataPrinter.cuh"
#include "Memory/CADataDomain.cuh"
#include "Memory/CAStateGenerator.cuh"
class CASolver {
   protected:
    CADataDomain<int>* hostVisibleData;

    virtual void fillBoundaryConditions();

    virtual void CAStepAlgorithm() = 0;
    virtual void swapPointers() = 0;
    virtual void fillHorizontalBoundaryConditions() = 0;
    virtual void fillVerticalBoundaryConditions() = 0;

    virtual void copyCurrentStateToHostVisibleData() = 0;
    virtual void copyHostVisibleDataToCurrentState() = 0;

   public:
    virtual void resetState(int seed, float density);
    virtual void doStep();
    virtual void doSteps(int stepsNumber);
    virtual CADataDomain<int>* getCurrentState();
    virtual void printCurrentState();
};

#endif