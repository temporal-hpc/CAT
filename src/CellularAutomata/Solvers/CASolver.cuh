#pragma once

#include "Memory/CADataDomain.cuh"

class CASolver {
   protected:
    virtual void swapPointers() = 0;
    virtual void fillBoundaryConditions() = 0;

   public:
    virtual void resetState(int seed, float density) = 0;
    virtual void doStep() = 0;
    virtual void doSteps(int stepsNumber) = 0;
    virtual CADataDomain* getCurrentState() = 0;
    virtual void printCurrentState() = 0;
};