#pragma once

#include "CellularAutomata/Solvers/HostSolver.cuh"
#include "Defines.h"
#include "Memory/Allocators/CPUAllocator.cuh"
#include "Memory/Allocators/GPUAllocator.cuh"

class CASolverFactory {
   public:
    static CASolver* createSolver(int SOLVER_CODE, int sideLength, int haloWidth);
};