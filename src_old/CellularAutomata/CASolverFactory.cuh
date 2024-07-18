#pragma once

#include "CellularAutomata/Solvers/BASE.cuh"
#include "CellularAutomata/Solvers/CAT.cuh"
#include "CellularAutomata/Solvers/COARSE.cuh"
#include "CellularAutomata/Solvers/HostSolver.cuh"
#include "CellularAutomata/Solvers/MCELL.cuh"
#include "CellularAutomata/Solvers/PACK.cuh"
#include "CellularAutomata/Solvers/SHARED.cuh"
#include "Debug.h"
#include "Defines.h"
#include "Memory/Allocators/CPUAllocator.cuh"
#include "Memory/Allocators/GPUAllocator.cuh"

class CASolverFactory
{
  public:
    static CASolver *createSolver(int SOLVER_CODE, int deviceId, int sideLength, int haloWidth);
};