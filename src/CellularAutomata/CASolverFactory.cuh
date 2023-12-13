#pragma once

#include "CellularAutomata/Solvers/CagigasPacketCoding64GPUSolver.cuh"
#include "CellularAutomata/Solvers/CoalescedTensorCoreGPUSolver.cuh"
#include "CellularAutomata/Solvers/FastTensorCoreGPUSolver.cuh"
#include "CellularAutomata/Solvers/GlobalMemoryGPUSolver.cuh"
#include "CellularAutomata/Solvers/HostSolver.cuh"
#include "CellularAutomata/Solvers/MillanCellsGPUSolver.cuh"
#include "CellularAutomata/Solvers/MillanTopaGPUSolver.cuh"
#include "CellularAutomata/Solvers/SharedMemoryGPUSolver.cuh"
#include "CellularAutomata/Solvers/TensorCoreGPUSolver.cuh"
#include "Debug.h"
#include "Defines.h"
#include "Memory/Allocators/CPUAllocator.cuh"
#include "Memory/Allocators/GPUAllocator.cuh"

#define BEST_CELLS_PER_THREAD (2)

class CASolverFactory {
   public:
    static CASolver* createSolver(int SOLVER_CODE, int deviceId, int sideLength, int haloWidth);
};