#include "CellularAutomata/CASolverFactory.cuh"
#include "Memory/CADataDomain.cuh"

CASolver* CASolverFactory::createSolver(int SOLVER_CODE, int sideLength, int haloWidth) {
    CASolver* solver = nullptr;
    switch (SOLVER_CODE) {
        case 0: {
            CPUAllocator<int>* cpuAllocator = new CPUAllocator<int>();
            Allocator<int>* allocator = reinterpret_cast<Allocator<int>*>(cpuAllocator);
            CADataDomain<int>* dataDomain = new CADataDomain<int>(allocator, sideLength, haloWidth);
            dataDomain->allocate();
            CADataDomain<int>* dataDomainBuffer = new CADataDomain<int>(allocator, sideLength, haloWidth);
            dataDomainBuffer->allocate();
            solver = new HostSolver<int>(dataDomain, dataDomainBuffer);
        }
    }
    return solver;
}
