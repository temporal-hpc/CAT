#include "CellularAutomata/CASolverFactory.cuh"
#include "Memory/CADataDomain.cuh"

static const int TENSOR_HALO_SIZE = 16;

CASolver* CASolverFactory::createSolver(int SOLVER_CODE, int deviceId, int fullHorizontalSize, int horizontalHaloSize) {
    CASolver* solver = nullptr;

    switch (SOLVER_CODE) {
        case 0: {
            CPUAllocator<int>* cpuAllocator = new CPUAllocator<int>();
            Allocator<int>* allocator = reinterpret_cast<Allocator<int>*>(cpuAllocator);
            CADataDomain<int>* dataDomain = new CADataDomain<int>(allocator, fullHorizontalSize, horizontalHaloSize);
            dataDomain->allocate();

            CADataDomain<int>* dataDomainBuffer = new CADataDomain<int>(allocator, fullHorizontalSize, horizontalHaloSize);
            dataDomainBuffer->allocate();

            solver = new HostSolver<int>(dataDomain, dataDomainBuffer);
            lDebug(1, "Solver of type HostSolver created");
            break;
        }
        case 1: {
            GPUAllocator<MTYPE>* gpuAllocator = new GPUAllocator<MTYPE>();
            Allocator<MTYPE>* allocator = reinterpret_cast<Allocator<MTYPE>*>(gpuAllocator);

            CADataDomain<MTYPE>* dataDomain = new CADataDomain<MTYPE>(allocator, fullHorizontalSize, horizontalHaloSize);
            dataDomain->allocate();

            CADataDomain<MTYPE>* dataDomainBuffer = new CADataDomain<MTYPE>(allocator, fullHorizontalSize, horizontalHaloSize);
            dataDomainBuffer->allocate();

            solver = new GlobalMemoryGPUSolver<MTYPE>(deviceId, dataDomain, dataDomainBuffer);
            lDebug(1, "Solver of type GlobalMemoryGPUSolver created");
            break;
        }
        case 2: {
            GPUAllocator<MTYPE>* gpuAllocator = new GPUAllocator<MTYPE>();
            Allocator<MTYPE>* allocator = reinterpret_cast<Allocator<MTYPE>*>(gpuAllocator);

            CADataDomain<MTYPE>* dataDomain = new CADataDomain<MTYPE>(allocator, fullHorizontalSize, horizontalHaloSize);
            dataDomain->allocate();

            CADataDomain<MTYPE>* dataDomainBuffer = new CADataDomain<MTYPE>(allocator, fullHorizontalSize, horizontalHaloSize);
            dataDomainBuffer->allocate();

            solver = new SharedMemoryGPUSolver<MTYPE>(deviceId, dataDomain, dataDomainBuffer);
            lDebug(1, "Solver of type SharedMemoryGPUSolver created");
            break;
        }
        case 3: {
            GPUAllocator<FTYPE>* gpuAllocator = new GPUAllocator<FTYPE>();
            Allocator<FTYPE>* allocator = reinterpret_cast<Allocator<FTYPE>*>(gpuAllocator);

            CADataDomain<FTYPE>* dataDomain = new CADataDomain<FTYPE>(allocator, fullHorizontalSize, TENSOR_HALO_SIZE);
            dataDomain->allocate();

            CADataDomain<FTYPE>* dataDomainBuffer = new CADataDomain<FTYPE>(allocator, fullHorizontalSize, TENSOR_HALO_SIZE);
            dataDomainBuffer->allocate();

            solver = new TensorCoreGPUSolver<FTYPE>(deviceId, dataDomain, dataDomainBuffer);
            lDebug(1, "Solver of type TensorCoreGPUSolver created");
            break;
        }
        case 4: {
            GPUAllocator<FTYPE>* gpuAllocator = new GPUAllocator<FTYPE>();
            Allocator<FTYPE>* allocator = reinterpret_cast<Allocator<FTYPE>*>(gpuAllocator);

            CADataDomain<FTYPE>* dataDomain = new CADataDomain<FTYPE>(allocator, fullHorizontalSize, TENSOR_HALO_SIZE);
            dataDomain->allocate();

            CADataDomain<FTYPE>* dataDomainBuffer = new CADataDomain<FTYPE>(allocator, fullHorizontalSize, TENSOR_HALO_SIZE);
            dataDomainBuffer->allocate();

            solver = new CoalescedTensorCoreGPUSolver<FTYPE>(deviceId, dataDomain, dataDomainBuffer);
            lDebug(1, "Solver of type CoalescedTensorCoreGPUSolver created");
            break;
        }
        case 5: {
            GPUAllocator<FTYPE>* gpuAllocator = new GPUAllocator<FTYPE>();
            Allocator<FTYPE>* allocator = reinterpret_cast<Allocator<FTYPE>*>(gpuAllocator);

            CADataDomain<FTYPE>* dataDomain = new CADataDomain<FTYPE>(allocator, fullHorizontalSize, TENSOR_HALO_SIZE);
            dataDomain->allocate();

            CADataDomain<FTYPE>* dataDomainBuffer = new CADataDomain<FTYPE>(allocator, fullHorizontalSize, TENSOR_HALO_SIZE);
            dataDomainBuffer->allocate();

            solver = new FastTensorCoreGPUSolver<FTYPE>(deviceId, dataDomain, dataDomainBuffer);
            lDebug(1, "Solver of type FastTensorCoreGPUSolver created");
            break;
        }
        case 6: {
            lDebug(1, "Creating solver of type MillanCellsGPUSolver");
            GPUAllocator<MTYPE>* gpuAllocator = new GPUAllocator<MTYPE>();
            Allocator<MTYPE>* allocator = reinterpret_cast<Allocator<MTYPE>*>(gpuAllocator);

            CADataDomain<MTYPE>* dataDomain = new CADataDomain<MTYPE>(allocator, fullHorizontalSize, horizontalHaloSize);
            dataDomain->allocate();

            CADataDomain<MTYPE>* dataDomainBuffer = new CADataDomain<MTYPE>(allocator, fullHorizontalSize, horizontalHaloSize);
            dataDomainBuffer->allocate();

            solver = new MillanCellsGPUSolver<MTYPE>(deviceId, dataDomain, dataDomainBuffer, BEST_CELLS_PER_THREAD);
            lDebug(1, "Solver of type MillanCellsGPUSolver created");
            break;
        }
        case 7: {
            lDebug(1, "Creating solver of type MillanTopaGPUSolver");
            GPUAllocator<MTYPE>* gpuAllocator = new GPUAllocator<MTYPE>();
            Allocator<MTYPE>* allocator = reinterpret_cast<Allocator<MTYPE>*>(gpuAllocator);

            CADataDomain<MTYPE>* dataDomain = new CADataDomain<MTYPE>(allocator, fullHorizontalSize, horizontalHaloSize);
            dataDomain->allocate();

            CADataDomain<MTYPE>* dataDomainBuffer = new CADataDomain<MTYPE>(allocator, fullHorizontalSize, horizontalHaloSize);
            dataDomainBuffer->allocate();

            solver = new MillanTopaGPUSolver<MTYPE>(deviceId, dataDomain, dataDomainBuffer);
            lDebug(1, "Solver of type MillanTopaGPUSolver created");
            break;
        }
        case 8: {
            lDebug(1, "Creating solver of type CagigasPacketCoding64GPUSolver");
            GPUAllocator<uint64_t>* gpuAllocator = new GPUAllocator<uint64_t>();
            Allocator<uint64_t>* allocator = reinterpret_cast<Allocator<uint64_t>*>(gpuAllocator);

            int packedSideLength = fullHorizontalSize / CagigasPacketCoding64GPUSolver::elementsPerCel;
            float packedHaloWidth = horizontalHaloSize / CagigasPacketCoding64GPUSolver::elementsPerCel;

            CADataDomain<uint64_t>* dataDomain = new CADataDomain<uint64_t>(allocator, packedSideLength, fullHorizontalSize, (int)ceil(packedHaloWidth), horizontalHaloSize);
            dataDomain->allocate();

            lDebug(1, "Size of dataDomain: %llu ((%i+%i)x(%i+%i))", dataDomain->getTotalSize(), dataDomain->getInnerHorizontalSize(), dataDomain->getHorizontalHaloSize(), dataDomain->getInnerVerticalSize(), dataDomain->getVerticalHaloSize());

            CADataDomain<uint64_t>* dataDomainBuffer = new CADataDomain<uint64_t>(allocator, packedSideLength, fullHorizontalSize, (int)ceil(packedHaloWidth), horizontalHaloSize);
            dataDomainBuffer->allocate();

            solver = new CagigasPacketCoding64GPUSolver(deviceId, dataDomain, dataDomainBuffer);
            lDebug(1, "Solver of type CagigasPacketCoding64GPUSolver created");
            break;
        }
    }
    return solver;
}
