#include "CellularAutomata/CASolverFactory.cuh"
#include "Memory/CADataDomain.cuh"

static const int TENSOR_HALO_SIZE = 16;

CASolver *CASolverFactory::createSolver(int SOLVER_CODE, int deviceId, int fullHorizontalSize, int horizontalHaloSize)
{
    CASolver *solver = nullptr;

    switch (SOLVER_CODE)
    {
    case 0: {
        GPUAllocator<MTYPE> *gpuAllocator = new GPUAllocator<MTYPE>();
        Allocator<MTYPE> *allocator = reinterpret_cast<Allocator<MTYPE> *>(gpuAllocator);

        CADataDomain<MTYPE> *dataDomain = new CADataDomain<MTYPE>(allocator, fullHorizontalSize, horizontalHaloSize);
        dataDomain->allocate();

        CADataDomain<MTYPE> *dataDomainBuffer =
            new CADataDomain<MTYPE>(allocator, fullHorizontalSize, horizontalHaloSize);
        dataDomainBuffer->allocate();

        solver = new BASE<MTYPE>(deviceId, dataDomain, dataDomainBuffer);
        lDebug(1, "Solver of type BASE created");
        break;
    }
    case 1: {
        lDebug(1, "Creating solver of type SHARED");
        GPUAllocator<MTYPE> *gpuAllocator = new GPUAllocator<MTYPE>();
        Allocator<MTYPE> *allocator = reinterpret_cast<Allocator<MTYPE> *>(gpuAllocator);

        CADataDomain<MTYPE> *dataDomain = new CADataDomain<MTYPE>(allocator, fullHorizontalSize, horizontalHaloSize);
        dataDomain->allocate();

        CADataDomain<MTYPE> *dataDomainBuffer =
            new CADataDomain<MTYPE>(allocator, fullHorizontalSize, horizontalHaloSize);
        dataDomainBuffer->allocate();

        solver = new SHARED<MTYPE>(deviceId, dataDomain, dataDomainBuffer);
        lDebug(1, "Solver of type SHARED created");
        break;
    }
    case 2: {
        GPUAllocator<MTYPE> *gpuAllocator = new GPUAllocator<MTYPE>();
        Allocator<MTYPE> *allocator = reinterpret_cast<Allocator<MTYPE> *>(gpuAllocator);

        CADataDomain<MTYPE> *dataDomain = new CADataDomain<MTYPE>(allocator, fullHorizontalSize, horizontalHaloSize);
        dataDomain->allocate();

        CADataDomain<MTYPE> *dataDomainBuffer =
            new CADataDomain<MTYPE>(allocator, fullHorizontalSize, horizontalHaloSize);
        dataDomainBuffer->allocate();

        solver = new COARSE<MTYPE>(deviceId, dataDomain, dataDomainBuffer);
        lDebug(1, "Solver of type COARSE created");
        break;
    }
    case 3: {
        GPUAllocator<FTYPE> *gpuAllocator = new GPUAllocator<FTYPE>();
        Allocator<FTYPE> *allocator = reinterpret_cast<Allocator<FTYPE> *>(gpuAllocator);

        CADataDomain<FTYPE> *dataDomain = new CADataDomain<FTYPE>(allocator, fullHorizontalSize, TENSOR_HALO_SIZE);
        dataDomain->allocate();

        CADataDomain<FTYPE> *dataDomainBuffer =
            new CADataDomain<FTYPE>(allocator, fullHorizontalSize, TENSOR_HALO_SIZE);
        dataDomainBuffer->allocate();

        solver = new CAT<FTYPE>(deviceId, dataDomain, dataDomainBuffer);
        lDebug(1, "Solver of type CAT created");
        break;
    }
    case 4: {
        lDebug(1, "Creating solver of type MCELL");
        GPUAllocator<MTYPE> *gpuAllocator = new GPUAllocator<MTYPE>();
        Allocator<MTYPE> *allocator = reinterpret_cast<Allocator<MTYPE> *>(gpuAllocator);

        CADataDomain<MTYPE> *dataDomain = new CADataDomain<MTYPE>(allocator, fullHorizontalSize, horizontalHaloSize);
        dataDomain->allocate();

        CADataDomain<MTYPE> *dataDomainBuffer =
            new CADataDomain<MTYPE>(allocator, fullHorizontalSize, horizontalHaloSize);
        dataDomainBuffer->allocate();

        solver = new MCELL<MTYPE>(deviceId, dataDomain, dataDomainBuffer, BEST_CELLS_PER_THREAD);
        lDebug(1, "Solver of type MCELL created");
        break;
    }

    case 5: {
        lDebug(1, "Creating solver of type PACK");
        GPUAllocator<uint64_t> *gpuAllocator = new GPUAllocator<uint64_t>();
        Allocator<uint64_t> *allocator = reinterpret_cast<Allocator<uint64_t> *>(gpuAllocator);

        int packedSideLength = fullHorizontalSize / PACK::elementsPerCel;
        float packedHaloWidth = horizontalHaloSize / PACK::elementsPerCel;

        CADataDomain<uint64_t> *dataDomain = new CADataDomain<uint64_t>(allocator, packedSideLength, fullHorizontalSize,
                                                                        (int)ceil(packedHaloWidth), horizontalHaloSize);
        dataDomain->allocate();

        lDebug(1, "Size of dataDomain: %llu ((%i+%i)x(%i+%i))", dataDomain->getTotalSize(),
               dataDomain->getInnerHorizontalSize(), dataDomain->getHorizontalHaloSize(),
               dataDomain->getInnerVerticalSize(), dataDomain->getVerticalHaloSize());

        CADataDomain<uint64_t> *dataDomainBuffer = new CADataDomain<uint64_t>(
            allocator, packedSideLength, fullHorizontalSize, (int)ceil(packedHaloWidth), horizontalHaloSize);
        dataDomainBuffer->allocate();

        solver = new PACK(deviceId, dataDomain, dataDomainBuffer);
        lDebug(1, "Solver of type PACK created");
        break;
    }
    case 7: {
        GPUAllocator<FTYPE> *gpuAllocator = new GPUAllocator<FTYPE>();
        Allocator<FTYPE> *allocator = reinterpret_cast<Allocator<FTYPE> *>(gpuAllocator);

        CADataDomain<FTYPE> *dataDomain = new CADataDomain<FTYPE>(allocator, fullHorizontalSize, TENSOR_HALO_SIZE);
        dataDomain->allocate();

        CADataDomain<FTYPE> *dataDomainBuffer =
            new CADataDomain<FTYPE>(allocator, fullHorizontalSize, TENSOR_HALO_SIZE);
        dataDomainBuffer->allocate();

        solver = new CATWithoutCAT<FTYPE>(deviceId, dataDomain, dataDomainBuffer);
        lDebug(1, "Solver of type CATWithoutCAT created");
        break;
    }

    }
    return solver;
}
