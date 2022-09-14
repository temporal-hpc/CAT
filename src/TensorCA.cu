#include "TensorCA.cuh"

#include "GPUKernels.cuh"
#include "GPUTools.cuh"

#include <random>
//#define OFFSET 0.0f
#define REAL float

TensorCA::TensorCA(uint32_t deviceId, uint32_t n, uint32_t dimensions, float density, uint32_t seed)
    : deviceId(deviceId)
    , n(n)
    , dimensions(dimensions)
    , density(density)
    , seed(seed) {

    this->nWithHalo = n + HALO_SIZE;
    this->hasBeenAllocated = false;
    // Change default random engine to mt19937
    // rseed(seed);
    switch (dimensions) {
    case 1:
        this->mode = Mode::CA1D;
        this->nElements = this->nWithHalo;
        break;
    case 2:
        this->mode = Mode::CA2D;
        this->nElements = this->nWithHalo * this->nWithHalo;
        break;
    case 3:
        this->mode = Mode::CA3D;
        this->nElements = this->nWithHalo * this->nWithHalo * this->nWithHalo;
        break;
    default:
        this->mode = Mode::NOT_IMPLEMENTED;
        this->nElements = 0;
        break;
    }

    lDebug(1, "Created TensorCA: n=%u, nWithHalo=%u, nElements=%lu, dimensions=%i", n, nWithHalo, nElements, dimensions);
}

TensorCA::~TensorCA() {
    if (this->hasBeenAllocated) {
        freeMemory();
    }
}

void TensorCA::allocateMemory() {
    if (this->hasBeenAllocated) {
        lDebug(1, "Memory already allocated.");
        return;
    }
    this->hostData = (MTYPE*)malloc(sizeof(MTYPE) * this->nElements);
    cudaMalloc(&devData, sizeof(MTYPE) * nElements);
    gpuErrchk(cudaPeekAtLastError());
    this->hasBeenAllocated = true;
}

void TensorCA::freeMemory() {
    // clear
    free(hostData);
    cudaFree(devData);
    gpuErrchk(cudaPeekAtLastError());
    this->hasBeenAllocated = false;
}

bool TensorCA::init() {

    lDebug(1, "Selecting device %i", this->deviceId);
    gpuErrchk(cudaSetDevice(this->deviceId));

    lDebug(1, "Allocating memory.");
    this->allocateMemory();
    lDebug(1, "Memory allocated.");

    switch (this->mode) {
    case Mode::CA1D:
        this->GPUBlock = dim3(BSIZE3DX);
        this->GPUGrid = dim3((n / 2 + GPUBlock.x - 1) / GPUBlock.x);
        break;
    case Mode::CA2D:
        this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
        this->GPUGrid = dim3((n / 2 + GPUBlock.x - 1) / GPUBlock.x, (n / 2 + GPUBlock.y - 1) / GPUBlock.y);
        break;

    case Mode::CA3D:
        this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY, BSIZE3DZ);
        this->GPUGrid = dim3((n / 2 + GPUBlock.x - 1) / GPUBlock.x, (n / 2 + GPUBlock.y - 1) / GPUBlock.y, (n / 2 + GPUBlock.z - 1) / GPUBlock.z);
        break;
    }

    lDebug(1, "Parallel space: b(%i, %i, %i) g(%i, %i, %i)", GPUBlock.x, GPUBlock.y, GPUBlock.z, GPUGrid.x, GPUGrid.y, GPUGrid.z);

    this->reset();

    lDebug(1, "Transfering data to device.");
    this->transferHostToDevice();
    lDebug(1, "Done.");

    return true;
}

void TensorCA::transferHostToDevice() {
    cudaMemcpy(this->devData, this->hostData, sizeof(MTYPE) * this->nElements, cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}
void TensorCA::transferDeviceToHost() {
    cudaMemcpy(this->hostData, this->devData, sizeof(MTYPE) * this->nElements, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

float TensorCA::doBenchmarkAction(uint32_t nTimes) {

    lDebug(1, "Mapping to simplex of n=%lu   nWithHalo = %lu   nElements = %lu\n", this->n, this->nWithHalo, this->nElements);
    lDebug(1, "Cube size is %f MB\n", (float)this->nElements * sizeof(MTYPE) / (1024.0 * 1024.0f));

    // begin performance tests
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    lDebug(1, "Kernel (map=%i, rep=%i)", this->mode, nTimes);

    cudaEventRecord(start);
#ifdef MEASURE_POWER
    GPUPowerBegin(this->n, 100, 0, std::string("AutomataTC-") + std::to_string(this->deviceId));
#endif

    for (uint32_t i = 0; i < nTimes; ++i) {
        // KERNEL GOES HERE
        gpuErrchk(cudaDeviceSynchronize());
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#ifdef MEASURE_POWER
    GPUPowerEnd();
#endif
    lDebug(1, "Done");

    // return computing time
    float msecs = 0;
    cudaEventElapsedTime(&msecs, start, stop);
    return msecs / ((float)nTimes);
}

// Should we incvlude the seed in this function call
void TensorCA::reset() {
    // rseed()
    lDebug(1, "Resetting data to the initial state.");

    for (size_t i = 0; i < this->nElements; ++i) {
        if (rand() / (double)RAND_MAX < density) {
            this->hostData[i] = (MTYPE)1;

        } else {
            this->hostData[i] = (MTYPE)0;
        }
    }
    lDebug(1, "Transfering data to device.");
    this->transferHostToDevice();
    lDebug(1, "Done.");
}

void TensorCA::printHostData() {
    for (size_t i = 0; i < nElements; i++) {

        if ((int)this->hostData[i] == 0) {
            printf("  ");
        } else {
            printf("%i ", (int)this->hostData[i]);
        }

        if (i % (nWithHalo) == nWithHalo - 1) {
            printf("\n");
        }
        if (i % (nWithHalo * nWithHalo) == nWithHalo * nWithHalo - 1) {
            printf("\n");
        }
    }
}

void TensorCA::printDeviceData() {
    transferDeviceToHost();
    printHostData();
}

bool TensorCA::compare(TensorCA* a, TensorCA* b) {
    bool res = true;
    if (a->nElements != b->nElements) {
        return false;
    }
    for (size_t i = 0; i < a->nElements; ++i) {
        if (a->hostData[i] != b->hostData[i]) {
            // printf("a[%lu, %lu, %lu] (%lu) = %i != %i b\n", x, i, z, i, a->hostData[i], b->hostData[i]);
            res = false;
        }
    }

    return res;
}
