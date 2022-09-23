#include "TensorCA2D.cuh"

#include "GPUKernels.cuh"
#include "GPUTools.cuh"
#include "gif.h"
#include <random>
//#define OFFSET 0.0f
#define REAL float

TensorCA2D::TensorCA2D(uint32_t deviceId, uint32_t n, uint32_t modeCode, float density)
    : deviceId(deviceId)
    , n(n)
    , density(density) {

    this->nWithHalo = n + HALO_SIZE;
    this->hasBeenAllocated = false;
    this->nElements = this->nWithHalo * this->nWithHalo;

    switch (modeCode) {
    case 1:
        this->mode = Mode::CLASSICV1;
        break;
    case 2:
        this->mode = Mode::CLASSICV2;
        break;
    case 3:
        this->mode = Mode::TENSORCA;
        break;
    default:
        this->mode = Mode::NOT_IMPLEMENTED;
        this->nElements = 0;
        break;
    }

    lDebug(1, "Created TensorCA2D: n=%u, nWithHalo=%u, nElements=%lu, modeCode=%i", n, nWithHalo, nElements, modeCode);
}

TensorCA2D::~TensorCA2D() {
    if (this->hasBeenAllocated) {
        freeMemory();
    }
}

void TensorCA2D::allocateMemory() {
    if (this->hasBeenAllocated) {
        lDebug(1, "Memory already allocated.");
        return;
    }
    this->hostData = (MTYPE*)malloc(sizeof(MTYPE) * this->nElements);
    cudaMalloc(&devDataPing, sizeof(MTYPE) * nElements);
    cudaMalloc(&devDataPong, sizeof(MTYPE) * nElements);
    gpuErrchk(cudaPeekAtLastError());
    this->hasBeenAllocated = true;
}

void TensorCA2D::freeMemory() {
    // clear
    free(hostData);
    cudaFree(devDataPing);
    cudaFree(devDataPong);
    gpuErrchk(cudaPeekAtLastError());
    this->hasBeenAllocated = false;
}

bool TensorCA2D::init(uint32_t seed) {
    // Change default random engine to mt19937
    srand(seed);
    lDebug(1, "Selecting device %i", this->deviceId);
    gpuErrchk(cudaSetDevice(this->deviceId));

    lDebug(1, "Allocating memory.");
    this->allocateMemory();
    lDebug(1, "Memory allocated.");

    switch (this->mode) {
    case Mode::CLASSICV1:
        this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
        this->GPUGrid = dim3((n + GPUBlock.x - 1) / GPUBlock.x, (n + GPUBlock.y - 1) / GPUBlock.y);
        break;
    case Mode::CLASSICV2:
        if (BSIZE3DX < 3 || BSIZE3DY < 3) {
            lDebug(1, "Error. ClassicV2 mode requires a square block with sides >= 3");
            return false;
        }
        this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
        this->GPUGrid = dim3((n + (GPUBlock.x - HALO_SIZE) - 1) / (GPUBlock.x - HALO_SIZE), (n + (GPUBlock.y - HALO_SIZE) - 1) / (GPUBlock.y - HALO_SIZE));
        break;
    case Mode::TENSORCA:
        if (BSIZE3DX * BSIZE3DY % 32 != 0) {
            lDebug(1, "Error. TENSORCA mode requires a CTA size such that size%32 == 0");
            return false;
        }
        this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
        this->GPUGrid = dim3((n + GPUBlock.x - 1) / GPUBlock.x, (n + GPUBlock.y - 1) / GPUBlock.y);
        break;
    }

    lDebug(1, "Parallel space: b(%i, %i, %i) g(%i, %i, %i)", GPUBlock.x, GPUBlock.y, GPUBlock.z, GPUGrid.x, GPUGrid.y, GPUGrid.z);

    this->reset();

    lDebug(1, "Transfering data to device.");
    this->transferHostToDevice();
    lDebug(1, "Done.");

    return true;
}

void TensorCA2D::transferHostToDevice() {
    cudaMemcpy(this->devDataPing, this->hostData, sizeof(MTYPE) * this->nElements, cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}
void TensorCA2D::transferDeviceToHost() {
    cudaMemcpy(this->hostData, this->devDataPing, sizeof(MTYPE) * this->nElements, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

float TensorCA2D::doBenchmarkAction(uint32_t nTimes) {

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
    // int width = this->nWithHalo;
    // int height = this->nWithHalo;

    // auto fileName = "bwgif.gif";
    // int delay = 10;
    // GifWriter g;
    // GifBegin(&g, fileName, width, height, delay);
    // std::vector<uint8_t> frame(nElements * 4);
    switch (this->mode) {
    case Mode::CLASSICV1:
        for (uint32_t i = 0; i < nTimes; ++i) {
            ClassicV1GoLStep<<<this->GPUGrid, this->GPUBlock>>>(this->devDataPing, this->devDataPong, this->n, this->nWithHalo);
            gpuErrchk(cudaDeviceSynchronize());
            std::swap(this->devDataPing, this->devDataPong);
            // this->transferDeviceToHost();
            // for (int l = 0; l < nElements; l++) {
            //     frame[l * 4 + 0] = (uint8_t)this->hostData[l] * 255;
            //     frame[l * 4 + 1] = (uint8_t)this->hostData[l] * 255;
            //     frame[l * 4 + 2] = (uint8_t)this->hostData[l] * 255;
            //     frame[l * 4 + 3] = (uint8_t)this->hostData[l] * 255;
            // }
            // GifWriteFrame(&g, frame.data(), width, height, delay);
        }
        break;
    case Mode::CLASSICV2:
        for (uint32_t i = 0; i < nTimes; ++i) {
            ClassicV2GoLStep<<<this->GPUGrid, this->GPUBlock>>>(this->devDataPing, this->devDataPong, this->n, this->nWithHalo);
            gpuErrchk(cudaDeviceSynchronize());
            std::swap(this->devDataPing, this->devDataPong);
            // this->transferDeviceToHost();
            // for (int l = 0; l < nElements; l++) {
            //     frame[l * 4 + 0] = (uint8_t)this->hostData[l] * 255;
            //     frame[l * 4 + 1] = (uint8_t)this->hostData[l] * 255;
            //     frame[l * 4 + 2] = (uint8_t)this->hostData[l] * 255;
            //     frame[l * 4 + 3] = (uint8_t)this->hostData[l] * 255;
            // }
            // GifWriteFrame(&g, frame.data(), width, height, delay);
        }
        break;
    }

    cudaEventRecord(stop);
    // GifEnd(&g);

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

bool TensorCA2D::isInHalo(size_t i) {
    uint32_t x = i % this->nWithHalo;
    uint32_t y = i / this->nWithHalo;
    return (x == 0 || y == 0 || x == nWithHalo - 1 || y == nWithHalo - 1);
}
// Should we incvlude the seed in this function call
void TensorCA2D::reset() {
    // rseed()
    lDebug(1, "Resetting data to the initial state.");

    for (size_t i = 0; i < this->nElements; ++i) {

        if (!this->isInHalo(i) && rand() / (double)RAND_MAX < density) {
            this->hostData[i] = (MTYPE)1;

        } else {
            this->hostData[i] = (MTYPE)0;
        }

        // uint32_t x = i % this->nWithHalo;
        // uint32_t y = i / this->nWithHalo;
        // if (x == 0 || y == 0 || x == nWithHalo - 1 || y == nWithHalo - 1) {
        //     this->hostData[i] = (MTYPE)1;

        // } else {
        //     this->hostData[i] = (MTYPE)0;
        // }
    }
    lDebug(1, "Transfering data to device.");
    this->transferHostToDevice();
    lDebug(1, "Done.");

    lDebug(1, "Initial status in Host:");

    fDebug(1, this->printHostData());
}

void TensorCA2D::printHostData() {
    for (size_t i = 0; i < nElements; i++) {

        if ((int)this->hostData[i] == 99999) {
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

void TensorCA2D::printDeviceData() {
    transferDeviceToHost();
    printHostData();
}

bool TensorCA2D::compare(TensorCA2D* a, TensorCA2D* b) {
    bool res = true;
    if (a->nElements != b->nElements) {
        return false;
    }
    for (size_t i = 0; i < a->nElements; ++i) {
        if (a->hostData[i] != b->hostData[i]) {
            printf("a(%lu) = %i != %i b\n", i, a->hostData[i], b->hostData[i]);
            res = false;
        }
    }

    return res;
}
