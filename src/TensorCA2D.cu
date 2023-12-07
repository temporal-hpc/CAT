#include "TensorCA2D.cuh"

#include <random>
#include "GPUKernels.cuh"
#include "GPUTools.cuh"
#include "gif.h"
// #define OFFSET 0.0f
#define REAL float

TensorCA2D::TensorCA2D(uint32_t deviceId, uint32_t n, uint32_t modeCode, float density) : deviceId(deviceId), n(n), density(density) {
    this->hasBeenAllocated = false;

    switch (modeCode) {
        case 0:
            this->mode = Mode::CLASSICGBMEM;
            this->haloWidth = R;
            this->nWithHalo = n + HALO_SIZE;
            break;
        case 1:
            this->mode = Mode::CLASSICV1;
            this->haloWidth = R;
            this->nWithHalo = n + HALO_SIZE;
            break;
        case 2:
            this->mode = Mode::CLASSICV2;
            this->haloWidth = R;
            this->nWithHalo = n + HALO_SIZE;
            break;
        case 3:
            this->mode = Mode::TENSORCA;
            //                ⮟ due to fragment size
            this->haloWidth = 16;
            this->nWithHalo = n + (this->haloWidth * 2);
            if (NREGIONS_H * 16 > this->n) {
                lDebug(1, "NREGIONSH*16 < n. Shared memory will be significatly larger\n");
            }
            if (NREGIONS_V * 16 > this->n) {
                lDebug(1, "NREGIONSV*16 < n. Shared memory will be significatly larger\n");
            }
            break;
        case 4:
            this->mode = Mode::TENSORCACOALESCED;
            //                ⮟ due to fragment size
            this->haloWidth = 16;
            this->nWithHalo = n + (this->haloWidth * 2);
            if (NREGIONS_H * 16 > this->n) {
                lDebug(1, "NREGIONSH*16 < n. Shared memory will be significatly larger\n");
            }
            if (NREGIONS_V * 16 > this->n) {
                lDebug(1, "NREGIONSV*16 < n. Shared memory will be significatly larger\n");
            }
            break;
        case 5:
            this->mode = Mode::CLASSICGBMEMHALF;
            this->haloWidth = R;
            this->nWithHalo = n + HALO_SIZE;
            break;
        case 6:
            this->mode = Mode::TENSORCACOALESCEDMORETHREADS;
            this->haloWidth = 16;
            this->nWithHalo = n + (this->haloWidth * 2);
            if (NREGIONS_H * 16 > this->n) {
                lDebug(1, "NREGIONSH*16 < n. Shared memory will be significatly larger\n");
            }
            if (NREGIONS_V * 16 > this->n) {
                lDebug(1, "NREGIONSV*16 < n. Shared memory will be significatly larger\n");
            }
            break;
        case 7:
            this->mode = Mode::TENSORCACOALESCEDLESSSHMEM;
            this->haloWidth = 16;
            this->nWithHalo = n + (this->haloWidth * 2);
            if (NREGIONS_H * 16 > this->n) {
                lDebug(1, "NREGIONSH*16 < n. Shared memory will be significatly larger\n");
            }
            if (NREGIONS_V * 16 > this->n) {
                lDebug(1, "NREGIONSV*16 < n. Shared memory will be significatly larger\n");
            }
            break;
        case 8:
            this->mode = Mode::TENSORCACOALESCEDNOSHMEM;
            this->haloWidth = 16;
            this->nWithHalo = n + (this->haloWidth * 2);
            if (NREGIONS_H * 16 > this->n) {
                lDebug(1, "NREGIONSH*16 < n. Shared memory will be significatly larger\n");
            }
            if (NREGIONS_V * 16 > this->n) {
                lDebug(1, "NREGIONSV*16 < n. Shared memory will be significatly larger\n");
            }
            break;
        case 9:
            this->mode = Mode::TENSORCACOALESCEDLESSSHMEMINT4;
            this->haloWidth = 32;
            // this->haloWidthY = 32;
            this->nWithHalo = n + (this->haloWidth * 2);
            if (n % 8 != 0 || nWithHalo % 8 != 0) {
                lDebug(1, "n is not a multiple of 8. Exiting...\n");
                exit(-1);
            }
            if (NREGIONS_H * 32 > this->n) {
                lDebug(1, "NREGIONSH*32 < n. Shared memory will be significatly larger\n");
            }
            if (NREGIONS_V * 32 > this->n) {
                lDebug(1, "NREGIONSV*32 < n. Shared memory will be significatly larger\n");
            }
            break;
        case 10:
            this->mode = Mode::TENSORCACOALESCEDLESSSHMEMINT8;
            this->haloWidth = 16;
            this->nWithHalo = n + (this->haloWidth * 2);
            if (NREGIONS_H * 16 > this->n) {
                lDebug(1, "NREGIONSH*16 < n. Shared memory will be significatly larger\n");
            }
            if (NREGIONS_V * 16 > this->n) {
                lDebug(1, "NREGIONSV*16 < n. Shared memory will be significatly larger\n");
            }
            break;
        default:
            this->mode = Mode::NOT_IMPLEMENTED;
            this->nElements = 0;
            break;
    }
    this->nElements = this->nWithHalo * this->nWithHalo;

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
    lDebug(1, "Allocating %.2f MB in Host [hostData]", (sizeof(MTYPE) * this->nElements) / (double)1000000.0);
    this->hostData = (MTYPE*)malloc(sizeof(MTYPE) * this->nElements);
    if (this->mode == Mode::TENSORCA || this->mode == Mode::TENSORCACOALESCED || this->mode == Mode::CLASSICGBMEMHALF || this->mode == Mode::TENSORCACOALESCEDMORETHREADS || this->mode == Mode::TENSORCACOALESCEDLESSSHMEM || this->mode == Mode::TENSORCACOALESCEDNOSHMEM || this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT4V2) {
        lDebug(1, "Allocating %.2f MB in Device [devDataBufferTensor]", (sizeof(MTYPE) * nElements) / (double)1000000.0);
        cudaMalloc(&devDataBufferTensor, sizeof(MTYPE) * nElements);
        lDebug(1, "Allocating %.2f MB in Device [devDataPingTensor]", (sizeof(FTYPE) * nElements) / (double)1000000.0);
        cudaMalloc(&devDataPingTensor, sizeof(FTYPE) * nElements);
        lDebug(1, "Allocating %.2f MB in Device [devDataPongTensor]", (sizeof(FTYPE) * nElements) / (double)1000000.0);
        cudaMalloc(&devDataPongTensor, sizeof(FTYPE) * nElements);
    } else if (this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT4) {
        lDebug(1, "Allocating %.2f MB in Device [devDataBufferTensor]", (sizeof(int) * nElements) / (double)1000000.0);
        cudaMalloc(&devDataBufferTensor, sizeof(int) * nElements);
        lDebug(1, "Allocating %.2f MB in Device [devDataBufferTensor2]", (sizeof(int) * nElements) / (double)1000000.0);
        cudaMalloc(&devDataBufferTensor2, sizeof(int) * nElements);
        lDebug(1, "Allocating %.2f MB in Device [devDataPingTensor]", (sizeof(int) * nElements / 8) / (double)1000000.0);
        cudaMalloc(&devDataPingTensorInt4, sizeof(int) * nElements / 8);
        cudaMemset(devDataPingTensorInt4, 0, sizeof(int) * nElements / 8);
        cudaMemset(devDataBufferTensor, 0, sizeof(int) * nElements);
        cudaMemset(devDataBufferTensor2, 0, sizeof(int) * nElements);
        // lDebug(1, "Allocating %.2f MB in Device [devDataPongTensor]", (sizeof(int) * nElements/8) / (double)1000000.0);
        // cudaMalloc(&devDataPongTensorInt4, sizeof(int) * nElements/8);
    } else if (this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT8) {
        lDebug(1, "Allocating %.2f MB in Device [devDataBufferTensorInt8]", (sizeof(int) * nElements) / (double)1000000.0);
        cudaMalloc(&devDataBufferTensorInt8, sizeof(int) * nElements);
        lDebug(1, "Allocating %.2f MB in Device [devDataPingTensorInt8]", (sizeof(char) * nElements) / (double)1000000.0);
        cudaMalloc(&devDataPingTensorInt8, sizeof(char) * nElements);
        lDebug(1, "Allocating %.2f MB in Device [devDataPongTensorInt8]", (sizeof(char) * nElements) / (double)1000000.0);
        cudaMalloc(&devDataPongTensorInt8, sizeof(char) * nElements);

    } else {
        lDebug(1, "Allocating %.2f MB in Device [devDataPing]", (sizeof(MTYPE) * nElements) / (double)1000000.0);
        cudaMalloc(&devDataPing, sizeof(MTYPE) * nElements);
        lDebug(1, "Allocating %.2f MB in Device [devDataPong]", (sizeof(MTYPE) * nElements) / (double)1000000.0);
        cudaMalloc(&devDataPong, sizeof(MTYPE) * nElements);
    }
    gpuErrchk(cudaPeekAtLastError());
    this->hasBeenAllocated = true;
}

void TensorCA2D::freeMemory() {
    // clear
    free(hostData);
    if (this->mode == Mode::TENSORCA || this->mode == Mode::TENSORCACOALESCED || this->mode == Mode::CLASSICGBMEMHALF || this->mode == Mode::TENSORCACOALESCEDMORETHREADS || this->mode == Mode::TENSORCACOALESCEDLESSSHMEM || this->mode == Mode::TENSORCACOALESCEDNOSHMEM || this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT4V2) {
        cudaFree(devDataPingTensor);
        cudaFree(devDataPongTensor);
        cudaFree(devDataBufferTensor);
    } else if (this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT4) {
        cudaFree(devDataBufferTensor);
        cudaFree(devDataBufferTensor2);
        cudaFree(devDataPingTensorInt4);
    } else if (this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT8) {
        cudaFree(devDataPingTensorInt8);
        cudaFree(devDataPongTensorInt8);
        // cudaFree(devDataPongTensorInt4);
        cudaFree(devDataBufferTensorInt8);
    } else {
        cudaFree(devDataPing);
        cudaFree(devDataPong);
    }
    gpuErrchk(cudaPeekAtLastError());
    this->hasBeenAllocated = false;
}

bool TensorCA2D::init(uint32_t seed) {
    // Change default random engine to mt19937
    this->seed = seed;
    lDebug(1, "Selecting device %i", this->deviceId);
    gpuErrchk(cudaSetDevice(this->deviceId));

    lDebug(1, "Allocating memory.");
    //                                          Fragment size
    if ((this->mode == Mode::TENSORCA || this->mode == Mode::TENSORCACOALESCED || this->mode == Mode::TENSORCACOALESCEDMORETHREADS || this->mode == Mode::TENSORCACOALESCEDLESSSHMEM || this->mode == Mode::TENSORCACOALESCEDNOSHMEM || this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT8) && this->n % 16 != 0) {
        lDebug(1, "Error, n must be a multiple of 16 for this to work properly.");
        return false;
    } else if ((this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT4 || this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT4V2) && this->n % 32 != 0) {
        lDebug(1, "Error, n must be a multiple of 32 for this to work properly.");
        return false;
    }
    this->allocateMemory();
    lDebug(1, "Memory allocated.");

    switch (this->mode) {
        case Mode::CLASSICGBMEM:
            this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
            this->GPUGrid = dim3((n + GPUBlock.x - 1) / GPUBlock.x, (n + GPUBlock.y - 1) / GPUBlock.y);
            break;
        case Mode::CLASSICV1:
            this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
            this->GPUGrid = dim3((n + 80 - 1) / 80, (n + 80 - 1) / 80);
            break;
        case Mode::CLASSICV2:
            if (BSIZE3DX < HALO_SIZE + 1 || BSIZE3DY < HALO_SIZE + 1) {
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
            this->GPUGrid = dim3((n + (NREGIONS_H * 16) - 1) / (NREGIONS_H * 16), (n + (NREGIONS_V * 16) - 1) / (NREGIONS_V * 16));
            break;
        case Mode::TENSORCACOALESCED:
            if (BSIZE3DX * BSIZE3DY % 32 != 0) {
                lDebug(1, "Error. TENSORCA mode requires a CTA size such that size%32 == 0");
                return false;
            }
            this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
            this->GPUGrid = dim3((n + (NREGIONS_H * 16) - 1) / (NREGIONS_H * 16), (n + (NREGIONS_V * 16) - 1) / (NREGIONS_V * 16));
            break;
        case Mode::CLASSICGBMEMHALF:
            this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
            this->GPUGrid = dim3((n + GPUBlock.x - 1) / GPUBlock.x, (n + GPUBlock.y - 1) / GPUBlock.y);
            break;
        case Mode::TENSORCACOALESCEDMORETHREADS:
            if (BSIZE3DX * BSIZE3DY % 32 != 0) {
                lDebug(1, "Error. TENSORCA mode requires a CTA size such that size%32 == 0");
                return false;
            }
            if (NREGIONS_H - 2 <= 0 || NREGIONS_V - 2 <= 0) {
                lDebug(1, "Error. TENSORCA mode requires a NREGION_X mayor a 2");
                return false;
            }
            this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
            this->GPUGrid = dim3((n + ((NREGIONS_H - 2) * 16) - 1) / ((NREGIONS_H - 2) * 16), (n + ((NREGIONS_V - 2) * 16) - 1) / ((NREGIONS_V - 2) * 16));
            break;
        case Mode::TENSORCACOALESCEDLESSSHMEM:
            if (BSIZE3DX * BSIZE3DY % 32 != 0) {
                lDebug(1, "Error. TENSORCA mode requires a CTA size such that size%32 == 0");
                return false;
            }
            this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
            this->GPUGrid = dim3((n + (NREGIONS_H * 16) - 1) / (NREGIONS_H * 16), (n + (NREGIONS_V * 16) - 1) / (NREGIONS_V * 16));
            break;
        case Mode::TENSORCACOALESCEDNOSHMEM:
            if (BSIZE3DX * BSIZE3DY % 32 != 0) {
                lDebug(1, "Error. TENSORCA mode requires a CTA size such that size%32 == 0");
                return false;
            }
            this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
            this->GPUGrid = dim3((n + (NREGIONS_H * 16) - 1) / (NREGIONS_H * 16), (n + (NREGIONS_V * 16) - 1) / (NREGIONS_V * 16));
            break;
        case Mode::TENSORCACOALESCEDLESSSHMEMINT4:
            if (BSIZE3DX * BSIZE3DY % 32 != 0) {
                lDebug(1, "Error. TENSORCA mode requires a CTA size such that size%32 == 0");
                return false;
            }
            this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
            this->GPUGrid = dim3((n + (NREGIONS_H * 32) - 1) / (NREGIONS_H * 32), (n + (NREGIONS_V * 32) - 1) / (NREGIONS_V * 32));
            break;
        case Mode::TENSORCACOALESCEDLESSSHMEMINT4V2:
            if (BSIZE3DX * BSIZE3DY % 32 != 0) {
                lDebug(1, "Error. TENSORCA mode requires a CTA size such that size%32 == 0");
                return false;
            }
            this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
            this->GPUGrid = dim3((n + (NREGIONS_H * 32) - 1) / (NREGIONS_H * 32), (n + (NREGIONS_V * 32) - 1) / (NREGIONS_V * 32));
            break;
        case Mode::TENSORCACOALESCEDLESSSHMEMINT8:
            if (BSIZE3DX * BSIZE3DY % 32 != 0) {
                lDebug(1, "Error. TENSORCA mode requires a CTA size such that size%32 == 0");
                return false;
            }
            this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY);
            this->GPUGrid = dim3((n + (NREGIONS_H * 16) - 1) / (NREGIONS_H * 16), (n + (NREGIONS_V * 16) - 1) / (NREGIONS_V * 16));
            break;

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
    if (this->mode == Mode::TENSORCA || this->mode == Mode::CLASSICGBMEMHALF) {
        dim3 cblock(16, 16, 1);
        dim3 cgrid((this->nWithHalo + cblock.x - 1) / cblock.x, (this->nWithHalo + cblock.y - 1) / cblock.y);
        lDebug(1, "Copying to buffer.");
        cudaMemcpy(this->devDataBufferTensor, this->hostData, sizeof(MTYPE) * this->nElements, cudaMemcpyHostToDevice);
        lDebug(1, "Casting to half and storing in ping matrix.");
        convertFp32ToFp16<<<cgrid, cblock>>>(this->devDataPingTensor, this->devDataBufferTensor, this->nWithHalo);
    } else if (this->mode == Mode::TENSORCACOALESCED || this->mode == Mode::TENSORCACOALESCEDMORETHREADS || this->mode == Mode::TENSORCACOALESCEDLESSSHMEM || this->mode == Mode::TENSORCACOALESCEDNOSHMEM) {
        dim3 cblock(16, 16, 1);
        dim3 cgrid((this->nWithHalo + cblock.x - 1) / cblock.x, (this->nWithHalo + cblock.y - 1) / cblock.y);
        lDebug(1, "Copying to buffer.");
        cudaMemcpy(this->devDataBufferTensor, this->hostData, sizeof(MTYPE) * this->nElements, cudaMemcpyHostToDevice);
        lDebug(1, "Casting to half and storing in ping matrix.");
        convertFp32ToFp16AndDoChangeLayout<<<cgrid, cblock>>>(this->devDataPingTensor, this->devDataBufferTensor, this->nWithHalo);
    } else if (this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT4) {
        dim3 cblock(32 / 8, 32);
        dim3 cgrid((this->nWithHalo + cblock.x * 8 - 1) / (cblock.x * 8), (this->nWithHalo + cblock.y - 1) / (cblock.y));
        printf("Grid(%i,%i)\n", cgrid.x, cgrid.y);
        lDebug(1, "Copying to buffer.");
        cudaMemcpy(this->devDataBufferTensor, this->hostData, sizeof(int) * this->nElements, cudaMemcpyHostToDevice);
        // printDeviceData();
        // cudaMemcpy(this->devDataBufferTensor2, this->hostData, sizeof(int) * this->nElements, cudaMemcpyHostToDevice);
        lDebug(1, "Casting to int4 and storing in ping matrix.");
        convertUInt32ToUInt4AndDoChangeLayout<<<cgrid, cblock>>>(this->devDataPingTensorInt4, this->devDataBufferTensor, this->nWithHalo);
        // cudaMemset(this->devDataBufferTensor, 0, sizeof(int) * this->nElements);
        // cudaMemset(this->devDataBufferTensor2, 0, sizeof(int) * this->nElements);
    } else if (this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT8) {
        dim3 cblock(16, 16, 1);
        dim3 cgrid((this->nWithHalo + cblock.x - 1) / cblock.x, (this->nWithHalo + cblock.y - 1) / cblock.y);
        lDebug(1, "Copying to buffer.");
        cudaMemcpy(this->devDataBufferTensorInt8, this->hostData, sizeof(int) * this->nElements, cudaMemcpyHostToDevice);
        lDebug(1, "Casting to half and storing in ping matrix.");
        convertInt32ToInt8AndDoChangeLayout<<<cgrid, cblock>>>(this->devDataPingTensorInt8, this->devDataBufferTensorInt8, this->nWithHalo);

    } else {
        cudaMemcpy(this->devDataPing, this->hostData, sizeof(MTYPE) * this->nElements, cudaMemcpyHostToDevice);
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}
void TensorCA2D::transferDeviceToHost() {
    if (this->mode == Mode::TENSORCA || this->mode == Mode::CLASSICGBMEMHALF) {
        dim3 cblock(16, 16);
        dim3 cgrid((this->nWithHalo + cblock.x - 1) / cblock.x, (this->nWithHalo + cblock.y - 1) / cblock.y);
        lDebug(1, "Casting to half and storing in buffer matrix.");
        convertFp16ToFp32<<<cgrid, cblock>>>(this->devDataBufferTensor, this->devDataPingTensor, this->nWithHalo);
        lDebug(1, "Copying to host.");
        cudaMemcpy(this->hostData, this->devDataBufferTensor, sizeof(MTYPE) * this->nElements, cudaMemcpyDeviceToHost);
    } else if (this->mode == Mode::TENSORCACOALESCED || this->mode == Mode::TENSORCACOALESCEDMORETHREADS || this->mode == Mode::TENSORCACOALESCEDLESSSHMEM || this->mode == Mode::TENSORCACOALESCEDNOSHMEM) {
        dim3 cblock(16, 16);
        dim3 cgrid((this->nWithHalo + cblock.x - 1) / cblock.x, (this->nWithHalo + cblock.y - 1) / cblock.y);
        lDebug(1, "Casting to half and storing in buffer matrix.");
        convertFp16ToFp32AndUndoChangeLayout<<<cgrid, cblock>>>(this->devDataBufferTensor, this->devDataPingTensor, this->nWithHalo);
        lDebug(1, "Copying to host.");
        cudaMemcpy(this->hostData, this->devDataBufferTensor, sizeof(MTYPE) * this->nElements, cudaMemcpyDeviceToHost);
    } else if (this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT4) {
        dim3 cblock(32, 32);
        dim3 cgrid((this->nWithHalo + cblock.x - 1) / (cblock.x), (this->nWithHalo + cblock.y - 1) / (cblock.y));
        lDebug(1, "Casting to int4 and storing in buffer matrix.");
        cudaMemcpy(this->hostData, this->devDataBufferTensor, sizeof(int) * this->nElements, cudaMemcpyDeviceToHost);
        lDebug(1, "Copying to host.");
    } else if (this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT8) {
        dim3 cblock(16, 16);
        dim3 cgrid((this->nWithHalo + cblock.x - 1) / cblock.x, (this->nWithHalo + cblock.y - 1) / cblock.y);
        lDebug(1, "Casting to half and storing in buffer matrix.");
        convertInt8ToInt32AndUndoChangeLayout<<<cgrid, cblock>>>(this->devDataBufferTensorInt8, this->devDataPingTensorInt8, this->nWithHalo);
        lDebug(1, "Copying to host.");
        cudaMemcpy(this->hostData, this->devDataBufferTensorInt8, sizeof(int) * this->nElements, cudaMemcpyDeviceToHost);

    } else {
        cudaMemcpy(this->hostData, this->devDataPing, sizeof(MTYPE) * this->nElements, cudaMemcpyDeviceToHost);
    }
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
    cudaStream_t stream;
    size_t shmem_size = ((NREGIONS_H + 2) * (NREGIONS_V + 2) * 16 * 16 * 2 + 256 * 2) * sizeof(FTYPE);
    size_t shmem_size2 = ((NREGIONS_H + 2) * (NREGIONS_V + 2) * 16 * 16 + 256 * 2) * sizeof(FTYPE);
    size_t shmem_sizeInt4 = ((NREGIONS_H + 2) * (NREGIONS_V + 2) * 32 * 32 + 256 / 8 * 6) * sizeof(int) + ((NREGIONS_H + 2) * (NREGIONS_V + 2) * 32 * 32 / 8) * sizeof(int);

    size_t shmem_sizeInt8 = ((NREGIONS_H + 2) * (NREGIONS_V + 2) * 16 * 16) * sizeof(char) * 5;  // one copy char and one int = 5 bytes

    if (this->mode == Mode::TENSORCA || this->mode == Mode::TENSORCACOALESCED || this->mode == Mode::TENSORCACOALESCEDMORETHREADS || this->mode == Mode::TENSORCACOALESCEDLESSSHMEM || this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT8) {
        cudaFuncSetAttribute(TensorV1GoLStep, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
        cudaFuncSetAttribute(TensorCoalescedV1GoLStep, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
        cudaFuncSetAttribute(TensorCoalescedV2GoLStep, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
        cudaFuncSetAttribute(TensorCoalescedV3GoLStep, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size2);
        cudaFuncSetAttribute(TensorCoalescedInt8, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sizeInt8);
        if (shmem_size2 > 100000) {
            int carveout = int(60 + ((shmem_size2 - 100000) / 64000.0) * 40.0);
            carveout = carveout > 100 ? 100 : carveout;
            cudaFuncSetAttribute(TensorCoalescedV3GoLStep, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
        }
        if (shmem_sizeInt8 > 100000) {
            int carveout = int(60 + ((shmem_sizeInt8 - 100000) / 64000.0) * 40.0);
            carveout = carveout > 100 ? 100 : carveout;
            cudaFuncSetAttribute(TensorCoalescedInt8, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
        }
        if (this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT8) {
            lDebug(1, "Setted shared memory size to %f KiB", shmem_sizeInt8 / 1024.f);

        } else {
            lDebug(1, "Setted shared memory size to %f KiB", shmem_size / 1024.f);
        }
        cudaStreamCreate(&stream);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    } else if (this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT4) {
        cudaFuncSetAttribute(TensorCoalescedSubTypeGoLStep, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sizeInt4);
        if (shmem_sizeInt4 > 100000) {
            int carveout = int(60 + ((shmem_sizeInt4 - 100000) / 64000.0) * 40.0);
            carveout = carveout > 100 ? 100 : carveout;
            cudaFuncSetAttribute(TensorCoalescedSubTypeGoLStep, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
        }
        lDebug(1, "Setted shared memory size to %f KiB", shmem_sizeInt4 / 1024.f);
        cudaStreamCreate(&stream);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }
    dim3 cblock(256);
    dim3 cgrid((this->nWithHalo * this->nWithHalo + 256 - 1) / (256), 1, 1);
    dim3 cGridHorizontal((int)ceil(n / (float)cblock.x), 1, 1);
    dim3 cGridVertical((int)ceil((this->nWithHalo) / (float)cblock.x), 1, 1);

    dim3 cblockCoales(16, 16);
    dim3 cGridHorizontalCoales(2 * (int)ceil(n / (float)cblockCoales.x), 1, 1);
    dim3 cGridVerticalCoales(2 * (int)ceil((this->nWithHalo) / (float)cblockCoales.x), 1, 1);

    if (this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT4) {
        cudaMemset(this->devDataBufferTensor, 0, sizeof(int) * this->nElements);
        cudaMemset(this->devDataBufferTensor2, 0, sizeof(int) * this->nElements);
    }
    // printf("%p, %p\n", );
    cudaEventRecord(start);
#ifdef MEASURE_POWER
    GPUPowerBegin(this->n, 100, 0, std::string("AutomataTC-") + std::to_string(this->deviceId));
#endif
    // int width = this->nWithHalo;
    // int height = this->nWithHalo;

    // auto fileName = "bwgif2.gif";
    // int delay = 10;
    // GifWriter g;
    // GifBegin(&g, fileName, width, height, delay);
    // std::vector<uint8_t> frame(nElements * 4);
    size_t shmemm = (size_t)(shmem_size / 2);
    switch (this->mode) {
        case Mode::NOT_IMPLEMENTED:
            lDebug(1, "METHOD NOT IMPLEMENTED");
            break;
        case Mode::CLASSICGBMEM:
            for (uint32_t i = 0; i < nTimes; ++i) {
                // printDeviceData();
                copyHorizontalHalo<<<cGridHorizontal, cblock>>>(this->devDataPing, n, nWithHalo);
                copyVerticalHalo<<<cGridVertical, cblock>>>(this->devDataPing, n, nWithHalo);
                // printDeviceData();
                // getchar();
                ClassicGlobalMemGoLStep<<<this->GPUGrid, this->GPUBlock>>>(this->devDataPing, this->devDataPong, this->n, this->nWithHalo);
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
        case Mode::CLASSICV1:
            for (uint32_t i = 0; i < nTimes; ++i) {
                // printDeviceData();
                copyHorizontalHalo<<<cGridHorizontal, cblock>>>(this->devDataPing, n, nWithHalo);
                copyVerticalHalo<<<cGridVertical, cblock>>>(this->devDataPing, n, nWithHalo);
                // printDeviceData();
                // getchar();

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
                // gpuErrchk(cudaDeviceSynchronize());

                // for (int l = 0; l < nElements; l++) {
                //     frame[l * 4 + 0] = (uint8_t)this->hostData[l] * 255;
                //     frame[l * 4 + 1] = (uint8_t)this->hostData[l] * 255;
                //     frame[l * 4 + 2] = (uint8_t)this->hostData[l] * 255;
                //     frame[l * 4 + 3] = (uint8_t)this->hostData[l] * 255;
                // }
                // GifWriteFrame(&g, frame.data(), width, height, delay);
            }
            break;
        case Mode::TENSORCA:

            for (uint32_t i = 0; i < nTimes; ++i) {
                printDeviceData();
                copyHorizontalHaloTensor<<<cGridHorizontal, cblock>>>(this->devDataPingTensor, n, nWithHalo);
                copyVerticalHaloTensor<<<cGridVertical, cblock>>>(this->devDataPingTensor, n, nWithHalo);
                printDeviceData();
                getchar();

                TensorV1GoLStep<<<this->GPUGrid, this->GPUBlock, shmem_size, stream>>>(this->devDataPingTensor, this->devDataPongTensor, this->n, this->nWithHalo);
                gpuErrchk(cudaDeviceSynchronize());
                std::swap(this->devDataPingTensor, this->devDataPongTensor);
                // this->transferDeviceToHost();
                // for (int l = 0; l < nElements; l++) {
                //     frame[l * 4 + 0] = (uint8_t)this->hostData[l] * 255;
                //     frame[l * 4 + 1] = (uint8_t)this->hostData[l] * 255;
                //     frame[l * 4 + 2] = (uint8_t)this->hostData[l] * 255;
                //     frame[l * 4 + 3] = 255;
                // }
                // GifWriteFrame(&g, frame.data(), width, height, delay);
            }
            break;
        case Mode::TENSORCACOALESCED:
            for (uint32_t i = 0; i < nTimes; ++i) {
                // printDeviceData();
                copyHorizontalHaloCoalescedVersion<<<cGridHorizontalCoales, cblockCoales>>>(this->devDataPingTensor, n, nWithHalo);
                copyVerticalHaloCoalescedVersion<<<cGridVerticalCoales, cblockCoales>>>(this->devDataPingTensor, n, nWithHalo);
                // printDeviceData();
                // getchar();

                TensorCoalescedV1GoLStep<<<this->GPUGrid, this->GPUBlock, shmem_size, stream>>>(this->devDataPingTensor, this->devDataPongTensor, this->n, this->nWithHalo);
                gpuErrchk(cudaDeviceSynchronize());
                std::swap(this->devDataPingTensor, this->devDataPongTensor);
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
        case Mode::CLASSICGBMEMHALF:
            for (uint32_t i = 0; i < nTimes; ++i) {
                // printDeviceData();
                copyHorizontalHaloHalf<<<cGridHorizontal, cblock>>>(this->devDataPingTensor, n, nWithHalo);
                copyVerticalHaloHalf<<<cGridVertical, cblock>>>(this->devDataPingTensor, n, nWithHalo);
                // printDeviceData();
                // getchar();

                ClassicGlobalMemHALFGoLStep<<<this->GPUGrid, this->GPUBlock>>>(this->devDataPingTensor, this->devDataPongTensor, this->n, this->nWithHalo);
                gpuErrchk(cudaDeviceSynchronize());
                std::swap(this->devDataPingTensor, this->devDataPongTensor);
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
        case Mode::TENSORCACOALESCEDMORETHREADS:
            for (uint32_t i = 0; i < nTimes; ++i) {
                TensorCoalescedV2GoLStep<<<this->GPUGrid, this->GPUBlock, shmem_size, stream>>>(this->devDataPingTensor, this->devDataPongTensor, this->n, this->nWithHalo);
                gpuErrchk(cudaDeviceSynchronize());
                std::swap(this->devDataPingTensor, this->devDataPongTensor);
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
        case Mode::TENSORCACOALESCEDLESSSHMEM:
            for (uint32_t i = 0; i < nTimes; ++i) {
                // printDeviceData();
                copyHorizontalHaloCoalescedVersion<<<cGridHorizontalCoales, cblockCoales>>>(this->devDataPingTensor, n, nWithHalo);
                copyVerticalHaloCoalescedVersion<<<cGridVerticalCoales, cblockCoales>>>(this->devDataPingTensor, n, nWithHalo);
                // printDeviceData();
                // getchar();
                TensorCoalescedV3GoLStep<<<this->GPUGrid, this->GPUBlock, shmemm, stream>>>(this->devDataPingTensor, this->devDataPongTensor, this->n, this->nWithHalo);
                // gpuErrchk(cudaDeviceSynchronize());
                std::swap(this->devDataPingTensor, this->devDataPongTensor);
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
        case Mode::TENSORCACOALESCEDNOSHMEM:
            for (uint32_t i = 0; i < nTimes; ++i) {
                TensorCoalescedV4GoLStep_Step1<<<this->GPUGrid, this->GPUBlock>>>(this->devDataPingTensor, this->devDataPongTensor, this->n, this->nWithHalo);
                gpuErrchk(cudaDeviceSynchronize());
                TensorCoalescedV4GoLStep_Step2<<<this->GPUGrid, this->GPUBlock>>>(this->devDataPongTensor, this->devDataPingTensor, this->n, this->nWithHalo);
                gpuErrchk(cudaDeviceSynchronize());

                // std::swap(this->devDataPingTensor, this->devDataPongTensor);
                //  this->transferDeviceToHost();
                //  for (int l = 0; l < nElements; l++) {
                //      frame[l * 4 + 0] = (uint8_t)this->hostData[l] * 255;
                //      frame[l * 4 + 1] = (uint8_t)this->hostData[l] * 255;
                //      frame[l * 4 + 2] = (uint8_t)this->hostData[l] * 255;
                //      frame[l * 4 + 3] = (uint8_t)this->hostData[l] * 255;
                //  }
                //  GifWriteFrame(&g, frame.data(), width, height, delay);
            }
            break;
        case Mode::TENSORCACOALESCEDLESSSHMEMINT4:
            for (uint32_t i = 0; i < nTimes; ++i) {
                TensorCoalescedSubTypeGoLStep<<<this->GPUGrid, this->GPUBlock, shmem_sizeInt4, stream>>>(this->devDataPingTensorInt4, this->n, this->nWithHalo, this->devDataBufferTensor);
                gpuErrchk(cudaDeviceSynchronize());
                onlyConvertUInt32ToUInt4<<<cgrid, cblock>>>(this->devDataPingTensorInt4, this->devDataBufferTensor, this->nWithHalo);
                gpuErrchk(cudaDeviceSynchronize());
                // printDeviceData();
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
        case Mode::TENSORCACOALESCEDLESSSHMEMINT8:
            for (uint32_t i = 0; i < nTimes; ++i) {
                TensorCoalescedInt8<<<this->GPUGrid, this->GPUBlock, (size_t)shmem_sizeInt8, stream>>>(this->devDataPingTensorInt8, this->devDataPongTensorInt8, this->n, this->nWithHalo);
                gpuErrchk(cudaDeviceSynchronize());
                std::swap(this->devDataPingTensorInt8, this->devDataPongTensorInt8);
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

    if (this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT4) {
        dim3 cblock2(32, 32);
        dim3 cgrid2((this->nWithHalo + 32 - 1) / (32), (this->nWithHalo + 32 - 1) / (32));

        UndoChangeLayout<<<cgrid2, cblock2>>>(this->devDataBufferTensor2, this->devDataBufferTensor, this->nWithHalo);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());

        cudaMemcpy(this->devDataBufferTensor, this->devDataBufferTensor2, sizeof(int) * this->nElements, cudaMemcpyDeviceToDevice);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
    }
    // return computing time
    float msecs = 0;
    cudaEventElapsedTime(&msecs, start, stop);
    return msecs / ((float)nTimes);
}

bool TensorCA2D::isInHalo(size_t i) {
    uint32_t x = i % this->nWithHalo;
    uint32_t y = i / this->nWithHalo;
    // uint32_t HALO_SIZE = (HALO_SIZE / 2) * this->haloWidth;
    return (x < this->haloWidth || y < this->haloWidth || x >= nWithHalo - this->haloWidth || y >= nWithHalo - this->haloWidth);
}

int random_val(std::mt19937& rng, float prob) {
    return static_cast<int>(static_cast<double>(rng() - rng.min()) / (rng.max() - rng.min() + 1) < prob);
}
// Should we incvlude the seed in this function call
void TensorCA2D::reset() {
    // rseed()
    std::mt19937 rng(seed);

    lDebug(1, "Resetting data to the initial state.");

    int ii = 0;
    for (size_t i = 0; i < this->nElements; ++i) {
        if (!this->isInHalo(i)) {
            this->hostData[i] = (MTYPE)random_val(rng, density);
            ii++;
        } else {
            this->hostData[i] = 0;
        }

        // uint32_t x = i % this->nWithHalo;
        // uint32_t y = i / this->nWithHalo;
        // if (x == 0 || y == 0 || x == nWithHalo - 1 || y == nWithHalo - 1) {
        //     this->hostData[i] = (MTYPE)1;

        // } else {
        //     this->hostData[i] = (MTYPE)0;
        // }
    }
    lDebug(1, "Newly host data:\n");
    fDebug(1, this->printHostData());

    lDebug(1, "Transfering data to device.");
    this->transferHostToDevice();
    lDebug(1, "Done.");

    lDebug(1, "Setting Pong elements to 0.");
    if (this->mode == Mode::TENSORCA || this->mode == Mode::TENSORCACOALESCED || this->mode == Mode::TENSORCACOALESCEDMORETHREADS || this->mode == Mode::CLASSICGBMEMHALF || this->mode == Mode::TENSORCACOALESCEDLESSSHMEM || this->mode == Mode::TENSORCACOALESCEDNOSHMEM) {
        cudaMemset(this->devDataPongTensor, 0, sizeof(char) * this->nElements);

    } else if (this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT8) {
        cudaMemset(this->devDataPongTensorInt8, 0, sizeof(char) * this->nElements);
    } else if (this->mode == Mode::TENSORCACOALESCEDLESSSHMEMINT4) {
        // cudaMemset(this->devDataPongTensorInt4, 0, sizeof(int) * this->nElements/8);
    } else {
        cudaMemset(this->devDataPong, 0, sizeof(MTYPE) * this->nElements);
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    lDebug(1, "Initial status in Host:");

    fDebug(1, this->printDeviceData());
}

void TensorCA2D::printHostData() {
    if (n > pow(2, 7)) {
        return;
    }
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

void TensorCA2D::printDeviceData() {
    transferDeviceToHost();
    printHostData();
}

bool TensorCA2D::compare(TensorCA2D* a) {
    bool res = true;

    for (size_t i = 0; i < this->n; ++i) {
        for (size_t j = 0; j < this->n; ++j) {
            size_t a_index = (i + a->haloWidth) * a->nWithHalo + j + a->haloWidth;
            size_t ref_index = (i + this->haloWidth) * this->nWithHalo + j + this->haloWidth;
            if (a->hostData[a_index] != this->hostData[ref_index]) {
                // printf("a(%llu) = %i != %i this(%llu)\n", a_index, a->hostData[a_index], this->hostData[ref_index], ref_index);
                //  printf("1 ");
                res = false;
            }
        }
    }

    return res;
}
