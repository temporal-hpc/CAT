#include "CudaTimer.cuh"

CudaTimer::CudaTimer() {
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
    elapsedTimeMiliseconds = 0.0f;
}

CudaTimer::~CudaTimer() {
    cudaEventDestroy(startTime);
    cudaEventDestroy(stopTime);
}

void CudaTimer::start() {
    cudaEventRecord(startTime);
}

void CudaTimer::stop() {
    cudaEventRecord(stopTime);
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&elapsedTimeMiliseconds, startTime, stopTime);
}

float CudaTimer::getElapsedTimeMiliseconds() const {
    return elapsedTimeMiliseconds;
}
