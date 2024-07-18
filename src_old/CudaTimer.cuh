
#include <cuda_runtime.h>

class CudaTimer {
   private:
    cudaEvent_t startTime;
    cudaEvent_t stopTime;
    float elapsedTimeMiliseconds;

   public:
    CudaTimer();
    ~CudaTimer();

    void start();
    void stop();
    float getElapsedTimeMiliseconds() const;
};
