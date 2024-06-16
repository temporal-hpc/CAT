#include "GPUBenchmark.cuh"

#ifdef MEASURE_POWER
#include "nvmlPower.hpp"
#endif

GPUBenchmark::GPUBenchmark(CASolver* pSolver, int n, int pRepeats, int pSteps, int pSeed, float pDensity) {
    solver = pSolver;
    this->n = n;
    steps = pSteps;
    repeats = pRepeats;
    seed = pSeed;
    density = pDensity;

    timer = new CudaTimer();
    stats = new StatsCollector();
}

void GPUBenchmark::reset() {
	int s = rand()%1000000;

        lDebug(1, "Resetting state:");
    solver->resetState(s, density);
}

//void GPUBenchmark::run() {
//        reset();
//    for (int i = 0; i < repeats/2; i++) { //realizations
//    solver->doSteps(steps);
//    
//    }
//        lDebug(1, "Benchmark started");
//	srand(seed);
//    for (int i = 0; i < repeats; i++) { //realizations
//        reset();
//        lDebug(1, "Initial state:");
//        if (n <= PRINT_LIMIT) {
//            fDebug(1, solver->printCurrentState());
//        }
//#ifdef MEASURE_POWER
//    GPUPowerBegin("0", 100);
//#endif
//        doOneRun();
//#ifdef MEASURE_POWER
//    GPUPowerEnd();
//#endif
//        lDebug(1, "Benchmark finished. Results:");
//        //printf("PRINT_LIMIT %i\n", PRINT_LIMIT);
//        if (n <= PRINT_LIMIT) {
//            fDebug(1, solver->printCurrentState());
//        }
//    }
//}

void GPUBenchmark::run() {
	reset();
	lDebug(1, "Benchmark started");
	srand(seed);
	// WARMUP for STEPS/4
	solver->doSteps(steps >> 2);
	for (int i = 0; i < repeats; i++) { //realizations
		//reset();
		lDebug(1, "Initial state:");
		if (n <= PRINT_LIMIT) {
		    fDebug(1, solver->printCurrentState());
		}
		#ifdef MEASURE_POWER
		    GPUPowerBegin("0", 100);
		#endif
	        doOneRun();
		#ifdef MEASURE_POWER
		    GPUPowerEnd();
		#endif
	        lDebug(1, "Benchmark finished. Results:");
	        //printf("PRINT_LIMIT %i\n", PRINT_LIMIT);
	        if (n <= PRINT_LIMIT) {
	            fDebug(1, solver->printCurrentState());
       		 }
    	}
}
void GPUBenchmark::doOneRun() {
    //solver->doSteps(steps);
    timer->start();


    solver->doSteps(steps);


    timer->stop();
    registerElapsedTime(timer->getElapsedTimeMiliseconds() / steps);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}

void GPUBenchmark::registerElapsedTime(float milliseconds) {
    stats->add(milliseconds);
}

StatsCollector* GPUBenchmark::getStats() {
    return stats;
}
