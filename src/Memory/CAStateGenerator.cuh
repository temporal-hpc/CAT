#pragma once

#include <random>
#include "Memory/CADataDomain.cuh"
#include "Memory/DataDomain.cuh"
#include "Debug.h"
#include <omp.h>

class CAStateGenerator {
   public:
    static void generateRandomState(CADataDomain<int>* data, int seed, float density) {
        int numThreads = omp_get_max_threads();
	lDebug(0, "Initializing random state with %i threads", numThreads);
        #pragma omp parallel
        {
            // Use thread ID as seed
            unsigned int seed2 = static_cast<unsigned int>(seed + omp_get_thread_num());
            std::mt19937 rng(seed2);
            
            // Determine the chunk size for each thread
	    size_t totalSize = data->getInnerHorizontalSize()*(size_t)data->getInnerVerticalSize() ;
            size_t chunkSize = (totalSize + numThreads - 1) / numThreads;
            size_t threadID = omp_get_thread_num();

            // Calculate the starting index for each thread
            int startIdx = threadID * chunkSize;

            for (size_t i = startIdx; i < totalSize && i < startIdx + chunkSize; ++i) {
               int value = 0;
               //if (!isInHaloI(i, data->getFullHorizontalSize(), data->getHorizontalHaloSize())) {
               value = randomVal(rng, density);
               //}

	       data->setInnerElementAt(i/data->getInnerHorizontalSize(),i%data->getInnerHorizontalSize() , value);
            }
        } 
    }

    static bool isInHalo(size_t pos, size_t sideLengthWithHalo, int haloWidth) {
        uint32_t x = pos % sideLengthWithHalo;
        uint32_t y = pos / sideLengthWithHalo;
        return (x < haloWidth || y < haloWidth || x >= sideLengthWithHalo - haloWidth || y >= sideLengthWithHalo - haloWidth);
    }

    static int randomVal(std::mt19937& rng, float prob) { 
	    return static_cast<int>(static_cast<double>(rng() - rng.min()) / (rng.max() - rng.min() + 1) < prob); 
    }
};
