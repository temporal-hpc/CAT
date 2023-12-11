#pragma once

#include <random>
#include "Memory/CADataDomain.cuh"
#include "Memory/DataDomain.cuh"

class CAStateGenerator {
   public:
    static void generateRandomState(CADataDomain<int>* data, int seed, float density) {
        std::mt19937 rng(seed);
        for (size_t i = 0; i < data->getTotalSize(); ++i) {
            int value = 0;
            if (!isInHalo(i, data->getSideLength(), data->getHaloWidth())) {
                value = randomVal(rng, density);
            }

            data->setElementAt(i, value);
        }
    }

    static bool isInHalo(size_t pos, size_t sideLengthWithHalo, int haloWidth) {
        uint32_t x = pos % sideLengthWithHalo;
        uint32_t y = pos / sideLengthWithHalo;
        return (x < haloWidth || y < haloWidth || x >= sideLengthWithHalo - haloWidth || y >= sideLengthWithHalo - haloWidth);
    }

    static int randomVal(std::mt19937& rng, float prob) { return static_cast<int>(static_cast<double>(rng() - rng.min()) / (rng.max() - rng.min() + 1) < prob); }
};