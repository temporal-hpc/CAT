#ifndef STATS_COLLECTOR_H
#define STATS_COLLECTOR_H
#include <cinttypes>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
class StatsCollector {
    std::vector<float> runs;
    float average;
    float standardDeviation;
    float standardError;
    float variance;

   public:
    StatsCollector();

    void add(float val);
    float getAverage();
    float getStandardDeviation();
    float getStandardError();
    float getVariance();

    bool isInvalid(float var);

    void printStats();
    void printShortStats();
};
#endif