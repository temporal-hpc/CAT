#include "StatsCollector.hpp"
#define NOT_CALCULATED HUGE_VAL

StatsCollector::StatsCollector() {
    this->average = NOT_CALCULATED;
    this->standardDeviation = NOT_CALCULATED;
    this->standardError = NOT_CALCULATED;
    this->variance = NOT_CALCULATED;
}

void StatsCollector::add(float val) {
    if (val > 0) {
        this->runs.push_back(val);
    }
    this->average = NOT_CALCULATED;
    this->variance = NOT_CALCULATED;
    this->standardDeviation = NOT_CALCULATED;
    this->standardError = NOT_CALCULATED;
}

float StatsCollector::getAverage() {
    if (isInvalid(this->average) && this->runs.size() != 0) {
        this->average = std::reduce(this->runs.begin(), this->runs.end(), 0.0) / this->runs.size();
    }
    return this->average;
}

float StatsCollector::getStandardDeviation() {
    if (isInvalid(this->standardDeviation) && this->runs.size() != 0) {
        const float variance = getVariance();
        this->standardDeviation = sqrt(variance);
    }
    return this->standardDeviation;
}

float StatsCollector::getStandardError() {
    if (isInvalid(this->standardError) && this->runs.size() != 0) {
        const float standarDeviation = getStandardDeviation();
        this->standardError = standarDeviation / sqrt(this->runs.size());
    }
    return this->standardError;
}

float StatsCollector::getVariance() {
    if (isInvalid(this->variance) && this->runs.size() != 0) {
        const float mean = getAverage();
        auto sz = runs.size();
        auto variance_func = [&mean, &sz](float accumulator, const float& val) { return accumulator + ((val - mean) * (val - mean) / (sz - 1)); };

        this->variance = std::accumulate(runs.begin(), runs.end(), 0.0, variance_func);
    }
    return this->variance;
}

bool StatsCollector::isInvalid(float var) {
    return var == NOT_CALCULATED;
}

void StatsCollector::printStats() {
    printf("Average: %f\n", getAverage());
    printf("Standard Deviation: %f\n", getStandardDeviation());
    printf("Standard Error: %f\n", getStandardError());
    printf("Variance: %f\n", getVariance());
}
void StatsCollector::printShortStats() {
    printf("%f,%f,%f,%f\n", getAverage(), getStandardDeviation(), getStandardError(), getVariance());
}