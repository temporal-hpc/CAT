#include "CellularAutomata/Solvers/CASolver.cuh"

void CASolver::resetState(int seed, float density) {
    CAStateGenerator::generateRandomState(hostVisibleData, seed, density);
    copyHostVisibleDataToCurrentState();
}

CADataDomain<int>* CASolver::getCurrentState() {
    copyCurrentStateToHostVisibleData();
    return hostVisibleData;
}

void CASolver::doSteps(int stepNumber) {
    for (int i = 0; i < stepNumber; i++) {
        doStep();
    }
}
void CASolver::doStep() {
    fillBoundaryConditions();
    CAStepAlgorithm();
    swapPointers();
}
void CASolver::fillBoundaryConditions() {
    fillHorizontalBoundaryConditions();
    fillVerticalBoundaryConditions();
}

void CASolver::printCurrentState() {
    copyCurrentStateToHostVisibleData();
    CADataPrinter::printCADataWithoutHalo(hostVisibleData);
}
