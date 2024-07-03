#include "CellularAutomata/CADataDomainComparator.cuh"
#include "Debug.h"
CADataDomainComparator::CADataDomainComparator(CASolver* pSolver, CASolver* pReferenceSolver) {
    this->solver = pSolver;
    this->referenceSolver = pReferenceSolver;
}

bool CADataDomainComparator::compareCurrentStates() {
    CADataDomain<int>* data = solver->getCurrentState();
    CADataDomain<int>* referenceData = referenceSolver->getCurrentState();

    if (areDifferentSize()) {
        return false;
    }

    for (int i = 0; i < data->getInnerHorizontalSize(); i++) {
        for (int j = 0; j < data->getInnerHorizontalSize(); j++) {
            int element = (int)data->getInnerElementAt(i, j);
            int referenceElement = (int)referenceData->getInnerElementAt(i, j);
            if (element != referenceElement) {
                lDebug(1, "Element at (%d, %d) is %d, but should be %d\n", i, j, element, referenceElement);
                return false;
            }
        }
    }
    return true;
}

bool CADataDomainComparator::areDifferentSize() {
    CADataDomain<int>* data = solver->getCurrentState();
    CADataDomain<int>* referenceData = referenceSolver->getCurrentState();

    return data->getInnerHorizontalSize() != referenceData->getInnerHorizontalSize();
}
