#include "CellularAutomata/CADataDomainComparator.cuh"

CADataDomainComparator::CADataDomainComparator(CASolver* pSolver, CASolver* pReferenceSolver) {
    this->solver = pSolver;
    this->referenceSolver = pReferenceSolver;
}

bool CADataDomainComparator::compareCurrentStates() {
    CADataDomain* data = solver->getCurrentState();
    CADataDomain* referenceData = referenceSolver->getCurrentState();

    if (areDifferentSize()) {
        return false;
    }

    for (int i = 0; i < data->getSideLengthWithoutHalo(); i++) {
        for (int j = 0; j < data->getSideLengthWithoutHalo(); j++) {
            int element = (int)data->getInnerElementAt(i, j);
            int referenceElement = (int)referenceData->getInnerElementAt(i, j);
            if (element != (int)referenceElement) {
                return false;
            }
        }
    }
    return true;
}

bool CADataDomainComparator::areDifferentSize() {
    return data->getSideLengthWithoutHalo() != referenceData->getSideLengthWithoutHalo();
}
