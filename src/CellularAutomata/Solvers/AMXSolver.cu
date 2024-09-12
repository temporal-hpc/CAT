#include "CellularAutomata/CADataPrinter.cuh"
#include "CellularAutomata/Solvers/AMXSolver.cuh"
#include "Memory/Allocators/CPUAllocator.cuh"

AMXSolver::AMXSolver(CADataDomain<uint8_t>* domain, CADataDomain<uint8_t>* domainBuffer) {
    dataDomain = domain;
    dataDomainBuffer = domainBuffer;

    CPUAllocator<int>* cpuAllocator = new CPUAllocator<int>();
    Allocator<int>* allocator = reinterpret_cast<Allocator<int>*>(cpuAllocator);
    hostVisibleData = new CADataDomain<int>(allocator, dataDomain->getInnerHorizontalSize(), dataDomain->getHorizontalHaloSize());
    hostVisibleData->allocate();

    setupAMX();
    
    fillTridiag();
}

void AMXSolver::setupAMX() {
    __tilecfg tile_config;

    tile_config->palette_id = 1;
    tile_config->start_row = 0;

    // Configure tiles for block_size x block_size matrices
    for (int i = 0; i < 2; ++i) {
        tile_config->colsb[i] = 16 ;
        tile_config->rows[i] = 16;
    }
    for (int i = 2; i < 8; ++i) {
        tile_config->colsb[i] = 64;
        tile_config->rows[i] = 16;
    }

    _tile_loadconfig(tile_config);
}


void AMXSolver::fillTridiag() {
    int i;

    for (i = 0; i < 16*64; i += 1)
    {
        int col = i & 15;
        int row = i >> 4;
        if (col -15+RADIUS> row){
            tridiag[i] = 1;
        } else {
            tridiag[i] = 0;
        }
    }

    for (i = 0; i < 16*64; i += 1)
    {
        int col = i & 15;
        int row = i >> 4;
        if (abs(col - row) <= RADIUS){
            tridiag[i+16*64] = 1;
        } else {
            tridiag[i+16*64] = 0;
        }
    }
    for (i = 0; i < 16*64; i += 1)
    {
        int col = i & 15;
        int row = i >> 4;
        if (col + 15 - RADIUS < row){
            tridiag[i+16*64*2] = 1;
        } else {
            tridiag[i+16*64*2] = 0;
        }
    }

    // // debug print tridiag in 2d
    // for (int i = 0; i < 16; i++) {
    //     for (int j = 0; j < 16; j++) {
    //         std::cout << (int)tridiag[i * 16 + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // for (int i = 0; i < 16; i++) {
    //     for (int j = 0; j < 16; j++) {
    //         std::cout << (int)tridiag[i * 16 + j + 256] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // for (int i = 0; i < 16; i++) {
    //     for (int j = 0; j < 16; j++) {
    //         std::cout << (int)tridiag[i * 16 + j + 512] << " ";
    //     }
    //     std::cout << std::endl;
    // }


}

void AMXSolver::copyCurrentStateToHostVisibleData() {
    for (int i = 0; i < dataDomain->getTotalSize(); ++i) {
        uint8_t value = dataDomain->getElementAt(i);
        hostVisibleData->setElementAt(i, (int)value);
    }
}
void AMXSolver::copyHostVisibleDataToCurrentState() {
    for (int i = 0; i < hostVisibleData->getTotalSize(); ++i) {
        int value = hostVisibleData->getElementAt(i);
        dataDomain->setElementAt(i, value);
    }
}

void AMXSolver::swapPointers() {
    CADataDomain<uint8_t>* temp = dataDomain;
    dataDomain = dataDomainBuffer;
    dataDomainBuffer = temp;
}

uint8_t AMXSolver::transitionFunction(int k, int a, int b) {
    return (1 - (((k - a) >> 31) & 0x1)) * (1 - (((b - k) >> 31) & 0x1));
}

void AMXSolver::CAStepAlgorithm() {

    for (int i = 0; i < dataDomain->getHorizontalHaloSize(); i+=16) {
        for (int j = 0; j < dataDomain->getFullHorizontalSize(); j+=16) {
            //take three continuous 16x16 blocks and load them into amx
            
            _tile_loadd(2, tridiag, 64);
            _tile_loadd(3, tridiag + 16*64, 64);
            _tile_loadd(4, tridiag + 16*64*2, 64);

            _tile_loadd(dataDomain->getData()
            


        }
    }


    for (int i = 0; i < dataDomain->getInnerHorizontalSize(); ++i) {
        for (int j = 0; j < dataDomain->getInnerHorizontalSize(); ++j) {
            int liveNeighbors = countAliveNeighbors(i, j);
            uint8_t cellValue = dataDomain->getInnerElementAt(i, j);
            uint8_t result = cellValue * transitionFunction(liveNeighbors, SMIN, SMAX) + (1 - cellValue) * transitionFunction(liveNeighbors, BMIN, BMAX);

            dataDomainBuffer->setInnerElementAt(i, j, result);
        }
    }
}

int AMXSolver::countAliveNeighbors(int y, int x) {
    int aliveNeighbors = 0;

    for (int i = -RADIUS; i <= RADIUS; ++i) {
        for (int j = -RADIUS; j <= RADIUS; ++j) {
            if (i == 0 && j == 0)
                continue;
            aliveNeighbors += dataDomain->getInnerElementAt(y + i, x + j);
        }
    }

    return aliveNeighbors;
}

void AMXSolver::fillHorizontalBoundaryConditions() {
    for (int h = 0; h < dataDomain->getHorizontalHaloSize(); ++h) {
        for (int j = 0; j < dataDomain->getInnerHorizontalSize(); ++j) {
            size_t topIndex = (dataDomain->getHorizontalHaloSize() + h) * dataDomain->getFullHorizontalSize() + dataDomain->getHorizontalHaloSize() + j;
            size_t bottomIndex = topIndex + (dataDomain->getInnerHorizontalSize()) * dataDomain->getFullHorizontalSize();
            uint8_t value = dataDomain->getElementAt(topIndex);
            dataDomain->setElementAt(bottomIndex, value);
        }

        for (int j = 0; j < dataDomain->getInnerHorizontalSize(); ++j) {
            size_t topIndex = (h)*dataDomain->getFullHorizontalSize() + dataDomain->getHorizontalHaloSize() + j;
            size_t bottomIndex = topIndex + (dataDomain->getInnerHorizontalSize()) * dataDomain->getFullHorizontalSize();

            uint8_t value = dataDomain->getElementAt(bottomIndex);
            dataDomain->setElementAt(topIndex, value);
        }
    }
}

void AMXSolver::fillVerticalBoundaryConditions() {
    for (int h = 0; h < dataDomain->getHorizontalHaloSize(); ++h) {
        for (int i = 0; i < dataDomain->getFullHorizontalSize(); ++i) {
            size_t leftIndex = i * dataDomain->getFullHorizontalSize() + h;
            size_t rightIndex = leftIndex + dataDomain->getInnerHorizontalSize();
            uint8_t value = dataDomain->getElementAt(rightIndex);
            dataDomain->setElementAt(leftIndex, value);
        }

        for (int i = 0; i < dataDomain->getFullHorizontalSize(); ++i) {
            size_t leftIndex = i * dataDomain->getFullHorizontalSize() + dataDomain->getHorizontalHaloSize() + h;
            size_t rightIndex = leftIndex + dataDomain->getInnerHorizontalSize();
            uint8_t value = dataDomain->getElementAt(leftIndex);
            dataDomain->setElementAt(rightIndex, value);
        }
    }
}
