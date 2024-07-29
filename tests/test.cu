#include "CAT.h"
#include <iostream>

#include <cuda.h>

#define N 32
#define RADIUS 1

using namespace Temporal;

void InitWithValue(uint8_t *input, size_t size, int value)
{
    for (size_t i = 0; i < size; i++)
    {
        input[i] = value;
    }
}

void RandomFill(uint8_t *input, size_t n, int radius, float density)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            if (rand() % 100 < density * 100)
            {
                input[i * (n + 2 * radius) + j + radius] = 1;
            }
        }
    }
}

void Print(uint8_t *input, size_t n, int radius)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (input[i * (n + 2 * radius) + j + radius] == 0)
            {
                std::cout << "  ";
            }
            else
            {
                std::cout << "X ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T> void SimulateInput(Solver<T> *solver, uint8_t *input1, size_t n, int radius, int numSteps)
{
    size_t nWithHalo = n + 2 * radius;

    InitWithValue(input1, nWithHalo * nWithHalo, 0);
    srand(0);
    RandomFill(input1, n, radius, 0.3);
    uint8_t *d_input1;
    uint8_t *d_input2;
    cudaMalloc(&d_input1, nWithHalo * nWithHalo);
    cudaMalloc(&d_input2, nWithHalo * nWithHalo);
    cudaMemcpy(d_input1, input1, nWithHalo * nWithHalo, cudaMemcpyHostToDevice);

    for (int i = 0; i < numSteps; i++)
    {
        solver->StepSimulation(d_input1, d_input2, n, radius);
        std::swap(d_input1, d_input2);
    }
    cudaMemcpy(input1, d_input1, nWithHalo * nWithHalo, cudaMemcpyDeviceToHost);

    cudaFree(d_input1);
    cudaFree(d_input2);
}
template <typename T> void SimulateInputCAT(Solver<T> *solver, uint8_t *input1, size_t n, int radius, int numSteps)
{
    size_t nWithHalo = n + 2 * radius;
    size_t nWithHaloCat = n + 2 * 16;

    InitWithValue(input1, nWithHaloCat * nWithHaloCat, 0);
    srand(0);
    RandomFill(input1, n, radius, 0.3);
    half *d_input1;
    half *d_input2;
    uint8_t *d_input1_uint8;
    cudaMalloc(&d_input1, nWithHaloCat * nWithHaloCat);
    cudaMalloc(&d_input2, nWithHaloCat * nWithHaloCat);
    cudaMalloc(&d_input1_uint8, nWithHalo * nWithHalo);
    cudaMemcpy(d_input1_uint8, input1, nWithHaloCat * nWithHaloCat, cudaMemcpyHostToDevice);
    solver->prepareData(d_input1_uint8, d_input1, n, radius);

    for (int i = 0; i < numSteps; i++)
    {
        solver->StepSimulation(d_input1, d_input2, n, radius);
        std::swap(d_input1, d_input2);
    }
    solver->unprepareData(d_input1, d_input1_uint8, n, radius);
    cudaMemcpy(input1, d_input1_uint8, nWithHalo * nWithHalo, cudaMemcpyDeviceToHost);

    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_input1_uint8);
}
bool Compare(uint8_t *input1, uint8_t *input2, size_t n, int radius)
{
    size_t nWithHalo = n + 2 * radius;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (input1[i * nWithHalo + j + radius] != input2[i * nWithHalo + j + radius])
            {
                std::cout << "Mismatch at " << i << " " << j << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main()
{
    std::cout << "Testing CAT library..." << std::endl;
    // Example usage of the library
    // CAT::Function(); // Replace with actual function call
    int radius = RADIUS;
    size_t nWithHalo = N + 2 * radius;

    BASESolver *solver = new BASESolver();
    solver->setBlockSize(16, 16);
    solver->setGridSize(N, 1);

    COARSESolver *coarseSolver = new COARSESolver();
    coarseSolver->setBlockSize(16, 16);
    coarseSolver->setGridSize(N, 1);

    CATSolver *catSolver = new CATSolver(1, 1);
    catSolver->setBlockSize(16, 16);
    catSolver->setGridSize(N, 1);

    MCELLSolver *mcellSolver = new MCELLSolver(radius);
    mcellSolver->setBlockSize(32, 32);
    mcellSolver->setGridSize(N, 1);

    SHAREDSolver *sharedSolver = new SHAREDSolver();
    sharedSolver->setBlockSize(16, 16);
    sharedSolver->setGridSize(N, 1);

    PACKSolver *packSolver = new PACKSolver(N, radius);
    packSolver->setBlockSize(16, 16);
    packSolver->setGridSize(N, 1);

    uint8_t *inputBASE = new uint8_t[nWithHalo * nWithHalo];
    uint8_t *inputCOARSE = new uint8_t[nWithHalo * nWithHalo];
    size_t nWithHaloCat = N + 2 * 16;
    uint8_t *inputCAT = new uint8_t[nWithHaloCat * nWithHaloCat];
    // uint64_t *inputPACK = new uint64_t[nWithHalo * nWithHalo];
    uint8_t *inputMCELL = new uint8_t[nWithHalo * nWithHalo];
    uint8_t *inputSHARED = new uint8_t[nWithHalo * nWithHalo];
    SimulateInput(solver, inputBASE, N, radius, 20);
    SimulateInput(coarseSolver, inputCOARSE, N, radius, 20);
    SimulateInputCAT(catSolver, inputCAT, N, radius, 20);
    // SimulateInput(packSolver, inputPACK, N, radius, 20);
    SimulateInput(mcellSolver, inputMCELL, N, radius, 20);
    SimulateInput(sharedSolver, inputSHARED, N, radius, 20);

    if (Compare(inputBASE, inputCOARSE, N, radius))
    {
        std::cout << "BASE and COARSE match" << std::endl;
    }
    else
    {
        std::cout << "BASE and COARSE do not match" << std::endl;
    }

    if (Compare(inputBASE, inputMCELL, N, radius))
    {
        std::cout << "BASE and MCELL match" << std::endl;
    }
    else
    {
        std::cout << "BASE and MCELL do not match" << std::endl;
    }

    if (Compare(inputBASE, inputCAT, N, radius))
    {
        std::cout << "BASE and CAT match" << std::endl;
    }
    else
    {
        std::cout << "BASE and CAT do not match" << std::endl;
    }

    if (Compare(inputBASE, inputSHARED, N, radius))
    {
        std::cout << "BASE and SHARED match" << std::endl;
    }
    else
    {
        std::cout << "BASE and SHARED do not match" << std::endl;
    }

    // if (Compare(inputBASE, inputPACK, N, radius))
    // {
    //     std::cout << "BASE and PACK match" << std::endl;
    // }
    // else
    // {
    //     std::cout << "BASE and PACK do not match" << std::endl;
    // }

    return 0;
}