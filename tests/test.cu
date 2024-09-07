#include "CAT.h"
#include <iostream>

#include <cuda.h>
#include <mma.h>
#include <vector>
__forceinline__ unsigned char getSubCellH(uint64_t cell, unsigned char pos)
{
    return (cell >> (8 - 1 - pos) * 8);
}

using namespace nvcuda;

#define N 2048
#define Z 10
#define RADIUS 1

using namespace Temporal;

void InitWithValue(uint8_t *input, size_t n, int halo, int value)
{
    size_t nWithHalo = n + 2 * halo;
    for (size_t i = 0; i < (nWithHalo)*nWithHalo; i++)
    {
        input[i] = value;
    }
}

void RandomFill(uint8_t *input, size_t n, int halo, float density)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            if (rand() % 100 < density * 100)
            {
                input[(i + halo) * (n + 2 * halo) + j + halo] = 1;
            }
        }
    }
}

void Print(std::vector<uint8_t *> input, size_t n, int halo, int radius, int nTiles)
{
    for (int z = 0; z < Z; z++)
    {
        std::cout << "Z = " << z << std::endl;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (input[z][(i + halo) * (n + 2 * halo) + j + halo] == 0)
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
    }
    std::cout << std::endl;
}

template <typename T>
void SimulateInput(Solver<T> *solver, std::vector<uint8_t *> &input1, size_t n, int halo, int radius, int numSteps)
{
    size_t nWithHalo = n + 2 * radius;

    for (int i = 0; i < Z; i++)
    {
        InitWithValue(input1[i], n, halo, 0);
        srand(i);
        RandomFill(input1[i], n, halo, 0.3);
    }
    uint8_t **d_inputs1;
    uint8_t **d_inputs2;

    cudaMalloc(&d_inputs1, Z * sizeof(uint8_t *));
    cudaMalloc(&d_inputs2, Z * sizeof(uint8_t *));
    uint8_t **h_ptrArray1 = new uint8_t *[Z];
    uint8_t **h_ptrArray2 = new uint8_t *[Z];

    for (int i = 0; i < Z; i++)
    {
        uint8_t *d_input1;
        uint8_t *d_input2;
        cudaMalloc(&d_input1, nWithHalo * nWithHalo * sizeof(uint8_t));
        cudaMalloc(&d_input2, nWithHalo * nWithHalo * sizeof(uint8_t));
        cudaMemcpy(d_input1, input1[i], nWithHalo * nWithHalo, cudaMemcpyHostToDevice);
        h_ptrArray1[i] = d_input1;
        h_ptrArray2[i] = d_input2;
    }

    cudaMemcpy(d_inputs1, h_ptrArray1, Z * sizeof(uint8_t *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs2, h_ptrArray2, Z * sizeof(uint8_t *), cudaMemcpyHostToDevice);

    for (int i = 0; i < numSteps; i++)
    {
        solver->StepSimulation(d_inputs1, d_inputs2, n, halo, radius, Z);
        std::swap(d_inputs1, d_inputs2);
        std::swap(h_ptrArray1, h_ptrArray2);
    }

    for (int i = 0; i < Z; i++)
    {
        cudaMemcpy(input1[i], h_ptrArray1[i], nWithHalo * nWithHalo, cudaMemcpyDeviceToHost);
        cudaFree(h_ptrArray1[i]);
        cudaFree(h_ptrArray2[i]);
    }
    cudaFree(d_inputs1);
    cudaFree(d_inputs2);
}

template <typename T>
void SimulateInputCAT(Solver<T> *solver, std::vector<uint8_t *> &input1, size_t n, int halo, int radius, int numSteps)
{
    // size_t nWithHalo = n + 2 * halo;

    // for (int i = 0; i < Z; i++)
    // {
    //     InitWithValue(input1[i], n, halo, 0);
    //     srand(i);
    //     RandomFill(input1[i], n, radius, 0.3);
    // }
    // half *d_input1;
    // half *d_input2;
    // uint8_t *d_input1_uint8;
    // cudaMalloc(&d_input1, nWithHalo * nWithHalo * sizeof(half));
    // cudaMalloc(&d_input2, nWithHalo * nWithHalo * sizeof(half));
    // cudaMalloc(&d_input1_uint8, nWithHalo * nWithHalo * sizeof(uint8_t));
    // cudaMemcpy(d_input1_uint8, input1, nWithHalo * nWithHalo, cudaMemcpyHostToDevice);
    // solver->prepareData(d_input1_uint8, d_input1, n, halo, radius);

    // for (int i = 0; i < numSteps; i++)
    // {
    //     solver->StepSimulation(d_input1, d_input2, n, halo, radius);
    //     std::swap(d_input1, d_input2);
    // }
    // solver->unprepareData(d_input1, d_input1_uint8, n, halo, radius);
    // cudaMemcpy(input1, d_input1_uint8, nWithHalo * nWithHalo, cudaMemcpyDeviceToHost);

    // cudaFree(d_input1);
    // cudaFree(d_input2);
    // cudaFree(d_input1_uint8);
    size_t nWithHalo = n + 2 * halo;

    for (int i = 0; i < Z; i++)
    {
        InitWithValue(input1[i], n, halo, 0);
        srand(i);
        RandomFill(input1[i], n, halo, 0.3);
    }
    half **d_inputs1;
    half **d_inputs2;
    uint8_t **d_input1s_uint8;

    cudaMalloc(&d_inputs1, Z * sizeof(half *));
    cudaMalloc(&d_inputs2, Z * sizeof(half *));
    cudaMalloc(&d_input1s_uint8, Z * sizeof(uint8_t *));
    void **h_ptrArray1 = new void *[Z];
    void **h_ptrArray2 = new void *[Z];
    void **h_ptrArray3_uint8 = new void *[Z];

    for (int i = 0; i < Z; i++)
    {
        half *d_input1;
        half *d_input2;
        uint8_t *d_input1_uint8;

        cudaMalloc(&d_input1, nWithHalo * nWithHalo * sizeof(half));
        cudaMalloc(&d_input2, nWithHalo * nWithHalo * sizeof(half));
        cudaMalloc(&d_input1_uint8, nWithHalo * nWithHalo * sizeof(uint8_t));
        cudaMemcpy(d_input1_uint8, input1[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyHostToDevice);
        h_ptrArray1[i] = d_input1;
        h_ptrArray2[i] = d_input2;
        h_ptrArray3_uint8[i] = d_input1_uint8;
    }

    cudaMemcpy(d_inputs1, h_ptrArray1, Z * sizeof(half *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs2, h_ptrArray2, Z * sizeof(half *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input1s_uint8, h_ptrArray3_uint8, Z * sizeof(uint8_t *), cudaMemcpyHostToDevice);

    solver->prepareData(d_input1s_uint8, (void **)d_inputs1, n, halo, radius, Z);

    for (int i = 0; i < numSteps; i++)
    {
        solver->StepSimulation((void **)d_inputs1, (void **)d_inputs2, n, halo, radius, Z);
        std::swap(d_inputs1, d_inputs2);
        std::swap(h_ptrArray1, h_ptrArray2);
    }
    solver->unprepareData((void **)d_inputs1, d_input1s_uint8, n, halo, radius, Z);

    for (int i = 0; i < Z; i++)
    {
        cudaMemcpy(input1[i], h_ptrArray3_uint8[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        cudaFree(h_ptrArray1[i]);
        cudaFree(h_ptrArray2[i]);
        cudaFree(h_ptrArray3_uint8[i]);
    }
    cudaFree(d_inputs1);
    cudaFree(d_inputs2);
    cudaFree(d_input1s_uint8);
}

void SimulateInputPACK(PACKSolver *solver, std::vector<uint8_t *> input1, size_t n, int halo, int radius, int numSteps)
{
    size_t nWithHalo = n + 2 * halo;
    size_t nx = n / 8 + 2 * (int)ceil(halo / 8.0f);
    size_t ny = n + 2 * halo;
    for (int i = 0; i < Z; i++)
    {
        InitWithValue(input1[i], n, halo, 0);
        srand(i);
        RandomFill(input1[i], n, radius, 0.3);
    }
    uint64_t *d_input1;
    uint64_t *h_input1 = new uint64_t[nx * ny];
    uint64_t *d_input2;
    uint8_t *d_input1_uint8;
    cudaMalloc(&d_input1, nx * ny * sizeof(uint64_t));
    cudaMalloc(&d_input2, nx * ny * sizeof(uint64_t));
    cudaMalloc(&d_input1_uint8, nWithHalo * nWithHalo * sizeof(uint8_t));
    // cudaMemcpy(d_input1_uint8, input1, nWithHalo * nWithHalo, cudaMemcpyHostToDevice);

    // solver->prepareData(d_input1_uint8, d_input1, n, halo, radius);
    solver->fillHorizontalBoundaryConditions(d_input1, n, radius);
    solver->fillVerticalBoundaryConditions(d_input1, n, radius);
    // cudaMemcpy(h_input1, d_input1, nx * ny * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    // // print it
    // for (int i = 0; i < ny; i++)
    // {
    //     for (int j = 0; j < nx; j++)
    //     {
    //         for (int k = 0; k < 8; k++)
    //         {
    //             std::cout << (int)getSubCellH(h_input1[i * nx + j], k) << " ";
    //         }
    //         // std::cout << h_input1[i * ny + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // for (int i = 0; i < numSteps; i++)
    // {
    //     solver->StepSimulation(d_input1, d_input2, n, halo, radius);
    //     std::swap(d_input1, d_input2);
    // }
    // solver->unprepareData(d_input1, d_input1_uint8, n, halo, radius);
    // cudaMemcpy(input1, d_input1_uint8, nWithHalo * nWithHalo, cudaMemcpyDeviceToHost);

    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_input1_uint8);
}

bool Compare(std::vector<uint8_t *> input1, std::vector<uint8_t *> input2, size_t n, int halo1, int halo2, int radius)
{
    size_t nWithHalo1 = n + 2 * halo1;
    size_t nWithHalo2 = n + 2 * halo2;
    bool fullOfZeros = true;
    for (int z = 0; z < Z; z++)
    {
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                if (input1[z][(i + halo1) * nWithHalo1 + j + halo1] != input2[z][(i + halo2) * nWithHalo2 + j + halo2])
                {
                    std::cout << "Mismatch at " << i << " " << j << std::endl;
                    return false;
                }
                if (input1[z][(i + halo1) * nWithHalo1 + j + halo1] != 0 ||
                    input2[z][(i + halo2) * nWithHalo2 + j + halo2] != 0)
                {
                    fullOfZeros = false;
                }
            }
        }
    }
    if (fullOfZeros)
    {
        std::cout << "Both are full of zeros" << std::endl;
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
    solver->prepareGrid(N, 1);

    COARSESolver *coarseSolver = new COARSESolver();
    coarseSolver->setBlockSize(16, 16);
    coarseSolver->prepareGrid(N, 1);

    CATSolver *catSolver = new CATSolver(1, 13);
    catSolver->setBlockSize(16, 16);
    catSolver->prepareGrid(N, 16);

    MCELLSolver *mcellSolver = new MCELLSolver(radius);
    mcellSolver->setBlockSize(32, 32);
    mcellSolver->prepareGrid(N, 1);

    SHAREDSolver *sharedSolver = new SHAREDSolver();
    sharedSolver->setBlockSize(16, 16);
    sharedSolver->prepareGrid(N, 1);

    // PACKSolver *packSolver = new PACKSolver(radius);
    // packSolver->setBlockSize(16, 16);
    // packSolver->prepareGrid(N, 1);

    std::vector<uint8_t *> inputBASE = std::vector<uint8_t *>(Z);
    std::vector<uint8_t *> inputCOARSE = std::vector<uint8_t *>(Z);
    std::vector<uint8_t *> inputCAT = std::vector<uint8_t *>(Z);
    // std::vector<uint8_t *> inputPACK = std::vector<uint8_t *>(Z);
    std::vector<uint8_t *> inputMCELL = std::vector<uint8_t *>(Z);
    std::vector<uint8_t *> inputSHARED = std::vector<uint8_t *>(Z);

    size_t nWithHaloCat = N + 2 * 16;
    for (int i = 0; i < Z; i++)
    {
        inputBASE[i] = new uint8_t[nWithHalo * nWithHalo];
        inputCOARSE[i] = new uint8_t[nWithHalo * nWithHalo];
        inputCAT[i] = new uint8_t[nWithHaloCat * nWithHaloCat];
        // inputPACK[i] = new uint8_t[nWithHalo * nWithHalo];
        inputMCELL[i] = new uint8_t[nWithHalo * nWithHalo];
        inputSHARED[i] = new uint8_t[nWithHalo * nWithHalo];
    }

    SimulateInput(solver, inputBASE, N, radius, radius, 10);
    SimulateInput(coarseSolver, inputCOARSE, N, radius, radius, 10);
    SimulateInputCAT(catSolver, inputCAT, N, 16, radius, 10);
    // SimulateInputPACK(packSolver, inputPACK, N, radius, radius, 10);
    SimulateInput(mcellSolver, inputMCELL, N, radius, radius, 10);
    SimulateInput(sharedSolver, inputSHARED, N, radius, radius, 10);
    // Print(inputBASE, N, radius, radius);
    // Print(inputCOARSE, N, radius, radius);
    if (Compare(inputBASE, inputCOARSE, N, radius, radius, radius))
    {
        std::cout << "BASE and COARSE match" << std::endl;
    }
    else
    {
        std::cout << "BASE and COARSE do not match" << std::endl;
    }

    if (Compare(inputBASE, inputMCELL, N, radius, radius, radius))
    {
        std::cout << "BASE and MCELL match" << std::endl;
    }
    else
    {
        std::cout << "BASE and MCELL do not match" << std::endl;
    }
    // Print(inputBASE, N, radius, radius);
    // Print(inputCAT, N, 16, radius);
    if (Compare(inputBASE, inputCAT, N, radius, 16, radius))
    {
        std::cout << "BASE and CAT match" << std::endl;
    }
    else
    {
        std::cout << "BASE and CAT do not match" << std::endl;
    }

    if (Compare(inputBASE, inputSHARED, N, radius, radius, radius))
    {
        std::cout << "BASE and SHARED match" << std::endl;
    }
    else
    {
        std::cout << "BASE and SHARED do not match" << std::endl;
    }

    // if (Compare(inputBASE, inputPACK, N, radius, radius, radius))
    // {
    //     std::cout << "BASE and PACK match" << std::endl;
    // }
    // else
    // {
    //     std::cout << "BASE and PACK do not match" << std::endl;
    // }

    return 0;
}