#include <iostream>

#include <cuda.h>
#include <mma.h>
#include <vector>
#include <string>
#include <fstream>
#include <temporal/CAT.h>
#include <iomanip>

// Define this to enable animation frame export
#define OUTPUT_ANIMATION 
bool WritePackedBinaryFrame(std::ofstream &out, const std::vector<uint8_t *> &input, size_t n, int halo);

__forceinline__ unsigned char getSubCellH(uint64_t cell, unsigned char pos)
{
    return (cell >> (8 - 1 - pos) * 8);
}

using namespace Temporal;

struct PerfResult {
    float totalKernelMs;
};

#define CAT_HALO 16

static int gN = 32;
static int gZ = 1;
static int gRadius = 1;
static float gDensity = 0.3f;
static int gSteps = 2;
static int gInnerSteps = 2;
static int gRegionsX = 1;
static int gRegionsY = 13;


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
    for (int z = 0; z < gZ; z++)
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


PerfResult SimulateInputCAT(CATSolver *solver, std::vector<uint8_t *> &input1, size_t n, int halo, int radius, int numSteps)
{
    size_t nWithHalo = n + 2 * halo;

    half **d_inputs1;
    half **d_inputs2;
    uint8_t **d_input1s_uint8;

    cudaMalloc(&d_inputs1, gZ * sizeof(half *));
    cudaMalloc(&d_inputs2, gZ * sizeof(half *));
    cudaMalloc(&d_input1s_uint8, gZ * sizeof(uint8_t *));
    void **h_ptrArray1 = new void *[gZ];
    void **h_ptrArray2 = new void *[gZ];
    void **h_ptrArray3_uint8 = new void *[gZ];

    for (int i = 0; i < gZ; i++)
    {
        half *d_input1;
        half *d_input2;
        uint8_t *d_input1_uint8;

        cudaMalloc(&d_input1, nWithHalo * nWithHalo * sizeof(half));
        cudaMalloc(&d_input2, nWithHalo * nWithHalo * sizeof(half));
        cudaMalloc(&d_input1_uint8, nWithHalo * nWithHalo * sizeof(uint8_t));
        h_ptrArray1[i] = d_input1;
        h_ptrArray2[i] = d_input2;
        h_ptrArray3_uint8[i] = d_input1_uint8;
    }

    cudaMemcpy(d_inputs1, h_ptrArray1, gZ * sizeof(half *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs2, h_ptrArray2, gZ * sizeof(half *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input1s_uint8, h_ptrArray3_uint8, gZ * sizeof(uint8_t *), cudaMemcpyHostToDevice);

    for (int i = 0; i < gZ; i++)
    {
        cudaMemcpy(h_ptrArray3_uint8[i], input1[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyHostToDevice);
    }

    solver->prepareData(d_input1s_uint8, (void **)d_inputs1, n, halo, radius, gZ);

#ifdef OUTPUT_ANIMATION
    // Open animation file and write header
    std::ofstream animFile("cat_animation.bin", std::ios::binary);
    if (!animFile)
    {
        std::cerr << "Failed to open animation file" << std::endl;
        return PerfResult{0.0f};
    }
    
    // Write header: N, Z, and number of frames
    const uint32_t nHeader = static_cast<uint32_t>(n);
    const uint32_t zHeader = static_cast<uint32_t>(gZ);
    const uint32_t numFrames = static_cast<uint32_t>(numSteps + 1);
    animFile.write(reinterpret_cast<const char *>(&nHeader), sizeof(nHeader));
    animFile.write(reinterpret_cast<const char *>(&zHeader), sizeof(zHeader));
    animFile.write(reinterpret_cast<const char *>(&numFrames), sizeof(numFrames));
    
    // Export initial state (frame 0)
    solver->unprepareData((void **)d_inputs1, d_input1s_uint8, n, halo, radius, gZ);
    for (int i = 0; i < gZ; i++)
    {
        cudaMemcpy(input1[i], h_ptrArray3_uint8[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    }
    
    WritePackedBinaryFrame(animFile, input1, n, halo);
    
    // Re-prepare data for simulation
    for (int i = 0; i < gZ; i++)
    {
        cudaMemcpy(h_ptrArray3_uint8[i], input1[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyHostToDevice);
    }
    solver->prepareData(d_input1s_uint8, (void **)d_inputs1, n, halo, radius, gZ);
#endif

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float totalKernelMs = 0.0f;
    cudaDeviceSynchronize();
    cudaEventRecord(startEvent);

    for (int step = 0; step < numSteps; step++)
    {
        solver->fillPeriodicBoundaryConditions((void **)d_inputs1, n, halo, gZ);
        solver->StepSimulation((void **)d_inputs1, (void **)d_inputs2, n, halo, radius, gZ);
        std::swap(d_inputs1, d_inputs2);

#ifdef OUTPUT_ANIMATION
        // Export frame after each step
        solver->unprepareData((void **)d_inputs1, d_input1s_uint8, n, halo, radius, gZ);
        for (int i = 0; i < gZ; i++)
        {
            cudaMemcpy(input1[i], h_ptrArray3_uint8[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        }
        
        WritePackedBinaryFrame(animFile, input1, n, halo);
        
        // Re-prepare data for next step
        for (int i = 0; i < gZ; i++)
        {
            cudaMemcpy(h_ptrArray3_uint8[i], input1[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyHostToDevice);
        }
        solver->prepareData(d_input1s_uint8, (void **)d_inputs1, n, halo, radius, gZ);
#endif
    }

#ifdef OUTPUT_ANIMATION
    animFile.close();
#endif

#ifndef OUTPUT_ANIMATION
    // Only unprepare and copy back once at the end if not outputting animation
    solver->unprepareData((void **)d_inputs1, d_input1s_uint8, n, halo, radius, gZ);
    for (int i = 0; i < gZ; i++)
    {
        cudaMemcpy(input1[i], h_ptrArray3_uint8[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    }
#endif

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&totalKernelMs, startEvent, stopEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    for (int i = 0; i < gZ; i++)
    {
        cudaFree(h_ptrArray1[i]);
        cudaFree(h_ptrArray2[i]);
        cudaFree(h_ptrArray3_uint8[i]);
    }
    cudaFree(d_inputs1);
    cudaFree(d_inputs2);
    cudaFree(d_input1s_uint8);

    return PerfResult{totalKernelMs};
}

PerfResult SimulateInputCATMultiStep(CATMultiStepSolver *solver, std::vector<uint8_t *> &input1, size_t n, int halo, int radius, int numSteps)
{
    size_t nWithHalo = n + 2 * halo;

    half **d_inputs1;
    half **d_inputs2;
    uint8_t **d_input1s_uint8;

    cudaMalloc(&d_inputs1, gZ * sizeof(half *));
    cudaMalloc(&d_inputs2, gZ * sizeof(half *));
    cudaMalloc(&d_input1s_uint8, gZ * sizeof(uint8_t *));
    void **h_ptrArray1 = new void *[gZ];
    void **h_ptrArray2 = new void *[gZ];
    void **h_ptrArray3_uint8 = new void *[gZ];

    for (int i = 0; i < gZ; i++)
    {
        half *d_input1;
        half *d_input2;
        uint8_t *d_input1_uint8;

        cudaMalloc(&d_input1, nWithHalo * nWithHalo * sizeof(half));
        cudaMalloc(&d_input2, nWithHalo * nWithHalo * sizeof(half));
        cudaMalloc(&d_input1_uint8, nWithHalo * nWithHalo * sizeof(uint8_t));
        h_ptrArray1[i] = d_input1;
        h_ptrArray2[i] = d_input2;
        h_ptrArray3_uint8[i] = d_input1_uint8;
    }

    cudaMemcpy(d_inputs1, h_ptrArray1, gZ * sizeof(half *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs2, h_ptrArray2, gZ * sizeof(half *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input1s_uint8, h_ptrArray3_uint8, gZ * sizeof(uint8_t *), cudaMemcpyHostToDevice);

    for (int i = 0; i < gZ; i++)
    {
        cudaMemcpy(h_ptrArray3_uint8[i], input1[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyHostToDevice);
    }

    solver->prepareData(d_input1s_uint8, (void **)d_inputs1, n, halo, radius, gZ);

#ifdef OUTPUT_ANIMATION
    // Open animation file and write header
    std::ofstream animFile("cat_multi_animation.bin", std::ios::binary);
    if (!animFile)
    {
        std::cerr << "Failed to open animation file" << std::endl;
        return PerfResult{0.0f};
    }
    
    // Write header: N, Z, and number of frames
    const uint32_t nHeader = static_cast<uint32_t>(n);
    const uint32_t zHeader = static_cast<uint32_t>(gZ);
    const uint32_t numFrames = static_cast<uint32_t>(numSteps + 1);
    animFile.write(reinterpret_cast<const char *>(&nHeader), sizeof(nHeader));
    animFile.write(reinterpret_cast<const char *>(&zHeader), sizeof(zHeader));
    animFile.write(reinterpret_cast<const char *>(&numFrames), sizeof(numFrames));
    
    // Export initial state (frame 0)
    solver->unprepareData((void **)d_inputs1, d_input1s_uint8, n, halo, radius, gZ);
    for (int i = 0; i < gZ; i++)
    {
        cudaMemcpy(input1[i], h_ptrArray3_uint8[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    }
    
    WritePackedBinaryFrame(animFile, input1, n, halo);
    
    // Re-prepare data for simulation
    for (int i = 0; i < gZ; i++)
    {
        cudaMemcpy(h_ptrArray3_uint8[i], input1[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyHostToDevice);
    }
    solver->prepareData(d_input1s_uint8, (void **)d_inputs1, n, halo, radius, gZ);
#endif

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float totalKernelMs = 0.0f;
    cudaDeviceSynchronize();
    cudaEventRecord(startEvent);

    int innerSteps = gInnerSteps;
    for (int step = 0; step < numSteps; step += innerSteps)
    {
        solver->fillPeriodicBoundaryConditions((void **)d_inputs1, n, halo, gZ);
        solver->StepSimulationMulti((void **)d_inputs1, (void **)d_inputs2, n, halo, radius, gZ, innerSteps);
        std::swap(d_inputs1, d_inputs2);

#ifdef OUTPUT_ANIMATION
        // Export frame after each step
        solver->unprepareData((void **)d_inputs1, d_input1s_uint8, n, halo, radius, gZ);
        for (int i = 0; i < gZ; i++)
        {
            cudaMemcpy(input1[i], h_ptrArray3_uint8[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        }
        
        WritePackedBinaryFrame(animFile, input1, n, halo);
        
        // Re-prepare data for next step
        for (int i = 0; i < gZ; i++)
        {
            cudaMemcpy(h_ptrArray3_uint8[i], input1[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyHostToDevice);
        }
        solver->prepareData(d_input1s_uint8, (void **)d_inputs1, n, halo, radius, gZ);
#endif
    }

#ifdef OUTPUT_ANIMATION
    animFile.close();
#endif

#ifndef OUTPUT_ANIMATION
    // Only unprepare and copy back once at the end if not outputting animation
    solver->unprepareData((void **)d_inputs1, d_input1s_uint8, n, halo, radius, gZ);
    for (int i = 0; i < gZ; i++)
    {
        cudaMemcpy(input1[i], h_ptrArray3_uint8[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    }
#endif

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&totalKernelMs, startEvent, stopEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    for (int i = 0; i < gZ; i++)
    {
        cudaFree(h_ptrArray1[i]);
        cudaFree(h_ptrArray2[i]);
        cudaFree(h_ptrArray3_uint8[i]);
    }
    cudaFree(d_inputs1);
    cudaFree(d_inputs2);
    cudaFree(d_input1s_uint8);

    return PerfResult{totalKernelMs};
}


PerfResult SimulateInputCATMultiStep2(CATMultiStepSolver2 *solver, std::vector<uint8_t *> &input1, size_t n, int halo, int radius, int numSteps)
{
    size_t nWithHalo = n + 2 * (halo); 

    half **d_inputs1;
    half **d_inputs2;
    uint8_t **d_input1s_uint8;

    cudaMalloc(&d_inputs1, gZ * sizeof(half *));
    cudaMalloc(&d_inputs2, gZ * sizeof(half *));
    cudaMalloc(&d_input1s_uint8, gZ * sizeof(uint8_t *));
    void **h_ptrArray1 = new void *[gZ];
    void **h_ptrArray2 = new void *[gZ];
    void **h_ptrArray3_uint8 = new void *[gZ];

    for (int i = 0; i < gZ; i++)
    {
        half *d_input1;
        half *d_input2;
        uint8_t *d_input1_uint8;

        cudaMalloc(&d_input1, nWithHalo * nWithHalo * sizeof(half));
        cudaMalloc(&d_input2, nWithHalo * nWithHalo * sizeof(half));
        cudaMalloc(&d_input1_uint8, nWithHalo * nWithHalo * sizeof(uint8_t));
        h_ptrArray1[i] = d_input1;
        h_ptrArray2[i] = d_input2;
        h_ptrArray3_uint8[i] = d_input1_uint8;
    }

    cudaMemcpy(d_inputs1, h_ptrArray1, gZ * sizeof(half *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs2, h_ptrArray2, gZ * sizeof(half *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input1s_uint8, h_ptrArray3_uint8, gZ * sizeof(uint8_t *), cudaMemcpyHostToDevice);

    for (int i = 0; i < gZ; i++)
    {
        cudaMemcpy(h_ptrArray3_uint8[i], input1[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyHostToDevice);
    }

    solver->prepareData(d_input1s_uint8, (void **)d_inputs1, n, halo, radius, gZ);


#ifdef OUTPUT_ANIMATION
    // Open animation file and write header
    std::ofstream animFile("cat_multi2_animation.bin", std::ios::binary);
    if (!animFile)
    {
        std::cerr << "Failed to open animation file" << std::endl;
        return PerfResult{0.0f};
    }
    
    // Write header: N, Z, and number of frames
    const uint32_t nHeader = static_cast<uint32_t>(n); // Output only the original N region
    const uint32_t zHeader = static_cast<uint32_t>(gZ);
    const uint32_t numFrames = static_cast<uint32_t>(numSteps + 1);
    animFile.write(reinterpret_cast<const char *>(&nHeader), sizeof(nHeader));
    animFile.write(reinterpret_cast<const char *>(&zHeader), sizeof(zHeader));
    animFile.write(reinterpret_cast<const char *>(&numFrames), sizeof(numFrames));
    
    // Export initial state (frame 0)
    solver->unprepareData((void **)d_inputs1, d_input1s_uint8, n, halo, radius, gZ);
    for (int i = 0; i < gZ; i++)
    {
        cudaMemcpy(input1[i], h_ptrArray3_uint8[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    }
    
    WritePackedBinaryFrame(animFile, input1, n, halo); // Only write the original N region
    
    // Re-prepare data for simulation
    for (int i = 0; i < gZ; i++)
    {
        cudaMemcpy(h_ptrArray3_uint8[i], input1[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyHostToDevice);
    }
    solver->prepareData(d_input1s_uint8, (void **)d_inputs1, n, halo, radius, gZ);
#endif

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float totalKernelMs = 0.0f;
    cudaDeviceSynchronize();
    cudaEventRecord(startEvent, solver->getStream());

    int innerSteps = gInnerSteps;
    for (int step = 0; step < numSteps; step += innerSteps)
    {
        // No fillPeriodicBoundaryConditions needed – CG3 handles it internally
        solver->StepSimulationMulti((void **)d_inputs1, (void **)d_inputs2, n, halo, radius, gZ, innerSteps);
        std::swap(d_inputs1, d_inputs2);

#ifdef OUTPUT_ANIMATION
        // Export frame after each step
        solver->unprepareData((void **)d_inputs1, d_input1s_uint8, n, halo, radius, gZ);
        for (int i = 0; i < gZ; i++)
        {
            cudaMemcpy(input1[i], h_ptrArray3_uint8[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        }
        
        WritePackedBinaryFrame(animFile, input1, n, halo); // Only write the original N region
        
        // Re-prepare data for next step
        for (int i = 0; i < gZ; i++)
        {
            cudaMemcpy(h_ptrArray3_uint8[i], input1[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyHostToDevice);
        }
        solver->prepareData(d_input1s_uint8, (void **)d_inputs1, n, halo, radius, gZ);
#endif
    }

#ifdef OUTPUT_ANIMATION
    animFile.close();
#endif

#ifndef OUTPUT_ANIMATION
    solver->unprepareData((void **)d_inputs1, d_input1s_uint8, n, halo, radius, gZ);
    for (int i = 0; i < gZ; i++)
    {
        cudaMemcpy(input1[i], h_ptrArray3_uint8[i], nWithHalo * nWithHalo * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    }
#endif

    cudaEventRecord(stopEvent, solver->getStream());
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&totalKernelMs, startEvent, stopEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    solver->resetL2Persistence();

    for (int i = 0; i < gZ; i++)
    {
        cudaFree(h_ptrArray1[i]);
        cudaFree(h_ptrArray2[i]);
        cudaFree(h_ptrArray3_uint8[i]);
    }
    cudaFree(d_inputs1);
    cudaFree(d_inputs2);
    cudaFree(d_input1s_uint8);

    return PerfResult{totalKernelMs};
}


bool Compare(std::vector<uint8_t *> input1, std::vector<uint8_t *> input2, size_t n, int halo1, int halo2, int radius)
{
    size_t nWithHalo1 = n + 2 * halo1;
    size_t nWithHalo2 = n + 2 * halo2;
    bool fullOfZeros = true;
    for (int z = 0; z < gZ; z++)
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

bool WritePackedBinary(const std::string &filePath, const std::vector<uint8_t *> &input, size_t n, int halo)
{
    const size_t totalBits = static_cast<size_t>(gZ) * n * n;
    const size_t totalBytes = (totalBits + 7) / 8;
    std::vector<uint8_t> buffer(totalBytes, 0);

    size_t bitIndex = 0;
    const size_t stride = n + 2 * halo;
    for (int z = 0; z < gZ; z++)
    {
        for (size_t i = 0; i < n; i++)
        {
            const size_t rowBase = (i + halo) * stride + halo;
            for (size_t j = 0; j < n; j++)
            {
                if (input[z][rowBase + j] != 0)
                {
                    buffer[bitIndex >> 3] |= static_cast<uint8_t>(1u << (bitIndex & 7));
                }
                ++bitIndex;
            }
        }
    }

    std::ofstream out(filePath, std::ios::binary);
    if (!out)
    {
        return false;
    }
    const uint32_t nHeader = static_cast<uint32_t>(n);
    out.write(reinterpret_cast<const char *>(&nHeader), sizeof(nHeader));
    out.write(reinterpret_cast<const char *>(buffer.data()), buffer.size());
    return out.good();
}

bool WritePackedBinaryFrame(std::ofstream &out, const std::vector<uint8_t *> &input, size_t n, int halo)
{
    const size_t totalBits = static_cast<size_t>(gZ) * n * n;
    const size_t totalBytes = (totalBits + 7) / 8;
    std::vector<uint8_t> buffer(totalBytes, 0);

    size_t bitIndex = 0;
    const size_t stride = n + 2 * halo;
    for (int z = 0; z < gZ; z++)
    {
        for (size_t i = 0; i < n; i++)
        {
            const size_t rowBase = (i + halo) * stride + halo;
            for (size_t j = 0; j < n; j++)
            {
                if (input[z][rowBase + j] != 0)
                {
                    buffer[bitIndex >> 3] |= static_cast<uint8_t>(1u << (bitIndex & 7));
                }
                ++bitIndex;
            }
        }
    }

    out.write(reinterpret_cast<const char *>(buffer.data()), buffer.size());
    return out.good();
}

int main(int argc, char **argv)
{
    std::cout << "Testing CAT library..." << std::endl;
    if (argc < 1 || argc > 9)
    {
        std::cout << "Invalid arguments. " << argc << std::endl;
        std::cout << "  Usage: " << argv[0] << " [N] [density] [radius] [Z] [steps] [inner] [regions_x] [regions_y]" << std::endl;
        exit(1);
    }
    if (argc > 1)
    {
        gN = std::stoi(argv[1]);
    }
    if (argc > 2)
    {
        gDensity = std::stof(argv[2]);
    }
    if (argc > 3)
    {
        gRadius = std::stoi(argv[3]);
    }
    if (argc > 4)
    {
        gZ = std::stoi(argv[4]);
    }
    if (argc > 5)
    {
        gSteps = std::stoi(argv[5]);
    }
    if (argc > 6)
    {
        gInnerSteps = std::stoi(argv[6]);
    }
    if (argc > 8)
    {
        gRegionsX = std::stoi(argv[7]);
        gRegionsY = std::stoi(argv[8]);
    }
    std::cout << "  N: " << gN << ", density: " << gDensity << ", radius: " << gRadius << ", Z: " << gZ
              << ", steps: " << gSteps << ", inner steps: " << gInnerSteps
              << ", regions: " << gRegionsX << " x " << gRegionsY << std::endl;

    int radius = gRadius;

    size_t nWithHalo = gN + 2 * radius;
    
    CATSolver *catSolver = new CATSolver(1, 13, 1,3,3,3);
    catSolver->setBlockSize(16, 16);
    catSolver->prepareGrid(gN, CAT_HALO);
    
    CATMultiStepSolver *catMultiStepSolver = new CATMultiStepSolver(gRegionsX, gRegionsY, 1,3,3,3);
    catMultiStepSolver->setBlockSize(16, 16);
    catMultiStepSolver->prepareGrid(gN, CAT_HALO);

    CATMultiStepSolver2 *catMultiStepSolver2 = new CATMultiStepSolver2(gRegionsX, gRegionsY, 1,3,3,3, gN, CAT_HALO);
    catMultiStepSolver2->setBlockSize(16, 16);
    catMultiStepSolver2->prepareGrid(gN, CAT_HALO);

    std::vector<uint8_t *> inputCAT = std::vector<uint8_t *>(gZ);
    std::vector<uint8_t *> inputCATMultiStep = std::vector<uint8_t *>(gZ);
    std::vector<uint8_t *> inputCATMultiStep2 = std::vector<uint8_t *>(gZ);

    size_t nWithHaloCat = gN + 2 * CAT_HALO;
    size_t nWithHaloCat2 = (gN) + 2 * (CAT_HALO);
    for (int i = 0; i < gZ; i++)
    {
        inputCAT[i] = new uint8_t[nWithHaloCat * nWithHaloCat];
        inputCATMultiStep[i] = new uint8_t[nWithHaloCat * nWithHaloCat];
        inputCATMultiStep2[i] = new uint8_t[nWithHaloCat2 * nWithHaloCat2];
    }

    printf("Initializing input data with density %.2f...\n", gDensity);
    printf("Tile size for CAT: %llux%llu = %llu cells\n", nWithHaloCat, nWithHaloCat, nWithHaloCat * nWithHaloCat);
    printf("Tile size for CATMultiStep2: %llux%llu = %llu cells\n", nWithHaloCat2, nWithHaloCat2, nWithHaloCat2 * nWithHaloCat2);

    for (int i = 0; i < gZ; i++)
    {
        InitWithValue(inputCAT[i], gN, CAT_HALO, 0);
        InitWithValue(inputCATMultiStep[i], gN, CAT_HALO, 0);
        InitWithValue(inputCATMultiStep2[i], gN, CAT_HALO, 0);
        srand(i);
        RandomFill(inputCAT[i], gN, CAT_HALO, gDensity);
        std::copy(inputCAT[i], inputCAT[i] + nWithHaloCat * nWithHaloCat, inputCATMultiStep[i]);

        for (int row = 0; row < nWithHaloCat; row++)
        {
            for (int col = 0; col < nWithHaloCat; col++)
            {
                inputCATMultiStep2[i][(row) * nWithHaloCat2 + col] = inputCAT[i][row * nWithHaloCat + col];
            }
        }
    }

#ifdef OUTPUT_ANIMATION
    std::cout << "Animation export enabled - will generate " << (gSteps + 1) << " frames" << std::endl;
#endif

    PerfResult catMultiResult2 = SimulateInputCATMultiStep2(catMultiStepSolver2, inputCATMultiStep2, gN, CAT_HALO, radius, gSteps);
    PerfResult catMultiResult = SimulateInputCATMultiStep(catMultiStepSolver, inputCATMultiStep, gN, CAT_HALO, radius, gSteps);
    PerfResult catResult = SimulateInputCAT(catSolver, inputCAT, gN, CAT_HALO, radius, gSteps);

    if (Compare(inputCATMultiStep, inputCAT, gN, CAT_HALO, CAT_HALO, radius))
    {
        std::cout << "CATMultiStep and CAT match" << std::endl;
    }
    else
    {
        std::cout << "CATMultiStep and CAT do not match" << std::endl;
    }

    // Performance Report
    double cellsPerStep = static_cast<double>(gN) * gN * gZ;
    double totalCells = cellsPerStep * gSteps;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n========== Performance Report ==========" << std::endl;
    std::cout << "  Grid:  " << gN << " x " << gN << " x " << gZ
              << " (" << std::setprecision(2) << cellsPerStep / 1e6 << " Mcells/step)" << std::endl;
    std::cout << "  Steps: " << gSteps << ",  Radius: " << gRadius << std::endl;

    std::cout << std::setprecision(3);
    std::cout << "\n  Timing summary (total for " << gSteps << " steps):" << std::endl;
    std::cout << "    " << std::left << std::setw(22) << "Solver"
              << std::right << std::setw(14) << "Kernel ms"
              << std::setw(18) << "Avg ms/step" << std::endl;
    std::cout << "    " << std::left << std::setw(22) << "CATSolver"
              << std::right << std::setw(14) << catResult.totalKernelMs
              << std::setw(18) << (catResult.totalKernelMs / gSteps) << std::endl;
    std::cout << "    " << std::left << std::setw(22) << "CATMultiStepSolver"
              << std::right << std::setw(14) << catMultiResult.totalKernelMs
              << std::setw(18) << (catMultiResult.totalKernelMs / gSteps) << std::endl;
    std::cout << "    " << std::left << std::setw(22) << "CATMultiStepSolver2"
              << std::right << std::setw(14) << catMultiResult2.totalKernelMs
              << std::setw(18) << (catMultiResult2.totalKernelMs / gSteps) << std::endl;

    std::cout << std::setprecision(4);
    std::cout << "\n  Throughput (kernel):" << std::endl;
    std::cout << "    CATSolver:        "
              << totalCells / (catResult.totalKernelMs / 1000.0) / 1e9 << " Gcells/s" << std::endl;
    std::cout << "    CATMultiStep:     "
              << totalCells / (catMultiResult.totalKernelMs / 1000.0) / 1e9 << " Gcells/s" << std::endl;
    std::cout << "    CATMultiStep2:    "
              << totalCells / (catMultiResult2.totalKernelMs / 1000.0) / 1e9 << " Gcells/s" << std::endl;

    std::cout << std::setprecision(2);
    std::cout << "\n  Speedup vs CATSolver:" << std::endl;
    std::cout << "    CATMultiStep:     " << catResult.totalKernelMs / catMultiResult.totalKernelMs << "x" << std::endl;
    std::cout << "    CATMultiStep2:    " << catResult.totalKernelMs / catMultiResult2.totalKernelMs << "x" << std::endl;
    std::cout << "=========================================" << std::endl;

    return 0;
}