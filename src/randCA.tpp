#include "randCA.cuh"

template <typename T>
//__global__ void kernel_random_array(uint64_t n, int halo, float density, T max, curandState *states, T *array)
__global__ void kernel_random_array(uint64_t n, int halo, float density, T max, T *array)
{
    curandState s;
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t id = (uint64_t)id_y * (uint64_t)gridDim.x * (uint64_t)blockDim.x + (uint64_t)id_x;
    uint64_t id_with_halo = (id_y + halo) * ((uint64_t)gridDim.x * (uint64_t)blockDim.x + 2 * halo) + id_x + halo;

    if (id_x >= n || id_y >= n)
    {
        return;
    }
    curand_init(10, id, 0, &s);
    //float x = curand_uniform(&states[id]);
    float x = curand_uniform(&s);
    if (x <= density)
    {
        array[id_with_halo] = 1;
    }
    else
    {
        array[id_with_halo] = 0;
    }
}

template <typename T> inline curandState *setup_curand(uint64_t n, int seed, curandState *devStates)
{
    dim3 block(RANDBSIZE_X, RANDBSIZE_Y, 1);
    dim3 grid((n + RANDBSIZE_X - 1) / RANDBSIZE_X, (n + RANDBSIZE_Y - 1) / RANDBSIZE_Y, 1);
    // printf("grid: %d %d %d\n", grid.x, grid.y, grid.z);
    // printf("block: %d %d %d\n", block.x, block.y, block.z);
    kernel_setup_prngs<<<grid, block>>>(n, seed, devStates);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return devStates;
}

//template <typename T> inline T *genRandCA(T *d_array, size_t n, int halo, float density, T max, curandState *devStates)
template <typename T> inline T *genRandCA(T *d_array, size_t n, int halo, float density, T max)
{
    size_t size = n * n;
    dim3 block(RANDBSIZE_X, RANDBSIZE_Y, 1);
    dim3 grid((n + RANDBSIZE_X - 1) / RANDBSIZE_X, (n + RANDBSIZE_Y - 1) / RANDBSIZE_Y, 1);
    //kernel_random_array<<<grid, block>>>(n, halo, density, max, devStates, d_array);
    kernel_random_array<<<grid, block>>>(n, halo, density, max, d_array);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return d_array;
}

template <typename T> inline T *genRandCA_LTL(T *d_array, size_t n, int halo, float density, uint64_t seed)
{
    // devstates
    size_t size = (n) * (n);
    /*
    curandState *devStates;
    cudaMalloc((void **)&devStates, size * sizeof(curandState));
    printf("memory used for randStates = %f GBytes\n", (double)size*sizeof(curandState)/1e9);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    printf("Generating PRNGs...."); fflush(stdout);
    devStates = setup_curand<T>(n, seed, devStates);
    printf("done\n"); fflush(stdout);
    */

    //printf("memory used for CA = %f GBytes (type %i bytes)\n", (double)size*sizeof(T)/1e9);
    // random generation
    //printf("Generating random CA...."); fflush(stdout);
    //genRandCA<T>(d_array, n, halo, density, 2, devStates);
    genRandCA<T>(d_array, n, halo, density, 2);
    //printf("done\n"); fflush(stdout);

    // clean memory
    //cudaFree(devStates);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return d_array;
}
