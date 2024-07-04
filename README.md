# CAT: Cellular Automata on Tensor Cores

CAT is a WIP library to perform Fast Larger than Life (LtL) Cellular Automata (CA)simulations using tensor cores. It's programmed using CUDA and C++.

#### Requirements

- CUDA 11+
- C++17
- A tensor core NVIDIA GPU

#### Compilation

- Modify the Makefile to suite your needs. Architecture microcode, Radius, LtL rules, GPU Block size (we recomed using `16x16`), and the number of regions to use for CAT (we recomend using `NREGIONS_H=2, NREGIONS_V=5` for low shared memory GPUs and `NREGIONS_H=1, NREGIONS_V=13` otherwise, although result may vary).
- Perform the compilation `make -j 10`. If debug information is needed, compile using `make debug -j 10`.

#### Execution

Depending on the compilation performed, the main executable should be `bin/prog` or `debug/prog`. To execute, simply run `bin/prog <method> <n> <steps>`, where `method` is the solver algorithm, `n` is the side length without halo, and `steps` is the number of steps to simulate. Below is a complete list of the supported arguments which can be accesed using `bin/prog -h`:

```
Usage: CAT: Celular Automata on Tensor Cores [--help] [--version] [--deviceId VAR] [--density VAR] [--seed VAR] [--doVerify] n solver steps

Positional arguments:
  n               Size of the data domain 
  solver          Solver to use:
                        0 - BASE: Global Memory
                        1 - SHARED: Shared Memory
                        2 - CAT: Fast Tensor Core
                        3 - COARSE: Thread Coarsening
                        4 - MCELL: Multiple cells per thread
                        5 - PACK: uint64 Packet Coding 
  steps           Number of steps of the CA simulation 

Optional arguments:
  -h, --help      shows help message and exits 
  -v, --version   prints version information and exits 
  -g, --deviceId  Device ID [nargs=0..1] [default: 0]
  -d, --density   Density of the data domain 
  --seed          Seed for the random number generator [nargs=0..1] [default: 0]
  --doVerify      Verify the results? WARNING: memory requirements double 
```


#### Additional Information

All of the kernel code is in `src/GPUKernels.cu`.
In order to measure power, the compile macro `MEASURE_POWER` must be set to a value other than `NO`.


*Temporal research group from the Universidad Austral de Chile.*