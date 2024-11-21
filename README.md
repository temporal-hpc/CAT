# CAT: Cellular Automata on Tensor Cores
## CPU version of CAT here: https://github.com/kezada94/CAT_cpu

## Overview

The CAT project is a CUDA-based library designed to simulate Larger than Life Cellular automata using GPU tensor cores.

A CPU version of CAT can be found here: https://github.com/kezada94/CAT_cpu

## Abstract
Cellular automata (CA) are simulation models that can produce complex emergent behaviors from simple local rules. Although state-of-the-art GPU solutions are already fast due to their data-parallel nature, their performance can rapidly degrade in CA with a large neighborhood radius. With the inclusion of tensor cores across the entire GPU ecosystem,  interest has grown in finding ways to leverage these fast units outside the field of artificial intelligence, which was their original purpose. 
In this work, we present CAT, a GPU tensor core approach that can accelerate CA in which the cell transition function acts on a weighted summation of its neighborhood. CAT is evaluated theoretically, using an extended PRAM cost model, as well as empirically using the Larger Than Life (LTL) family of CA as case studies. The results confirm that the cost model is accurate, showing that CAT exhibits constant time throughout the entire radius range $1 \le r \le 16$, and its theoretical speedups agree with the empirical results. At low radius $r=1,2$, CAT is competitive and is only surpassed by the fastest state-of-the-art GPU solution. Starting from $r=3$, CAT progressively outperforms all other approaches, reaching speedups of up to $101\times$ over a GPU baseline and up to $\sim 14\times$ over the fastest state-of-the-art GPU approach. In terms of energy efficiency, CAT is competitive in the range $1 \le r \le 4$ and from $r \ge 5$ it is the most energy efficient approach. As for performance scaling across GPU architectures, CAT shows a promising trend that, if continues for future generations, it would increase its performance at a higher rate than classical GPU solutions. A CPU version of CAT was also explored, using the recently introduced AMX instructions. Although its performance is still below GPU tensor cores, it is a promising approach as it can still outperform some GPU approaches at large radius. The results obtained in this work put CAT as an approach with great potential for scientists who need to study emerging phenomena in CA with a large neighborhood radius, both in the GPU and in the CPU. 

## Building the library

To build the project, you need to have CMake and CUDA installed on your system. Follow these steps to build the project:

Clone the repository:
```
git clone <repository_url>
cd CAT
```
Create a build directory and navigate into it:
```
mkdir build
cd build
```
Run CMake to configure the project:
```
cmake ..
```
Build the project using Make:
```
make
```

## Running Tests

After building the project, you can run the tests using the following command:
```
./tests/test_exe
```
## Citation

goes here

