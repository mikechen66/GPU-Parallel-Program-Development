# CUDA Thrust 
Nvidia 

## Introduction 

Thrust is a C++ parallel programming library which resembles the C++ Standard Library. Thrust's high-level interface greatly enhances programmer productivity while enabling performance portability between GPUs and multicore CPUs. Interoperability with established technologies (such as CUDA, TBB, and OpenMP) facilitates integration with existing software. Develop high-performance applications rapidly with Thrust. Thrust is included in the NVIDIA HPC SDK and the CUDA Toolkit. Please have a look at the repository as follows. 

https://github.com/NVIDIA/thrust

## Quick Start: Using Thrust From Your Project

To use Thrust from your project, first recursively clone the Thrust Github repository:

``` 
git clone --recursive https://github.com/NVIDIA/thrust.git 
```
Thrust is a header-only library; there is no need to build or install the project unless you want to run the Thrust unit tests. For CMake-based projects, Nvidia provide a CMake package for use with find_package. See the CMake README for more information. Thrust can also be added via add_subdirectory or tools like the CMake Package Manager.

## Examples 
Examples
Thrust is best explained through examples. The following source code generates random numbers serially and then transfers them to a parallel device where they are sorted.

```
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

int main(void) {
  // generate 32M random numbers serially
  thrust::host_vector<int> h_vec(32 << 20);
  std::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer data to the device
  thrust::device_vector<int> d_vec = h_vec;

  // sort data on the device (846M keys per second on GeForce GTX 480)
  thrust::sort(d_vec.begin(), d_vec.end());

  // transfer data back to host
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

  return 0;
}
```

This code sample computes the sum of 100 random numbers in parallel as follows. 

```
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <algorithm>
#include <cstdlib>

int main(void) {
  // generate random data serially
  thrust::host_vector<int> h_vec(100);
  std::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer to device and compute sum
  thrust::device_vector<int> d_vec = h_vec;
  int x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());
  return 0;
}
```
