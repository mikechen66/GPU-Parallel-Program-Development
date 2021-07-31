# Hands-On GPU-Accelerated CV with OpenCV and CUDA
Bhaumik Vaidya

## Introduction 

OpenCV is the most widely chosen tool for computer vision with its ability to work in multiple 
programming languages. This is where CUDA comes into the picture, allowing OpenCV to leverage 
powerful NVDIA GPUs. This book provides a detailed overview of integrating OpenCV with CUDA 
for practical applications.

## Instructions

With exception of some unique (jargon) functionalities, the CUDA program is full compliance with 
C and C++ but ended as .cu rather than .c or .cpp. Users need to install a cuda plug-in such as 
CUDA Snippets for Sublime Text. The program of hello.cu looks like the following:

```
#include <iostream>
#include <stdio.h>

__global__ void my_kernel(void) {
}

int main(void) {
    my_kernel <<<1, 1 >>>();
    printf("Hello, CUDA!\n");
    return 0;
}
```

## CUDA Features

### Kernel function 

The CUDA has a hierarchical architecture in terms of parallel execution with millions of threads. 
The kernel execution can be done in parallel with multiple blocks. Each block is further divided 
into multiple threads. It has the following expression. 

```
kernel<<<grid_size, block_size>>>
```

### Jargons

There are three qualifier keywords for CPU and GPU communicationL  "__global__" for a function call 
from cpu to gpu, "__device__" for GPU and "__host__" for CPU. 

```
#include <stdio.h>
#define N 5

__global__ void gpu_global_memory(int *d_a) { / a global function 
    d_a[threadIdx.x] = threadIdx.x; //d_a for device (GPU)
}

int main(int argc, char **argv) {
    int h_a[N]; //h_a for host (CPU)
    int *d_a;
    cudaMalloc((void **)&d_a, sizeof(int) *N);
    cudaMemcpy((void *)d_a, (void *)h_a, sizeof(int) *N, cudaMemcpyHostToDevice);
    gpu_global_memory << <1, N >> >(d_a);
    cudaMemcpy((void *)h_a, (void *)d_a, sizeof(int) *N, cudaMemcpyDeviceToHost);
    printf("Array in Global Memory is: \n");
    for (int i = 0; i < N; i++) {
        printf("At Index: %d --> %d \n", i, h_a[i]);
    }
    return
}
```

Atomic snippet for fine-grained operation

```
// Initialize GPU memory with zero value.
cudaMemset((void *)d_a, 0, ARRAY_BYTES);
gpu_increment_without_atomic <<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_a);
```

Synchronization between host and device 

```
cudaDeviceSynchronize();
cudaThreadSynchronize();
__syncthreads();
cudaEventSynchronize
cudaDeviceSynchronize();
cudaStreamSynchronize(stream0);
cudaStreamSynchronize(stream1);
```

### Dynamic memory management

The memory management inlcudes cudaMalloc, cudaMemcpy and cudaFree. It is similar to
the related functions in C. 

```
cudaMalloc: for dynamic memory allocation
cudaMemcpy: being similar to the funcion of Memcpy in C 
cudaFree: as similar as above
```

## CUDA Module for OpenCV 

### GPUMat

```
cv::cuda::GpuMat d_result1,d_img1, d_img2;
```

### Pointer, arithmetic and bitwise Operation 

```
cv::Ptr<cv::cuda::Filter> filter3x3,filter5x5,filter7x7;
cv::cuda::add(d_img1,d_img2, d_result1);
cv::cuda::subtract(d_img1, d_img2,d_result1);
cv::cuda::bitwise_not(d_img1,d_result1);
cv::cuda::bitwise_and(d_thresc[0], d_thresc[1],d_intermediate);
```

### Image processing 

```
cv::cuda::cvtColor(d_img1, d_result1,cv::COLOR_BGR2GRAY);
cv::cuda::threshold(d_img1, d_result2, 128.0, 255.0, cv::THRESH_BINARY_INV);
cv::cuda::equalizeHist(d_img1, d_result1);
cv::cuda::resize(d_img1,d_result1,cv::Size(200, 200), cv::INTER_CUBIC);
v::cuda::warpAffine(d_img1,d_result1,trans_mat,d_img1.size());
filter3x3 = cv::cuda::createBoxFilter(CV_8UC1,CV_8UC1,cv::Size(3,3));
filter5x5 = cv::cuda::createGaussianFilter(CV_8UC1,CV_8UC1,cv::Size(5,5),1);
filter1 = cv::cuda::createLaplacianFilter(CV_8UC1,CV_8UC1,1);
filterd = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE,CV_8UC1,element);
```

### Object Detection 

```
cv::Ptr<cv::cuda::ORB> detector = cv::cuda::ORB::create();
cv::Ptr<cv::cuda::CannyEdgeDetector> canny_edge = cv::cuda::createCannyEdgeDetector(2.0, 100.0, 3, false);
Ptr<cuda::CascadeClassifier> cascade = cuda::CascadeClassifier::create("haarcascade_eye.xml");
Ptr<BackgroundSubtractor> mog = cuda::createBackgroundSubtractorMOG();
```

## PyCUDA

There are three steps to develop PyCUDA code, inlcuding Import pycuda.driver as drv, Import 
pycuda.autoinit and From pycuda.compiler import SourceModule. The most important kernels include
map, reducton and scan. It is the simple example for the above hello.cu. 

```
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
    #include <stdio.h>

     __global__ void myfirst_kernel() {
        printf("Hello,PyCUDA!!!");
      }
""")

function = mod.get_function("myfirst_kernel")
function(block=(1,1,1))
```

## Resources 

Please Visit the following link to check out videos of the code.  

Video: 

http://bit.ly/2PZOYcH

PyCUDA

https://github.com/inducer/pycuda

Pypi: 

https://pypi.org/project/pycuda/

