# Hands On GPU Accelerated Computer Vision with OpenCV and CUDA
Bhaumik Vaidya

## Introduction 

OpenCV is the most widely chosen tool for computer vision with its ability to work in multiple 
programming languages. This is where CUDA comes into the picture, allowing OpenCV to leverage 
powerful NVDIA GPUs. This book provides a detailed overview of integrating OpenCV with CUDA 
for practical applications.

## Instructions and Navigations

All of the code is organized into folders. For example, Chapter02.

The code will look like the following:
```
while (tid < N)
    {
       d_c[tid] = d_a[tid] + d_b[tid];
       tid += blockDim.x * gridDim.x;
    }
```

## Resources 

Please Visit the following link to check out videos of the code.  

Video: 

http://bit.ly/2PZOYcH

PyCUDA

https://github.com/inducer/pycuda

Pypi: 

https://pypi.org/project/pycuda/

