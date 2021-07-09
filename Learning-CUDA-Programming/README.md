# Learning CUDA Programming
Jaegeun Han

## Introduction

CUDA is designed to work with programming languages such as C, C++, and Python. Developera 
can use CUDA to leverage a GPU's parallel computing power for a range of high-performance 
computing applications in the fields of science, healthcare, and deep learning.

## Contents

Understand general GPU operations and programming patterns in CUDA; uncover the difference between GPU programming and CPU programming;
analyze GPU application performance and implement optimization strategies; explore GPU programming, profiling, and debugging tools; grasp parallel programming algorithms and how to implement them; scale GPU-accelerated applications with multi-GPU and multi-nodes; delve into GPU programming platforms with accelerated libraries, Python, and OpenACC Gain insights into deep learning accelerators in deep learning.

## Instructions

All of the code is organized into folders. For example, Chapter02. The code will look like the following:


```
#include<stdio.h>
#include<stdlib.h>

__global__ void print_from_gpu(void) {
    printf("Hello World! from thread [%d,%d] \
        From device\n", threadIdx.x,blockIdx.x);
}
```
