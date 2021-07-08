# PyCUDA

## Introduction

PyCUDA lets developers access Nvidia's CUDA parallel computation API from Python. It is one of 
the best wrappers of the CUDA APIs. Please have a look at the repository as follows. 

https://github.com/inducer/pycuda

## Features

Object cleanup: Tied to lifetime of objects. This idiom, often called RAII in C++, makes it much 
easier to write correct, leak- and crash-free code. 

Convenience: Abstractions like pycuda.driver.SourceModule and pycuda.gpuarray.GPUArray make CUDA
programming more convenient than with Nvidia's C-based runtime.

Completeness: PyCUDA puts the full power of CUDA's driver API at your disposal. It includes code 
for interoperability with OpenGL.

Automatic Error Checking: All CUDA errors are automatically translated into Python exceptions.

Speed: PyCUDA's base layer is written in C++, so all the niceties above are virtually free. Helpful 
Documentation and a Wiki.
