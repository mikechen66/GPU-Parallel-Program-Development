# Nvidia CUDA Samples by Nvidia 

## Introduction to the Repository

Samples for CUDA Developers which demonstrates features in CUDA Toolkit. This version 
supports CUDA Toolkit 11.4.

https://github.com/NVIDIA/cuda-samples/tree/master/Samples

https://github.com/NVIDIA/cuda-samples

## Getting Started

Prerequisites

Download and install the CUDA Toolkit 11.4 for your corresponding platform. For system 
requirements and installation instructions of cuda toolkit, please refer to the Linux 
Installation Guide, and the Windows Installation Guide.

Getting the CUDA Samples
Using git clone the repository of CUDA Samples using the command below.

git clone https://github.com/NVIDIA/cuda-samples.git

Without using git the easiest way to use these samples is to download the zip file 
containing the current version by clicking the "Download ZIP" button on the repo page. 
You can then unzip the entire archive and use the samples.

## Building CUDA Samples for Linux

The Linux samples are built using makefiles. To use the makefiles, change the current 
directory to the sample directory you wish to build, and run make:

$ cd <sample_dir>

$ make

The samples makefiles can take advantage of certain options:

1.TARGET_ARCH= - cross-compile targeting a specific architecture. Allowed architectures 
are x86_64, ppc64le, armv7l, aarch64. By default, TARGET_ARCH is set to HOST_ARCH. On 
a x86_64 machine, not setting TARGET_ARCH is the equivalent of setting TARGET_ARCH=x86_64.

$ make TARGET_ARCH=x86_64

$ make TARGET_ARCH=ppc64le

$ make TARGET_ARCH=armv7l

$ make TARGET_ARCH=aarch64

Please see the below weblink for more details on cross platform compilation of the cuda 
samples.

https://docs.nvidia.com/cuda/cuda-samples/index.html#cross-samples

2.dbg=1 - build with debug symbols

$ make dbg=1

3.SMS="A B ..." - override the SM architectures for which the sample will be built, where 
"A B ..." is a space-delimited list of SM architectures. For example, to generate SASS 
for SM 50 and SM 60, use SMS="50 60".

$ make SMS="50 60"

4.HOST_COMPILER=<host_compiler> - override the default g++ host compiler. See the Linux 
Installation Guide for a list of supported host compilers.

$ make HOST_COMPILER=g++

## Dependencies

Some of the CUDA Samples rely on third-party applications and/or libraries or features 
provided by the CUDA Toolkit and Driver, to either build or execute. These dependencies 
are listed in the above weblink. 

If a sample has a third-party dependency being available but is not installed, it will 
waive itself at build time. Each sample's dependencies are listed in its README's 
Dependencies section.
