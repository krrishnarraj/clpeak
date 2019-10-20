# clpeak

[![Build Status](https://travis-ci.com/krrishnarraj/clpeak.svg?branch=master)](https://travis-ci.com/krrishnarraj/clpeak)
[![Snap Status](https://build.snapcraft.io/badge/krrishnarraj/clpeak.svg)](https://build.snapcraft.io/user/krrishnarraj/clpeak)

A synthetic benchmarking tool to measure peak capabilities of opencl devices. It only measures the peak metrics that can be achieved using vector operations and does not represent a real-world use case

## Building

```console
git submodule update --init --recursive --remote
mkdir build
cd build
cmake ..
cmake --build .
```

## Sample

```text
Platform: NVIDIA CUDA
  Device: Tesla V100-SXM2-16GB
    Driver version  : 390.77 (Linux x64)
    Compute units   : 80
    Clock frequency : 1530 MHz

    Global memory bandwidth (GBPS)
      float   : 767.48
      float2  : 810.81
      float4  : 843.06
      float8  : 726.12
      float16 : 735.98

    Single-precision compute (GFLOPS)
      float   : 15680.96
      float2  : 15674.50
      float4  : 15645.58
      float8  : 15583.27
      float16 : 15466.50

    No half precision support! Skipped

    Double-precision compute (GFLOPS)
      double   : 7859.49
      double2  : 7849.96
      double4  : 7832.96
      double8  : 7799.82
      double16 : 7740.88

    Integer compute (GIOPS)
      int   : 15653.47
      int2  : 15654.40
      int4  : 15655.21
      int8  : 15659.04
      int16 : 15608.65

    Transfer bandwidth (GBPS)
      enqueueWriteBuffer         : 10.64
      enqueueReadBuffer          : 11.92
      enqueueMapBuffer(for read) : 9.97
        memcpy from mapped ptr   : 8.62
      enqueueUnmap(after write)  : 11.04
        memcpy to mapped ptr     : 9.16

    Kernel launch latency : 7.22 us
```
