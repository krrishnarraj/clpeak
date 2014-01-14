clpeak
======

A tool which profiles OpenCL devices to find their peak capacities like bandwidth & compute

eg:

1.
Platform: AMD Accelerated Parallel Processing

  Device: Cayman

    Driver version: 1348.4 (Linux x64)

    Global memory bandwidth (GBPS)
      float   : 130.97
      float2  : 131.36
      float4  : 90.50
      float8  : 69.91
      float16 : 35.27

    Single-precision compute (GFLOPS)
      float   : 674.44
      float2  : 1345.68
      float4  : 2601.47
      float8  : 2586.69
      float16 : 2573.38

    Double-precision compute (GFLOPS)
      double   : 671.24
      double2  : 671.59
      double4  : 670.93
      double8  : 669.51
      double16 : 666.50

    Transfer bandwidth (GBPS)
      enqueueWriteBuffer         : 3.53
      enqueueReadBuffer          : 4.43
      enqueueMapBuffer(for read) : 152.89
        memcpy from mapped ptr   : 4.40
      enqueueUnmap(after write)  : 1781.26
        memcpy to mapped ptr     : 4.42

    Kernel launch latency : 44.22 us


2.
Platform: ARM Platform

  Device: Mali-T604

    Driver version: 1.1 (Linux ARM)

    Global memory bandwidth (GBPS)
      float   : 1.56
      float2  : 4.41
      float4  : 5.75
      float8  : Out of resources! Skipped

    Single-precision compute (GFLOPS)
      float   : 2.38
      float2  : 16.40
      float4  : 8.07
      float8  : 21.84
      float16 : 16.37

    No double precision support! Skipped

    Transfer bandwidth (GBPS)
      enqueueWriteBuffer         : 5.84
      enqueueReadBuffer          : 2.59
      enqueueMapBuffer(for read) : 934.99
        memcpy from mapped ptr   : 2.83
      enqueueUnmap(after write)  : 1813.75
        memcpy to mapped ptr     : 2.85

    Kernel launch latency : 149.46 us


Send in results of your device to krrishnarraj@gmail.com
