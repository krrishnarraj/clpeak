#ifndef MTL_INTERNAL_H
#define MTL_INTERNAL_H

// ---------------------------------------------------------------------------
// Internal header for Metal backend .mm files.
//
// Provides the ObjC imports, pimpl definitions, and shared helpers that
// every benchmark category file needs.  Not included from the public
// mtl_peak.h — that header stays pure C++ with only forward declarations.
// ---------------------------------------------------------------------------

#ifdef ENABLE_METAL

#include <metal/mtl_peak.h>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>

#include <common/common.h>
#include <common/options.h>
#include <common/inventory.h>

#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <cstring>

// Workload constant for simdgroup_matrix kernels: 1024 outer iters of 4
// independent 8x8x8 matmul chains per simdgroup.  Per simdgroup ops =
// 1024 * 4 * 8*8*8*2 = 4,194,304; per thread (32 threads/simdgroup) =
// 131,072 ops.  Distinct from COOPMAT_WORK_PER_WI (which assumes 16x16x16)
// because Apple silicon's simdgroup_matrix is fixed at 8x8x8.
static const uint32_t MTL_SIMDGROUP_WORK_PER_WI = 131072;

// ---------------------------------------------------------------------------
// Pimpl definitions (forward-declared in mtl_peak.h)
// ---------------------------------------------------------------------------

struct MetalDeviceImpl {
    id<MTLDevice>             device;
    id<MTLCommandQueue>       queue;
    NSMutableDictionary<NSValue*, id<MTLLibrary>>           *libraryCache;
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>> *pipelineCache;
};

struct MetalPeakImpl {
    NSArray<id<MTLDevice>> *allDevices;
};

// ---------------------------------------------------------------------------
// Shared helpers used by benchmark files
// ---------------------------------------------------------------------------

id<MTLLibrary> mtlGetLibrary(MetalDevice &dev, const char *src, const char *srcName);

id<MTLComputePipelineState> mtlGetPipeline(MetalDevice &dev, const char *src,
                                           const char *srcName, const char *fnName);

float mtlRunDispatches(MetalDevice &dev, id<MTLComputePipelineState> pso,
                       id<MTLBuffer> outBuf, const void *scalarArg, uint32_t scalarSize,
                       id<MTLBuffer> secondBuf,
                       MTLSize gridSize, MTLSize tgSize,
                       unsigned int warmup,
                       unsigned int targetTimeUs, unsigned int forcedIters);

#endif // ENABLE_METAL
#endif // MTL_INTERNAL_H
