#ifdef ENABLE_METAL

#include "mtl_internal.h"

// Get an MTLLibrary for the given source text, compiling on first miss.
id<MTLLibrary> mtlGetLibrary(MetalDevice &dev, const char *src, const char *srcName)
{
    NSValue *key = [NSValue valueWithPointer:src];
    id<MTLLibrary> lib = dev.impl->libraryCache[key];
    if (lib) return lib;

    NSError *err = nil;
    NSString *srcStr = [NSString stringWithUTF8String:src];
    MTLCompileOptions *opts = [MTLCompileOptions new];
    // languageVersion is a property on MTLCompileOptions; pin to 3.0 so
    // simdgroup_matrix compiles even when the SDK default is older. bf16
    // needs 3.1, set conditionally below.
    opts.languageVersion = MTLLanguageVersion3_0;
#if TARGET_OS_IPHONE
    if (@available(iOS 17.0, *))
        opts.languageVersion = MTLLanguageVersion3_1;
#else
    if (@available(macOS 14.0, *))
        opts.languageVersion = MTLLanguageVersion3_1;
#endif

    lib = [dev.impl->device newLibraryWithSource:srcStr options:opts error:&err];
    if (!lib)
    {
        NSLog(@"Metal compile of %s failed: %@", srcName, err);
        return nil;
    }
    dev.impl->libraryCache[key] = lib;
    return lib;
}

id<MTLComputePipelineState> mtlGetPipeline(MetalDevice &dev, const char *src,
                                               const char *srcName, const char *fnName)
{
    NSString *cacheKey = [NSString stringWithFormat:@"%p#%s", (void*)src, fnName];
    id<MTLComputePipelineState> pso = dev.impl->pipelineCache[cacheKey];
    if (pso) return pso;

    id<MTLLibrary> lib = mtlGetLibrary(dev, src, srcName);
    if (!lib) return nil;

    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:fnName]];
    if (!fn)
    {
        NSLog(@"Metal: function %s not found in %s", fnName, srcName);
        return nil;
    }

    NSError *err = nil;
    pso = [dev.impl->device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso)
    {
        NSLog(@"Metal: pipeline create for %s failed: %@", fnName, err);
        return nil;
    }
    dev.impl->pipelineCache[cacheKey] = pso;
    return pso;
}

#endif // ENABLE_METAL
