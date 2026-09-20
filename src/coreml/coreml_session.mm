#ifdef ENABLE_COREML

#include "coreml_session.h"
#include "coreml_internal.h"

#include <common/console_mute.h>
#include <common/coreml_cache.h>

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <signal.h>
#include <sys/statvfs.h>
#include <unistd.h>

namespace
{

double nowUs()
{
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string nsErrorText(NSError *err)
{
  if (!err)
    return "unknown error";
  std::string s = err.localizedDescription ? err.localizedDescription.UTF8String : "";
  // Core ML puts the compiler's actual complaint one level down, and it is
  // the part worth reading: the top-level text is usually just "Unable to
  // compile model" or "Failed to build the model execution plan".
  NSError *under = err.userInfo[NSUnderlyingErrorKey];
  if (under && under.localizedDescription)
  {
    std::string u = under.localizedDescription.UTF8String;
    if (!u.empty() && s.find(u) == std::string::npos)
      s += " (" + u + ")";
  }
  NSString *reason = err.userInfo[NSLocalizedFailureReasonErrorKey];
  if (reason && s.find(reason.UTF8String) == std::string::npos)
    s += " (" + std::string(reason.UTF8String) + ")";
  // One line: the compiler's messages come with their own newlines.
  for (char &c : s)
    if (c == '\n' || c == '\r')
      c = ' ';
  return s;
}

MLMultiArrayDataType mlDataType(int dtype)
{
  switch (dtype)
  {
  case CML_FP32:  return MLMultiArrayDataTypeFloat32;
  case CML_INT32: return MLMultiArrayDataTypeInt32;
  default:        return MLMultiArrayDataTypeFloat16;
  }
}

std::atomic<unsigned> g_packageSeq{0};

// Every session's files carry this process's pid: clpeak-coreml-<pid>-<n>.
NSString *packagePrefix()
{
  return [NSString stringWithFormat:@"clpeak-coreml-%d-", (int)getpid()];
}

// Bytes free on the volume holding the temporary directory, or 0 when it
// cannot be asked.  What matters is real free space, not what a Finder
// window shows: macOS reports purgeable space as available, and a volume
// showing 500 GB "available" has been seen with 12 GB actually free.
uint64_t tempFreeBytes()
{
  struct statvfs st;
  if (statvfs(NSTemporaryDirectory().fileSystemRepresentation, &st) != 0)
    return 0;
  return (uint64_t)st.f_bavail * (uint64_t)st.f_frsize;
}

// Remove packages and compiled models left by clpeak processes that are no
// longer running.  A session cleans up after itself, but a process that
// died mid-session -- a compiler abort, a kill -- leaves up to two
// gigabytes behind, and enough of those fill a volume.  Runs once per
// process, before the first session.
void sweepStalePackages()
{
  static std::once_flag once;
  std::call_once(once, []() {
    @autoreleasepool
    {
      NSFileManager *fm = NSFileManager.defaultManager;
      NSString *tmp = NSTemporaryDirectory();
      NSArray<NSString *> *entries = [fm contentsOfDirectoryAtPath:tmp error:nil];
      for (NSString *name in entries)
      {
        if (![name hasPrefix:@"clpeak-coreml-"])
          continue;
        // clpeak-coreml-<pid>-<n>.<ext>
        NSArray<NSString *> *parts = [[name stringByDeletingPathExtension] componentsSeparatedByString:@"-"];
        if (parts.count < 4)
          continue;
        const int pid = parts[2].intValue;
        if (pid <= 0 || pid == (int)getpid())
          continue;
        if (kill(pid, 0) == 0 || errno != ESRCH)
          continue;   // still running (or not ours to judge)
        [fm removeItemAtPath:[tmp stringByAppendingPathComponent:name] error:nil];
        CLPEAK_VLOG("coreml: removed stale %s left by process %d\n", name.UTF8String, pid);
      }
    }
  });
}

} // namespace

CoremlDeviceKind coremlKindOf(id<MLComputeDeviceProtocol> d)
{
  if ([d isKindOfClass:[MLNeuralEngineComputeDevice class]])
    return CoremlDeviceKind::NeuralEngine;
  if ([d isKindOfClass:[MLGPUComputeDevice class]])
    return CoremlDeviceKind::Gpu;
  return CoremlDeviceKind::Cpu;
}

const char *coremlKindName(CoremlDeviceKind k)
{
  switch (k)
  {
  case CoremlDeviceKind::NeuralEngine: return "Neural Engine";
  case CoremlDeviceKind::Gpu:          return "GPU";
  case CoremlDeviceKind::Cpu:          return "CPU";
  }
  return "?";
}

MLModelConfiguration *coremlConfigurationFor(const coreml_device_info_t &dev)
{
  MLModelConfiguration *cfg = [MLModelConfiguration new];
  switch (dev.kind)
  {
  case CoremlDeviceKind::NeuralEngine:
    // The strictest request Core ML offers: there is no Neural-Engine-only
    // mode, the CPU stays as the fallback, and the compute plan says per
    // operation whether it was needed.
    cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    break;
  case CoremlDeviceKind::Gpu:
    cfg.computeUnits = MLComputeUnitsCPUAndGPU;
    if (dev.gpuIndex >= 0)
    {
      // Several GPUs enumerate as several devices; pin the one this row is for.
      int idx = 0;
      for (id<MLComputeDeviceProtocol> d in MLAllComputeDevices())
        if ([d isKindOfClass:[MLGPUComputeDevice class]])
        {
          if (idx == dev.gpuIndex)
          {
            cfg.preferredMetalDevice = [(MLGPUComputeDevice *)d metalDevice];
            break;
          }
          idx++;
        }
    }
    break;
  case CoremlDeviceKind::Cpu:
    cfg.computeUnits = MLComputeUnitsCPUOnly;
    break;
  }
  return cfg;
}

struct CoremlSession::Impl
{
  coreml_device_info_t dev;
  NSURL *packageURL = nil;
  NSURL *compiledURL = nil;
  MLModel *model = nil;
  NSMutableDictionary<NSString *, MLFeatureValue *> *features = nil;
  MLDictionaryFeatureProvider *provider = nil;
  id<MLFeatureProvider> lastOutput = nil;
  // Input buffers the MLMultiArrays wrap; owned here so they outlive them.
  std::vector<std::unique_ptr<std::vector<uint8_t>>> buffers;

  ~Impl()
  {
    model = nil;
    lastOutput = nil;
    provider = nil;
    features = nil;
    NSFileManager *fm = NSFileManager.defaultManager;
    if (compiledURL)
      [fm removeItemAtURL:compiledURL error:nil];
    if (packageURL)
      [fm removeItemAtURL:packageURL error:nil];
  }
};

std::unique_ptr<CoremlSession> CoremlSession::create(const coreml_device_info_t &dev,
                                                     const CoremlProgram &prog,
                                                     std::string &error)
{
  @autoreleasepool
  {
    std::unique_ptr<CoremlSession> s(new CoremlSession());
    s->impl = new Impl();
    s->impl->dev = dev;

    sweepStalePackages();

    // ---- Is there room? ------------------------------------------------------
    // A session's transient footprint on disk is several times the model:
    // the package, the compiler's copy of it (.mlmodelc), and Core ML's own
    // cache entry, which holds the weights twice more (see
    // include/common/coreml_cache.h) -- 4.5 GB was measured for a 768 MB
    // model.  A volume that runs out under the Metal compiler at load aborts
    // the process ("LLVM ERROR: IO failure on output stream: No space left
    // on device") with no way to catch it, so six times the model plus
    // headroom is checked against real free space and the rung reports
    // itself unsupported instead.
    const std::string model = prog.buildModel();
    const uint64_t modelBytes = (uint64_t)model.size() + (uint64_t)prog.weightBytes().size();
    const uint64_t needBytes = 6 * modelBytes + (1ull << 30);
    const uint64_t freeBytes = tempFreeBytes();
    if (freeBytes && freeBytes < needBytes)
    {
      error = "not enough free disk space for the model files: the temporary volume has " +
              std::to_string(freeBytes >> 20) + " MB free and this model needs about " +
              std::to_string(needBytes >> 20) +
              " MB while it is compiled (what the system reports as available "
              "includes purgeable space, which is not this)";
      return nullptr;
    }

    // ---- Write the package -----------------------------------------------
    // Manifest.json, Data/com.apple.CoreML/model.mlmodel and
    // Data/com.apple.CoreML/weights/weight.bin -- the layout coremltools
    // writes and the compiler expects.  The item identifiers only have to be
    // unique within the manifest.
    const double t0 = nowUs();
    NSString *tmp = NSTemporaryDirectory();
    NSString *dir = [tmp stringByAppendingPathComponent:
                             [packagePrefix() stringByAppendingFormat:@"%u.mlpackage", g_packageSeq++]];
    NSFileManager *fm = NSFileManager.defaultManager;
    [fm removeItemAtPath:dir error:nil];
    NSString *dataDir = [dir stringByAppendingPathComponent:@"Data/com.apple.CoreML/weights"];
    NSError *err = nil;
    if (![fm createDirectoryAtPath:dataDir withIntermediateDirectories:YES attributes:nil error:&err])
    {
      error = "cannot create a temporary model package: " + nsErrorText(err);
      return nullptr;
    }
    s->impl->packageURL = [NSURL fileURLWithPath:dir];
    {
      NSData *md = [NSData dataWithBytesNoCopy:(void *)model.data() length:model.size() freeWhenDone:NO];
      if (![md writeToFile:[dir stringByAppendingPathComponent:@"Data/com.apple.CoreML/model.mlmodel"]
                   options:0 error:&err])
      {
        error = "cannot write the model: " + nsErrorText(err);
        return nullptr;
      }
      const std::string &w = prog.weightBytes();
      NSData *wd = [NSData dataWithBytesNoCopy:(void *)w.data() length:w.size() freeWhenDone:NO];
      if (![wd writeToFile:[dataDir stringByAppendingPathComponent:@"weight.bin"] options:0 error:&err])
      {
        error = "cannot write the weights: " + nsErrorText(err);
        return nullptr;
      }
      NSString *manifest =
          @"{\"fileFormatVersion\":\"1.0.0\",\"itemInfoEntries\":{"
           "\"C1A5EA0B-0000-4000-8000-000000000001\":{\"author\":\"com.apple.CoreML\","
           "\"description\":\"CoreML Model Specification\",\"name\":\"model.mlmodel\","
           "\"path\":\"com.apple.CoreML/model.mlmodel\"},"
           "\"C1A5EA0B-0000-4000-8000-000000000002\":{\"author\":\"com.apple.CoreML\","
           "\"description\":\"CoreML Model Weights\",\"name\":\"weights\","
           "\"path\":\"com.apple.CoreML/weights\"}},"
           "\"rootModelIdentifier\":\"C1A5EA0B-0000-4000-8000-000000000001\"}";
      if (![manifest writeToFile:[dir stringByAppendingPathComponent:@"Manifest.json"] atomically:NO
                        encoding:NSUTF8StringEncoding error:&err])
      {
        error = "cannot write the package manifest: " + nsErrorText(err);
        return nullptr;
      }
    }

    // The Neural Engine's compiler prints its failures straight to stderr
    // ("E5RT encountered an STL exception ... ANECCompile() FAILED") below
    // any log level, and a failure there is not an error here: it is how a
    // shape the ANE cannot take ends up on the CPU, which the plan reports.
    // Muted for the compile, load and plan; a no-op under --verbose.
    clpeak::ScopedConsoleMute mute;

    // ---- Compile -----------------------------------------------------------
    // Produces the .mlmodelc the runtime loads.  This is the generic
    // front-end; the per-device compilers (the Neural Engine's included) run
    // at load, which is why the load is timed separately.
    {
      // The synchronous form, deliberately.  Core ML's model parser raises an
      // NSException on some inputs it has no error path for -- an 8-bit float
      // weight in a blockwise decompression, on macOS 26.6 -- and the
      // asynchronous form runs on a dispatch worker where nothing can catch
      // it and the whole process dies.  On this thread it is an error string.
      NSURL *compiled = nil;
      NSError *cerr = nil;
      @try
      {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        compiled = [MLModel compileModelAtURL:s->impl->packageURL error:&cerr];
#pragma clang diagnostic pop
      }
      @catch (NSException *ex)
      {
        error = "Core ML's compiler raised " + std::string(ex.name ? ex.name.UTF8String : "an exception") +
                (ex.reason ? ": " + std::string(ex.reason.UTF8String) : std::string());
        return nullptr;
      }
      if (!compiled)
      {
        error = nsErrorText(cerr);
        return nullptr;
      }
      s->impl->compiledURL = compiled;
    }
    // The compiler copied everything it needs into the .mlmodelc; the
    // package is dead weight on disk from here, and at a gigabyte a side
    // that is the difference between a rung fitting and not.
    [fm removeItemAtURL:s->impl->packageURL error:nil];
    s->impl->packageURL = nil;
    const double t1 = nowUs();
    s->compileUs = t1 - t0;

    // ---- Load --------------------------------------------------------------
    MLModelConfiguration *cfg = coremlConfigurationFor(dev);
    {
      NSError *lerr = nil;
      MLModel *m = [MLModel modelWithContentsOfURL:s->impl->compiledURL configuration:cfg error:&lerr];
      if (!m)
      {
        error = nsErrorText(lerr);
        return nullptr;
      }
      s->impl->model = m;
    }
    const double t2 = nowUs();
    s->loadUs = t2 - t1;

    // ---- Compute plan ------------------------------------------------------
    // Per-operation placement, which is the CPU-fallback guard of this
    // backend: Core ML never refuses a model for lack of a Neural Engine
    // kernel, it moves the operation, and only the plan says so.
    if (@available(macOS 14.4, iOS 17.4, *))
    {
      dispatch_semaphore_t sem = dispatch_semaphore_create(0);
      __block MLComputePlan *plan = nil;
      __block NSError *perr = nil;
      [MLComputePlan loadContentsOfURL:s->impl->compiledURL
                         configuration:cfg
                     completionHandler:^(MLComputePlan *cp, NSError *e) {
                       plan = cp;
                       perr = e;
                       dispatch_semaphore_signal(sem);
                     }];
      dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
      if (!plan && dev.kind != CoremlDeviceKind::Cpu)
      {
        // Without the plan nothing proves where the work ran, and an
        // accelerator row without that proof is the number this backend
        // exists not to publish.
        error = "Core ML produced no compute plan for this model, so nothing can prove "
                "where it would run: " + nsErrorText(perr);
        return nullptr;
      }
      if (plan)
      {
        MLModelStructureProgram *program = plan.modelStructure.program;
        MLModelStructureProgramFunction *fn = program ? program.functions[@"main"] : nil;
        if (fn)
        {
          s->planKnown = true;
          for (MLModelStructureProgramOperation *op in fn.block.operations)
          {
            MLComputePlanDeviceUsage *du = [plan computeDeviceUsageForMLProgramOperation:op];
            if (!du || !du.preferredComputeDevice)
              continue;   // constants and compile-time decompressions
            CoremlPlacement pl;
            pl.opType = op.operatorName ? op.operatorName.UTF8String : "?";
            pl.preferred = coremlKindOf(du.preferredComputeDevice);
            pl.capable = false;
            for (id<MLComputeDeviceProtocol> d in du.supportedComputeDevices)
              if (coremlKindOf(d) == dev.kind)
                pl.capable = true;
            MLComputePlanCost *cost = [plan estimatedCostOfMLProgramOperation:op];
            pl.weight = cost ? cost.weight : -1.0;
            s->placement.push_back(pl);
          }
          // A plan whose every operation is costed at zero has said where
          // the work goes and nothing about how much of it -- Core ML
          // answers that for a model it keeps on the CPU end to end (a
          // 64-token transformer block under the Neural Engine
          // configuration read 26 operations, all CPU, all 0.000000).
          // Weighed, such a plan would pass onDevice() with every operation
          // off the device; so the weights are marked unknown, which counts
          // each operation whole.
          bool anyCost = false;
          for (const auto &pl : s->placement)
            if (pl.weight > 0.0)
              anyCost = true;
          if (!anyCost)
            for (auto &pl : s->placement)
              pl.weight = -1.0;
        }
      }
    }
    s->planUs = nowUs() - t2;

    s->impl->features = [NSMutableDictionary new];
    return s;
  }
}

CoremlSession::~CoremlSession()
{
  @autoreleasepool
  {
    delete impl;
  }
  // The model is gone; so is any use for what Core ML's runtime cached
  // about it, which is the model again, weights included (see
  // include/common/coreml_cache.h).  Every session is a different model
  // here, so the cache never repays keeping and would otherwise grow by
  // gigabytes a run.
  clpeak::purgeCoreMLCompileCache();
}

namespace
{

// "ios17.matmul" -> "matmul": the version prefix says nothing a reader needs.
std::string bareOp(const std::string &opType)
{
  const size_t dot = opType.find('.');
  return dot == std::string::npos ? opType : opType.substr(dot + 1);
}

// The operations of `placement` off `kind`, joined, either those worth
// `significant` cost or the rest.
std::string joinOps(const std::vector<CoremlPlacement> &placement, CoremlDeviceKind kind,
                    bool significant)
{
  std::string s;
  for (const auto &pl : placement)
  {
    if (pl.preferred == kind)
      continue;
    const bool big = pl.weight < 0.0 || pl.weight > kCoremlOffDeviceShare;
    if (big != significant)
      continue;
    const std::string t = bareOp(pl.opType);
    if (s.find(t) != std::string::npos)
      continue;
    if (!s.empty())
      s += ", ";
    s += t;
  }
  return s;
}

} // namespace

bool CoremlSession::onDevice() const
{
  if (impl->dev.kind == CoremlDeviceKind::Cpu)
    return true;
  double off = 0.0;
  for (const auto &pl : placement)
    if (pl.preferred != impl->dev.kind)
      off += pl.weight < 0.0 ? 1.0 : pl.weight;
  return off <= kCoremlOffDeviceShare;
}

bool CoremlSession::offDeviceCapable() const
{
  bool any = false;
  for (const auto &pl : placement)
    if (pl.preferred != impl->dev.kind)
    {
      any = true;
      if (!pl.capable)
        return false;
    }
  return any;
}

std::string CoremlSession::offDevice() const
{
  // Name what moved: the significant operations when the session failed the
  // guard, everything otherwise.
  std::string s = joinOps(placement, impl->dev.kind, true);
  if (s.empty())
    s = joinOps(placement, impl->dev.kind, false);
  return s;
}

std::string CoremlSession::glue() const
{
  if (impl->dev.kind == CoremlDeviceKind::Cpu)
    return std::string();
  return joinOps(placement, impl->dev.kind, false);
}

void *CoremlSession::bindInput(const std::string &name, int dtype,
                               const std::vector<int64_t> &dims, size_t size,
                               std::string &error)
{
  @autoreleasepool
  {
    auto buf = std::make_unique<std::vector<uint8_t>>(size, 0);
    void *ptr = buf->data();
    NSMutableArray<NSNumber *> *shape = [NSMutableArray new];
    NSMutableArray<NSNumber *> *strides = [NSMutableArray new];
    int64_t stride = 1;
    std::vector<int64_t> st(dims.size(), 1);
    for (size_t i = dims.size(); i-- > 0;)
    {
      st[i] = stride;
      stride *= dims[i];
    }
    for (size_t i = 0; i < dims.size(); i++)
    {
      [shape addObject:@(dims[i])];
      [strides addObject:@(st[i])];
    }
    NSError *err = nil;
    // Wraps the session's own buffer: no copy per run, and the pointer the
    // caller writes through is the one the model reads.
    MLMultiArray *arr = [[MLMultiArray alloc] initWithDataPointer:ptr
                                                            shape:shape
                                                         dataType:mlDataType(dtype)
                                                          strides:strides
                                                      deallocator:nil
                                                            error:&err];
    if (!arr)
    {
      error = "cannot create an input array: " + nsErrorText(err);
      return nullptr;
    }
    impl->buffers.push_back(std::move(buf));
    impl->features[[NSString stringWithUTF8String:name.c_str()]] =
        [MLFeatureValue featureValueWithMultiArray:arr];
    impl->provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:impl->features error:&err];
    if (!impl->provider)
    {
      error = "cannot create the feature provider: " + nsErrorText(err);
      return nullptr;
    }
    return ptr;
  }
}

bool CoremlSession::run(std::string &error)
{
  @autoreleasepool
  {
    NSError *err = nil;
    id<MLFeatureProvider> out = [impl->model predictionFromFeatures:impl->provider error:&err];
    if (!out)
    {
      error = nsErrorText(err);
      return false;
    }
    impl->lastOutput = out;
    return true;
  }
}

double CoremlSession::timeRuns(unsigned n, std::string &error)
{
  const double t0 = nowUs();
  for (unsigned i = 0; i < n; i++)
  {
    @autoreleasepool
    {
      NSError *err = nil;
      id<MLFeatureProvider> out = [impl->model predictionFromFeatures:impl->provider error:&err];
      if (!out)
      {
        error = nsErrorText(err);
        return -1.0;
      }
      if (i + 1 == n)
        impl->lastOutput = out;
    }
  }
  return (nowUs() - t0) / (double)n;
}

bool CoremlSession::outputBytes(const std::string &name, std::vector<uint8_t> &out,
                                std::string &error)
{
  @autoreleasepool
  {
    if (!impl->lastOutput)
    {
      error = "no prediction has run";
      return false;
    }
    MLFeatureValue *fv = [impl->lastOutput featureValueForName:[NSString stringWithUTF8String:name.c_str()]];
    MLMultiArray *arr = fv ? fv.multiArrayValue : nil;
    if (!arr)
    {
      error = "output '" + name + "' is missing from the prediction";
      return false;
    }
    __block std::vector<uint8_t> *dst = &out;
    __block bool ok = false;
    // A prediction's output may arrive strided; walk it through the
    // strides rather than assuming it is packed.
    const NSInteger count = arr.count;
    const NSUInteger rank = arr.shape.count;
    std::vector<int64_t> shape(rank), strides(rank);
    for (NSUInteger i = 0; i < rank; i++)
    {
      shape[i] = arr.shape[i].integerValue;
      strides[i] = arr.strides[i].integerValue;
    }
    size_t es = 2;
    if (arr.dataType == MLMultiArrayDataTypeFloat32 || arr.dataType == MLMultiArrayDataTypeInt32)
      es = 4;
    else if (arr.dataType == MLMultiArrayDataTypeDouble)
      es = 8;
    [arr getBytesWithHandler:^(const void *bytes, NSInteger size) {
      (void)size;
      dst->resize((size_t)count * es);
      const uint8_t *src = static_cast<const uint8_t *>(bytes);
      // Packed fast path.
      bool packed = true;
      int64_t expect = 1;
      for (size_t i = rank; i-- > 0;)
      {
        if (strides[i] != expect)
          packed = false;
        expect *= shape[i];
      }
      if (packed)
        std::memcpy(dst->data(), src, dst->size());
      else
      {
        std::vector<int64_t> idx(rank, 0);
        for (int64_t k = 0; k < count; k++)
        {
          int64_t off = 0;
          for (size_t i = 0; i < rank; i++)
            off += idx[i] * strides[i];
          std::memcpy(dst->data() + (size_t)k * es, src + (size_t)off * es, es);
          for (size_t i = rank; i-- > 0;)
          {
            if (++idx[i] < shape[i])
              break;
            idx[i] = 0;
          }
        }
      }
      ok = true;
    }];
    if (!ok)
      error = "cannot read output '" + name + "'";
    return ok;
  }
}

#endif // ENABLE_COREML
