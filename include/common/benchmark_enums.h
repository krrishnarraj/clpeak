#ifndef BENCHMARK_ENUMS_H
#define BENCHMARK_ENUMS_H

// --------------------------------------------------------------------------
// Neutral enums shared by every backend.  No backend-specific includes.
// --------------------------------------------------------------------------

// Neutral device type — replaces cl_device_type / VkPhysicalDeviceType.
enum class DeviceType : unsigned int {
    Cpu         = 1 << 0,
    Gpu         = 1 << 1,
    Accelerator = 1 << 2,
    Unknown     = 0
};

// Every backend clpeak can be built with, in the order they appear on the
// command line and in the result document.  A backend's flag is its name
// in lower case (--opencl, --coreml); the table in options.cpp holds both.
// Adding one is a value here, a row in that table, and a BackendEntry in
// src/cli/main.cpp -- the parser, the help and the GUI need nothing else.
enum class Backend : unsigned int {
    OpenCL = 0,
    Vulkan,
    Cuda,
    Rocm,
    Metal,
    Oneapi,
    Cpu,
    Onnx,
    Coreml,
    COUNT
};

// Every measurable test across all backends.  A value names WHAT is
// measured, never which backend measures it: the backend is picked by its
// own flag, and a test flag then applies to every backend that runs.  So
// there is one Gemm (cuBLASLt, hipBLASLt, oneMKL, MPS, Accelerate, an ONNX
// MatMul, a Core ML matmul are all "the vendor's tuned matmul"), one
// MatrixCompute (mma.sync, MFMA, cooperative matrix, simdgroup_matrix,
// joint_matrix, AMX/SME), and the NPU runtimes' transfer and dispatch rows
// share TransferBW and KernelLatency with the GPUs.
enum class Benchmark : unsigned int {
    // ---- Compute --------------------------------------------------------
    ComputeSP = 0,
    ComputeHP,
    ComputeDP,
    ComputeMP,
    ComputeBF16,
    ComputeInt,
    ComputeIntFast,     // mad24 (OpenCL)
    ComputeChar,        // 8-bit integer vectors (OpenCL)
    ComputeShort,       // 16-bit integer vectors (OpenCL)
    ComputeInt8DP,      // int8 dot product (DP4a / VNNI / SDOT / dot(): packed)
    ComputeInt16DP,     // int16 dot product (x86 VPDPWSSD / AVX-VNNI-INT16)
    ComputeFP8DP,       // fp8 dot product (ARM FEAT_FP8DOT4)
    ComputeDivSqrt,     // fp divide + sqrt throughput (CPU)
    ComputeIntDiv,      // scalar u64 integer divide throughput (CPU)
    MatrixCompute,      // matrix engine via intrinsics: tensor cores, MFMA/WMMA, coopmat, simdgroup_matrix, joint_matrix, AMX/SME
    Gemm,               // the vendor library's / runtime's tuned matmul: cuBLASLt, hipBLASLt, oneMKL, MPS, Accelerate/BNNS, ONNX MatMul, Core ML matmul
    Attention,          // scaled-dot-product attention through the vendor library (MPSGraph)
    Conv,               // 2-D convolution peak through a graph runtime (ONNX / Core ML)
    NumericError,       // accuracy cost of each dtype vs an fp32 reference (ONNX / Core ML)
    SmtScaling,         // CPU fp32 FMA at 1 thread/core vs all SMT threads (GFLOPS)

    // ---- Crypto / string (CPU fixed-function and SIMD) --------------------
    CryptoAes,          // AES-128 encrypt throughput (AES-NI / VAES-512 / ARM FEAT_AES)
    CryptoSha256,       // SHA-256 compression throughput (SHA-NI / ARM FEAT_SHA256)
    CryptoSha512,       // SHA-512 compression throughput (ARM FEAT_SHA512)
    CryptoCrc32c,       // CRC32-C throughput (SSE4.2 CRC32 / ARM FEAT_CRC32)
    StringScan,         // memchr-style SIMD byte scan, L1-resident (CPU; GB/s)
    Utf8Validate,       // UTF-8 validation via lookup-shuffle PSHUFB/TBL (CPU; GB/s)

    // ---- Bandwidth --------------------------------------------------------
    GlobalBW,
    LocalBW,
    ImageBW,
    TransferBW,         // host<->device, on GPUs over the bus and on NPU runtimes through the framework
    TensorBW,           // resident-tensor read bandwidth through a graph runtime (ONNX / Core ML)
    Activation,         // softmax / layernorm / gate throughput through a graph runtime (ONNX / Core ML)
    CacheBandwidth,     // CPU per-level cache bandwidth (L1/L2/L3/DRAM)
    TextureSample,      // bilinear texture sample rate (Metal; GTexels/s)

    // ---- Latency ----------------------------------------------------------
    KernelLatency,      // the fixed cost of handing a device one piece of work: a kernel launch, a session run, a prediction
    MemoryLatency,      // CPU pointer-chase latency (L1/L2/L3/DRAM + MLP + TLB)
    Atomics,            // CPU atomic fetch-add: uncontended / contended (ns)
    BranchPenalty,      // CPU branch mispredict penalty (ns)
    StoreForward,       // CPU store-to-load forwarding roundtrip (ns)

    // ---- AI composite -----------------------------------------------------
    TransformerBlock,   // fixed transformer decoder block: prefill + decode (ONNX / Core ML)

    COUNT
};

// Test category — drives the run-order phase loop on every backend.
enum class Category {
    Compute,      // all compute (floating-point + integer)
    Crypto,       // fixed-function crypto/hash silicon (CPU: AES/SHA/CRC)
    String,       // string/text processing (CPU: byte scan, UTF-8 validation)
    Bandwidth,
    Latency,
    Ai,           // AI-composite tests (fixed transformer-block micro-graphs)
    Unknown
};

// Map every benchmark to its primary category.
inline Category categoryOf(Benchmark b)
{
    switch (b) {
    case Benchmark::ComputeSP:
    case Benchmark::ComputeHP:
    case Benchmark::ComputeDP:
    case Benchmark::ComputeMP:
    case Benchmark::ComputeBF16:
    case Benchmark::ComputeInt:
    case Benchmark::ComputeIntFast:
    case Benchmark::ComputeChar:
    case Benchmark::ComputeShort:
    case Benchmark::ComputeInt8DP:
    case Benchmark::ComputeInt16DP:
    case Benchmark::ComputeFP8DP:
    case Benchmark::ComputeDivSqrt:
    case Benchmark::ComputeIntDiv:
    case Benchmark::MatrixCompute:
    case Benchmark::Gemm:
    case Benchmark::Attention:
    case Benchmark::Conv:
    case Benchmark::NumericError:
    case Benchmark::SmtScaling:
        return Category::Compute;

    case Benchmark::CryptoAes:
    case Benchmark::CryptoSha256:
    case Benchmark::CryptoSha512:
    case Benchmark::CryptoCrc32c:
        return Category::Crypto;

    case Benchmark::StringScan:
    case Benchmark::Utf8Validate:
        return Category::String;

    case Benchmark::GlobalBW:
    case Benchmark::LocalBW:
    case Benchmark::ImageBW:
    case Benchmark::TransferBW:
    case Benchmark::TensorBW:
    case Benchmark::Activation:
    case Benchmark::CacheBandwidth:
    case Benchmark::TextureSample:
        return Category::Bandwidth;

    case Benchmark::KernelLatency:
    case Benchmark::MemoryLatency:
    case Benchmark::Atomics:
    case Benchmark::BranchPenalty:
    case Benchmark::StoreForward:
        return Category::Latency;

    case Benchmark::TransformerBlock:
        return Category::Ai;

    case Benchmark::COUNT:
        break;
    }
    return Category::Unknown;
}

#endif // BENCHMARK_ENUMS_H
