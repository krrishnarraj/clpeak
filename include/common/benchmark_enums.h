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

// Every measurable test across all backends.
enum class Benchmark : unsigned int {
    GlobalBW = 0,
    LocalBW,
    ImageBW,
    ComputeHP,
    ComputeMP,
    ComputeSP,
    ComputeDP,
    ComputeInt,
    ComputeIntFast,
    ComputeChar,
    ComputeShort,
    ComputeInt8DP,
    ComputeInt16DP,     // int16 dot product (x86 VPDPWSSD / AVX-VNNI-INT16)
    ComputeBF16,
    ComputeFP8DP,       // fp8 dot product (ARM FEAT_FP8DOT4)
    ComputeDivSqrt,     // fp divide + sqrt throughput (CPU)
    ComputeIntDiv,      // scalar u64 integer divide throughput (CPU)
    CoopMatrix,
    Wmma,
    SimdgroupMatrix,
    MpsGemm,
    Cublas,
    Rocwmma,
    Mfma,
    Rocblas,
    JointMatrix,
    Onemkl,
    Amx,                // CPU matrix engine (Intel AMX / ARM I8MM)
    CryptoAes,          // AES-128 encrypt throughput (AES-NI / VAES-512 / ARM FEAT_AES)
    CryptoSha256,       // SHA-256 compression throughput (SHA-NI / ARM FEAT_SHA256)
    CryptoSha512,       // SHA-512 compression throughput (ARM FEAT_SHA512)
    CryptoCrc32c,       // CRC32-C throughput (SSE4.2 CRC32 / ARM FEAT_CRC32)
    TransferBW,
    CacheBandwidth,     // CPU per-level cache bandwidth (L1/L2/L3/DRAM)
    MemoryLatency,      // CPU pointer-chase latency (L1/L2/L3/DRAM)
    KernelLatency,
    COUNT
};

// Test category — drives the run-order phase loop on every backend.
enum class Category {
    FpCompute,
    IntCompute,
    Crypto,       // fixed-function crypto/hash silicon (CPU: AES/SHA/CRC)
    Bandwidth,
    Latency,
    Unknown
};

// Map every benchmark to its primary category.  Tensor / vendor-library
// tests that span both fp and int variants (Wmma, CoopMatrix, SimdgroupMatrix,
// Cublas, MpsGemm, Rocwmma, Mfma, Rocblas, Amx) are listed under their fp form here; backends iterate
// them again in the int_compute phase emitting only int variants there.
inline Category categoryOf(Benchmark b)
{
    switch (b) {
    case Benchmark::GlobalBW:
    case Benchmark::LocalBW:
    case Benchmark::ImageBW:
    case Benchmark::TransferBW:
    case Benchmark::CacheBandwidth:
        return Category::Bandwidth;

    case Benchmark::ComputeSP:
    case Benchmark::ComputeHP:
    case Benchmark::ComputeDP:
    case Benchmark::ComputeMP:
    case Benchmark::ComputeBF16:
    case Benchmark::ComputeFP8DP:
    case Benchmark::ComputeDivSqrt:
    case Benchmark::Wmma:
    case Benchmark::CoopMatrix:
    case Benchmark::SimdgroupMatrix:
    case Benchmark::Cublas:
    case Benchmark::MpsGemm:
    case Benchmark::Rocwmma:
    case Benchmark::Mfma:
    case Benchmark::Rocblas:
    case Benchmark::JointMatrix:
    case Benchmark::Onemkl:
    case Benchmark::Amx:
        return Category::FpCompute;

    case Benchmark::ComputeInt:
    case Benchmark::ComputeIntFast:
    case Benchmark::ComputeChar:
    case Benchmark::ComputeShort:
    case Benchmark::ComputeInt8DP:
    case Benchmark::ComputeInt16DP:
    case Benchmark::ComputeIntDiv:
        return Category::IntCompute;

    case Benchmark::CryptoAes:
    case Benchmark::CryptoSha256:
    case Benchmark::CryptoSha512:
    case Benchmark::CryptoCrc32c:
        return Category::Crypto;

    case Benchmark::KernelLatency:
    case Benchmark::MemoryLatency:
        return Category::Latency;

    case Benchmark::COUNT:
        break;
    }
    return Category::Unknown;
}

#endif // BENCHMARK_ENUMS_H
