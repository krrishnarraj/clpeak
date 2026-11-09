#include <common/vocabulary.h>
#include <common/options.h>   // benchmarkFlag, benchmarkFromFlag

#include <cstddef>

// Every list below is append-only (include/common/vocabulary.h): a new value
// goes at the end of its table and of its enum, and a string here is never
// changed.  The static_asserts keep each table one string per enum value.

namespace xpupeak {

namespace {

template <class E, std::size_t N>
const char *nameIn(const char *const (&names)[N], E v)
{
    const std::size_t i = static_cast<std::size_t>(v);
    return i < N ? names[i] : "";
}

template <class E, std::size_t N>
bool parseIn(const char *const (&names)[N], const std::string &s, E &out)
{
    for (std::size_t i = 0; i < N; i++)
        if (s == names[i])
        {
            out = static_cast<E>(i);
            return true;
        }
    return false;
}

#define XPUPEAK_VOCAB_LIST(E, names)                                          \
    static_assert(sizeof(names) / sizeof(names[0]) ==                        \
                      static_cast<std::size_t>(E::COUNT),                    \
                  #E " needs exactly one string per value")

// ── Data types ─────────────────────────────────────────────────────────────

const char *const kDTypes[] = {
    "fp64", "fp32", "tf32", "fp16", "bf16",
    "fp8_e4m3", "fp8_e5m2", "fp8_e4m3fnuz", "fp8_e5m2fnuz",
    "fp6_e3m2", "fp6_e2m3", "fp4_e2m1",
    "mxfp8_e4m3", "mxfp8_e5m2", "mxfp6_e3m2", "mxfp6_e2m3", "mxfp4", "nvfp4",
    "int64", "int32", "int24", "int16", "int8", "int4",
    "uint64", "uint16", "uint8", "uint4",
    "uint32",
};
XPUPEAK_VOCAB_LIST(DType, kDTypes);

// ── Dim keys and their values ──────────────────────────────────────────────

const char *const kQuant[]       = {"none", "qdq", "lut", "block", "channel"};
const char *const kSparsity[]    = {"dense", "2:4"};
const char *const kScope[]       = {"core", "device"};
const char *const kLevel[]       = {"l1", "l2", "l3", "slc", "dram"};
const char *const kPattern[]     = {"random", "linear", "page_stride", "predictable"};
const char *const kOp[]          = {"read", "write", "copy", "triad", "fdiv", "fsqrt", "silu",
                                    "softmax", "layernorm", "dispatch", "roundtrip", "mispredict"};
const char *const kDirection[]   = {"h2d", "d2h", "roundtrip"};
const char *const kMemory[]      = {"pinned", "pageable"};
const char *const kLayer[]       = {"conv3x3s2", "conv1x1", "depthwise3x3"};
const char *const kPixelFormat[] = {"rgba32f", "rgba16f", "rgba8", "r32f"};
const char *const kPhase[]       = {"prefill", "decode"};
const char *const kContention[]  = {"none", "all"};
const char *const kWorkload[]    = {"trivial_op", "small_matmul", "session_create", "model_create"};
const char *const kIsa[]         = {"scalar", "sse2", "ssse3", "sse4_2", "avx", "avx2", "avx512",
                                    "avx10", "amx", "neon", "sve", "sve2", "sme", "sme2"};
const char *const kImpl[]        = {"cublaslt", "hipblaslt", "rocblas", "rocwmma", "onemkl", "mps",
                                    "mpsgraph", "accelerate", "bnns"};
XPUPEAK_VOCAB_LIST(dim::Quant, kQuant);
XPUPEAK_VOCAB_LIST(dim::Sparsity, kSparsity);
XPUPEAK_VOCAB_LIST(dim::Scope, kScope);
XPUPEAK_VOCAB_LIST(dim::Level, kLevel);
XPUPEAK_VOCAB_LIST(dim::Pattern, kPattern);
XPUPEAK_VOCAB_LIST(dim::Op, kOp);
XPUPEAK_VOCAB_LIST(dim::Direction, kDirection);
XPUPEAK_VOCAB_LIST(dim::Memory, kMemory);
XPUPEAK_VOCAB_LIST(dim::Layer, kLayer);
XPUPEAK_VOCAB_LIST(dim::PixelFormat, kPixelFormat);
XPUPEAK_VOCAB_LIST(dim::Phase, kPhase);
XPUPEAK_VOCAB_LIST(dim::Contention, kContention);
XPUPEAK_VOCAB_LIST(dim::Workload, kWorkload);
XPUPEAK_VOCAB_LIST(dim::Isa, kIsa);
XPUPEAK_VOCAB_LIST(dim::Impl, kImpl);

using DC = DimClass;
using DK = DimKind;
using DD = DimDefault;

// Indexed by Dim.  A key's class and default never change; a correction is
// a new key (docs/format-v4.md, Compatibility).
const DimKeyInfo kDims[] = {
    {"dtype", DC::Identity, DK::DataType, DD::FamilyDtype, "",
     "the data type the operation computes on: its inputs'"},
    {"weights", DC::Identity, DK::DataType, DD::SameAsDtype, "",
     "the stored weights' type, where it is not the activations'"},
    {"acc", DC::Identity, DK::DataType, DD::FamilyAcc, "",
     "the accumulator's type"},
    {"quant", DC::Identity, DK::Word, DD::Value, "none",
     "how quantized weights are scaled: quantize-dequantize pairs, a lookup table, "
     "a scale per block, a scale per output channel"},
    {"sparsity", DC::Identity, DK::Word, DD::Value, "dense",
     "structured sparsity"},
    {"scope", DC::Identity, DK::Word, DD::Value, "device",
     "how much of the processor ran it: one thread on one core, or all of it"},
    {"threads_per_core", DC::Identity, DK::Integer, DD::None, "",
     "threads per core, where SMT is what is measured"},
    {"level", DC::Identity, DK::Word, DD::None, "",
     "where the data lived"},
    {"pattern", DC::Identity, DK::Word, DD::Value, "random",
     "the access pattern; page_stride puts every read on another page, predictable "
     "is a branch the CPU guesses right"},
    {"in_flight", DC::Identity, DK::Integer, DD::Value, "1",
     "independent accesses in flight at once"},
    {"op", DC::Identity, DK::Word, DD::None, "",
     "the operation"},
    {"direction", DC::Identity, DK::Word, DD::None, "",
     "which way the data moved between the host and the device"},
    {"memory", DC::Identity, DK::Word, DD::None, "",
     "the host memory a transfer used"},
    {"working_set", DC::Identity, DK::Integer, DD::None, "",
     "bytes the reading's data occupies"},
    {"layer", DC::Identity, DK::Word, DD::None, "",
     "a convolution layer's shape"},
    {"pixel_format", DC::Identity, DK::Word, DD::None, "",
     "an image's texel format"},
    {"phase", DC::Identity, DK::Word, DD::None, "",
     "a transformer block's phase"},
    {"context", DC::Identity, DK::Integer, DD::None, "",
     "tokens: a prefill's sequence length, a decode's KV-cache length"},
    {"contention", DC::Identity, DK::Word, DD::Value, "none",
     "whether every other thread competed for the same data"},
    {"workload", DC::Identity, DK::Word, DD::None, "",
     "what a dispatch-latency reading submits"},
    {"width", DC::Variant, DK::Integer, DD::None, "",
     "vector width, in elements"},
    {"chains", DC::Variant, DK::Integer, DD::None, "",
     "independent dependency chains in flight"},
    {"isa", DC::Variant, DK::Word, DD::None, "",
     "the instruction set a CPU kernel was written in"},
    {"simd_bits", DC::Variant, DK::Integer, DD::None, "",
     "vector register width in bits: 128, 256, 512, or the SVE or SME vector length"},
    {"impl", DC::Variant, DK::Word, DD::None, "",
     "the library that ran it"},
    {"form", DC::Variant, DK::Text, DD::None, "",
     "a spelling of the same work: a race's winner, a layout, a graph form"},
    {"tile", DC::Variant, DK::Text, DD::None, "",
     "a matrix tile or instruction shape"},
    {"problem_size", DC::Variant, DK::Integer, DD::None, "",
     "a GEMM's edge, where a test sweeps sizes"},
    {"kv", DC::Identity, DK::DataType, DD::SameAsDtype, "",
     "a transformer block's KV-cache type"},
};
XPUPEAK_VOCAB_LIST(Dim, kDims);

struct WordList {
    const char *const *names;
    unsigned           count;
};

template <std::size_t N>
constexpr WordList words(const char *const (&names)[N])
{
    return {names, static_cast<unsigned>(N)};
}

WordList wordsOf(Dim d)
{
    switch (d)
    {
    case Dim::Quant:       return words(kQuant);
    case Dim::Sparsity:    return words(kSparsity);
    case Dim::Scope:       return words(kScope);
    case Dim::Level:       return words(kLevel);
    case Dim::Pattern:     return words(kPattern);
    case Dim::Op:          return words(kOp);
    case Dim::Direction:   return words(kDirection);
    case Dim::Memory:      return words(kMemory);
    case Dim::Layer:       return words(kLayer);
    case Dim::PixelFormat: return words(kPixelFormat);
    case Dim::Phase:       return words(kPhase);
    case Dim::Contention:  return words(kContention);
    case Dim::Workload:    return words(kWorkload);
    case Dim::Isa:         return words(kIsa);
    case Dim::Impl:        return words(kImpl);
    default:               return {nullptr, 0};
    }
}

// ── Families ───────────────────────────────────────────────────────────────

using FM = FamilyMethod;
using AR = AccRule;

// Indexed by Benchmark.
constexpr FamilyInfo kFamilies[] = {
    {Benchmark::ComputeSP,        FM::Vector,  true,  DType::Fp32,   AR::SameAsDtype, DType::Fp32},
    {Benchmark::ComputeHP,        FM::Vector,  true,  DType::Fp16,   AR::SameAsDtype, DType::Fp32},
    {Benchmark::ComputeDP,        FM::Vector,  true,  DType::Fp64,   AR::SameAsDtype, DType::Fp32},
    // fp16 inputs, fp32 accumulate: what "mixed" means.
    {Benchmark::ComputeMP,        FM::Vector,  true,  DType::Fp16,   AR::Fixed,       DType::Fp32},
    {Benchmark::ComputeBF16,      FM::Vector,  true,  DType::Bf16,   AR::SameAsDtype, DType::Fp32},
    {Benchmark::ComputeInt,       FM::Vector,  true,  DType::Int32,  AR::SameAsDtype, DType::Fp32},
    {Benchmark::ComputeIntFast,   FM::Vector,  true,  DType::Int24,  AR::SameAsDtype, DType::Fp32},
    {Benchmark::ComputeChar,      FM::Vector,  true,  DType::Int8,   AR::SameAsDtype, DType::Fp32},
    {Benchmark::ComputeShort,     FM::Vector,  true,  DType::Int16,  AR::SameAsDtype, DType::Fp32},
    // Dot products sum into a wider accumulator.
    {Benchmark::ComputeInt8DP,    FM::Vector,  true,  DType::Int8,   AR::Usual,       DType::Fp32},
    {Benchmark::ComputeInt16DP,   FM::Vector,  true,  DType::Int16,  AR::Usual,       DType::Fp32},
    // E4M3 and E5M2 both: no one data type.
    {Benchmark::ComputeFP8DP,     FM::Vector,  false, DType::Fp32,   AR::Usual,       DType::Fp32},
    // fp32 and fp64 rows.
    {Benchmark::ComputeDivSqrt,   FM::Vector,  false, DType::Fp32,   AR::SameAsDtype, DType::Fp32},
    {Benchmark::ComputeIntDiv,    FM::Vector,  true,  DType::Uint64, AR::SameAsDtype, DType::Fp32},
    {Benchmark::MatrixCompute,    FM::Matrix,  false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::Gemm,             FM::Library, false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::Attention,        FM::Library, false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::Conv,             FM::Graph,   false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::NumericError,     FM::Graph,   false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::SmtScaling,       FM::Vector,  true,  DType::Fp32,   AR::SameAsDtype, DType::Fp32},
    {Benchmark::CryptoAes,        FM::Crypto,  false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::CryptoSha256,     FM::Crypto,  false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::CryptoSha512,     FM::Crypto,  false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::CryptoCrc32c,     FM::Crypto,  false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::StringScan,       FM::String,  false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::Utf8Validate,     FM::String,  false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::GlobalBW,         FM::Memory,  false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::LocalBW,          FM::Memory,  false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::ImageBW,          FM::Memory,  false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::TransferBW,       FM::Memory,  false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::TensorBW,         FM::Graph,   false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::Activation,       FM::Graph,   false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::CacheBandwidth,   FM::Memory,  false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::TextureSample,    FM::Memory,  false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::KernelLatency,    FM::Latency, false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::MemoryLatency,    FM::Latency, false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::Atomics,          FM::Latency, false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::BranchPenalty,    FM::Latency, false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::StoreForward,     FM::Latency, false, DType::Fp32,   AR::Usual,       DType::Fp32},
    {Benchmark::TransformerBlock, FM::Graph,   false, DType::Fp32,   AR::Usual,       DType::Fp32},
};
static_assert(sizeof(kFamilies) / sizeof(kFamilies[0]) ==
                  static_cast<std::size_t>(Benchmark::COUNT),
              "every Benchmark needs exactly one family row");

constexpr bool familiesInEnumOrder()
{
    for (std::size_t i = 0; i < sizeof(kFamilies) / sizeof(kFamilies[0]); i++)
        if (kFamilies[i].benchmark != static_cast<Benchmark>(i)) return false;
    return true;
}
static_assert(familiesInEnumOrder(), "family rows must be in Benchmark enum order");

const char *const kMethods[] = {"vector", "matrix", "library", "graph", "memory", "latency",
                                "crypto", "string"};
XPUPEAK_VOCAB_LIST(FamilyMethod, kMethods);

// ── Hardware, system, backends ─────────────────────────────────────────────

const char *const kProcessorKinds[] = {"cpu", "gpu", "npu", "accelerator"};
const char *const kPackageKinds[]   = {"soc", "cpu", "discrete"};
const char *const kLinks[]          = {"uuid", "luid", "pci_bus", "registry", "host", "soc",
                                       "sole", "own"};
const char *const kClusterKinds[]   = {"prime", "performance", "efficiency", "low_power"};
const char *const kCacheLevels[]    = {"l1d", "l1i", "l2", "l3", "slc"};
const char *const kEngineKinds[]    = {"matrix"};
const char *const kEngineScopes[]   = {"core", "cluster"};
const char *const kEngineReaches[]  = {"instructions", "library"};
const char *const kVendors[]        = {"amd", "apple", "arm", "google", "imagination", "intel",
                                       "mediatek", "nvidia", "qualcomm", "samsung", "microsoft",
                                       "broadcom", "huawei", "mesa", "pocl", "ampere", "hygon",
                                       "zhaoxin", "fujitsu", "marvell"};
const char *const kFormFactors[]    = {"mobile", "laptop", "desktop", "server"};
const char *const kFormDetails[]    = {"phone", "tablet", "embedded"};
const char *const kFormEvidence[]   = {"model", "os", "characteristics", "battery",
                                       "platform_role", "chassis", "device_tree", "default"};
const char *const kPowerSources[]   = {"ac", "battery"};
const char *const kBackendKinds[]   = {"gpu", "cpu", "ai"};
XPUPEAK_VOCAB_LIST(ProcessorKind, kProcessorKinds);
XPUPEAK_VOCAB_LIST(PackageKind, kPackageKinds);
XPUPEAK_VOCAB_LIST(LinkEvidence, kLinks);
XPUPEAK_VOCAB_LIST(ClusterKind, kClusterKinds);
XPUPEAK_VOCAB_LIST(CacheLevel, kCacheLevels);
XPUPEAK_VOCAB_LIST(EngineKind, kEngineKinds);
XPUPEAK_VOCAB_LIST(EngineScope, kEngineScopes);
XPUPEAK_VOCAB_LIST(EngineReach, kEngineReaches);
XPUPEAK_VOCAB_LIST(Vendor, kVendors);
XPUPEAK_VOCAB_LIST(FormFactor, kFormFactors);
XPUPEAK_VOCAB_LIST(FormFactorDetail, kFormDetails);
XPUPEAK_VOCAB_LIST(FormFactorEvidence, kFormEvidence);
XPUPEAK_VOCAB_LIST(PowerSource, kPowerSources);
XPUPEAK_VOCAB_LIST(BackendKind, kBackendKinds);

// PCI vendor ids, and the ids Khronos assigns (VkVendorId, above 0xFFFF) to
// vendors that have none.  Several ids can name one vendor.
struct PciVendor {
    std::uint32_t id;
    Vendor        vendor;
};
const PciVendor kPciVendors[] = {
    {0x1002, Vendor::Amd},          // ATI / AMD graphics
    {0x1022, Vendor::Amd},          // AMD CPUs and chipsets
    {0x106b, Vendor::Apple},
    {0x13b5, Vendor::Arm},          // Mali
    {0x1ae0, Vendor::Google},
    {0x1010, Vendor::Imagination},  // PowerVR
    {0x8086, Vendor::Intel},
    {0x8087, Vendor::Intel},
    {0x14c3, Vendor::Mediatek},
    {0x10de, Vendor::Nvidia},
    {0x5143, Vendor::Qualcomm},     // Adreno, as Vulkan and OpenCL report it
    {0x17cb, Vendor::Qualcomm},
    {0x144d, Vendor::Samsung},      // Xclipse
    {0x1414, Vendor::Microsoft},    // WARP, the Basic Render Driver
    {0x14e4, Vendor::Broadcom},     // VideoCore
    {0x19e5, Vendor::Huawei},
    {0x10005, Vendor::Mesa},        // VK_VENDOR_ID_MESA: llvmpipe, lavapipe
    {0x10006, Vendor::Pocl},        // VK_VENDOR_ID_POCL
};

struct CpuidVendor {
    const char *id;
    Vendor      vendor;
};
const CpuidVendor kCpuidVendors[] = {
    {"GenuineIntel", Vendor::Intel},
    {"AuthenticAMD", Vendor::Amd},
    {"HygonGenuine", Vendor::Hygon},
    {"  Shanghai  ", Vendor::Zhaoxin},
    {"CentaurHauls", Vendor::Zhaoxin},
};

struct MidrVendor {
    unsigned implementer;
    Vendor   vendor;
};
const MidrVendor kMidrVendors[] = {
    {0x41, Vendor::Arm},
    {0x42, Vendor::Broadcom},
    {0x43, Vendor::Marvell},        // Cavium
    {0x46, Vendor::Fujitsu},
    {0x48, Vendor::Huawei},         // HiSilicon
    {0x4e, Vendor::Nvidia},
    {0x51, Vendor::Qualcomm},
    {0x53, Vendor::Samsung},
    {0x56, Vendor::Marvell},
    {0x61, Vendor::Apple},
    {0x69, Vendor::Intel},
    {0x6d, Vendor::Microsoft},
    {0xc0, Vendor::Ampere},
};

} // namespace

// ── Data types ─────────────────────────────────────────────────────────────

const char *toString(DType t) { return nameIn(kDTypes, t); }
bool fromString(const std::string &s, DType &out) { return parseIn(kDTypes, s, out); }

bool isInteger(DType t)
{
    switch (t)
    {
    case DType::Int64: case DType::Int32: case DType::Int24: case DType::Int16:
    case DType::Int8:  case DType::Int4:  case DType::Uint64: case DType::Uint32:
    case DType::Uint16: case DType::Uint8: case DType::Uint4:
        return true;
    default:
        return false;
    }
}

bool isFloat(DType t) { return t != DType::COUNT && !isInteger(t); }

DType usualAccumulator(DType t)
{
    if (t == DType::Fp64) return DType::Fp64;
    if (t == DType::Int64 || t == DType::Uint64) return DType::Int64;
    return isInteger(t) ? DType::Int32 : DType::Fp32;
}

// ── Dims ───────────────────────────────────────────────────────────────────

const DimKeyInfo &dimInfo(Dim d)
{
    const std::size_t i = static_cast<std::size_t>(d);
    return kDims[i < static_cast<std::size_t>(Dim::COUNT) ? i : 0];
}

const char *toString(Dim d)
{
    return d < Dim::COUNT ? dimInfo(d).key : "";
}

bool fromString(const std::string &s, Dim &out)
{
    for (std::size_t i = 0; i < static_cast<std::size_t>(Dim::COUNT); i++)
        if (s == kDims[i].key)
        {
            out = static_cast<Dim>(i);
            return true;
        }
    return false;
}

const char *dimWord(Dim key, unsigned index)
{
    const WordList w = wordsOf(key);
    return index < w.count ? w.names[index] : nullptr;
}

unsigned dimWordCount(Dim key) { return wordsOf(key).count; }

std::string dimDefaultFor(const std::string &key, const std::string &family,
                          const std::string &dtype)
{
    Dim d;
    if (!fromString(key, d)) return "";
    const DimKeyInfo &info = dimInfo(d);

    Benchmark b = Benchmark::COUNT;
    const bool knownFamily = familyFromString(family, b);
    const FamilyInfo *fam = knownFamily ? &familyInfo(b) : nullptr;

    // The reading's data type, or else its family's.
    std::string type = dtype;
    if (type.empty() && fam && fam->hasDtype) type = toString(fam->dtype);

    switch (info.def)
    {
    case DimDefault::None:
        return "";
    case DimDefault::Value:
        return info.defaultValue;
    case DimDefault::FamilyDtype:
        return (fam && fam->hasDtype) ? toString(fam->dtype) : "";
    case DimDefault::SameAsDtype:
        return type;
    case DimDefault::FamilyAcc:
    {
        if (fam && fam->acc == AccRule::Fixed) return toString(fam->accType);
        DType t;
        if (!fromString(type, t)) return "";
        if (fam && fam->acc == AccRule::SameAsDtype) return toString(t);
        return toString(usualAccumulator(t));
    }
    }
    return "";
}

// ── Families ───────────────────────────────────────────────────────────────

const char *toString(FamilyMethod m) { return nameIn(kMethods, m); }
bool fromString(const std::string &s, FamilyMethod &out) { return parseIn(kMethods, s, out); }

const FamilyInfo &familyInfo(Benchmark b)
{
    const std::size_t i = static_cast<std::size_t>(b);
    return kFamilies[i < static_cast<std::size_t>(Benchmark::COUNT) ? i : 0];
}

bool familyFromString(const std::string &flag, Benchmark &out)
{
    return benchmarkFromFlag(flag, out);
}

// ── Hardware, system, backends ─────────────────────────────────────────────

const char *toString(ProcessorKind k) { return nameIn(kProcessorKinds, k); }
bool fromString(const std::string &s, ProcessorKind &out) { return parseIn(kProcessorKinds, s, out); }
const char *toString(PackageKind k) { return nameIn(kPackageKinds, k); }
bool fromString(const std::string &s, PackageKind &out) { return parseIn(kPackageKinds, s, out); }
const char *toString(LinkEvidence e) { return nameIn(kLinks, e); }
bool fromString(const std::string &s, LinkEvidence &out) { return parseIn(kLinks, s, out); }
const char *toString(ClusterKind k) { return nameIn(kClusterKinds, k); }
bool fromString(const std::string &s, ClusterKind &out) { return parseIn(kClusterKinds, s, out); }
const char *toString(CacheLevel l) { return nameIn(kCacheLevels, l); }
bool fromString(const std::string &s, CacheLevel &out) { return parseIn(kCacheLevels, s, out); }
const char *toString(EngineKind k) { return nameIn(kEngineKinds, k); }
const char *toString(EngineScope s) { return nameIn(kEngineScopes, s); }
const char *toString(EngineReach r) { return nameIn(kEngineReaches, r); }
bool fromString(const std::string &s, EngineKind &out) { return parseIn(kEngineKinds, s, out); }
bool fromString(const std::string &s, EngineScope &out) { return parseIn(kEngineScopes, s, out); }
bool fromString(const std::string &s, EngineReach &out) { return parseIn(kEngineReaches, s, out); }
const char *toString(Vendor v) { return nameIn(kVendors, v); }
bool fromString(const std::string &s, Vendor &out) { return parseIn(kVendors, s, out); }
const char *toString(FormFactor f) { return nameIn(kFormFactors, f); }
bool fromString(const std::string &s, FormFactor &out) { return parseIn(kFormFactors, s, out); }
const char *toString(FormFactorDetail d) { return nameIn(kFormDetails, d); }
bool fromString(const std::string &s, FormFactorDetail &out) { return parseIn(kFormDetails, s, out); }
const char *toString(FormFactorEvidence e) { return nameIn(kFormEvidence, e); }
bool fromString(const std::string &s, FormFactorEvidence &out) { return parseIn(kFormEvidence, s, out); }
const char *toString(PowerSource p) { return nameIn(kPowerSources, p); }
bool fromString(const std::string &s, PowerSource &out) { return parseIn(kPowerSources, s, out); }
const char *toString(BackendKind k) { return nameIn(kBackendKinds, k); }
bool fromString(const std::string &s, BackendKind &out) { return parseIn(kBackendKinds, s, out); }

bool vendorFromPciId(std::uint32_t id, Vendor &out)
{
    for (const PciVendor &v : kPciVendors)
        if (v.id == id)
        {
            out = v.vendor;
            return true;
        }
    return false;
}

bool vendorFromCpuid(const std::string &vendorString, Vendor &out)
{
    for (const CpuidVendor &v : kCpuidVendors)
        if (vendorString == v.id)
        {
            out = v.vendor;
            return true;
        }
    return false;
}

bool vendorFromMidrImplementer(unsigned implementer, Vendor &out)
{
    for (const MidrVendor &v : kMidrVendors)
        if (v.implementer == implementer)
        {
            out = v.vendor;
            return true;
        }
    return false;
}

BackendKind backendKind(Backend b)
{
    switch (b)
    {
    case Backend::Cpu:    return BackendKind::Cpu;
    case Backend::Coreml:
    case Backend::Litert:
    case Backend::Onnx:   return BackendKind::Ai;
    default:              return BackendKind::Gpu;
    }
}

bool backendPortable(Backend b)
{
    return b == Backend::Vulkan || b == Backend::OpenCL || b == Backend::Onnx;
}

} // namespace xpupeak
