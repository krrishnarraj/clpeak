#ifndef XPUPEAK_VOCABULARY_H
#define XPUPEAK_VOCABULARY_H

#include <cstdint>
#include <string>
#include <type_traits>

#include <common/benchmark_enums.h>

// ── The vocabulary ─────────────────────────────────────────────────────────
//
// The words a result document uses for what was measured and what it ran
// on: data types, the keys of a reading's `dims` and their values, kinds of
// processor and package, vendors, form factors, how a device was linked to
// its processor, and the families tests belong to.  docs/format-v4.md lists
// every one; the Dart side mirrors them (app/lib/src/model/).
//
// The file carries the strings.  The enums are for the C++ that writes them,
// so a producer cannot misspell one, and for the C++ that reasons about them.
// A reader meets strings it does not know -- a newer build's -- and keeps
// them as written: nothing here is ever the only way to read a file.
//
// Every list is append-only (docs/format-v4.md, Compatibility).  A value's
// string never changes and is never reused, and a dim key's class and
// default never change.  A missing value goes at the end of its list, in a
// commit of its own, so tasks working in parallel merge cleanly.  Only the
// strings reach a file, so an enum's numeric values are free to follow the
// list's order.
//
// Everything is in namespace xpupeak: the ONNX and Core ML convolution tests
// have a `DType` of their own.  A file that authors dims says
// `using xpupeak::Dim; using xpupeak::DType; namespace dim = xpupeak::dim;`
// and then writes the compact form, `{{Dim::Dtype, DType::Fp16},
// {Dim::Acc, DType::Fp16}}` (include/common/AGENTS.md).

namespace xpupeak {

// ── Data types ─────────────────────────────────────────────────────────────
//
// Element types, block-scaled formats as values of their own (an MX or NV
// block carries its scales, so `mxfp4` is not `fp4_e2m1` with a dim), and
// integer types.  A mixed format is described with dims instead -- `weights`,
// `acc`, `kv`, `quant`, `sparsity` -- never with a value of its own.
enum class DType : unsigned char {
    // Float
    Fp64, Fp32, Tf32, Fp16, Bf16,
    Fp8E4m3, Fp8E5m2, Fp8E4m3Fnuz, Fp8E5m2Fnuz,
    Fp6E3m2, Fp6E2m3, Fp4E2m1,
    // Block-scaled
    Mxfp8E4m3, Mxfp8E5m2, Mxfp6E3m2, Mxfp6E2m3, Mxfp4, Nvfp4,
    // Integer
    Int64, Int32, Int24, Int16, Int8, Int4,
    Uint64, Uint16, Uint8, Uint4,
    // Appended after the seed list
    Uint32,             // oneAPI joint_matrix and Vulkan cooperative-matrix component types
    COUNT
};

const char *toString(DType t);
bool        fromString(const std::string &s, DType &out);
bool        isFloat(DType t);         // element and block-scaled floats
bool        isInteger(DType t);

// The accumulator a matrix, dot-product or library operation on `t` uses
// unless it says otherwise: fp64 for fp64, fp32 for every other float,
// int64 for the 64-bit integers, int32 for the rest.
DType usualAccumulator(DType t);

// ── Dims ───────────────────────────────────────────────────────────────────
//
// A reading's coordinates.  Every key has one fixed class:
//
//   Identity -- must match for two readings to be comparable: a data type,
//               a cache level, a direction.
//   Variant  -- an implementation choice, over which the best reading is
//               taken: a vector width, an ISA, a library, a race's winner.
//
// Where one word means both, it is two keys: a GEMM's `problem_size` is
// variant, a bandwidth test's `working_set` is identity.  An identity key's
// default is what an absent key means; a key with none is not applicable
// where absent, and keyed wherever present.  A reader that meets a key it
// does not know treats it as identity, so it never merges what it does not
// understand.
enum class Dim : unsigned char {
    Dtype, Weights, Acc, Quant, Sparsity, Scope, ThreadsPerCore, Level,
    Pattern, InFlight, Op, Direction, Memory, WorkingSet, Layer, PixelFormat,
    Phase, Context, Contention, Workload,
    Width, Chains, Isa, SimdBits, Impl, Form, Tile, ProblemSize,
    // Appended after the seed list
    Kv,                 // a transformer block's KV-cache data type (`int8_kv`)
    COUNT
};

enum class DimClass : unsigned char { Identity, Variant };

// What a value is.  Values are strings or whole numbers, and stay so.
enum class DimKind : unsigned char {
    DataType,           // a DType string
    Integer,            // a whole number: a count, bytes, tokens, bits
    Word,               // one of the key's own list (dimWord)
    Text,               // free text: a race's winner, a layout, a tile shape
};

// Where an identity key's default comes from.
enum class DimDefault : unsigned char {
    None,               // no default: absent means not applicable
    Value,              // a fixed value (DimKeyInfo::defaultValue)
    FamilyDtype,        // the family's data type (FamilyInfo::dtype)
    FamilyAcc,          // the family's accumulator rule (FamilyInfo::acc)
    SameAsDtype,        // the reading's data type
};

struct DimKeyInfo {
    const char *key;            // "dtype", "working_set"
    DimClass    cls;
    DimKind     kind;
    DimDefault  def;
    const char *defaultValue;   // DimDefault::Value only, else ""
    const char *meaning;        // one line, as docs/format-v4.md says it
};

const DimKeyInfo &dimInfo(Dim d);
const char       *toString(Dim d);
bool              fromString(const std::string &s, Dim &out);

// A Word key's values, in list order; nullptr past the end or for a key
// that is not a Word key.
const char *dimWord(Dim key, unsigned index);
unsigned    dimWordCount(Dim key);

// The default of `key` for a reading of `family` (a Benchmark's flag name)
// whose data type is `dtype` (its dims' or else its family's): the string an
// absent key means, or "" for a key with none -- and for a key this build
// does not know, which a reader treats as identity.  `family` and `dtype`
// may be ones this build does not know either.
std::string dimDefaultFor(const std::string &key, const std::string &family,
                          const std::string &dtype);

// The values of the Word keys.  One enum per key, so a list grows without
// touching another's, and DimEntry checks the pairing (logger.h).
namespace dim {

enum class Quant : unsigned char { None, Qdq, Lut, Block, Channel, COUNT };
enum class Sparsity : unsigned char { Dense, S2_4 /* "2:4" */, COUNT };
enum class Scope : unsigned char { Core, Device, COUNT };
enum class Level : unsigned char { L1, L2, L3, Slc, Dram, COUNT };
enum class Pattern : unsigned char { Random, Linear, PageStride, Predictable, COUNT };
enum class Op : unsigned char {
    Read, Write, Copy, Triad, Fdiv, Fsqrt, Silu, Softmax, Layernorm,
    Dispatch, Roundtrip, Mispredict, COUNT
};
enum class Direction : unsigned char { H2d, D2h, Roundtrip, COUNT };
enum class Memory : unsigned char { Pinned, Pageable, COUNT };
enum class Layer : unsigned char { Conv3x3s2, Conv1x1, Depthwise3x3, COUNT };
enum class PixelFormat : unsigned char { Rgba32f, Rgba16f, Rgba8, R32f, COUNT };
enum class Phase : unsigned char { Prefill, Decode, COUNT };
enum class Contention : unsigned char { None, All, COUNT };
enum class Workload : unsigned char { TrivialOp, SmallMatmul, SessionCreate, ModelCreate, COUNT };
enum class Isa : unsigned char {
    Scalar, Sse2, Ssse3, Sse4_2, Avx, Avx2, Avx512, Avx10, Amx,
    Neon, Sve, Sve2, Sme, Sme2, COUNT
};
enum class Impl : unsigned char {
    Cublaslt, Hipblaslt, Rocblas, Rocwmma, Onemkl, Mps, Mpsgraph, Accelerate, Bnns, COUNT
};

// Which key each value enum belongs to.
template <class E> struct WordOf {};
template <> struct WordOf<Quant>       { static constexpr Dim key = Dim::Quant; };
template <> struct WordOf<Sparsity>    { static constexpr Dim key = Dim::Sparsity; };
template <> struct WordOf<Scope>       { static constexpr Dim key = Dim::Scope; };
template <> struct WordOf<Level>       { static constexpr Dim key = Dim::Level; };
template <> struct WordOf<Pattern>     { static constexpr Dim key = Dim::Pattern; };
template <> struct WordOf<Op>          { static constexpr Dim key = Dim::Op; };
template <> struct WordOf<Direction>   { static constexpr Dim key = Dim::Direction; };
template <> struct WordOf<Memory>      { static constexpr Dim key = Dim::Memory; };
template <> struct WordOf<Layer>       { static constexpr Dim key = Dim::Layer; };
template <> struct WordOf<PixelFormat> { static constexpr Dim key = Dim::PixelFormat; };
template <> struct WordOf<Phase>       { static constexpr Dim key = Dim::Phase; };
template <> struct WordOf<Contention>  { static constexpr Dim key = Dim::Contention; };
template <> struct WordOf<Workload>    { static constexpr Dim key = Dim::Workload; };
template <> struct WordOf<Isa>         { static constexpr Dim key = Dim::Isa; };
template <> struct WordOf<Impl>        { static constexpr Dim key = Dim::Impl; };

template <class E, class = void> struct IsWord : std::false_type {};
template <class E> struct IsWord<E, decltype(void(WordOf<E>::key))> : std::true_type {};

template <class E, std::enable_if_t<IsWord<E>::value, int> = 0>
const char *toString(E v) { return dimWord(WordOf<E>::key, static_cast<unsigned>(v)); }

} // namespace dim

// ── Families ───────────────────────────────────────────────────────────────
//
// A family is a Benchmark, named in a file by its flag (benchmarkFlag(),
// options.h): what is measured, never which backend measures it.  Each has
// its category (categoryOf()), its method, and for a family of one data type
// that type, which its readings' `dtype` defaults to.  `acc` is how its
// accumulator defaults: a vector family computes in its own type, so an fp16
// FMA accumulates in fp16; a matrix, dot-product, library or graph family
// accumulates as usualAccumulator() says.

enum class FamilyMethod : unsigned char {
    Vector, Matrix, Library, Graph, Memory, Latency, Crypto, String, COUNT
};
const char *toString(FamilyMethod m);
bool        fromString(const std::string &s, FamilyMethod &out);

enum class AccRule : unsigned char {
    Usual,              // usualAccumulator(dtype)
    SameAsDtype,        // the operation's own type
    Fixed,              // FamilyInfo::accType
};

struct FamilyInfo {
    Benchmark    benchmark;
    FamilyMethod method;
    bool         hasDtype;      // a family of one data type
    DType        dtype;         // that type, when hasDtype
    AccRule      acc;
    DType        accType;       // AccRule::Fixed only
};

const FamilyInfo &familyInfo(Benchmark b);
// A family by its flag name; false for one this build does not know.
bool familyFromString(const std::string &flag, Benchmark &out);

// ── Hardware ───────────────────────────────────────────────────────────────

// A processor's kind -- and a device's `type`, which uses the same strings
// (DeviceType, benchmark_enums.h): an ML runtime's NPU is `npu`, and
// `accelerator` is for the rest.
enum class ProcessorKind : unsigned char { Cpu, Gpu, Npu, Accelerator, COUNT };
const char *toString(ProcessorKind k);
bool        fromString(const std::string &s, ProcessorKind &out);

// A package: one chip or card.  An SoC holds the host CPU and whatever
// shares its die or package (an integrated GPU, an on-die NPU); a discrete
// GPU is a package of its own; `cpu` is a host CPU package with nothing else
// joined to it.
enum class PackageKind : unsigned char { Soc, Cpu, Discrete, COUNT };
const char *toString(PackageKind k);
bool        fromString(const std::string &s, PackageKind &out);

// What linked a device to its processor (docs/format-v4.md, Linking): an
// equal identifier (the strong four), executing on the host CPU, being the
// SoC's NPU or GPU, being the only candidate of its kind and vendor -- or
// nothing, which leaves the device a processor of its own.
enum class LinkEvidence : unsigned char {
    Uuid, Luid, PciBus, Registry, Host, Soc, Sole, Own, COUNT
};
const char *toString(LinkEvidence e);
bool        fromString(const std::string &s, LinkEvidence &out);

// A CPU's cluster: Cortex-X / big / LITTLE, Apple's P and E, Intel's P, E
// and low-power E cores.  A CPU of one kind of core may leave it absent.
enum class ClusterKind : unsigned char { Prime, Performance, Efficiency, LowPower, COUNT };
const char *toString(ClusterKind k);
bool        fromString(const std::string &s, ClusterKind &out);

// A cache in a processor's facts.  Not the `level` dim, whose values name
// where a reading's data lived (l1 there is l1d here).
enum class CacheLevel : unsigned char { L1d, L1i, L2, L3, Slc, COUNT };
const char *toString(CacheLevel l);
bool        fromString(const std::string &s, CacheLevel &out);

// A CPU's matrix engine: Intel's AMX tiles in every core, Apple's AMX and
// SME units one per cluster.  A per-cluster engine's all-core number does
// not scale with cores.
enum class EngineKind : unsigned char { Matrix, COUNT };
enum class EngineScope : unsigned char { Core, Cluster, COUNT };
enum class EngineReach : unsigned char { Instructions, Library, COUNT };
const char *toString(EngineKind k);
const char *toString(EngineScope s);
const char *toString(EngineReach r);
bool        fromString(const std::string &s, EngineKind &out);
bool        fromString(const std::string &s, EngineScope &out);
bool        fromString(const std::string &s, EngineReach &out);

// Vendors, of processors and of the APIs' devices.
enum class Vendor : unsigned char {
    Amd, Apple, Arm, Google, Imagination, Intel, Mediatek, Nvidia, Qualcomm,
    Samsung, Microsoft, Broadcom, Huawei, Mesa, Pocl, Ampere, Hygon, Zhaoxin,
    Fujitsu, Marvell, COUNT
};
const char *toString(Vendor v);
bool        fromString(const std::string &s, Vendor &out);

// The vendor an API's vendor id names: a PCI vendor id, or one of the ids
// Khronos assigns above 0xFFFF to vendors with none (Mesa, PoCL).  False for
// an id not in the table.
bool vendorFromPciId(std::uint32_t id, Vendor &out);
// A CPU's: the x86 CPUID vendor string ("GenuineIntel"), or an ARM MIDR's
// implementer byte (0x61 is Apple).
bool vendorFromCpuid(const std::string &vendorString, Vendor &out);
bool vendorFromMidrImplementer(unsigned implementer, Vendor &out);

// ── System ─────────────────────────────────────────────────────────────────

enum class FormFactor : unsigned char { Mobile, Laptop, Desktop, Server, COUNT };
const char *toString(FormFactor f);
bool        fromString(const std::string &s, FormFactor &out);

// The finer class under a form factor, when known: phone or tablet under
// mobile, tablet under laptop (a Windows slate), embedded under desktop (a
// Raspberry Pi, a Jetson).
enum class FormFactorDetail : unsigned char { Phone, Tablet, Embedded, COUNT };
const char *toString(FormFactorDetail d);
bool        fromString(const std::string &s, FormFactorDetail &out);

// What the form factor was read from, in the order docs/format-v4.md's
// System section gives per OS; `default` is a desktop nothing else spoke for.
enum class FormFactorEvidence : unsigned char {
    Model, Os, Characteristics, Battery, PlatformRole, Chassis, DeviceTree, Default, COUNT
};
const char *toString(FormFactorEvidence e);
bool        fromString(const std::string &s, FormFactorEvidence &out);

enum class PowerSource : unsigned char { Ac, Battery, COUNT };
const char *toString(PowerSource p);
bool        fromString(const std::string &s, PowerSource &out);

// ── Backends ───────────────────────────────────────────────────────────────
//
// What a backend reaches: `gpu` for the GPU compute APIs (a gpu backend may
// still reach a CPU through a CPU driver -- the device's `type` says what it
// ran on), `cpu` for the native CPU backend, `ai` for the ML runtimes.
// `portable` marks the layers that run on more than one vendor's silicon
// (Vulkan, OpenCL, ONNX Runtime), as against a vendor's own API.
enum class BackendKind : unsigned char { Gpu, Cpu, Ai, COUNT };
const char *toString(BackendKind k);
bool        fromString(const std::string &s, BackendKind &out);

BackendKind backendKind(Backend b);
bool        backendPortable(Backend b);

} // namespace xpupeak

#endif  // XPUPEAK_VOCABULARY_H
