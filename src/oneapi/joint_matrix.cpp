#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#ifdef CLPEAK_ONEAPI_HAS_JOINT_MATRIX
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
#include <sycl/ext/oneapi/bfloat16.hpp>
#define CLPEAK_ONEAPI_JM_HAS_BF16 1
#endif
#endif

// XMX matrix engine peak — Intel's analog of rocWMMA / cuda WMMA / Vulkan
// coopMatrix / Metal simdgroup_matrix.
//
// The row list is not hardcoded: every combination the device advertises in
// its `matrix_combinations` table becomes a row.  Each entry names the A, B
// and accumulator element types plus a tile shape, where a dimension the
// hardware leaves free is reported as 0 with a `max_*` bound (Xe reports M
// that way) and a fixed dimension carries its single legal size.  Alchemist
// (A380/A770) advertises seven: four 8-bit sign combinations at 8x8x32, fp16
// and bf16 at 8x8x16, and a second, large bf16 tile at 32x32x16.  PVC / Xe2 /
// Xe3 report N=16 and add tf32.
//
// joint_matrix needs M, N and K at compile time, so the shapes below are the
// compiled set; a device advertising anything else records a row naming the
// shape rather than dropping it silently, which is how we learn what to add.
//
// Ops accounting is per sub-group: each sub-group runs its own accumulator
// chain, and how many sub-groups the compiler packs into a work-group is its
// choice — see the note on JM_SG_COUNT_NOTE below.

#ifdef CLPEAK_ONEAPI_HAS_JOINT_MATRIX

namespace {

namespace syclex = sycl::ext::oneapi::experimental::matrix;

// MMA iterations per sub-group.  Fixed rather than scaled per tile: the block
// count is derived from a fixed output-byte budget (JM_OUT_BYTES), so blocks
// fall as the tile grows and the work per launch comes out the same for every
// row of a given K.  That keeps launch duration bounded on the large 32x32
// tile -- a fixed block count there would make the launch 16x longer, into
// Windows TDR range on a slow part -- and keeps the end-of-chain store the
// same small fraction of every row, so the rows compare directly.
constexpr uint32_t JM_ITERS = 256;

// Output budget for the test, clamped by device memory at the call site.
// Sized so the 8x8 tile keeps exactly the block count this benchmark has
// always used, which puts the single store at the end of the chain at ~1% of
// a launch.
constexpr uint64_t JM_OUT_BYTES = 512ull << 20;

// Per-instantiation kernel-name tag: SYCL needs a distinct type per
// parallel_for, and the template arguments already are the distinction.
template <typename At, typename Bt, typename Acc, int M, int N, int K> struct JmTag;

// Returned when the device advertises a tile whose shape has no compiled
// instantiation.  Distinct from -1 (launch failed) so the row can say which
// of the two actually happened.
constexpr float JM_NOT_COMPILED = -2.0f;

// Long spelling, for the verbose combination dump.
static const char *mtName(syclex::matrix_type t)
{
  using mt = syclex::matrix_type;
  switch (t)
  {
    case mt::bf16:   return "bf16";
    case mt::fp16:   return "fp16";
    case mt::tf32:   return "tf32";
    case mt::fp32:   return "fp32";
    case mt::fp64:   return "fp64";
    case mt::sint8:  return "sint8";
    case mt::sint16: return "sint16";
    case mt::sint32: return "sint32";
    case mt::sint64: return "sint64";
    case mt::uint8:  return "uint8";
    case mt::uint16: return "uint16";
    case mt::uint32: return "uint32";
    case mt::uint64: return "uint64";
    default:         return "?";
  }
}

// Compact spelling, for row names and tile labels ("s8xu8+s32").
static const char *mtShort(syclex::matrix_type t)
{
  using mt = syclex::matrix_type;
  switch (t)
  {
    case mt::bf16:   return "bf16";
    case mt::fp16:   return "fp16";
    case mt::tf32:   return "tf32";
    case mt::fp32:   return "fp32";
    case mt::fp64:   return "fp64";
    case mt::sint8:  return "s8";
    case mt::sint16: return "s16";
    case mt::sint32: return "s32";
    case mt::sint64: return "s64";
    case mt::uint8:  return "u8";
    case mt::uint16: return "u16";
    case mt::uint32: return "u32";
    case mt::uint64: return "u64";
    default:         return "x";
  }
}

// Pull the device's joint_matrix combination table.  Sets `threw` so callers
// can distinguish "queried OK, none" from "couldn't query at all".
static std::vector<syclex::combination> queryCombos(const sycl::device &d, bool &threw)
{
  threw = false;
  try {
    return d.get_info<
      sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
  } catch (const std::exception &e) {
    CLPEAK_VLOG("joint_matrix: matrix_combinations query threw: %s\n", e.what());
    threw = true;
    return {};
  }
}

// Verbose-only: print every (a/b/c/d type, M/N/K, max M/N/K) the device accepts.
// This is the ground truth the row list below is derived from.
static void dumpMatrixCombinations(const std::vector<syclex::combination> &combos)
{
  if (!::clpeak::verboseEnabled()) return;
  CLPEAK_VLOG("joint_matrix: device reports %zu matrix combination(s):\n",
              combos.size());
  for (const auto &c : combos)
    CLPEAK_VLOG("  a=%-5s b=%-5s c=%-5s d=%-5s  M=%zu N=%zu K=%zu  "
                "(max M=%zu N=%zu K=%zu)\n",
                mtName(c.atype), mtName(c.btype), mtName(c.ctype), mtName(c.dtype),
                c.msize, c.nsize, c.ksize,
                c.max_msize, c.max_nsize, c.max_ksize);
}

// ---------------------------------------------------------------------------
// One row: a combination from the device's table resolved to a concrete tile.
// ---------------------------------------------------------------------------

struct JmTile {
  syclex::matrix_type at{}, bt{}, ct{};
  uint32_t    M = 0, N = 0, K = 0;
  bool        advertised = true;  // false => placeholder for a canonical row
                                  // this device does not offer at all
  std::string metric;             // "joint_matrix_bf16", "joint_matrix_bf16_32x32x16"
  std::string shape;              // "8x8x16"
  std::string label;              // "bf16xbf16+fp32"

  uint64_t volume() const { return (uint64_t)M * N * K; }
};

// A free dimension is reported as 0 with a max_* bound; a fixed one carries
// its single legal size.  Take the largest tile the entry permits.
static uint32_t resolveDim(size_t fixed, size_t maxv)
{
  return (uint32_t)(fixed != 0 ? fixed : maxv);
}

// Row-name stem for an (A,B) type pair.  A same-type FP pair gets the bare
// dtype (bf16/fp16/tf32) and signed8 x signed8 keeps the historical "int8",
// so rows that already exist keep their names and saved baselines still line
// up.  Every other pair spells out both operands: "u8u8", "s8u8", "u8s8".
static std::string jmBaseName(syclex::matrix_type at, syclex::matrix_type bt)
{
  using mt = syclex::matrix_type;
  if (at == bt)
  {
    switch (at)
    {
      case mt::bf16: case mt::fp16: case mt::tf32:
      case mt::fp32: case mt::fp64:
        return mtShort(at);
      case mt::sint8:
        return "int8";
      default:
        break;
    }
  }
  return std::string(mtShort(at)) + mtShort(bt);
}

// Plain-language note for one row, with the tile shape appended so a reader
// can tell the two bf16 rows on Alchemist apart without decoding the name.
static std::string jmNote(const JmTile &t)
{
  using mt = syclex::matrix_type;
  std::string s;

  if (t.at == mt::bf16 && t.bt == mt::bf16)
    s = "bfloat16 inputs with 32-bit totals -- 16 bits arranged for AI work, "
        "trading digits of accuracy for the number range of a full float.";
  else if (t.at == mt::fp16 && t.bt == mt::fp16)
    s = "16-bit inputs with 32-bit totals -- the everyday precision of AI "
        "inference.";
  else if (t.at == mt::tf32 && t.bt == mt::tf32)
    s = "tf32, a trimmed-down stand-in for 32-bit float: it keeps the full "
        "number range but drops accuracy to fit the matrix engine.  Not every "
        "Intel part has it.";
  else if (t.at == mt::sint8 && t.bt == mt::sint8)
    s = "8-bit whole numbers with 32-bit totals, the format quantized neural "
        "networks use.";
  else if (t.at == mt::uint8 && t.bt == mt::uint8)
    s = "Unsigned 8-bit whole numbers with 32-bit totals -- the same engine as "
        "the signed row, reading both operands as 0..255.";
  else if ((t.at == mt::uint8 && t.bt == mt::sint8) ||
           (t.at == mt::sint8 && t.bt == mt::uint8))
    s = "Mixed-sign 8-bit whole numbers with 32-bit totals -- one operand "
        "unsigned, the other signed, which is what quantized networks get when "
        "unsigned activations meet signed weights.";
  else
    s = std::string(mtShort(t.at)) + " inputs against " + mtShort(t.bt) +
        ", accumulating in " + mtShort(t.ct) + ".";

  if (!t.shape.empty())
    s += "  Tile " + t.shape + ".";
  return s;
}

// True when this accumulator type makes the combination an integer one.
// The benchmark reports integer rows in TOPS and floating-point rows in TFLOPS
// within the single Compute category (integer rows carry a per-reading unit).
static bool jmIsIntAcc(syclex::matrix_type t)
{
  using mt = syclex::matrix_type;
  switch (t)
  {
    case mt::sint8:  case mt::uint8:
    case mt::sint16: case mt::uint16:
    case mt::sint32: case mt::uint32:
    case mt::sint64: case mt::uint64:
      return true;
    default:
      return false;
  }
}

// Resolve the device's table into the concrete tiles this category will run.
static std::vector<JmTile> resolveTiles(const std::vector<syclex::combination> &combos,
                                        bool wantInt)
{
  std::vector<JmTile> out;
  for (const auto &c : combos)
  {
    // c and d are the accumulator in and out.  Xe always reports them equal
    // and the kernel below feeds one accumulator back into itself, so a
    // combination that separates them is not something this test can express.
    if (c.ctype != c.dtype) continue;
    if (jmIsIntAcc(c.ctype) != wantInt) continue;

    JmTile t;
    t.at = c.atype;
    t.bt = c.btype;
    t.ct = c.ctype;
    t.M  = resolveDim(c.msize, c.max_msize);
    t.N  = resolveDim(c.nsize, c.max_nsize);
    t.K  = resolveDim(c.ksize, c.max_ksize);
    if (t.M == 0 || t.N == 0 || t.K == 0) continue;   // unresolvable entry

    // Some drivers list one shape twice (once fixed, once as a free dimension
    // whose max lands on the same size).  One row per distinct tile.
    const bool dup = std::any_of(out.begin(), out.end(), [&](const JmTile &o) {
      return o.at == t.at && o.bt == t.bt && o.ct == t.ct &&
             o.M == t.M && o.N == t.N && o.K == t.K;
    });
    if (!dup) out.push_back(t);
  }
  return out;
}

// Make sure the historical row set always appears.  An explicit "tf32 not in
// this device's combinations" row is more useful to a reader than a silently
// missing one, and it keeps the row set stable across devices so baselines
// still compare.
static void addMissingCanonical(std::vector<JmTile> &tiles, bool wantInt)
{
  using mt = syclex::matrix_type;
  struct Canon { mt at, bt, ct; };
  static const Canon fpCanon[]  = {{mt::bf16, mt::bf16, mt::fp32},
                                   {mt::fp16, mt::fp16, mt::fp32},
                                   {mt::tf32, mt::tf32, mt::fp32}};
  static const Canon intCanon[] = {{mt::sint8, mt::sint8, mt::sint32}};

  const Canon *list = wantInt ? intCanon : fpCanon;
  const size_t n    = wantInt ? (sizeof(intCanon) / sizeof(intCanon[0]))
                              : (sizeof(fpCanon)  / sizeof(fpCanon[0]));

  for (size_t i = 0; i < n; i++)
  {
    const std::string want = jmBaseName(list[i].at, list[i].bt);
    const bool have = std::any_of(tiles.begin(), tiles.end(), [&](const JmTile &o) {
      return jmBaseName(o.at, o.bt) == want;
    });
    if (have) continue;
    JmTile t;
    t.at = list[i].at;
    t.bt = list[i].bt;
    t.ct = list[i].ct;
    t.advertised = false;
    tiles.push_back(t);
  }
}

// Order the rows and name them.  Grouping by stem and putting the smallest
// tile of a stem first makes the naming rule below independent of the order
// the driver happened to list its combinations in.
static void finalizeTiles(std::vector<JmTile> &tiles)
{
  std::stable_sort(tiles.begin(), tiles.end(), [](const JmTile &a, const JmTile &b) {
    const std::string an = jmBaseName(a.at, a.bt);
    const std::string bn = jmBaseName(b.at, b.bt);
    if (an != bn) return an < bn;
    return a.volume() < b.volume();
  });

  for (size_t i = 0; i < tiles.size(); i++)
  {
    JmTile &t = tiles[i];
    if (t.advertised)
      t.shape = std::to_string(t.M) + "x" + std::to_string(t.N) + "x" + std::to_string(t.K);
    t.label = std::string(mtShort(t.at)) + "x" + mtShort(t.bt) + "+" + mtShort(t.ct);

    // Bare stem for the first tile of each stem, MxNxK suffix for any further
    // tile of the same stem — Alchemist advertises bf16 at both 8x8x16 and
    // 32x32x16, and the two need distinct row names.
    //
    // No "joint_matrix_" prefix: the test is called joint_matrix, so every
    // reading repeating it said nothing.  The prefix went when the fp and int
    // halves became one test.
    const std::string base = jmBaseName(t.at, t.bt);
    const bool dup = (i > 0) && jmBaseName(tiles[i - 1].at, tiles[i - 1].bt) == base;
    t.metric = base + (dup ? " " + t.shape : "");
  }
}

// ---------------------------------------------------------------------------
// The kernel.
// ---------------------------------------------------------------------------

// Run one matrix-engine variant.  At/Bt are the joint_matrix element types
// (bfloat16, sycl::half, precision::tf32, int8_t, uint8_t); FillA/FillB are
// the types used to fill A/B (float for tf32, otherwise == At/Bt); Acc is the
// accumulator/output element type.  M, N and K are the compile-time tile.
template <typename At, typename Bt, typename Acc, int M, int N, int K,
          typename FillA, typename FillB>
static float runJmVariant(OneapiPeak &peak, OneapiDevice &dev,
                          Acc *outBuf, uint32_t *sgCountBuf,
                          uint32_t numBlocks, uint32_t blockSize,
                          FillA aFill, FillB bFill,
                          unsigned int targetTimeUs, unsigned int forced)
{
  const uint64_t totalThreads = (uint64_t)numBlocks * blockSize;

  auto submit = [=](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<JmTag<At, Bt, Acc, M, N, K>>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          auto sg = it.get_sub_group();
          syclex::joint_matrix<sycl::sub_group, At,  syclex::use::a, M, K, syclex::layout::row_major>       a;
          // Intel XMX requires the B operand in VNNI/packed layout; a row_major
          // B is rejected at launch on Xe-HPG (Arc/DG2).
          syclex::joint_matrix<sycl::sub_group, Bt,  syclex::use::b, K, N, syclex::layout::ext_intel_packed> b;
          syclex::joint_matrix<sycl::sub_group, Acc, syclex::use::accumulator, M, N> c;
          syclex::joint_matrix_fill(sg, a, aFill);
          syclex::joint_matrix_fill(sg, b, bFill);
          syclex::joint_matrix_fill(sg, c, (Acc)0);
          #pragma unroll 1
          for (uint32_t i = 0; i < JM_ITERS; i++)
            syclex::joint_matrix_mad(sg, c, a, b, c);

          // JM_SG_COUNT_NOTE: the ops accounting is per sub-group, because
          // every sub-group in the work-group runs the whole chain above.  How
          // many sub-groups a work-group holds is IGC's choice: the work-group
          // is sized to the device's widest sub-group, but reqd_sub_group_size
          // cannot be used to pin it (the attribute triggers an IGC "Divide by
          // zero" internal compiler error on DG2 with any non-default size),
          // and DG2's dpas is a SIMD8 instruction, so a 32-wide work-group can
          // compile to four sub-groups.  Assuming one would under-report by
          // exactly that factor, so report the real count back to the host
          // instead of assuming.
          const uint32_t sgPerWG = (uint32_t)sg.get_group_range()[0];
          if (it.get_global_linear_id() == 0) *sgCountBuf = sgPerWG;

          // One output tile per work-group, written by every sub-group in it.
          // They all ran the same chain from the same fill values, so they
          // store identical bytes and the race is benign -- and sharing the
          // slot keeps both the buffer and the store traffic independent of
          // how many sub-groups the compiler decided to pack in.
          Acc *blockOut = outBuf + (size_t)it.get_group(0) * (size_t)(M * N);
          syclex::joint_matrix_store(sg, c,
            sycl::address_space_cast<sycl::access::address_space::global_space,
                                     sycl::access::decorated::no>(blockOut),
            N, syclex::layout::row_major);
        });
    });
  };
  return peak.runKernel(dev, submit, targetTimeUs, forced);
}

// Tile shapes with a compiled instantiation.  Covers every (M,N) Intel Xe
// advertises today: 8x8 (DG2/Alchemist), 8x16 (PVC/Xe2/Xe3), 16x16, and the
// large 32x32 bf16 tile Alchemist reports alongside its 8x8 one.  A shape
// outside this list records a row that names it rather than vanishing, so the
// next unusual part tells us exactly which instantiation to add.
#define CLPEAK_JM_SHAPES(X)  X(8, 8)  X(8, 16)  X(16, 16)  X(32, 32)

template <typename At, typename Bt, typename Acc, int K, typename FillA, typename FillB>
static float dispatchJmShape(OneapiPeak &peak, OneapiDevice &dev,
                             Acc *out, uint32_t *sgCount,
                             uint32_t numBlocks, uint32_t blockSize,
                             FillA aFill, FillB bFill,
                             unsigned int targetTimeUs, unsigned int forced,
                             uint32_t M, uint32_t N)
{
#define CLPEAK_JM_CASE(MM, NN)                                                  \
  if (M == (MM) && N == (NN))                                                   \
    return runJmVariant<At, Bt, Acc, MM, NN, K, FillA, FillB>(                   \
        peak, dev, out, sgCount, numBlocks, blockSize,                           \
        aFill, bFill, targetTimeUs, forced);
  CLPEAK_JM_SHAPES(CLPEAK_JM_CASE)
#undef CLPEAK_JM_CASE
  return JM_NOT_COMPILED;
}

// Floating-point tiles: fp32 accumulator, three A/B element types.  K is fixed
// by the element width on Xe (32 bytes per row of A), so a table entry that
// asks for a different K has no instantiation and says so.
static float runJmFp(OneapiPeak &peak, OneapiDevice &dev,
                     float *out, uint32_t *sgCount,
                     uint32_t numBlocks, uint32_t blockSize,
                     unsigned int targetTimeUs, unsigned int forced, const JmTile &t)
{
  using mt = syclex::matrix_type;
  if (t.ct != mt::fp32) return JM_NOT_COMPILED;
#ifdef CLPEAK_ONEAPI_JM_HAS_BF16
  using bfloat16 = sycl::ext::oneapi::bfloat16;
  if (t.at == mt::bf16 && t.bt == mt::bf16 && t.K == 16)
    return dispatchJmShape<bfloat16, bfloat16, float, 16>(
        peak, dev, out, sgCount, numBlocks, blockSize,
        (bfloat16)1.f, (bfloat16)1.f, targetTimeUs, forced, t.M, t.N);
#endif
  if (t.at == mt::fp16 && t.bt == mt::fp16 && t.K == 16)
    return dispatchJmShape<sycl::half, sycl::half, float, 16>(
        peak, dev, out, sgCount, numBlocks, blockSize,
        (sycl::half)1.f, (sycl::half)1.f, targetTimeUs, forced, t.M, t.N);

  // tf32: the matrix element type is precision::tf32, filled with a float.
  if (t.at == mt::tf32 && t.bt == mt::tf32 && t.K == 8)
    return dispatchJmShape<syclex::precision::tf32, syclex::precision::tf32, float, 8>(
        peak, dev, out, sgCount, numBlocks, blockSize,
        1.f, 1.f, targetTimeUs, forced, t.M, t.N);

  return JM_NOT_COMPILED;
}

// Integer tiles: int32 accumulator, all four signed/unsigned 8-bit A/B pairs.
static float runJmInt(OneapiPeak &peak, OneapiDevice &dev,
                      int32_t *out, uint32_t *sgCount,
                      uint32_t numBlocks, uint32_t blockSize,
                      unsigned int targetTimeUs, unsigned int forced, const JmTile &t)
{
  using mt = syclex::matrix_type;
  if (t.ct != mt::sint32 || t.K != 32) return JM_NOT_COMPILED;

  if (t.at == mt::sint8 && t.bt == mt::sint8)
    return dispatchJmShape<int8_t, int8_t, int32_t, 32>(
        peak, dev, out, sgCount, numBlocks, blockSize,
        (int8_t)1, (int8_t)1, targetTimeUs, forced, t.M, t.N);
  if (t.at == mt::uint8 && t.bt == mt::uint8)
    return dispatchJmShape<uint8_t, uint8_t, int32_t, 32>(
        peak, dev, out, sgCount, numBlocks, blockSize,
        (uint8_t)1, (uint8_t)1, targetTimeUs, forced, t.M, t.N);
  if (t.at == mt::uint8 && t.bt == mt::sint8)
    return dispatchJmShape<uint8_t, int8_t, int32_t, 32>(
        peak, dev, out, sgCount, numBlocks, blockSize,
        (uint8_t)1, (int8_t)1, targetTimeUs, forced, t.M, t.N);
  if (t.at == mt::sint8 && t.bt == mt::uint8)
    return dispatchJmShape<int8_t, uint8_t, int32_t, 32>(
        peak, dev, out, sgCount, numBlocks, blockSize,
        (int8_t)1, (uint8_t)1, targetTimeUs, forced, t.M, t.N);

  return JM_NOT_COMPILED;
}

} // namespace

#endif // CLPEAK_ONEAPI_HAS_JOINT_MATRIX

int OneapiPeak::runJointMatrix(OneapiDevice &dev, benchmark_config_t &cfg)
{
  // One test for all data types -- integer readings carry their own unit.
  auto test = currentDeviceScope->beginTest(
    {"joint_matrix", "joint_matrix peak",
     "flops", Category::Unknown,
     "Peak speed of Intel's XMX matrix engine -- dedicated units that multiply "
     "whole blocks of numbers in one step rather than one value at a time.  "
     "One reading per input type, sign combination and tile shape the device "
     "advertises, so the set of readings is itself a description of the "
     "hardware.",
     TestShape::Heterogeneous, "data type"});

  auto jmOpts = [&](const JmTile &t) {
    logger::EmitOptions o;
    o.description = jmNote(t);
    if (jmIsIntAcc(t.ct)) o.unit = "ops";
    return o;
  };

  // Note for the int8 row, used by the paths that skip before any enumeration
  // has happened.  The FP equivalents go through skipAll, which carries no
  // per-reading notes, so they have nothing to declare here.
  const char *int8Note = "8-bit whole numbers with 32-bit totals, the format "
                         "quantized neural networks use.";

  auto skipEverything = [&](ResultStatus status, const char *reason) {
    test.skip("bf16", status, reason);
    test.skip("fp16", status, reason);
    test.skip("tf32", status, reason);
    logger::EmitOptions o; o.description = int8Note; o.unit = "ops";
    test.skip("int8", status, reason, o);
  };

#ifndef CLPEAK_ONEAPI_HAS_JOINT_MATRIX
  skipEverything(ResultStatus::Unsupported,
                 "joint_matrix header not available in this oneAPI toolchain");
  return 0;
#else
  // Query the device's matrix-combination table ONCE: it is both the verbose
  // dump and the source of the row list below.
  bool combosThrew = false;
  const auto combos = queryCombos(dev.dev, combosThrew);

  if (!combosThrew)
    dumpMatrixCombinations(combos);

  if (!dev.info.xmxSupported)
  {
    skipEverything(ResultStatus::Unsupported,
                   "XMX matrix engine not available on this device");
    return 0;
  }

  std::vector<JmTile> tilesFp = resolveTiles(combos, false);
  std::vector<JmTile> tilesInt = resolveTiles(combos, true);
  bool haveFp = !tilesFp.empty();
  bool haveInt = !tilesInt.empty();
  if (!haveFp && !haveInt)
  {
    skipEverything(ResultStatus::Unsupported,
                   combosThrew
                     ? "device's matrix-combination table could not be queried"
                     : "device advertises no matrix combinations");
    return 0;
  }
  if (haveFp) { addMissingCanonical(tilesFp, false); finalizeTiles(tilesFp); }
  if (haveInt) { addMissingCanonical(tilesInt, true); finalizeTiles(tilesInt); }
  std::vector<JmTile> tiles;
  tiles.reserve(tilesFp.size() + tilesInt.size());
  tiles.insert(tiles.end(), tilesFp.begin(), tilesFp.end());
  tiles.insert(tiles.end(), tilesInt.begin(), tilesInt.end());

  // The work-group is the device's widest sub-group, but IGC decides the SIMD
  // width it compiles at, so a work-group may end up holding several sub-groups
  // -- see JM_SG_COUNT_NOTE in the kernel, which reports the real count back.
  // maxSgPerWG (widest work-group split into narrowest sub-groups) is the
  // sanity bound that readback is checked against.
  uint32_t blockSize = dev.info.preferredSubGroupSize;
  if (blockSize == 0) blockSize = 32;      // fallback
  uint32_t minSgSize = blockSize;
  for (size_t s : dev.info.subGroupSizes)
    if (s > 0 && (uint32_t)s < minSgSize) minSgSize = (uint32_t)s;
  const uint32_t maxSgPerWG =
      std::max<uint32_t>(1u, blockSize / std::max<uint32_t>(1u, minSgSize));

  // One allocation shared by every row, sized to a fixed byte budget; each row
  // then takes as many blocks as its own tile fits into it.  Blocks therefore
  // fall as the tile grows, which is what keeps the work per launch -- and so
  // the launch duration and the store overhead -- the same for every row.
  uint64_t maxTileElems = 0;
  for (const auto &t : tiles)
    if (t.advertised)
      maxTileElems = std::max<uint64_t>(maxTileElems, (uint64_t)t.M * t.N);

  const uint64_t globalThreads = targetGlobalThreads((uint32_t)dev.info.numCUs);
  const uint64_t wantBlocks    = globalThreads / blockSize;
  // fp32 and int32 accumulators are both 4 bytes, so one figure covers both.
  uint64_t outElems = std::min(JM_OUT_BYTES, dev.info.totalGlobalMem / 4) / 4;
  if (outElems < maxTileElems) outElems = maxTileElems;   // room for one block

  const unsigned int forced = forceIters ? specifiedIters : 0;

  auto skipRows = [&](ResultStatus status, const std::string &reason) {
    for (const auto &t : tiles)
      test.skip(t.metric, status, reason, jmOpts(t));
  };

  void *out = isInt ? (void *)sycl::malloc_device<int32_t>(outElems, dev.stream)
                    : (void *)sycl::malloc_device<float>(outElems, dev.stream);
  uint32_t *sgCount = sycl::malloc_device<uint32_t>(1, dev.stream);
  if (!out || !sgCount)
  {
    if (out)     sycl::free(out, dev.stream);
    if (sgCount) sycl::free(sgCount, dev.stream);
    skipRows(ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }

  for (const auto &t : tiles)
  {
    if (!t.advertised)
    {
      test.skip(t.metric, ResultStatus::Unsupported,
                t.label + " not in this device's matrix-engine combinations",
                jmOpts(t));
      continue;
    }

    // As many blocks as this tile fits into the shared buffer -- see the
    // sizing note above.
    const uint64_t tileElems = (uint64_t)t.M * t.N;
    uint64_t blocks = std::min<uint64_t>(wantBlocks, outElems / tileElems);
    if (blocks == 0) blocks = 1;
    const uint32_t numBlocks = (uint32_t)blocks;

    const float us = jmIsIntAcc(t.ct)
        ? runJmInt(*this, dev, (int32_t *)out, sgCount, numBlocks, blockSize,
                   cfg.targetTimeUs, forced, t)
        : runJmFp (*this, dev, (float *)out, sgCount, numBlocks, blockSize,
                   cfg.targetTimeUs, forced, t);

    if (us == JM_NOT_COMPILED)
    {
      test.skip(t.metric, ResultStatus::Unsupported,
                "tile " + t.label + " " + t.shape +
                  " has no compiled instantiation in this build",
                jmOpts(t));
      continue;
    }
    if (us <= 0.0f)
    {
      test.skip(t.metric, ResultStatus::Error, "kernel launch failed", jmOpts(t));
      continue;
    }

    // How many chains actually ran per work-group (JM_SG_COUNT_NOTE).
    uint32_t sgPerWG = 0;
    try {
      dev.stream.memcpy(&sgPerWG, sgCount, sizeof(uint32_t)).wait();
    } catch (const std::exception &e) {
      CLPEAK_VLOG("joint_matrix: sub-group count readback failed: %s\n", e.what());
      sgPerWG = 0;
    }
    // maxSgPerWG is a hard upper bound (widest work-group over narrowest
    // advertised sub-group), so a value outside it means the readback is not
    // to be trusted; fall back to the historical assumption and say so.
    if (sgPerWG == 0 || sgPerWG > maxSgPerWG)
    {
      CLPEAK_VLOG("joint_matrix: implausible sub-group count %u (bound %u), "
                  "assuming 1\n", sgPerWG, maxSgPerWG);
      sgPerWG = 1;
    }
    CLPEAK_VLOG("joint_matrix: %s %s -- %u work-items/work-group ran as %u "
                "sub-group(s), %u blocks\n",
                t.label.c_str(), t.shape.c_str(), blockSize, sgPerWG, numBlocks);

    const double ops = (double)numBlocks * (double)sgPerWG * (double)t.volume() *
                       2.0 * (double)JM_ITERS;
    test.emit(t.metric, (float)(ops * 1.0e6 / us), jmOpts(t));
  }

  sycl::free(out, dev.stream);
  sycl::free(sgCount, dev.stream);
  return 0;
#endif
}

#endif // ENABLE_ONEAPI
