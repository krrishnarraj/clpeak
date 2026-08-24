// mad_chain.metal -- the affine MAD chain every compute_*_alt kernel expands.
//
// Prepended to every embedded .metal source by EmbedMetalKernels.cmake, since
// newLibraryWithSource compiles the string as-is and cannot resolve #include
// of a sibling file.
//
// Each compute family defines its chain twice: the squaring recurrence
//
//     x = x*x + c                       (c a per-thread loop invariant)
//
// which the compute_* kernels have always used, and the affine recurrence over
// N independent accumulators
//
//     x_k = a*x_k + b                   (a, b per-thread loop invariants)
//
// which the compute_*_alt kernels add.  runComputeKernel times both and
// reports the faster.  Both spell 16 chain instructions per _16, so the
// per-thread op budget is identical and the two readings are comparable.
//
// Why two shapes, and why the integer families elsewhere in clpeak use a third
// one instead: the MAD chain block in include/common/common.h.  Apple GPUs are
// not the reason this exists -- they run the squaring chain at full rate -- but
// racing costs nothing here and keeps every backend on the same footing.
//
// N is 4 at vector width 1, 2 at width 2 and 1 from width 4 up, where the
// vector already supplies the instruction-level parallelism.  Live values per
// lane stay at ~5 either way, which is what the MAD chain rules require.

#define CH_MAD(d, m1, m2, ad)  d = fma(m1, m2, ad);

#define AF4_DECL(T, seed, inv) T a = (inv); T b = a + T(2); T x0 = (seed); T x1 = (seed) + T(1); T x2 = (seed) + T(2); T x3 = (seed) + T(3);
#define AF4_G                  CH_MAD(x0, a, x0, b) CH_MAD(x1, a, x1, b) CH_MAD(x2, a, x2, b) CH_MAD(x3, a, x3, b)
#define AF4_16                 AF4_G AF4_G AF4_G AF4_G
#define AF4_RES                ((x0 + x1) + (x2 + x3))

#define AF2_DECL(T, seed, inv) T a = (inv); T b = a + T(2); T x0 = (seed); T x1 = (seed) + T(1);
#define AF2_G                  CH_MAD(x0, a, x0, b) CH_MAD(x1, a, x1, b)
#define AF2_16                 AF2_G AF2_G AF2_G AF2_G AF2_G AF2_G AF2_G AF2_G
#define AF2_RES                (x0 + x1)

#define AF1_DECL(T, seed, inv) T a = (inv); T b = a + T(2); T x0 = (seed);
#define AF1_G                  CH_MAD(x0, a, x0, b)
#define AF1_16                 AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G
#define AF1_RES                (x0)
