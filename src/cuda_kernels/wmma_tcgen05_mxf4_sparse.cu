// Experimental Blackwell 5th-generation Tensor Core probe:
// sparse MXFP4 (E2M1 + UE8M0 block scale) via tcgen05.mma.sp.
//
// Target instruction:
//   tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32
//
// This is intentionally a minimal throughput probe, not a GEMM kernel.  A/B
// live in shared memory, D/scale/sparse metadata live in Tensor Memory, and a
// single thread issues a long dependent stream of tcgen05 MMA operations.  The
// result is arbitrary; the benchmark only needs the operation to be live and
// completed before Tensor Memory is deallocated.
//
// Shape encoded in idesc:
//   M=128, N=128, sparse K=128, A/B E2M1, scales UE8M0, sparse enabled.
// Nominal ops per tcgen05 instruction = 128*128*128*2 = 4,194,304.
// 128 instructions per CTA => 536,870,912 ops/block = 4,194,304 ops/thread
// when launched as one 128-thread warpgroup.

static __device__ __forceinline__ unsigned long long make_smem_desc(const void *ptr,
                                                                    unsigned lbo,
                                                                    unsigned sbo)
{
    unsigned addr = __cvta_generic_to_shared(ptr);
    unsigned long long desc = 0;
    desc |= (unsigned long long)((addr & 0x3ffffu) >> 4);
    desc |= (unsigned long long)((lbo  & 0x3ffffu) >> 4) << 16;
    desc |= (unsigned long long)((sbo  & 0x3ffffu) >> 4) << 32;
    desc |= 1ull << 46; // tcgen05 shared descriptor fixed constant field.
    return desc;        // no swizzle, relative leading-dimension mode.
}

extern "C" __global__ void wmma_tcgen05_mxf4_sparse(float *out, float A)
{
    __shared__ __align__(16) unsigned char smemA[4096]; // 128 x 64 packed sparse E2M1
    __shared__ __align__(16) unsigned char smemB[8192]; // 128 x 128 packed E2M1
    __shared__ __align__(8)  unsigned long long mbar;
    __shared__ __align__(4)  unsigned tmemBase;

    unsigned tid = threadIdx.x;
    unsigned seed = 0x44444444u ^ (__float_as_uint(A) & 0x11111111u);
    for (unsigned i = tid; i < sizeof(smemA) / sizeof(unsigned); i += blockDim.x)
        reinterpret_cast<unsigned *>(smemA)[i] = seed;
    for (unsigned i = tid; i < sizeof(smemB) / sizeof(unsigned); i += blockDim.x)
        reinterpret_cast<unsigned *>(smemB)[i] = seed ^ 0x11111111u;

    if (tid == 0)
    {
        unsigned long long mbarAddr = (unsigned long long)&mbar;
        asm volatile("mbarrier.init.b64 [%0], 1;" :: "l"(mbarAddr) : "memory");
    }
    __syncthreads();

    // tcgen05.alloc is issued by one warp; all lanes in warp 0 participate.
    if (tid < 32)
    {
        unsigned tmemAddrSlot = __cvta_generic_to_shared(&tmemBase);
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;\n"
            :: "r"(tmemAddrSlot) : "memory");
    }
    __syncthreads();

    unsigned tD = tmemBase;
    unsigned tSpMeta = tD + 128;
    unsigned tScaleA = tD + 192;
    unsigned tScaleB = tD + 256;

    // The exact values are not important for throughput.  The descriptors are
    // 16-byte aligned, no-swizzle shared-memory descriptors.
    unsigned long long aDesc = make_smem_desc(smemA, 64, 4096);
    unsigned long long bDesc = make_smem_desc(smemB, 64, 8192);
    unsigned idesc = 0x08a00484u;
    unsigned enableD = 0;

    if (tid == 0)
    {
        #pragma unroll 1
        for (int i = 0; i < 128; i++)
        {
            asm volatile(
                "{ .reg .pred p;\n"
                "  setp.ne.u32 p, %7, 0;\n"
                "tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::discard "
                "[%0], %1, %2, [%3], %4, [%5], [%6], p;\n"
                "}\n"
                :: "r"(tD), "l"(aDesc), "l"(bDesc), "r"(tSpMeta), "r"(idesc),
                   "r"(tScaleA), "r"(tScaleB), "r"(enableD)
                : "memory");
        }

        unsigned long long mbarAddr = (unsigned long long)&mbar;
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
            :: "l"(mbarAddr) : "memory");

        unsigned done = 0;
        do
        {
            asm volatile(
                "{ .reg .pred p;\n"
                "  mbarrier.try_wait.parity.b64 p, [%1], 0;\n"
                "  selp.u32 %0, 1, 0, p;\n"
                "}\n"
                : "=r"(done) : "l"(mbarAddr) : "memory");
        } while (!done);

        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
        out[blockIdx.x] = (float)(tD ^ tSpMeta ^ tScaleA ^ tScaleB);
    }
    __syncthreads();

    if (tid < 32)
    {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;\n"
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n"
            :: "r"(tmemBase) : "memory");
    }
}
