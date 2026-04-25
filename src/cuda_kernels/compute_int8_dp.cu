// INT8 dot-product throughput via __dp4a -- the canonical NVIDIA
// shader-core INT8 path on Pascal+ (sm_61+).  Each __dp4a call performs
// dot(char4, char4) into an int32 accumulator: 4 INT8 multiply-adds
// = 8 INT8 ops.
//
// Three variants for ILP scaling:
//   compute_int8_dp   -- 1 dependent chain (issue-rate floor)
//   compute_int8_dp2  -- 2 independent chains (more in-flight dp4as)
//   compute_int8_dp4  -- 4 independent chains (saturates dp4a issue rate
//                        on RTX 5060; matches Vulkan int8_dp4 variant)
//
// Op accounting: 8192 ops/thread = 1024 dp4a calls.  v1 = 64*16, v2 = each
// chain does 32*16 = 512 calls => 1024 total per thread.  v4 = each chain
// does 16*16 = 256 => 1024 total.

#define STEP(x, y, a)    a = __dp4a(x, y, a); y ^= a;
#define STEP_4(x, y, a)  STEP(x, y, a) STEP(x, y, a) STEP(x, y, a) STEP(x, y, a)
#define STEP_16(x, y, a) STEP_4(x, y, a) STEP_4(x, y, a) STEP_4(x, y, a) STEP_4(x, y, a)

extern "C" __global__ void compute_int8_dp(int *out, int A)
{
    int x = (A & 0xff)
          | (((A + 1) & 0xff) << 8)
          | (((A + 2) & 0xff) << 16)
          | (((A + 3) & 0xff) << 24);
    int y = (int)threadIdx.x;
    int a = (int)threadIdx.x;

    #pragma unroll 1
    for (int i = 0; i < 64; i++)
    {
        STEP_16(x, y, a)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = a;
}

extern "C" __global__ void compute_int8_dp2(int *out, int A)
{
    int x  = (A & 0xff) | (((A+1)&0xff)<<8) | (((A+2)&0xff)<<16) | (((A+3)&0xff)<<24);
    int y0 = (int)threadIdx.x;
    int y1 = (int)threadIdx.x + 1;
    int a0 = (int)threadIdx.x;
    int a1 = (int)threadIdx.x + 7;

    #pragma unroll 1
    for (int i = 0; i < 32; i++)
    {
        STEP_16(x, y0, a0)
        STEP_16(x, y1, a1)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1;
}

extern "C" __global__ void compute_int8_dp4(int *out, int A)
{
    int x  = (A & 0xff) | (((A+1)&0xff)<<8) | (((A+2)&0xff)<<16) | (((A+3)&0xff)<<24);
    int y0 = (int)threadIdx.x;
    int y1 = (int)threadIdx.x + 1;
    int y2 = (int)threadIdx.x + 2;
    int y3 = (int)threadIdx.x + 3;
    int a0 = (int)threadIdx.x;
    int a1 = (int)threadIdx.x + 7;
    int a2 = (int)threadIdx.x + 13;
    int a3 = (int)threadIdx.x + 19;

    #pragma unroll 1
    for (int i = 0; i < 16; i++)
    {
        STEP_16(x, y0, a0)
        STEP_16(x, y1, a1)
        STEP_16(x, y2, a2)
        STEP_16(x, y3, a3)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}
