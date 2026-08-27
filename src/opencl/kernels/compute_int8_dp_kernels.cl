MSTRINGIFY(

// INT8 dot-product compute throughput using cl_khr_integer_dot_product.
// Each dot_acc_sat(char4, char4, int) computes 4 signed INT8 multiply-adds
// into a 32-bit accumulator and is the hardware DP4a / XDL / INT8-tensor-core
// path on modern GPUs (NVIDIA Turing+, AMD RDNA2+, Intel Xe+, Adreno, Mali).

// OpenCL 3.0 exposes this as an optional *feature* rather than an extension,
// so a conforming compiler may define __opencl_c_integer_dot_product_input_4x8bit
// without ever defining cl_khr_integer_dot_product.  Checking only the
// extension macro compiled the kernels out on Intel's CPU runtime even though
// the device advertised the extension and reported the 4x8-bit capability,
// leaving clCreateKernel to fail with CL_INVALID_KERNEL_NAME.

// The chain shape.  Three constraints have to hold at once, and each one has
// already produced a wrong reading in another backend:
//
//  - Both multiplicands may not be loop-invariant.  a = dot_acc_sat(x, y, a)
//    with x and y both fixed is a + n*dot(x, y), which a compiler may and does
//    strength-reduce.  The Vulkan backend shipped that shape and read 74939
//    GOPS on an RTX 5060 whose dp4a peak, measured through CUDA's own __dp4a
//    on the same card, is 33928 -- 2.2x past the hardware.
//
//  - Nothing may run between the dots.  The obvious way to keep an operand
//    moving is to rewrite it from the accumulator (y ^= a), but that XOR is a
//    second dependent integer op per dot and the op budget credits none of it;
//    on an Arc A380 it cost more than half the rate (8832 GOPS against 19497
//    for the same instruction with nothing beside it).
//
//  - All three source operands must be distinct registers.  Intel Alchemist
//    halves a three-source op that reads the same register twice -- the rule
//    mad_chain.cl is built around -- so a = dot_acc_sat(x, as_char4(a), a) is
//    not the answer either.
//
// What satisfies all three: two accumulators feeding each other.  Each dot
// reads {x, the other accumulator, its own}, three distinct registers, and
// writes its own.  Every dot depends on the one before it, so a pair is one
// dependent chain, not two; and because the dot extracts the bytes of a value
// that is itself a 32-bit accumulator, the recurrence is not affine and has no
// closed form to fold to.  v2..v16 run 2..16 independent copies of the pair,
// which is what the ILP ladder measures.
//
// as_char4, not a cast.  The accumulator is int -- that is what dot_acc_sat
// returns -- and the 4x8-bit dot takes char4, so the other accumulator has to
// change type on the way in.  as_char4 reinterprets the same 32 bits: clang
// lowers it to `bitcast i32 to <4 x i8>`, one register read differently, no
// instruction issued, which is what keeps rule two.  The other spelling,
// (char4)a, is a conversion rather than a reinterpret -- it truncates the
// accumulator to its low byte and broadcasts that byte to all four lanes --
// so it would add arithmetic the budget does not count and throw away three
// quarters of the state that keeps the recurrence non-affine.
//
// Op accounting: 1024 dots per work-item, each 4 INT8 multiply-adds = 8 ops,
// = COMPUTE_INT8_DP_WORK_PER_WI (8192), the same for every variant.  One
// DP_STEP is two dots, so each variant issues 512 of them: v1 = 64 iters x 8,
// v2 = 64 x 4 per chain x 2 chains, v4 = 64 x 2 x 4, v8 = 64 x 1 x 8,
// v16 = 32 x 1 x 16.  Chain k holds accumulators a<k> and b<k>, seeded 4
// apart from every other accumulator in the work-item: chains that start on
// the same value stay bitwise equal forever and a compiler is free to keep
// only one of them.
\n#if defined(cl_khr_integer_dot_product)
\n  #pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
\n#endif
\n#if defined(cl_khr_integer_dot_product) || defined(__opencl_c_integer_dot_product_input_4x8bit)
\n  #define INT8_DP_AVAILABLE
\n#endif

\n#ifdef INT8_DP_AVAILABLE

\n#undef DP_STEP
\n#undef DP_STEP_2
\n#undef DP_STEP_4
\n#undef DP_STEP_8
\n
\n#define DP_STEP(x, p, q)    p = dot_acc_sat(x, as_char4(q), p);  q = dot_acc_sat(x, as_char4(p), q);
\n#define DP_STEP_2(x, p, q)  DP_STEP(x, p, q)  DP_STEP(x, p, q)
\n#define DP_STEP_4(x, p, q)  DP_STEP_2(x, p, q)  DP_STEP_2(x, p, q)
\n#define DP_STEP_8(x, p, q)  DP_STEP_4(x, p, q)  DP_STEP_4(x, p, q)
\n

__kernel void compute_int8_dp_v1(__global int *ptr, char _A)
{
    char4 x = (char4)(_A, _A+1, _A+2, _A+3);
    int lid = (int)get_local_id(0);
    int a0 = lid, b0 = lid + 4;

    for (int i = 0; i < 64; i++)
    {
        DP_STEP_8(x, a0, b0)
    }

    ptr[get_global_id(0)] = a0 + b0;
}

__kernel void compute_int8_dp_v2(__global int *ptr, char _A)
{
    char4 x = (char4)(_A, _A+1, _A+2, _A+3);
    int lid = (int)get_local_id(0);
    int a0 = lid,     b0 = lid + 4;
    int a1 = lid + 8, b1 = lid + 12;

    for (int i = 0; i < 64; i++)
    {
        DP_STEP(x, a0, b0)  DP_STEP(x, a1, b1)
        DP_STEP(x, a0, b0)  DP_STEP(x, a1, b1)
        DP_STEP(x, a0, b0)  DP_STEP(x, a1, b1)
        DP_STEP(x, a0, b0)  DP_STEP(x, a1, b1)
    }

    ptr[get_global_id(0)] = (a0 + b0) + (a1 + b1);
}

__kernel void compute_int8_dp_v4(__global int *ptr, char _A)
{
    char4 x = (char4)(_A, _A+1, _A+2, _A+3);
    int lid = (int)get_local_id(0);
    int a0 = lid + 0,  b0 = lid + 4;
    int a1 = lid + 8,  b1 = lid + 12;
    int a2 = lid + 16, b2 = lid + 20;
    int a3 = lid + 24, b3 = lid + 28;

    for (int i = 0; i < 64; i++)
    {
        DP_STEP(x, a0, b0)  DP_STEP(x, a1, b1)  DP_STEP(x, a2, b2)  DP_STEP(x, a3, b3)
        DP_STEP(x, a0, b0)  DP_STEP(x, a1, b1)  DP_STEP(x, a2, b2)  DP_STEP(x, a3, b3)
    }

    ptr[get_global_id(0)] = ((a0 + b0) + (a1 + b1)) + ((a2 + b2) + (a3 + b3));
}

__kernel void compute_int8_dp_v8(__global int *ptr, char _A)
{
    char4 x = (char4)(_A, _A+1, _A+2, _A+3);
    int lid = (int)get_local_id(0);
    int a0 = lid + 0,  b0 = lid + 4;
    int a1 = lid + 8,  b1 = lid + 12;
    int a2 = lid + 16, b2 = lid + 20;
    int a3 = lid + 24, b3 = lid + 28;
    int a4 = lid + 32, b4 = lid + 36;
    int a5 = lid + 40, b5 = lid + 44;
    int a6 = lid + 48, b6 = lid + 52;
    int a7 = lid + 56, b7 = lid + 60;

    for (int i = 0; i < 64; i++)
    {
        DP_STEP(x, a0, b0)  DP_STEP(x, a1, b1)  DP_STEP(x, a2, b2)  DP_STEP(x, a3, b3)
        DP_STEP(x, a4, b4)  DP_STEP(x, a5, b5)  DP_STEP(x, a6, b6)  DP_STEP(x, a7, b7)
    }

    ptr[get_global_id(0)] = (((a0 + b0) + (a1 + b1)) + ((a2 + b2) + (a3 + b3)))
                          + (((a4 + b4) + (a5 + b5)) + ((a6 + b6) + (a7 + b7)));
}

__kernel void compute_int8_dp_v16(__global int *ptr, char _A)
{
    char4 x = (char4)(_A, _A+1, _A+2, _A+3);
    int lid = (int)get_local_id(0);
    int a0 = lid + 0,   b0 = lid + 4;
    int a1 = lid + 8,   b1 = lid + 12;
    int a2 = lid + 16,  b2 = lid + 20;
    int a3 = lid + 24,  b3 = lid + 28;
    int a4 = lid + 32,  b4 = lid + 36;
    int a5 = lid + 40,  b5 = lid + 44;
    int a6 = lid + 48,  b6 = lid + 52;
    int a7 = lid + 56,  b7 = lid + 60;
    int a8 = lid + 64,  b8 = lid + 68;
    int a9 = lid + 72,  b9 = lid + 76;
    int a10 = lid + 80, b10 = lid + 84;
    int a11 = lid + 88, b11 = lid + 92;
    int a12 = lid + 96, b12 = lid + 100;
    int a13 = lid + 104, b13 = lid + 108;
    int a14 = lid + 112, b14 = lid + 116;
    int a15 = lid + 120, b15 = lid + 124;

    for (int i = 0; i < 32; i++)
    {
        DP_STEP(x, a0, b0)    DP_STEP(x, a1, b1)    DP_STEP(x, a2, b2)    DP_STEP(x, a3, b3)
        DP_STEP(x, a4, b4)    DP_STEP(x, a5, b5)    DP_STEP(x, a6, b6)    DP_STEP(x, a7, b7)
        DP_STEP(x, a8, b8)    DP_STEP(x, a9, b9)    DP_STEP(x, a10, b10)  DP_STEP(x, a11, b11)
        DP_STEP(x, a12, b12)  DP_STEP(x, a13, b13)  DP_STEP(x, a14, b14)  DP_STEP(x, a15, b15)
    }

    ptr[get_global_id(0)] =
        ((((a0 + b0) + (a1 + b1)) + ((a2 + b2) + (a3 + b3)))
       + (((a4 + b4) + (a5 + b5)) + ((a6 + b6) + (a7 + b7))))
      + ((((a8 + b8) + (a9 + b9)) + ((a10 + b10) + (a11 + b11)))
       + (((a12 + b12) + (a13 + b13)) + ((a14 + b14) + (a15 + b15))));
}

\n#endif  // INT8_DP_AVAILABLE

)
