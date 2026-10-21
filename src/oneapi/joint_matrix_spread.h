#ifndef CLPEAK_ONEAPI_JOINT_MATRIX_SPREAD_H
#define CLPEAK_ONEAPI_JOINT_MATRIX_SPREAD_H

// The per-lane refill the joint_matrix kernels in this backend run their A
// and B tiles through: the blocks in joint_matrix_block.cpp always, the
// chain in joint_matrix.cpp in one of the two forms it is raced in (why
// there are two: runJmVariant).  Include it after
// <sycl/ext/oneapi/matrix/matrix.hpp>.
//
// The kernels build their tiles with joint_matrix_fill from a kernel
// argument.  IGC lowers a fill to that one value inserted into every element
// of the lane's share (JointMatrixFuncsResolutionPass::ResolveFill), its
// uniformity analysis then holds the tile as one value for the whole
// sub-group, and its DPAS emitter copies a uniform A or B out to full width
// in front of every DPAS that reads it (EmitPass::emitDpas, through
// BroadcastIfUniform).  On Alchemist, where A and B are both eight dwords a
// lane, that is sixteen MOVs re-copying tiles that never change, ahead of
// every product the benchmark counts.  An A380 (driver 9034) read 22.4
// TFLOPS fp16 here with filled tiles, where Vulkan read 30.7 for the same
// sixteen-product block at the same SIMD8 width with its tiles spread as
// below.
//
// So each tile is also refilled element by element, every element a value
// of its own taken from the lane's id -- on a branch the host never takes
// (`spread`, a kernel argument it leaves 0) and no compiler can rule out.  A
// tile that may differ from lane to lane is one the compiler keeps whole in
// registers and hands to the DPAS as it is.  Vulkan's coopmat shaders
// build theirs the same way (CM_SPREAD in src/vulkan/shaders/coopmat_chain.glsl).
//
// A kernel's `spread` must reach it as a value the device compiler cannot
// see: a parameter of the host function that builds the kernel, which the
// kernel lambda captures.  A constant in that function would be folded into
// the kernel and the branch with it.

#include <sycl/sycl.hpp>

#include <cstdint>
#include <type_traits>

namespace clpeak_oneapi {

// Element `n` of a tile spread from its fill value `v`: `v` itself at n = 0
// and a value of its own at every other n, as CM_VAL(i, n) is in Vulkan.  A
// floating-point element steps its bit pattern rather than its value:
// arithmetic on a bfloat16 calls DPC++'s conversion library, a device image
// of its own the runtime would have to link into every bf16 kernel for a
// branch no run takes.  The steps stay small enough that every element is
// still a finite number near `v`.
template <typename T>
inline T jmSpreadValue(T v, uint32_t n)
{
  if constexpr (std::is_integral_v<T>)
    return (T)(v + (int)n);
  else
  {
    using Bits = std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>;
    return sycl::bit_cast<T>((Bits)(sycl::bit_cast<Bits>(v) + n));
  }
}

// Refill `tile` element by element: element e of this lane's share becomes
// number e * lanes + lane, so no two elements of the sub-group's tile share
// a number, in one lane or across lanes.
template <typename Tile>
inline void jmSpread(sycl::sub_group sg, Tile &tile)
{
  const uint32_t lanes = (uint32_t)sg.get_local_linear_range();
  const uint32_t lane = (uint32_t)sg.get_local_linear_id();
  uint32_t e = 0;
  sycl::ext::oneapi::experimental::matrix::joint_matrix_apply(sg, tile, [&](auto &x) {
    x = jmSpreadValue(x, e * lanes + lane);
    e++;
  });
}

} // namespace clpeak_oneapi

#endif // CLPEAK_ONEAPI_JOINT_MATRIX_SPREAD_H
