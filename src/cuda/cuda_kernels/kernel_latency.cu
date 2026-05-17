// Empty kernel for measuring launch+complete round-trip overhead.  Single
// block, single thread, no work -- whatever ms/launch this measures is
// pure CUDA dispatch + driver scheduling latency.

extern "C" __global__ void kernel_latency_noop()
{
}
