MSTRINGIFY(

// Empty kernel for the launch latency test, enqueued as one work-item: what a
// launch costs is then the runtime's submission and scheduling alone, as in
// every other backend's latency test (Vulkan's 1x1x1 no-op dispatch, CUDA's
// <<<1, 1>>>).  No arguments, so there is no buffer to bind or migrate.

__kernel void kernel_latency_noop(void)
{
}

)
