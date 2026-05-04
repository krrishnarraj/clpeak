package kr.clpeak

/**
 * Translates a [BenchmarkSelection] into the argv that the native CLI parser
 * (src/options.cpp) expects. The native parser flips to "deny by default;
 * enable picked" for tests on the first test flag, so we never emit per-test
 * flags here — all tests always run.
 *
 * Limitation: the native CLI's OpenCL platform/device flags select exactly one
 * platform and one device. If the user selects multiple OpenCL devices we omit
 * those flags and let native iterate every enumerated device. Same for Vulkan.
 */
object ArgvBuilder {

    fun build(sel: BenchmarkSelection, catalog: BackendCatalog): Array<String> {
        val args = mutableListOf("clpeak")

        val skipOpenCL = sel.openclDevices.isEmpty()
        val skipVulkan = sel.vulkanDevices.isEmpty()

        if (skipOpenCL) args += "--no-opencl"
        if (skipVulkan) args += "--no-vulkan"

        if (!skipOpenCL && sel.openclDevices.size == 1) {
            val (p, d) = sel.openclDevices.single()
            args += "--cl-platform"; args += p.toString()
            args += "--cl-device";   args += d.toString()
        }

        if (!skipVulkan && sel.vulkanDevices.size == 1) {
            args += "--vk-device"
            args += sel.vulkanDevices.single().toString()
        }

        return args.toTypedArray()
    }
}
