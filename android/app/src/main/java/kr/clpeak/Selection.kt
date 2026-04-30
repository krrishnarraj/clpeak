package kr.clpeak

/**
 * User's pick of which backend devices to benchmark. Tests are not user-selectable
 * — all applicable tests run on every selected device.
 *
 * OpenCL devices are identified by (platformIndex, deviceIndex). Vulkan devices
 * by deviceIndex only.
 */
data class BenchmarkSelection(
    val openclDevices: Set<Pair<Int, Int>>,
    val vulkanDevices: Set<Int>
) {
    val isEmpty: Boolean
        get() = openclDevices.isEmpty() && vulkanDevices.isEmpty()

    fun toggleOpenCl(platform: Int, device: Int): BenchmarkSelection {
        val key = platform to device
        return copy(
            openclDevices = if (key in openclDevices) openclDevices - key
                            else openclDevices + key
        )
    }

    fun toggleVulkan(device: Int): BenchmarkSelection {
        return copy(
            vulkanDevices = if (device in vulkanDevices) vulkanDevices - device
                            else vulkanDevices + device
        )
    }

    fun setOpenClBackend(catalog: OpenClCatalog, on: Boolean): BenchmarkSelection {
        val all = catalog.platforms.flatMap { p ->
            p.devices.map { d -> p.index to d.index }
        }.toSet()
        return copy(openclDevices = if (on) all else emptySet())
    }

    fun setVulkanBackend(catalog: VulkanCatalog, on: Boolean): BenchmarkSelection {
        return copy(
            vulkanDevices = if (on) catalog.devices.map { it.index }.toSet()
                            else emptySet()
        )
    }

    companion object {
        /** Default = run everything in the catalog. */
        fun allOf(catalog: BackendCatalog): BenchmarkSelection {
            val cl = catalog.opencl.platforms
                .flatMap { p -> p.devices.map { d -> p.index to d.index } }
                .toSet()
            val vk = catalog.vulkan.devices.map { it.index }.toSet()
            return BenchmarkSelection(cl, vk)
        }
    }
}
