package kr.clpeak

/**
 * User's pick of which backend devices to benchmark.
 *
 * Devices are identified by backend name and a composite key "platformIndex:deviceIndex"
 * (see [DeviceRef.key]).  Tests are not user-selectable — all applicable tests run on
 * every selected device.
 */
data class BenchmarkSelection(
    val devicesByBackend: Map<String, Set<String>> = emptyMap()
) {
    val isEmpty: Boolean get() = devicesByBackend.values.all { it.isEmpty() }

    fun isSelected(backend: String, deviceKey: String): Boolean =
        devicesByBackend[backend]?.contains(deviceKey) == true

    fun isBackendEnabled(backend: String): Boolean =
        !(devicesByBackend[backend] ?: emptySet()).isEmpty()

    fun toggleDevice(backend: String, deviceKey: String): BenchmarkSelection {
        val current = devicesByBackend[backend] ?: emptySet()
        val updated = if (deviceKey in current) current - deviceKey else current + deviceKey
        return copy(devicesByBackend = devicesByBackend + (backend to updated))
    }

    fun setBackend(backend: String, enabled: Boolean, allDeviceKeys: List<String>): BenchmarkSelection {
        return copy(
            devicesByBackend = devicesByBackend + (
                backend to if (enabled) allDeviceKeys.toSet() else emptySet()
            )
        )
    }

    companion object {
        fun allOf(catalog: BackendCatalog): BenchmarkSelection {
            val map = mutableMapOf<String, Set<String>>()
            for (be in catalog.backends) {
                map[be.name] = be.devices.map { it.key }.toSet()
            }
            return BenchmarkSelection(map)
        }
    }
}
