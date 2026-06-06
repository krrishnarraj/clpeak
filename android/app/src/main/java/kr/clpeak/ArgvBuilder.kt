package kr.clpeak

/**
 * Translates a [BenchmarkSelection] into the argv that the native CLI parser
 * (src/common/options.cpp) expects.  The native parser flips to "deny by default;
 * enable picked" for tests on the first test flag, so we never emit per-test
 * flags here — all tests always run.
 *
 * Device flags accept a comma-separated index list, so any subset of a backend's
 * devices can be expressed.  We emit a list only for a *partial* selection; a
 * fully-selected backend is left implicit (native runs every device).
 */
object ArgvBuilder {

    fun build(sel: BenchmarkSelection, catalog: BackendCatalog): Array<String> {
        val args = mutableListOf("clpeak")

        for (backend in catalog.backends) {
            val selectedKeys = sel.devicesByBackend[backend.name] ?: continue

            if (selectedKeys.isEmpty()) {
                when (backend.name) {
                    "OpenCL"  -> args += "--no-opencl"
                    "Vulkan"  -> args += "--no-vulkan"
                    "CUDA"    -> args += "--no-cuda"
                    "ROCm"    -> args += "--no-rocm"
                    "Metal"   -> args += "--no-metal"
                    "oneAPI"  -> args += "--no-oneapi"
                    "CPU"     -> args += "--no-cpu"
                }
                continue
            }

            // Full selection → no device flags; native iterates every device.
            if (selectedKeys == backend.devices.map { it.key }.toSet()) continue

            // Partial selection → list the chosen "platformIndex:deviceIndex" pairs.
            val pairs = selectedKeys.mapNotNull { key ->
                val dash = key.indexOf(':')
                if (dash < 0) return@mapNotNull null
                val p = key.substring(0, dash).toIntOrNull() ?: return@mapNotNull null
                val d = key.substring(dash + 1).toIntOrNull() ?: return@mapNotNull null
                p to d
            }
            if (pairs.isEmpty()) continue

            when (backend.name) {
                "OpenCL" -> {
                    // OpenCL filters platform and device independently, so a pick
                    // spanning platforms is the cross-product of the platform and
                    // device index lists (exact for the common single-platform case).
                    val platforms = pairs.map { it.first }.distinct().sorted()
                    val devices   = pairs.map { it.second }.distinct().sorted()
                    args += "--cl-platform"; args += platforms.joinToString(",")
                    args += "--cl-device";   args += devices.joinToString(",")
                }
                "Vulkan" -> {
                    val devices = pairs.map { it.second }.distinct().sorted()
                    args += "--vk-device"; args += devices.joinToString(",")
                }
                // Other backends aren't compiled into the Android binary
                // (see entry_android.cpp), so they need no device flags here.
            }
        }

        return args.toTypedArray()
    }
}
