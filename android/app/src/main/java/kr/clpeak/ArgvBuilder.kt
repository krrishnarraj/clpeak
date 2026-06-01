package kr.clpeak

/**
 * Translates a [BenchmarkSelection] into the argv that the native CLI parser
 * (src/common/options.cpp) expects.  The native parser flips to "deny by default;
 * enable picked" for tests on the first test flag, so we never emit per-test
 * flags here — all tests always run.
 *
 * Limitation: the native CLI's device flags select exactly one device.  If the
 * user selects multiple devices for a backend we omit those flags and let native
 * iterate every enumerated device.
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
                }
            } else if (selectedKeys.size == 1) {
                val key = selectedKeys.single()
                val dash = key.indexOf(':')
                if (dash < 0) continue
                val platformIdx = key.substring(0, dash).toIntOrNull() ?: continue
                val deviceIdx   = key.substring(dash + 1).toIntOrNull() ?: continue

                when (backend.name) {
                    "OpenCL" -> {
                        args += "--cl-platform"; args += platformIdx.toString()
                        args += "--cl-device";   args += deviceIdx.toString()
                    }
                    "Vulkan" -> {
                        args += "--vk-device";   args += deviceIdx.toString()
                    }
                    // Future backends (CUDA, Metal, etc.) can add their flags here.
                }
            }
            // else: multiple devices selected → omit flags, native iterates all
        }

        return args.toTypedArray()
    }
}
