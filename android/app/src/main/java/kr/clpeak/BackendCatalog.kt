package kr.clpeak

import org.json.JSONObject

data class DeviceRef(val index: Int, val name: String, val type: String = "")

data class OpenClPlatformInfo(
    val index: Int,
    val name: String,
    val devices: List<DeviceRef>
)

data class OpenClCatalog(
    val available: Boolean,
    val platforms: List<OpenClPlatformInfo>
) {
    val isEmpty: Boolean get() = platforms.all { it.devices.isEmpty() }
}

data class VulkanDevice(val index: Int, val name: String, val type: String)

data class VulkanCatalog(
    val available: Boolean,
    val devices: List<VulkanDevice>
) {
    val isEmpty: Boolean get() = devices.isEmpty()
}

data class BackendCatalog(
    val opencl: OpenClCatalog,
    val vulkan: VulkanCatalog
) {
    val hasAnyDevice: Boolean
        get() = !opencl.isEmpty || !vulkan.isEmpty

    companion object {
        val EMPTY = BackendCatalog(
            OpenClCatalog(available = false, platforms = emptyList()),
            VulkanCatalog(available = false, devices = emptyList())
        )

        /**
         * Parses the JSON shape emitted by inventoryToJson() in src/inventory.cpp:
         * `{"backends":[{"name":"OpenCL","available":bool,"platforms":[
         *    {"index":i,"name":s,"devices":[{"index":i,"name":s,"type":s,...}]}
         *  ]}, {"name":"Vulkan", ...}]}`.
         */
        fun fromJson(json: String): BackendCatalog {
            val root = JSONObject(json)
            val backends = root.optJSONArray("backends") ?: return EMPTY

            var opencl = OpenClCatalog(false, emptyList())
            var vulkan = VulkanCatalog(false, emptyList())

            for (i in 0 until backends.length()) {
                val b = backends.getJSONObject(i)
                when (b.optString("name")) {
                    "OpenCL" -> opencl = parseOpenCl(b)
                    "Vulkan" -> vulkan = parseVulkan(b)
                }
            }
            return BackendCatalog(opencl, vulkan)
        }

        private fun parseOpenCl(b: JSONObject): OpenClCatalog {
            val platforms = mutableListOf<OpenClPlatformInfo>()
            val plats = b.optJSONArray("platforms")
            if (plats != null) {
                for (i in 0 until plats.length()) {
                    val p = plats.getJSONObject(i)
                    val devs = p.optJSONArray("devices")
                    val deviceList = mutableListOf<DeviceRef>()
                    if (devs != null) {
                        for (j in 0 until devs.length()) {
                            val d = devs.getJSONObject(j)
                            deviceList.add(DeviceRef(
                                index = d.getInt("index"),
                                name  = d.getString("name"),
                                type  = d.optString("type", "")
                            ))
                        }
                    }
                    platforms.add(OpenClPlatformInfo(
                        index = p.getInt("index"),
                        name = p.getString("name"),
                        devices = deviceList
                    ))
                }
            }
            return OpenClCatalog(b.optBoolean("available", false), platforms)
        }

        private fun parseVulkan(b: JSONObject): VulkanCatalog {
            // Vulkan inventory has a single synthetic platform; flatten its devices.
            val devices = mutableListOf<VulkanDevice>()
            val plats = b.optJSONArray("platforms")
            if (plats != null) {
                for (i in 0 until plats.length()) {
                    val p = plats.getJSONObject(i)
                    val devs = p.optJSONArray("devices") ?: continue
                    for (j in 0 until devs.length()) {
                        val d = devs.getJSONObject(j)
                        devices.add(VulkanDevice(
                            index = d.getInt("index"),
                            name  = d.getString("name"),
                            type  = d.optString("type", "other")
                        ))
                    }
                }
            }
            return VulkanCatalog(b.optBoolean("available", false), devices)
        }
    }
}
