package kr.clpeak

import org.json.JSONObject

data class DeviceRef(val index: Int, val name: String)

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

        fun fromJson(json: String): BackendCatalog {
            val root = JSONObject(json)

            val cl = root.optJSONObject("opencl")
            val opencl = if (cl != null) {
                val plats = cl.optJSONArray("platforms")
                val platforms = mutableListOf<OpenClPlatformInfo>()
                if (plats != null) {
                    for (i in 0 until plats.length()) {
                        val p = plats.getJSONObject(i)
                        val devs = p.optJSONArray("devices")
                        val deviceList = mutableListOf<DeviceRef>()
                        if (devs != null) {
                            for (j in 0 until devs.length()) {
                                val d = devs.getJSONObject(j)
                                deviceList.add(DeviceRef(d.getInt("index"), d.getString("name")))
                            }
                        }
                        platforms.add(OpenClPlatformInfo(
                            index = p.getInt("index"),
                            name = p.getString("name"),
                            devices = deviceList
                        ))
                    }
                }
                OpenClCatalog(cl.optBoolean("available", false), platforms)
            } else {
                OpenClCatalog(false, emptyList())
            }

            val vk = root.optJSONObject("vulkan")
            val vulkan = if (vk != null) {
                val devs = vk.optJSONArray("devices")
                val deviceList = mutableListOf<VulkanDevice>()
                if (devs != null) {
                    for (i in 0 until devs.length()) {
                        val d = devs.getJSONObject(i)
                        deviceList.add(VulkanDevice(
                            index = d.getInt("index"),
                            name = d.getString("name"),
                            type = d.optString("type", "other")
                        ))
                    }
                }
                VulkanCatalog(vk.optBoolean("available", false), deviceList)
            } else {
                VulkanCatalog(false, emptyList())
            }

            return BackendCatalog(opencl, vulkan)
        }
    }
}
