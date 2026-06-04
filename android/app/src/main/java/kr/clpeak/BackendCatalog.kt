package kr.clpeak

import org.json.JSONObject

/**
 * A device discovered by a backend.  The composite [key] ("platformIndex:deviceIndex")
 * uniquely identifies this device across all backends and platforms.
 */
data class DeviceRef(
    val platformIndex: Int,
    val deviceIndex: Int,
    val name: String,
    val type: String = "",
    val apiVersion: String = ""
) {
    val key: String get() = "$platformIndex:$deviceIndex"
}

/** One backend (e.g. "Vulkan", "OpenCL") and the devices it found. */
data class BackendInfo(
    val name: String,
    val available: Boolean,
    val devices: List<DeviceRef>
) {
    val isEmpty: Boolean get() = devices.isEmpty()
}

/** Parsed from the C++ inventory JSON — a flat list of backends in enumeration order. */
data class BackendCatalog(
    val backends: List<BackendInfo>
) {
    val hasAnyDevice: Boolean get() = backends.any { !it.isEmpty }

    fun backend(named: String): BackendInfo? = backends.firstOrNull { it.name == named }

    companion object {
        val EMPTY = BackendCatalog(emptyList())

        /**
         * Parses the JSON shape emitted by inventoryToJson() in src/common/inventory.cpp:
         * `{"backends":[{"name":"OpenCL","available":bool,"platforms":[
         *    {"index":i,"name":s,"devices":[{"index":i,"name":s,"type":s,...}]}
         *  ]}, ...]}`.
         */
        fun fromJson(json: String): BackendCatalog {
            val root = JSONObject(json)
            val rawBackends = root.optJSONArray("backends") ?: return EMPTY

            val backends = mutableListOf<BackendInfo>()
            for (i in 0 until rawBackends.length()) {
                val b = rawBackends.getJSONObject(i)
                val name = b.optString("name")
                val available = b.optBoolean("available", false)
                val platforms = b.optJSONArray("platforms")
                val devices = mutableListOf<DeviceRef>()
                if (platforms != null) {
                    for (pi in 0 until platforms.length()) {
                        val p = platforms.getJSONObject(pi)
                        val pIndex = p.getInt("index")
                        val rawDevices = p.optJSONArray("devices") ?: continue
                        for (di in 0 until rawDevices.length()) {
                            val d = rawDevices.getJSONObject(di)
                            devices.add(
                                DeviceRef(
                                    platformIndex = pIndex,
                                    deviceIndex   = d.getInt("index"),
                                    name          = d.getString("name"),
                                    type          = d.optString("type", ""),
                                    apiVersion    = d.optString("api", "")
                                )
                            )
                        }
                    }
                }
                backends.add(BackendInfo(name, available, devices))
            }
            return BackendCatalog(backends)
        }
    }
}
