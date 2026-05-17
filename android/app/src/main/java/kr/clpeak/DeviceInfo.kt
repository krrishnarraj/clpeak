package kr.clpeak

data class DeviceInfo(
    val backend: String,
    val platformName: String,
    val deviceName: String,
    val driverVersion: String,
    val propsJson: String,       // [{"k":"Compute units","v":"16"},...]
    val platformIndex: Int = -1,
    val deviceIndex: Int = -1
) {
    /** Parse propsJson into a list of key-value pairs for display. */
    fun props(): List<Pair<String, String>> {
        val result = mutableListOf<Pair<String, String>>()
        // Simple parser for [{"k":"...","v":"..."},...]
        val regex = Regex("""\{"k":"([^"]*)","v":"([^"]*)"\}""")
        for (match in regex.findAll(propsJson)) {
            result.add(match.groupValues[1] to match.groupValues[2])
        }
        return result
    }
}
