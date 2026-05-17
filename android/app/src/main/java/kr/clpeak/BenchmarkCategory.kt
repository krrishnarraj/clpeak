package kr.clpeak

// Visual category used for tonal coloring + section grouping in the UI.
enum class TestType { FP_COMPUTE, INT_COMPUTE, BANDWIDTH, LATENCY, UNKNOWN }

fun testTypeFromCategory(category: String): TestType = when (category) {
    "fp_compute"  -> TestType.FP_COMPUTE
    "int_compute" -> TestType.INT_COMPUTE
    "bandwidth"   -> TestType.BANDWIDTH
    "latency"     -> TestType.LATENCY
    else          -> TestType.UNKNOWN
}

data class BenchmarkCategory(
    val backend: String,
    val testName: String,
    val category: String,
    val unit: String,
    val testType: TestType,
    val metrics: List<ResultEntry>,
    val isExpanded: Boolean = false
) {
    val displayName: String get() = metrics.firstOrNull()?.display ?: testName
    val peakValue: Float get() = metrics.filter { it.status == "ok" }.maxOfOrNull { it.value } ?: 0f
    val allSkipped: Boolean get() = metrics.all { it.status != "ok" }
    val skipReason: String get() = metrics.firstOrNull { it.reason.isNotEmpty() }?.reason ?: ""
    val skipStatus: String get() = metrics.firstOrNull { it.status != "ok" }?.status ?: ""
}
