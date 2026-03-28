package kr.clpeak

enum class TestType { BANDWIDTH, COMPUTE, LATENCY }

data class BenchmarkCategory(
    val testName: String,
    val displayName: String,
    val unit: String,
    val testType: TestType,
    val metrics: List<ResultEntry>,
    val isExpanded: Boolean = false
) {
    val peakValue: Float get() = metrics.maxOfOrNull { it.value } ?: 0f
}
