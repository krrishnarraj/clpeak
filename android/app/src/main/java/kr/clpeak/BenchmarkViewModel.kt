package kr.clpeak

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class BenchmarkViewModel : ViewModel() {

    private val _isRunning  = MutableLiveData(false)
    private val _deviceInfo = MutableLiveData<DeviceInfo?>(null)
    private val _categories = MutableLiveData<List<BenchmarkCategory>>(emptyList())
    private val _errorMsg   = MutableLiveData<String?>(null)

    val isRunning:  LiveData<Boolean>                 = _isRunning
    val deviceInfo: LiveData<DeviceInfo?>             = _deviceInfo
    val categories: LiveData<List<BenchmarkCategory>> = _categories
    val errorMsg:   LiveData<String?>                 = _errorMsg

    private val accumulated = mutableListOf<ResultEntry>()

    fun runBenchmarks() {
        if (_isRunning.value == true) return

        _isRunning.value = true
        _deviceInfo.value = null
        _categories.value = emptyList()
        _errorMsg.value = null
        accumulated.clear()

        val repo = BenchmarkRepository()

        viewModelScope.launch {
            // Collect structured metrics on the main thread
            val metricJob = launch {
                for (entry in repo.metricChannel) {
                    accumulated.add(entry)
                    if (_deviceInfo.value == null) {
                        _deviceInfo.value = DeviceInfo(entry.platform, entry.device, entry.driver)
                    }
                    _categories.value = buildCategories()
                }
            }

            // Run the benchmark on an IO thread (blocking JNI call)
            val result = withContext(Dispatchers.IO) { repo.runBenchmark() }

            // Wait for the metric collector to drain the channel
            metricJob.join()

            if (result != 0) {
                _errorMsg.value = "Benchmark exited with error ($result). " +
                    "Check that an OpenCL library is available."
            }
            _isRunning.value = false
            _categories.value = buildCategories()
        }
    }

    fun toggleCategory(testName: String) {
        _categories.value = _categories.value?.map { cat ->
            if (cat.testName == testName) cat.copy(isExpanded = !cat.isExpanded)
            else cat
        }
    }

    private fun buildCategories(): List<BenchmarkCategory> {
        return accumulated
            .groupBy { it.test }
            .entries
            .sortedBy { CATEGORY_ORDER.indexOf(it.key).let { i -> if (i < 0) Int.MAX_VALUE else i } }
            .map { (test, entries) ->
                val meta = CATEGORY_META[test] ?: CategoryMeta(
                    test.replace('_', ' ').replaceFirstChar { it.uppercase() },
                    TestType.COMPUTE
                )
                BenchmarkCategory(
                    testName    = test,
                    displayName = meta.displayName,
                    unit        = entries.firstOrNull()?.unit ?: "",
                    testType    = meta.type,
                    metrics     = entries,
                    isExpanded  = _categories.value?.find { it.testName == test }?.isExpanded ?: false
                )
            }
    }

    private data class CategoryMeta(val displayName: String, val type: TestType)

    companion object {
        private val CATEGORY_ORDER = listOf(
            "global_memory_bandwidth",
            "local_memory_bandwidth",
            "image_memory_bandwidth",
            "transfer_bandwidth",
            "single_precision_compute",
            "double_precision_compute",
            "half_precision_compute",
            "integer_compute",
            "integer_24bit_compute",
            "char_compute",
            "short_compute",
            "atomic_throughput",
            "kernel_launch_latency"
        )

        private val CATEGORY_META = mapOf(
            "global_memory_bandwidth"  to CategoryMeta("Global Memory Bandwidth",   TestType.BANDWIDTH),
            "local_memory_bandwidth"   to CategoryMeta("Local Memory Bandwidth",    TestType.BANDWIDTH),
            "image_memory_bandwidth"   to CategoryMeta("Image Memory Bandwidth",    TestType.BANDWIDTH),
            "transfer_bandwidth"       to CategoryMeta("Transfer Bandwidth",        TestType.BANDWIDTH),
            "single_precision_compute" to CategoryMeta("Single-Precision Compute",  TestType.COMPUTE),
            "double_precision_compute" to CategoryMeta("Double-Precision Compute",  TestType.COMPUTE),
            "half_precision_compute"   to CategoryMeta("Half-Precision Compute",    TestType.COMPUTE),
            "integer_compute"          to CategoryMeta("Integer Compute",           TestType.COMPUTE),
            "integer_24bit_compute"    to CategoryMeta("Integer 24-bit Compute",    TestType.COMPUTE),
            "char_compute"             to CategoryMeta("Char Compute",              TestType.COMPUTE),
            "short_compute"            to CategoryMeta("Short Compute",             TestType.COMPUTE),
            "atomic_throughput"        to CategoryMeta("Atomic Throughput",         TestType.COMPUTE),
            "kernel_launch_latency"    to CategoryMeta("Kernel Launch Latency",     TestType.LATENCY)
        )
    }
}
