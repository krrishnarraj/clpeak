package kr.clpeak

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class BenchmarkViewModel : ViewModel() {

    private val _isRunning       = MutableLiveData(false)
    private val _deviceInfoByBackend = MutableLiveData<Map<String, DeviceInfo>>(emptyMap())
    private val _backends        = MutableLiveData<List<String>>(emptyList())
    private val _categoriesByBackend = MutableLiveData<Map<String, List<BenchmarkCategory>>>(emptyMap())
    private val _errorMsg        = MutableLiveData<String?>(null)

    val isRunning:  LiveData<Boolean>                          = _isRunning
    val deviceInfoByBackend: LiveData<Map<String, DeviceInfo>> = _deviceInfoByBackend
    val backends:   LiveData<List<String>>                     = _backends
    val categoriesByBackend: LiveData<Map<String, List<BenchmarkCategory>>> = _categoriesByBackend
    val errorMsg:   LiveData<String?>                          = _errorMsg

    // Accumulate metrics preserving arrival order per (backend, test) so the
    // UI cards stay in the sequence they first appeared in.
    private val accumulated = mutableListOf<ResultEntry>()
    private val expandedKeys = mutableSetOf<String>()  // "backend|test"

    fun runBenchmarks() {
        if (_isRunning.value == true) return

        _isRunning.value = true
        _deviceInfoByBackend.value = emptyMap()
        _backends.value = emptyList()
        _categoriesByBackend.value = emptyMap()
        _errorMsg.value = null
        accumulated.clear()
        expandedKeys.clear()

        val repo = BenchmarkRepository()

        viewModelScope.launch {
            val metricJob = launch {
                for (entry in repo.metricChannel) {
                    accumulated.add(entry)
                    updateDeviceInfo(entry)
                    rebuild()
                }
            }

            val result = withContext(Dispatchers.IO) { repo.runBenchmark() }
            metricJob.join()

            if (result != 0) {
                _errorMsg.value = "Benchmark exited with error ($result). " +
                    "Check that an OpenCL library is available."
            }
            _isRunning.value = false
            rebuild()
        }
    }

    fun toggleCategory(backend: String, testName: String) {
        val key = "$backend|$testName"
        if (!expandedKeys.add(key)) expandedKeys.remove(key)
        rebuild()
    }

    private fun updateDeviceInfo(entry: ResultEntry) {
        val current = _deviceInfoByBackend.value ?: emptyMap()
        if (!current.containsKey(entry.backend)) {
            _deviceInfoByBackend.value = current + (entry.backend to
                DeviceInfo(entry.platform, entry.device, entry.driver))
        }
    }

    private fun rebuild() {
        // Preserve arrival order: build a backend list and per-backend test list
        // by first-seen position in accumulated.
        val backendOrder = linkedSetOf<String>()
        val perBackendTests = linkedMapOf<String, LinkedHashMap<String, MutableList<ResultEntry>>>()

        for (e in accumulated) {
            backendOrder.add(e.backend)
            val tests = perBackendTests.getOrPut(e.backend) { linkedMapOf() }
            tests.getOrPut(e.test) { mutableListOf() }.add(e)
        }

        val result = linkedMapOf<String, List<BenchmarkCategory>>()
        for (backend in backendOrder) {
            val tests = perBackendTests[backend] ?: continue
            val cats = tests.map { (test, entries) ->
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
                    isExpanded  = expandedKeys.contains("$backend|$test")
                )
            }
            result[backend] = cats
        }

        _backends.value = backendOrder.toList()
        _categoriesByBackend.value = result
    }

    private data class CategoryMeta(val displayName: String, val type: TestType)

    companion object {
        private val CATEGORY_META = mapOf(
            "global_memory_bandwidth"   to CategoryMeta("Global Memory Bandwidth",   TestType.BANDWIDTH),
            "local_memory_bandwidth"    to CategoryMeta("Local Memory Bandwidth",    TestType.BANDWIDTH),
            "image_memory_bandwidth"    to CategoryMeta("Image Memory Bandwidth",    TestType.BANDWIDTH),
            "transfer_bandwidth"        to CategoryMeta("Transfer Bandwidth",        TestType.BANDWIDTH),
            "single_precision_compute"  to CategoryMeta("Single-Precision Compute",  TestType.COMPUTE),
            "double_precision_compute"  to CategoryMeta("Double-Precision Compute",  TestType.COMPUTE),
            "half_precision_compute"    to CategoryMeta("Half-Precision Compute",    TestType.COMPUTE),
            "integer_compute"           to CategoryMeta("Integer Compute",           TestType.COMPUTE),
            "integer_24bit_compute"     to CategoryMeta("Integer 24-bit Compute",    TestType.COMPUTE),
            "char_compute"              to CategoryMeta("Char Compute",              TestType.COMPUTE),
            "short_compute"             to CategoryMeta("Short Compute",             TestType.COMPUTE),
            "integer_compute_int8_dp"   to CategoryMeta("INT8 Dot-Product Compute",  TestType.COMPUTE),
            "atomic_throughput"         to CategoryMeta("Atomic Throughput",         TestType.COMPUTE),
            "kernel_launch_latency"     to CategoryMeta("Kernel Launch Latency",     TestType.LATENCY)
        )
    }
}
