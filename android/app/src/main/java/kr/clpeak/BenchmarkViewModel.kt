package kr.clpeak

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class BenchmarkViewModel : ViewModel() {

    enum class Screen { SETUP, RESULTS }

    private val _isRunning       = MutableLiveData(false)
    private val _deviceInfoByBackend = MutableLiveData<Map<String, DeviceInfo>>(emptyMap())
    private val _backends        = MutableLiveData<List<String>>(emptyList())
    private val _categoriesByBackend = MutableLiveData<Map<String, List<BenchmarkCategory>>>(emptyMap())
    private val _errorMsg        = MutableLiveData<String?>(null)
    private val _catalog         = MutableLiveData(BackendCatalog.EMPTY)
    private val _selection       = MutableLiveData(BenchmarkSelection(emptySet(), emptySet()))
    private val _screen          = MutableLiveData(Screen.SETUP)

    val isRunning:  LiveData<Boolean>                          = _isRunning
    val deviceInfoByBackend: LiveData<Map<String, DeviceInfo>> = _deviceInfoByBackend
    val backends:   LiveData<List<String>>                     = _backends
    val categoriesByBackend: LiveData<Map<String, List<BenchmarkCategory>>> = _categoriesByBackend
    val errorMsg:   LiveData<String?>                          = _errorMsg
    val catalog:    LiveData<BackendCatalog>                   = _catalog
    val selection:  LiveData<BenchmarkSelection>               = _selection
    val screen:     LiveData<Screen>                           = _screen

    // Accumulate metrics preserving arrival order per (backend, test) so the
    // UI cards stay in the sequence they first appeared in.
    private val accumulated = mutableListOf<ResultEntry>()
    private val expandedKeys = mutableSetOf<String>()  // "backend|test"

    init {
        loadCatalog()
    }

    private fun loadCatalog() {
        viewModelScope.launch {
            val cat = withContext(Dispatchers.IO) {
                runCatching { BackendCatalog.fromJson(BenchmarkRepository().nativeEnumerateBackends()) }
                    .getOrDefault(BackendCatalog.EMPTY)
            }
            _catalog.value = cat
            // Pre-select everything so first-time users get the existing
            // "run all backends/devices" behavior with one tap.
            _selection.value = BenchmarkSelection.allOf(cat)
        }
    }

    fun toggleOpenClDevice(platform: Int, device: Int) {
        _selection.value = (_selection.value ?: BenchmarkSelection(emptySet(), emptySet()))
            .toggleOpenCl(platform, device)
    }

    fun toggleVulkanDevice(device: Int) {
        _selection.value = (_selection.value ?: BenchmarkSelection(emptySet(), emptySet()))
            .toggleVulkan(device)
    }

    fun setOpenClBackendEnabled(on: Boolean) {
        val cat = _catalog.value ?: return
        _selection.value = (_selection.value ?: BenchmarkSelection(emptySet(), emptySet()))
            .setOpenClBackend(cat.opencl, on)
    }

    fun setVulkanBackendEnabled(on: Boolean) {
        val cat = _catalog.value ?: return
        _selection.value = (_selection.value ?: BenchmarkSelection(emptySet(), emptySet()))
            .setVulkanBackend(cat.vulkan, on)
    }

    fun selectAll() {
        _selection.value = BenchmarkSelection.allOf(_catalog.value ?: BackendCatalog.EMPTY)
    }

    fun resetSelection() {
        _selection.value = BenchmarkSelection(emptySet(), emptySet())
    }

    fun returnToSetup() {
        if (_isRunning.value == true) return
        _screen.value = Screen.SETUP
    }

    fun runBenchmarks() {
        if (_isRunning.value == true) return
        val sel = _selection.value ?: return
        if (sel.isEmpty) {
            _errorMsg.value = "Select at least one device to benchmark."
            return
        }
        val cat = _catalog.value ?: BackendCatalog.EMPTY

        _isRunning.value = true
        _deviceInfoByBackend.value = emptyMap()
        _backends.value = emptyList()
        _categoriesByBackend.value = emptyMap()
        _errorMsg.value = null
        accumulated.clear()
        expandedKeys.clear()
        _screen.value = Screen.RESULTS

        val argv = ArgvBuilder.build(sel, cat)
        val repo = BenchmarkRepository()

        viewModelScope.launch {
            val metricJob = launch {
                for (entry in repo.metricChannel) {
                    accumulated.add(entry)
                    updateDeviceInfo(entry)
                    rebuild()
                }
            }

            val result = withContext(Dispatchers.IO) { repo.runBenchmark(argv) }
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
            "mixed_precision_compute"   to CategoryMeta("Mixed-Precision Compute (fp16×fp16+fp32)", TestType.COMPUTE),
            "integer_compute"           to CategoryMeta("Integer Compute",           TestType.COMPUTE),
            "integer_24bit_compute"     to CategoryMeta("Integer 24-bit Compute",    TestType.COMPUTE),
            "char_compute"              to CategoryMeta("Char Compute",              TestType.COMPUTE),
            "short_compute"             to CategoryMeta("Short Compute",             TestType.COMPUTE),
            "integer_compute_int8_dp"   to CategoryMeta("INT8 Dot-Product Compute",  TestType.COMPUTE),
            "int4_packed_compute"       to CategoryMeta("Packed INT4 Compute (emulated)", TestType.COMPUTE),
            "bfloat16_compute"          to CategoryMeta("BF16 Compute (bf16×bf16+fp32)", TestType.COMPUTE),
            "coopmat_fp16"              to CategoryMeta("Coop-Matrix FP16 (tensor cores)", TestType.COMPUTE),
            "coopmat_bf16"              to CategoryMeta("Coop-Matrix BF16 (tensor cores)", TestType.COMPUTE),
            "coopmat_int8"              to CategoryMeta("Coop-Matrix INT8 (tensor cores)", TestType.COMPUTE),
            "coopmat_fp8_e4m3"          to CategoryMeta("Coop-Matrix FP8 E4M3 (tensor cores)", TestType.COMPUTE),
            "coopmat_fp8_e5m2"          to CategoryMeta("Coop-Matrix FP8 E5M2 (tensor cores)", TestType.COMPUTE),
            "atomic_throughput"         to CategoryMeta("Atomic Throughput",         TestType.COMPUTE),
            "kernel_launch_latency"     to CategoryMeta("Kernel Launch Latency",     TestType.LATENCY)
        )
    }
}
