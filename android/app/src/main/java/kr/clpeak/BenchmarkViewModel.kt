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

    // Accumulate metrics preserving arrival order per (backend, category,
    // test) so the UI cards stay in the sequence they first appeared in.
    // Grouping key includes category because dual-category tests (wmma,
    // simdgroup_matrix, etc.) appear under both fp_compute and int_compute
    // and need to render as two separate cards.
    private val accumulated = mutableListOf<ResultEntry>()
    private val expandedKeys = mutableSetOf<String>()  // "backend|category|test"

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

    fun toggleCategory(backend: String, testName: String, category: String = "") {
        val key = "$backend|$category|$testName"
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
        // Preserve arrival order: build a backend list and per-(category,test)
        // metric list keyed by first-seen position in accumulated.  The same
        // test name (e.g. "wmma") may appear under both fp_compute and
        // int_compute on a single device -- they render as two cards.
        val backendOrder = linkedSetOf<String>()
        val perBackendCards = linkedMapOf<String, LinkedHashMap<Pair<String, String>, MutableList<ResultEntry>>>()

        for (e in accumulated) {
            backendOrder.add(e.backend)
            val cards = perBackendCards.getOrPut(e.backend) { linkedMapOf() }
            cards.getOrPut(e.category to e.test) { mutableListOf() }.add(e)
        }

        val result = linkedMapOf<String, List<BenchmarkCategory>>()
        for (backend in backendOrder) {
            val cards = perBackendCards[backend] ?: continue
            val list = cards.map { (key, entries) ->
                val (category, test) = key
                BenchmarkCategory(
                    testName    = test,
                    displayName = displayNameOf(test),
                    category    = category,
                    unit        = entries.firstOrNull()?.unit ?: "",
                    testType    = testTypeFromCategory(category),
                    metrics     = entries,
                    isExpanded  = expandedKeys.contains("$backend|$category|$test")
                )
            }
            result[backend] = list
        }

        _backends.value = backendOrder.toList()
        _categoriesByBackend.value = result
    }

    companion object {
        // Test-tag -> human label.  Much smaller than the previous
        // CATEGORY_META because category lookup now comes from the C++
        // emitter directly; this table is purely cosmetic.  Unknown tags
        // fall back to a title-cased tag form.
        private val DISPLAY_NAMES = mapOf(
            "global_memory_bandwidth"     to "Global Memory Bandwidth",
            "local_memory_bandwidth"      to "Local Memory Bandwidth",
            "image_memory_bandwidth"      to "Image Memory Bandwidth",
            "transfer_bandwidth"          to "Transfer Bandwidth",
            "single_precision_compute"    to "Single-Precision Compute",
            "double_precision_compute"    to "Double-Precision Compute",
            "half_precision_compute"      to "Half-Precision Compute",
            "mixed_precision_compute"     to "Mixed-Precision Compute (fp16×fp16+fp32)",
            "bfloat16_compute"            to "BF16 Compute (bf16×bf16+fp32)",
            "integer_compute"             to "Integer Compute",
            "integer_compute_fast"        to "Integer Compute (Fast 24-bit)",
            "integer_compute_char"        to "Char (8-bit) Integer Compute",
            "integer_compute_short"       to "Short (16-bit) Integer Compute",
            "integer_compute_int8_dp"     to "INT8 Dot-Product Compute",
            "int4_packed_compute"         to "Packed INT4 Compute (emulated)",
            "wmma"                        to "WMMA (tensor cores)",
            "bmma"                        to "BMMA (binary tensor cores)",
            "coopmat"                     to "Coop-Matrix (tensor cores)",
            "simdgroup_matrix"            to "Simdgroup Matrix",
            "cublas"                      to "cuBLASLt GEMM",
            "mps_gemm"                    to "MPS GEMM",
            "atomic_throughput"           to "Atomic Throughput",
            "kernel_launch_latency"       to "Kernel Launch Latency"
        )

        private fun displayNameOf(test: String): String =
            DISPLAY_NAMES[test] ?: test
                .replace('_', ' ')
                .split(' ')
                .joinToString(" ") { it.replaceFirstChar { c -> c.uppercase() } }
    }
}
