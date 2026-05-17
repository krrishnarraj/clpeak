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

    private val _isRunning          = MutableLiveData(false)
    private val _deviceInfoByBackend = MutableLiveData<Map<String, DeviceInfo>>(emptyMap())
    private val _backends            = MutableLiveData<List<String>>(emptyList())
    private val _categoriesByBackend = MutableLiveData<Map<String, List<BenchmarkCategory>>>(emptyMap())
    private val _errorMsg            = MutableLiveData<String?>(null)
    private val _catalog             = MutableLiveData(BackendCatalog.EMPTY)
    private val _selection           = MutableLiveData(BenchmarkSelection(emptySet(), emptySet()))
    private val _screen              = MutableLiveData(Screen.SETUP)

    val isRunning:  LiveData<Boolean>                          = _isRunning
    val deviceInfoByBackend: LiveData<Map<String, DeviceInfo>> = _deviceInfoByBackend
    val backends:   LiveData<List<String>>                     = _backends
    val categoriesByBackend: LiveData<Map<String, List<BenchmarkCategory>>> = _categoriesByBackend
    val errorMsg:   LiveData<String?>                          = _errorMsg
    val catalog:    LiveData<BackendCatalog>                   = _catalog
    val selection:  LiveData<BenchmarkSelection>               = _selection
    val screen:     LiveData<Screen>                           = _screen

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
            // Collect device info as it arrives
            val deviceJob = launch {
                for (info in repo.deviceInfoChannel) {
                    val current = _deviceInfoByBackend.value ?: emptyMap()
                    _deviceInfoByBackend.value = current + (info.backend to info)
                }
            }

            // Collect metrics
            val metricJob = launch {
                for (entry in repo.metricChannel) {
                    accumulated.add(entry)
                    rebuild()
                }
            }

            val result = withContext(Dispatchers.IO) { repo.runBenchmark(argv) }
            deviceJob.join()
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

    private fun rebuild() {
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
                    backend     = backend,
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
        private val DISPLAY_NAMES = mapOf(
            "global_memory_bandwidth"     to "Global Memory Bandwidth",
            "local_memory_bandwidth"      to "Local Memory Bandwidth",
            "image_memory_bandwidth"      to "Image Memory Bandwidth",
            "transfer_bandwidth"          to "Transfer Bandwidth",
            "single_precision_compute"    to "Single-Precision Compute",
            "double_precision_compute"    to "Double-Precision Compute",
            "half_precision_compute"      to "Half-Precision Compute",
            "mixed_precision_compute"     to "Mixed-Precision Compute",
            "bfloat16_compute"            to "BF16 Compute",
            "integer_compute"             to "Integer Compute",
            "integer_compute_fast"        to "Integer Compute (Fast 24-bit)",
            "integer_compute_char"        to "Char (8-bit) Integer Compute",
            "integer_compute_short"       to "Short (16-bit) Integer Compute",
            "integer_compute_int8_dp"     to "INT8 Dot-Product Compute",
            "int4_packed_compute"         to "Packed INT4 Compute",
            "wmma"                        to "WMMA",
            "bmma"                        to "BMMA",
            "coopmat"                     to "Cooperative Matrix",
            "simdgroup_matrix"            to "Simdgroup Matrix",
            "cublas"                      to "cuBLASLt GEMM",
            "mps_gemm"                    to "MPS GEMM",
            "atomic_throughput"           to "Atomic Throughput",
            "kernel_launch_latency"       to "Kernel Launch Latency",
            "wmma_fp16"                   to "WMMA fp16×fp16+fp32",
            "wmma_bf16"                   to "WMMA bf16×bf16+fp32",
            "wmma_tf32"                   to "WMMA tf32×tf32+fp32",
            "wmma_fp64"                   to "WMMA fp64×fp64+fp64",
            "wmma_fp8_e4m3"              to "FP8(E4M3) mma.sync",
            "wmma_fp8_e5m2"              to "FP8(E5M2) mma.sync",
            "wmma_fp4_e2m1"              to "FP4(E2M1) mma.sync",
            "wmma_mxf4_e2m1"             to "MXFP4(E2M1) mma.sync",
            "wmma_int8"                   to "WMMA int8×int8+int32",
            "wmma_int8_k32"              to "INT8 mma.sync K=32",
            "wmma_int8_sparse"           to "INT8 mma.sp 2:4",
            "wmma_int4"                   to "INT4 mma.sync",
            "wmma_bmma_b1"               to "BMMA b1 xor.popc",
            "cublas-fp"                   to "cuBLASLt GEMM",
            "cublas-int"                  to "cuBLASLt GEMM",
            "mps-gemm-fp"                 to "MPS GEMM",
            "mps-gemm-int"                to "MPS GEMM"
        )

        private fun displayNameOf(test: String): String =
            DISPLAY_NAMES[test] ?: test
                .replace('_', ' ')
                .split(' ')
                .joinToString(" ") { it.replaceFirstChar { c -> c.uppercase() } }
    }
}
