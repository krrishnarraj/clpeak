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
    private val _selection           = MutableLiveData(BenchmarkSelection())
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

    fun toggleDevice(backend: String, deviceKey: String) {
        _selection.value = (_selection.value ?: BenchmarkSelection()).toggleDevice(backend, deviceKey)
    }

    fun setBackendEnabled(backend: String, enabled: Boolean) {
        val cat = _catalog.value ?: return
        val info = cat.backend(backend) ?: return
        _selection.value = (_selection.value ?: BenchmarkSelection())
            .setBackend(backend, enabled, info.devices.map { it.key })
    }

    fun selectAll() {
        _selection.value = BenchmarkSelection.allOf(_catalog.value ?: BackendCatalog.EMPTY)
    }

    fun resetSelection() {
        _selection.value = BenchmarkSelection()
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
                    "Check that an OpenCL or Vulkan library is available."
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
}
