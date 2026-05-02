package kr.clpeak

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.recyclerview.widget.LinearLayoutManager
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.tabs.TabLayout
import kr.clpeak.databinding.FragmentResultsBinding

class ResultsFragment : Fragment() {

    private var _binding: FragmentResultsBinding? = null
    private val binding get() = _binding!!

    private val viewModel: BenchmarkViewModel by activityViewModels()
    private lateinit var adapter: ResultAdapter
    private var selectedBackend: String? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentResultsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        adapter = ResultAdapter { backend, testName, category ->
            viewModel.toggleCategory(backend, testName, category)
        }
        binding.recyclerResults.adapter = adapter
        binding.recyclerResults.layoutManager = LinearLayoutManager(requireContext())

        binding.backendTabs.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab) {
                selectedBackend = tab.text?.toString()
                refreshCurrentTab()
            }
            override fun onTabUnselected(tab: TabLayout.Tab) {}
            override fun onTabReselected(tab: TabLayout.Tab) {}
        })

        observeViewModel()
    }

    private fun observeViewModel() {
        viewModel.isRunning.observe(viewLifecycleOwner) { running ->
            binding.progressIndicator.visibility = if (running) View.VISIBLE else View.GONE
            val window = activity?.window ?: return@observe
            if (running) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            else         window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }

        viewModel.backends.observe(viewLifecycleOwner) { list ->
            syncTabs(list)
            refreshCurrentTab()
        }

        viewModel.categoriesByBackend.observe(viewLifecycleOwner) { refreshCurrentTab() }
        viewModel.deviceInfoByBackend.observe(viewLifecycleOwner) { refreshCurrentTab() }

        viewModel.errorMsg.observe(viewLifecycleOwner) { msg ->
            if (msg != null) {
                MaterialAlertDialogBuilder(requireContext())
                    .setTitle("Benchmark error")
                    .setMessage(msg)
                    .setPositiveButton("OK", null)
                    .show()
            }
        }
    }

    private fun syncTabs(backends: List<String>) {
        val tabs = binding.backendTabs
        val currentLabels = (0 until tabs.tabCount).map { tabs.getTabAt(it)?.text?.toString() }
        if (currentLabels == backends) return

        tabs.removeAllTabs()
        for (b in backends) tabs.addTab(tabs.newTab().setText(b), false)
        tabs.visibility = if (backends.isEmpty()) View.GONE else View.VISIBLE

        val target = backends.indexOf(selectedBackend).takeIf { it >= 0 } ?: 0
        if (backends.isNotEmpty()) {
            tabs.getTabAt(target)?.select()
            selectedBackend = backends[target]
        } else {
            selectedBackend = null
        }
    }

    private fun refreshCurrentTab() {
        val backend = selectedBackend
        val cats = viewModel.categoriesByBackend.value?.get(backend) ?: emptyList()
        adapter.submitList(cats.toList())

        val info = viewModel.deviceInfoByBackend.value?.get(backend)
        if (info != null) {
            binding.cardDeviceInfo.visibility = View.VISIBLE
            binding.tvPlatform.text = info.platformName
            binding.tvDevice.text   = info.deviceName
            binding.tvDriver.text   = info.driverVersion
        } else {
            binding.cardDeviceInfo.visibility = View.GONE
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
