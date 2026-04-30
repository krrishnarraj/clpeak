package kr.clpeak

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.view.setPadding
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import com.google.android.material.checkbox.MaterialCheckBox
import kr.clpeak.databinding.FragmentSetupBinding

class SetupFragment : Fragment() {

    private var _binding: FragmentSetupBinding? = null
    private val binding get() = _binding!!

    private val viewModel: BenchmarkViewModel by activityViewModels()

    // Track checkboxes so we can sync their state when selection changes
    // (e.g. master switch toggled, Select-all/Clear) without rebuilding the whole list.
    private val openclCheckboxes = mutableMapOf<Pair<Int, Int>, MaterialCheckBox>()
    private val vulkanCheckboxes = mutableMapOf<Int, MaterialCheckBox>()

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentSetupBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        binding.fabRun.setOnClickListener { viewModel.runBenchmarks() }
        binding.btnSelectAll.setOnClickListener { viewModel.selectAll() }
        binding.btnClear.setOnClickListener { viewModel.resetSelection() }

        binding.switchOpencl.setOnClickListener {
            viewModel.setOpenClBackendEnabled(binding.switchOpencl.isChecked)
        }
        binding.switchVulkan.setOnClickListener {
            viewModel.setVulkanBackendEnabled(binding.switchVulkan.isChecked)
        }

        viewModel.catalog.observe(viewLifecycleOwner) { catalog ->
            populateOpenCl(catalog.opencl)
            populateVulkan(catalog.vulkan)
            updateEmptyState(catalog)
        }
        viewModel.selection.observe(viewLifecycleOwner) { syncWithSelection(it) }
    }

    private fun populateOpenCl(catalog: OpenClCatalog) {
        binding.openclPlatforms.removeAllViews()
        openclCheckboxes.clear()

        if (catalog.platforms.isEmpty() || catalog.isEmpty) {
            binding.cardOpencl.visibility = View.GONE
            return
        }
        binding.cardOpencl.visibility = View.VISIBLE

        val ctx = requireContext()
        for (platform in catalog.platforms) {
            if (platform.devices.isEmpty()) continue

            if (catalog.platforms.size > 1) {
                val header = TextView(ctx).apply {
                    text = platform.name
                    setTextAppearance(
                        com.google.android.material.R.style.TextAppearance_Material3_LabelMedium
                    )
                    setPadding(dp(4), dp(8), dp(4), dp(2))
                }
                binding.openclPlatforms.addView(header)
            }

            for (device in platform.devices) {
                val cb = MaterialCheckBox(ctx).apply {
                    text = device.name
                    setOnClickListener {
                        viewModel.toggleOpenClDevice(platform.index, device.index)
                    }
                }
                binding.openclPlatforms.addView(
                    cb,
                    LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    )
                )
                openclCheckboxes[platform.index to device.index] = cb
            }
        }
    }

    private fun populateVulkan(catalog: VulkanCatalog) {
        binding.vulkanDevices.removeAllViews()
        vulkanCheckboxes.clear()

        if (catalog.devices.isEmpty()) {
            binding.cardVulkan.visibility = View.GONE
            return
        }
        binding.cardVulkan.visibility = View.VISIBLE

        val ctx = requireContext()
        for (device in catalog.devices) {
            val cb = MaterialCheckBox(ctx).apply {
                text = device.name
                setOnClickListener { viewModel.toggleVulkanDevice(device.index) }
            }
            binding.vulkanDevices.addView(
                cb,
                LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                )
            )
            vulkanCheckboxes[device.index] = cb
        }
    }

    private fun updateEmptyState(catalog: BackendCatalog) {
        val anyDevice = catalog.hasAnyDevice
        binding.emptyMessage.visibility = if (anyDevice) View.GONE else View.VISIBLE
        binding.fabRun.isEnabled = anyDevice
    }

    private fun syncWithSelection(selection: BenchmarkSelection) {
        for ((key, cb) in openclCheckboxes) {
            cb.isChecked = key in selection.openclDevices
        }
        for ((idx, cb) in vulkanCheckboxes) {
            cb.isChecked = idx in selection.vulkanDevices
        }
        binding.switchOpencl.isChecked = selection.openclDevices.isNotEmpty()
        binding.switchVulkan.isChecked = selection.vulkanDevices.isNotEmpty()
    }

    private fun dp(value: Int): Int =
        (value * resources.displayMetrics.density).toInt()

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
        openclCheckboxes.clear()
        vulkanCheckboxes.clear()
    }
}
