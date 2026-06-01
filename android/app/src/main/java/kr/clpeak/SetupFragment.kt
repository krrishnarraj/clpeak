package kr.clpeak

import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.view.setPadding
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import com.google.android.material.card.MaterialCardView
import com.google.android.material.checkbox.MaterialCheckBox
import com.google.android.material.materialswitch.MaterialSwitch
import kr.clpeak.databinding.FragmentSetupBinding

class SetupFragment : Fragment() {

    private var _binding: FragmentSetupBinding? = null
    private val binding get() = _binding!!

    private val viewModel: BenchmarkViewModel by activityViewModels()

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

        viewModel.catalog.observe(viewLifecycleOwner) { catalog ->
            populateBackends(catalog)
            updateEmptyState(catalog)
        }
        viewModel.selection.observe(viewLifecycleOwner) { syncSelection(it) }
    }

    // ---- Dynamic backend cards -----------------------------------------------

    private fun populateBackends(catalog: BackendCatalog) {
        binding.backendContainer.removeAllViews()

        val ctx = requireContext()
        for (backend in catalog.backends) {
            if (backend.devices.isEmpty()) continue

            val card = buildBackendCard(ctx, backend)
            binding.backendContainer.addView(card)
        }
    }

    private fun buildBackendCard(ctx: Context, backend: BackendInfo): MaterialCardView {
        val density = ctx.resources.displayMetrics.density

        val card = MaterialCardView(ctx).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                bottomMargin = (12 * density).toInt()
            }
            radius = (20 * density)
            cardElevation = 0f
        }

        val cardInner = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            setPadding((16 * density).toInt(), (16 * density).toInt(),
                       (16 * density).toInt(), (16 * density).toInt())
        }

        // Header row: label + master switch
        val header = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
        }

        val title = TextView(ctx).apply {
            text = backend.name
            setTextAppearance(
                com.google.android.material.R.style.TextAppearance_Material3_TitleMedium
            )
            layoutParams = LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                1f
            )
        }
        header.addView(title)

        // Master switch for the backend
        val masterSwitch = MaterialSwitch(ctx).apply {
            tag = "switch|${backend.name}"
            setOnCheckedChangeListener { _, isChecked ->
                viewModel.setBackendEnabled(backend.name, isChecked)
            }
        }
        header.addView(masterSwitch)
        cardInner.addView(header)

        // Device checkboxes
        val deviceContainer = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, (8 * density).toInt(), 0, 0)
            tag = "devices|${backend.name}"
        }
        for (device in backend.devices) {
            val cb = MaterialCheckBox(ctx).apply {
                text = device.name
                tag = "cb|${backend.name}|${device.key}"
                setOnClickListener {
                    viewModel.toggleDevice(backend.name, device.key)
                }
            }
            deviceContainer.addView(cb)
        }
        cardInner.addView(deviceContainer)
        card.addView(cardInner)

        return card
    }

    private fun syncSelection(selection: BenchmarkSelection) {
        val container = binding.backendContainer
        for (i in 0 until container.childCount) {
            val card = container.getChildAt(i) as? MaterialCardView ?: continue
            syncCardSelection(card, selection)
        }
    }

    private fun syncCardSelection(card: MaterialCardView, selection: BenchmarkSelection) {
        val inner = card.getChildAt(0) as? LinearLayout ?: return

        // Header row (index 0) contains the master switch
        val header = inner.getChildAt(0) as? LinearLayout ?: return
        val masterSwitch = findViewByTag(header, MaterialSwitch::class.java) ?: return
        val backendName = (masterSwitch.tag as? String)?.removePrefix("switch|") ?: return

        // Device container (index 1)
        val deviceContainer = inner.getChildAt(1) as? LinearLayout ?: return
        var anyChecked = false
        for (j in 0 until deviceContainer.childCount) {
            val cb = deviceContainer.getChildAt(j) as? MaterialCheckBox ?: continue
            val tag = cb.tag as? String ?: continue
            val parts = tag.split("|", limit = 3)
            if (parts.size < 3) continue
            val cbBackend = parts[1]
            val deviceKey = parts[2]
            if (cbBackend != backendName) continue
            val checked = selection.isSelected(backendName, deviceKey)
            cb.isChecked = checked
            if (checked) anyChecked = true
        }

        masterSwitch.setOnCheckedChangeListener(null)
        masterSwitch.isChecked = anyChecked
        masterSwitch.setOnCheckedChangeListener { _, isChecked ->
            viewModel.setBackendEnabled(backendName, isChecked)
        }
    }

    // ---- Helpers -------------------------------------------------------------

    private fun <T : View> findViewByTag(parent: ViewGroup, clazz: Class<T>): T? {
        for (i in 0 until parent.childCount) {
            val child = parent.getChildAt(i)
            if (clazz.isInstance(child)) {
                @Suppress("UNCHECKED_CAST")
                return child as T
            }
            if (child is ViewGroup) {
                val found = findViewByTag(child, clazz)
                if (found != null) return found
            }
        }
        return null
    }

    private fun updateEmptyState(catalog: BackendCatalog) {
        val anyDevice = catalog.hasAnyDevice
        binding.emptyMessage.visibility = if (anyDevice) View.GONE else View.VISIBLE
        binding.fabRun.isEnabled = anyDevice
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
