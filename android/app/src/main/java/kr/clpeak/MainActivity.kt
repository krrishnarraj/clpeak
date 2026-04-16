package kr.clpeak

import android.content.Intent
import android.content.res.ColorStateList
import android.net.Uri
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.view.WindowManager
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import androidx.recyclerview.widget.LinearLayoutManager
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import kr.clpeak.databinding.ActivityMainBinding
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val viewModel: BenchmarkViewModel by viewModels()
    private lateinit var adapter: ResultAdapter

    // Parallel lists: display names and their library paths
    private val openclDisplayNames = mutableListOf<String>()
    private val openclLibPaths     = mutableListOf<String>()
    private var selectedLibIndex   = 0

    private external fun nativeSetenv(key: String, value: String)

    companion object {
        init { System.loadLibrary("clpeak") }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)

        // Apply window insets for edge-to-edge
        ViewCompat.setOnApplyWindowInsetsListener(binding.coordinator) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            binding.fabRun.updatePadding(bottom = bars.bottom)
            // Let CoordinatorLayout handle the rest
            insets
        }

        adapter = ResultAdapter { testName -> viewModel.toggleCategory(testName) }
        binding.recyclerResults.adapter = adapter
        binding.recyclerResults.layoutManager = LinearLayoutManager(this)

        buildOpenclLibraryList()
        if (openclLibPaths.isNotEmpty()) {
            nativeSetenv("LIBOPENCL_SO_PATH", openclLibPaths[0])
        }

        binding.fabRun.setOnClickListener {
            if (viewModel.isRunning.value == true) return@setOnClickListener
            viewModel.runBenchmarks()
        }

        observeViewModel()
    }

    private fun observeViewModel() {
        viewModel.isRunning.observe(this) { running ->
            updateFabState(running)
            binding.progressIndicator.visibility = if (running) View.VISIBLE else View.GONE
            if (running) {
                binding.idleState.visibility = View.GONE
                window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            } else {
                window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
        }

        viewModel.deviceInfo.observe(this) { info ->
            if (info != null) {
                binding.cardDeviceInfo.visibility = View.VISIBLE
                binding.tvPlatform.text = info.platformName
                binding.tvDevice.text   = info.deviceName
                binding.tvDriver.text   = info.driverVersion
            }
        }

        viewModel.categories.observe(this) { cats ->
            adapter.submitList(cats.toList())
        }

        viewModel.errorMsg.observe(this) { msg ->
            if (msg != null) {
                MaterialAlertDialogBuilder(this)
                    .setTitle("Benchmark error")
                    .setMessage(msg)
                    .setPositiveButton("OK", null)
                    .show()
            }
        }
    }

    private fun updateFabState(running: Boolean) {
        if (running) {
            binding.fabRun.text = getString(R.string.stop_button_text)
            binding.fabRun.setIconResource(R.drawable.ic_stop)
            binding.fabRun.backgroundTintList =
                ColorStateList.valueOf(getColor(R.color.fab_stop))
            binding.fabRun.setTextColor(getColor(R.color.fab_on_stop))
            binding.fabRun.iconTint =
                ColorStateList.valueOf(getColor(R.color.fab_on_stop))
        } else {
            binding.fabRun.text = getString(R.string.run_button_text)
            binding.fabRun.setIconResource(R.drawable.ic_play_arrow)
            binding.fabRun.backgroundTintList =
                ColorStateList.valueOf(getColor(R.color.md_theme_primary))
            binding.fabRun.setTextColor(getColor(R.color.md_theme_on_primary))
            binding.fabRun.iconTint =
                ColorStateList.valueOf(getColor(R.color.md_theme_on_primary))
        }
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.main_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.menu_opencl_lib -> { showOpenclLibraryPicker(); true }
            R.id.menu_about      -> { showAbout(); true }
            else                 -> super.onOptionsItemSelected(item)
        }
    }

    private fun showAbout() {
        AboutBottomSheet().show(supportFragmentManager, AboutBottomSheet.TAG)
    }

    private fun buildOpenclLibraryList() {
        val allPaths = listOf(
            "/vendor/lib64/libOpenCL.so"              to "vendor lib64",
            "/system/lib64/libOpenCL.so"              to "system lib64",
            "/system/vendor/lib64/libOpenCL.so"       to "system vendor lib64",
            "/system/lib/libOpenCL.so"                to "system lib",
            "/system/vendor/lib/libOpenCL.so"         to "system vendor lib",
            "/system/vendor/lib64/egl/libGLES_mali.so" to "Mali",
            "/system/vendor/lib/egl/libGLES_mali.so"  to "Mali (32-bit)",
            "/system/vendor/lib/libPVROCL.so"         to "PowerVR",
            "/data/data/org.pocl.libs/files/lib/libpocl.so" to "POCL"
        )

        for ((path, label) in allPaths) {
            if (File(path).exists()) {
                openclLibPaths.add(path)
                openclDisplayNames.add(label)
            }
        }

        // Always include the system default
        openclLibPaths.add("libOpenCL.so")
        openclDisplayNames.add("Default")
    }

    private fun showOpenclLibraryPicker() {
        if (openclLibPaths.isEmpty()) return

        val items = openclDisplayNames.toTypedArray()
        val pocl = openclDisplayNames.indexOf("POCL")

        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.menu_opencl_lib)
            .setSingleChoiceItems(items, selectedLibIndex) { dialog, which ->
                // POCL special case: prompt to install if library file is missing
                if (which == pocl && !File(openclLibPaths[which]).exists()) {
                    dialog.dismiss()
                    MaterialAlertDialogBuilder(this)
                        .setTitle("POCL not installed")
                        .setMessage("Install POCL from the Play Store?")
                        .setPositiveButton("Install") { _, _ ->
                            startActivity(
                                Intent(Intent.ACTION_VIEW,
                                    Uri.parse("market://details?id=org.pocl.libs"))
                            )
                        }
                        .setNegativeButton("Cancel", null)
                        .show()
                    return@setSingleChoiceItems
                }
                selectedLibIndex = which
                nativeSetenv("LIBOPENCL_SO_PATH", openclLibPaths[which])
                dialog.dismiss()
            }
            .show()
    }
}
