package kr.clpeak

import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import androidx.activity.OnBackPressedCallback
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import kr.clpeak.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val viewModel: BenchmarkViewModel by viewModels()

    // Consumes back presses while a benchmark is running. Native code has no
    // cooperative cancel, so once a run starts the user must wait for it
    // to finish — there is nothing useful to navigate to.
    private val blockBackWhileRunning = object : OnBackPressedCallback(false) {
        override fun handleOnBackPressed() { /* swallow */ }
    }

    companion object {
        init { System.loadLibrary("clpeak") }
        private const val TAG_RESULTS = "results"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)

        ViewCompat.setOnApplyWindowInsetsListener(binding.coordinator) { _, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            binding.fragmentContainer.updatePadding(bottom = bars.bottom)
            insets
        }

        if (savedInstanceState == null) {
            supportFragmentManager.beginTransaction()
                .replace(R.id.fragment_container, SetupFragment())
                .commit()
        }

        supportFragmentManager.addOnBackStackChangedListener { updateUpAffordance() }
        onBackPressedDispatcher.addCallback(this, blockBackWhileRunning)

        viewModel.screen.observe(this) { screen ->
            when (screen) {
                BenchmarkViewModel.Screen.RESULTS -> showResultsIfNotShown()
                BenchmarkViewModel.Screen.SETUP   -> popToSetup()
                null -> {}
            }
        }

        viewModel.isRunning.observe(this) { running ->
            blockBackWhileRunning.isEnabled = running
            updateUpAffordance()
        }
    }

    private fun showResultsIfNotShown() {
        if (supportFragmentManager.findFragmentByTag(TAG_RESULTS) != null) return
        supportFragmentManager.beginTransaction()
            .replace(R.id.fragment_container, ResultsFragment(), TAG_RESULTS)
            .addToBackStack(TAG_RESULTS)
            .commit()
        binding.toolbar.title = getString(R.string.results_title)
    }

    private fun popToSetup() {
        if (supportFragmentManager.backStackEntryCount > 0) {
            supportFragmentManager.popBackStack(
                TAG_RESULTS,
                androidx.fragment.app.FragmentManager.POP_BACK_STACK_INCLUSIVE
            )
        }
        binding.toolbar.title = getString(R.string.app_name)
    }

    private fun updateUpAffordance() {
        val hasBackStack = supportFragmentManager.backStackEntryCount > 0
        val running = viewModel.isRunning.value == true
        supportActionBar?.setDisplayHomeAsUpEnabled(hasBackStack && !running)
        if (!hasBackStack) {
            binding.toolbar.title = getString(R.string.app_name)
        }
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.main_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            android.R.id.home -> { viewModel.returnToSetup(); true }
            R.id.menu_about   -> { showAbout(); true }
            else              -> super.onOptionsItemSelected(item)
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        viewModel.returnToSetup()
        return true
    }

    private fun showAbout() {
        AboutBottomSheet().show(supportFragmentManager, AboutBottomSheet.TAG)
    }
}
