package kr.clpeak

import android.content.Intent
import android.os.Bundle
import androidx.core.net.toUri
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import kr.clpeak.databinding.AboutFormBinding

class AboutBottomSheet : BottomSheetDialogFragment() {

    private external fun nativeGetVersion(): String

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?,
    ): View {
        val binding = AboutFormBinding.inflate(inflater, container, false)

        val version = try {
            nativeGetVersion()
        } catch (_: UnsatisfiedLinkError) {
            BuildConfig.VERSION_NAME
        }

        binding.tvVersion.text = getString(R.string.about_version, version)

        binding.btnGithub.setOnClickListener {
            startActivity(
                Intent(
                    Intent.ACTION_VIEW,
                    getString(R.string.about_url).toUri(),
                )
            )
        }

        return binding.root
    }

    companion object {
        const val TAG = "AboutBottomSheet"
    }
}
