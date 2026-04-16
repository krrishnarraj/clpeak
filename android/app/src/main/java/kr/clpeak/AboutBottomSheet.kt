package kr.clpeak

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import kr.clpeak.databinding.AboutFormBinding

class AboutBottomSheet : BottomSheetDialogFragment() {

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val binding = AboutFormBinding.inflate(inflater, container, false)

        binding.tvVersion.text = "Version ${BuildConfig.VERSION_NAME}"

        binding.btnGithub.setOnClickListener {
            startActivity(
                Intent(Intent.ACTION_VIEW, Uri.parse(getString(R.string.about_url)))
            )
        }

        return binding.root
    }

    companion object {
        const val TAG = "AboutBottomSheet"
    }
}
