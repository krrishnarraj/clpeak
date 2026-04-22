package kr.clpeak

import android.animation.ValueAnimator
import android.graphics.drawable.GradientDrawable
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.animation.DecelerateInterpolator
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import kr.clpeak.databinding.ItemBenchmarkCategoryBinding
import kr.clpeak.databinding.ItemBenchmarkMetricBinding

class ResultAdapter(
    private val onToggle: (String) -> Unit
) : ListAdapter<BenchmarkCategory, ResultAdapter.CategoryViewHolder>(DIFF_CALLBACK) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): CategoryViewHolder {
        val binding = ItemBenchmarkCategoryBinding.inflate(
            LayoutInflater.from(parent.context), parent, false
        )
        return CategoryViewHolder(binding)
    }

    override fun onBindViewHolder(holder: CategoryViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class CategoryViewHolder(
        private val binding: ItemBenchmarkCategoryBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        private var currentlyExpanded = false

        fun bind(category: BenchmarkCategory) {
            val ctx = binding.root.context

            binding.tvCategoryName.text = category.displayName
            binding.tvPeakValue.text    = "%.2f".format(category.peakValue)
            binding.tvUnit.text         = category.unit.uppercase()

            // Tonal card background based on test type
            val containerColor = when (category.testType) {
                TestType.BANDWIDTH -> R.color.test_bandwidth_container
                TestType.COMPUTE   -> R.color.test_compute_container
                TestType.LATENCY   -> R.color.test_latency_container
            }
            binding.root.setCardBackgroundColor(ctx.getColor(containerColor))

            // Type indicator dot color
            val indicatorColor = when (category.testType) {
                TestType.BANDWIDTH -> R.color.test_bandwidth
                TestType.COMPUTE   -> R.color.test_compute
                TestType.LATENCY   -> R.color.test_latency
            }
            val dot = binding.viewTypeIndicator.background
            if (dot is GradientDrawable) {
                dot.setColor(ctx.getColor(indicatorColor))
            } else {
                binding.viewTypeIndicator.background = GradientDrawable().apply {
                    shape = GradientDrawable.OVAL
                    setColor(ctx.getColor(indicatorColor))
                }
            }

            // Text color on the tonal container
            val onContainerColor = when (category.testType) {
                TestType.BANDWIDTH -> R.color.test_bandwidth_on_container
                TestType.COMPUTE   -> R.color.test_compute_on_container
                TestType.LATENCY   -> R.color.test_latency_on_container
            }
            binding.tvCategoryName.setTextColor(ctx.getColor(onContainerColor))

            val expanded = category.isExpanded
            val shouldAnimate = expanded != currentlyExpanded
            currentlyExpanded = expanded

            if (expanded) {
                bindMetricRows(category)
            }

            if (shouldAnimate && binding.root.width > 0) {
                animateExpandCollapse(binding.metricsContainer, expanded)
                binding.ivExpand.animate()
                    .rotation(if (expanded) 180f else 0f)
                    .setDuration(250)
                    .setInterpolator(DecelerateInterpolator())
                    .start()
            } else {
                binding.metricsContainer.visibility = if (expanded) View.VISIBLE else View.GONE
                binding.metricsContainer.layoutParams.height = ViewGroup.LayoutParams.WRAP_CONTENT
                binding.ivExpand.rotation = if (expanded) 180f else 0f
            }

            binding.root.setOnClickListener { onToggle(category.testName) }
        }

        private fun animateExpandCollapse(view: View, expand: Boolean) {
            if (expand) {
                view.visibility = View.VISIBLE
                val parent = view.parent as? View ?: return
                val widthSpec = View.MeasureSpec.makeMeasureSpec(
                    if (parent.width > 0) parent.width else 1000,
                    View.MeasureSpec.EXACTLY
                )
                view.measure(widthSpec, View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED))

                val targetHeight = view.measuredHeight
                view.layoutParams.height = 0
                view.requestLayout()
                view.alpha = 0f

                ValueAnimator.ofInt(0, targetHeight).apply {
                    duration = 250
                    interpolator = DecelerateInterpolator()
                    addUpdateListener { anim ->
                        view.layoutParams.height = anim.animatedValue as Int
                        view.requestLayout()
                    }
                    addListener(object : android.animation.AnimatorListenerAdapter() {
                        override fun onAnimationEnd(animation: android.animation.Animator) {
                            view.layoutParams.height = ViewGroup.LayoutParams.WRAP_CONTENT
                        }
                    })
                    start()
                }
                view.animate().alpha(1f).setDuration(200).setStartDelay(50).start()
            } else {
                val startHeight = view.height
                ValueAnimator.ofInt(startHeight, 0).apply {
                    duration = 200
                    interpolator = DecelerateInterpolator()
                    addUpdateListener { anim ->
                        view.layoutParams.height = anim.animatedValue as Int
                        view.requestLayout()
                        if (anim.animatedValue as Int == 0) {
                            view.visibility = View.GONE
                            view.layoutParams.height = ViewGroup.LayoutParams.WRAP_CONTENT
                        }
                    }
                    start()
                }
                view.animate().alpha(0f).setDuration(150).start()
            }
        }

        private fun bindMetricRows(category: BenchmarkCategory) {
            val container = binding.metricsContainer
            // Keep the divider (first child), remove dynamically added rows
            while (container.childCount > 1) {
                container.removeViewAt(container.childCount - 1)
            }

            val maxVal = category.metrics.maxOfOrNull { it.value }?.takeIf { it > 0f } ?: 1f
            val inflater = LayoutInflater.from(container.context)

            val barColorRes = when (category.testType) {
                TestType.BANDWIDTH -> R.color.test_bandwidth
                TestType.COMPUTE   -> R.color.test_compute
                TestType.LATENCY   -> R.color.test_latency
            }
            val barColor = container.context.getColor(barColorRes)

            for (entry in category.metrics) {
                val row = ItemBenchmarkMetricBinding.inflate(inflater, container, false)
                row.tvMetricName.text  = entry.metric
                if (entry.value == 0f && isZeroCopyMetric(entry.metric)) {
                    row.tvMetricValue.text = "zero-copy"
                    row.progressBar.max      = 1000
                    row.progressBar.progress = 0
                } else {
                    row.tvMetricValue.text = "%.2f".format(entry.value)
                    row.progressBar.max      = 1000
                    row.progressBar.progress = ((entry.value / maxVal) * 1000).toInt()
                }
                row.progressBar.setIndicatorColor(barColor)
                container.addView(row.root)
            }
        }
    }

    companion object {
        // Metrics that report 0 when the GPU uses shared memory (zero-copy)
        private val ZERO_COPY_METRICS = setOf("enqueuemapbuffer", "enqueueunmap")

        private fun isZeroCopyMetric(name: String) =
            name.lowercase().replace(" ", "") in ZERO_COPY_METRICS

        private val DIFF_CALLBACK = object : DiffUtil.ItemCallback<BenchmarkCategory>() {
            override fun areItemsTheSame(a: BenchmarkCategory, b: BenchmarkCategory) =
                a.testName == b.testName
            override fun areContentsTheSame(a: BenchmarkCategory, b: BenchmarkCategory) =
                a == b
        }
    }
}
