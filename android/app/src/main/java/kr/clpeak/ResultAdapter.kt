package kr.clpeak

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
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

        fun bind(category: BenchmarkCategory) {
            binding.tvCategoryName.text = category.displayName
            binding.tvPeakValue.text    = "%.2f".format(category.peakValue)
            binding.tvUnit.text         = category.unit.uppercase()

            val colorRes = when (category.testType) {
                TestType.BANDWIDTH -> R.color.test_bandwidth
                TestType.COMPUTE   -> R.color.test_compute
                TestType.LATENCY   -> R.color.test_latency
            }
            binding.viewTypeIndicator.setBackgroundResource(colorRes)

            val expanded = category.isExpanded
            binding.metricsContainer.visibility = if (expanded) View.VISIBLE else View.GONE
            binding.ivExpand.rotation = if (expanded) 180f else 0f

            if (expanded) {
                bindMetricRows(category)
            }

            binding.root.setOnClickListener { onToggle(category.testName) }
        }

        private fun bindMetricRows(category: BenchmarkCategory) {
            binding.metricsContainer.removeAllViews()
            val maxVal = category.metrics.maxOfOrNull { it.value }?.takeIf { it > 0f } ?: 1f
            val inflater = LayoutInflater.from(binding.root.context)

            val barColorRes = when (category.testType) {
                TestType.BANDWIDTH -> R.color.test_bandwidth
                TestType.COMPUTE   -> R.color.test_compute
                TestType.LATENCY   -> R.color.test_latency
            }
            val barColor = binding.root.context.getColor(barColorRes)

            for (entry in category.metrics) {
                val row = ItemBenchmarkMetricBinding.inflate(inflater, binding.metricsContainer, false)
                row.tvMetricName.text  = entry.metric
                row.tvMetricValue.text = "%.2f".format(entry.value)
                row.progressBar.max      = 1000
                row.progressBar.progress = ((entry.value / maxVal) * 1000).toInt()
                row.progressBar.setIndicatorColor(barColor)
                binding.metricsContainer.addView(row.root)
            }
        }
    }

    companion object {
        private val DIFF_CALLBACK = object : DiffUtil.ItemCallback<BenchmarkCategory>() {
            override fun areItemsTheSame(a: BenchmarkCategory, b: BenchmarkCategory) =
                a.testName == b.testName
            override fun areContentsTheSame(a: BenchmarkCategory, b: BenchmarkCategory) =
                a == b
        }
    }
}
