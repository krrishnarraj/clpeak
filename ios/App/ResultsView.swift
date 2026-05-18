import SwiftUI

struct ResultsView: View {
    @EnvironmentObject private var viewModel: BenchmarkViewModel

    var body: some View {
        VStack(spacing: 0) {
            if viewModel.isRunning {
                ProgressView()
                    .progressViewStyle(.linear)
            }

            if !viewModel.backendOrder.isEmpty {
                Picker("Backend", selection: selectedBackend) {
                    ForEach(viewModel.backendOrder, id: \.self) { backend in
                        Text(backend).tag(Optional(backend))
                    }
                }
                .pickerStyle(.segmented)
                .padding()
            }

            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    if let backend = viewModel.selectedBackend,
                       let info = viewModel.deviceInfoByBackend[backend] {
                        DeviceInfoPanel(info: info)
                    }

                    ForEach(currentCategories) { category in
                        ResultCategoryCard(category: category)
                    }

                    if currentCategories.isEmpty && viewModel.isRunning {
                        Text("Running benchmarks...")
                            .font(.body)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity)
                            .padding(.top, 48)
                    }
                }
                .padding(.horizontal)
                .padding(.bottom, 32)
            }
        }
        .alert("Benchmark error", isPresented: errorBinding) {
            Button("OK") { viewModel.errorMessage = nil }
        } message: {
            Text(viewModel.errorMessage ?? "")
        }
    }

    private var selectedBackend: Binding<String?> {
        Binding(
            get: { viewModel.selectedBackend },
            set: { viewModel.selectedBackend = $0 }
        )
    }

    private var currentCategories: [BenchmarkCategory] {
        guard let backend = viewModel.selectedBackend else { return [] }
        return viewModel.categoriesByBackend[backend] ?? []
    }

    private var errorBinding: Binding<Bool> {
        Binding(
            get: { viewModel.errorMessage != nil },
            set: { if !$0 { viewModel.errorMessage = nil } }
        )
    }
}

private struct DeviceInfoPanel: View {
    let info: DeviceInfo

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Device")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)

            InfoRow(label: "Platform", value: info.platformName)
            InfoRow(label: "Device", value: info.deviceName)
            InfoRow(label: "Driver", value: info.driverVersion)

            ForEach(Array(info.props().enumerated()), id: \.offset) { _, prop in
                InfoRow(label: prop.0, value: prop.1)
            }
        }
        .padding(16)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
    }
}

private struct InfoRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 82, alignment: .leading)
            Text(value.isEmpty ? "-" : value)
                .font(.subheadline)
                .textSelection(.enabled)
            Spacer(minLength: 0)
        }
    }
}

private struct ResultCategoryCard: View {
    let category: BenchmarkCategory
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            VStack(spacing: 8) {
                let maxValue = max(category.metrics.map(\.value).max() ?? 1, 1)
                ForEach(category.metrics) { entry in
                    MetricRow(entry: entry, maxValue: maxValue, tint: tint)
                }
            }
            .padding(.top, 10)
        } label: {
            HStack(spacing: 12) {
                Circle()
                    .fill(tint)
                    .frame(width: 10, height: 10)
                Text(category.displayName)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(2)
                Spacer()
                VStack(alignment: .trailing, spacing: 0) {
                    Text(String(format: "%.2f", category.peakValue))
                        .font(.headline.monospacedDigit())
                    Text(category.unit.uppercased())
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(14)
        .background(containerTint, in: RoundedRectangle(cornerRadius: 12))
    }

    private var tint: Color {
        switch category.testType {
        case .bandwidth: return .teal
        case .fpCompute: return .orange
        case .intCompute: return .indigo
        case .latency: return .pink
        case .unknown: return .gray
        }
    }

    private var containerTint: Color {
        tint.opacity(0.14)
    }
}

private struct MetricRow: View {
    let entry: ResultEntry
    let maxValue: Float
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(entry.metric)
                    .font(.caption)
                Spacer()
                Text(valueText)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule().fill(.secondary.opacity(0.16))
                    Capsule()
                        .fill(tint)
                        .frame(width: proxy.size.width * CGFloat(progress))
                }
            }
            .frame(height: 6)
        }
    }

    private var progress: Float {
        guard entry.value > 0 else { return 0 }
        return min(entry.value / maxValue, 1)
    }

    private var valueText: String {
        if entry.value == 0 && entry.metric.lowercased().replacingOccurrences(of: " ", with: "") == "enqueuemapbuffer" {
            return "zero-copy"
        }
        return String(format: "%.2f", entry.value)
    }
}
