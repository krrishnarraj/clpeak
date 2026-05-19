import SwiftUI

struct SetupView: View {
    @EnvironmentObject private var viewModel: BenchmarkViewModel

    var body: some View {
        ZStack(alignment: .bottom) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Backends")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .textCase(.uppercase)
                        .padding(.horizontal)

                    ForEach(viewModel.catalog.backends) { backend in
                        BackendSelectionSection(backend: backend)
                    }

                    if !viewModel.catalog.hasAnyDevice {
                        Text("No Vulkan or Metal devices were found.")
                            .font(.body)
                            .foregroundStyle(.red)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 24)
                    }

                    HStack {
                        Spacer()
                        Button("Clear") { viewModel.clearSelection() }
                            .buttonStyle(.bordered)
                        Button("Select All") { viewModel.selectAll() }
                            .buttonStyle(.bordered)
                    }
                    .padding(.horizontal)
                    .padding(.bottom, 96)
                }
                .padding(.top, 16)
            }

            Button {
                viewModel.runBenchmarks()
            } label: {
                Label("Run benchmark", systemImage: "play.fill")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(!viewModel.catalog.hasAnyDevice || viewModel.selection.isEmpty)
            .padding()
            .background(.regularMaterial)
        }
        .alert("Benchmark error", isPresented: errorBinding) {
            Button("OK") { viewModel.errorMessage = nil }
        } message: {
            Text(viewModel.errorMessage ?? "")
        }
    }

    private var errorBinding: Binding<Bool> {
        Binding(
            get: { viewModel.errorMessage != nil },
            set: { if !$0 { viewModel.errorMessage = nil } }
        )
    }
}

private struct BackendSelectionSection: View {
    @EnvironmentObject private var viewModel: BenchmarkViewModel
    let backend: BackendInfo

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Toggle(isOn: backendEnabled) {
                Text(backend.name)
                    .font(.headline)
            }
            .disabled(backend.devices.isEmpty)

            VStack(alignment: .leading, spacing: 10) {
                ForEach(backend.devices) { device in
                    Toggle(isOn: deviceSelected(device)) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(device.name)
                                .font(.body)
                            if !device.type.isEmpty || !device.apiVersion.isEmpty {
                                Text([device.type, device.apiVersion].filter { !$0.isEmpty }.joined(separator: " "))
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
            .padding(.top, 2)
        }
        .padding(16)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    private var backendEnabled: Binding<Bool> {
        Binding(
            get: { viewModel.selection.isBackendEnabled(backend) },
            set: { viewModel.setBackend(backend, enabled: $0) }
        )
    }

    private func deviceSelected(_ device: DeviceRef) -> Binding<Bool> {
        Binding(
            get: { viewModel.selection.isSelected(backend: backend.name, device: device.id) },
            set: { _ in viewModel.toggleDevice(backend: backend, device: device) }
        )
    }
}
