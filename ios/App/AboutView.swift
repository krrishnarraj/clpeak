import SwiftUI

struct AboutView: View {
    @EnvironmentObject private var viewModel: BenchmarkViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    LabeledContent("Version", value: viewModel.version)
                    Link("GitHub", destination: URL(string: "https://github.com/krrishnarraj/clpeak")!)
                }

                Section {
                    Text("Cross-API GPU benchmark for compute, bandwidth, and latency tests.")
                        .font(.body)
                }
            }
            .navigationTitle("About")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
