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
                    Text("A synthetic micro-benchmark that measures the peak achievable performance of GPU compute devices. It exercises tight vector / MAD / MMA loops to expose what the hardware is capable of in isolation, not real-world workload performance.")
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
