import SwiftUI

@main
struct ClpeakApp: App {
    @StateObject private var viewModel = BenchmarkViewModel()

    var body: some Scene {
        WindowGroup {
            NavigationStack {
                Group {
                    switch viewModel.screen {
                    case .setup:
                        SetupView()
                    case .results:
                        ResultsView()
                    }
                }
                .environmentObject(viewModel)
                .navigationTitle(viewModel.screen == .setup ? "clpeak" : "Results")
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        if viewModel.screen == .results && !viewModel.isRunning {
                            Button("Setup") { viewModel.returnToSetup() }
                        }
                    }
                    ToolbarItem(placement: .topBarTrailing) {
                        AboutButton()
                    }
                }
            }
        }
    }
}

private struct AboutButton: View {
    @State private var showingAbout = false

    var body: some View {
        Button {
            showingAbout = true
        } label: {
            Image(systemName: "info.circle")
        }
        .sheet(isPresented: $showingAbout) {
            AboutView()
        }
    }
}
