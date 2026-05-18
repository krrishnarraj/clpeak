import Foundation
import Darwin
import SwiftUI

private final class CallbackBox {
    let onDevice: (DeviceInfo) -> Void
    let onMetric: (ResultEntry) -> Void

    init(onDevice: @escaping (DeviceInfo) -> Void,
         onMetric: @escaping (ResultEntry) -> Void) {
        self.onDevice = onDevice
        self.onMetric = onMetric
    }
}

private func nativeString(_ value: UnsafePointer<CChar>?) -> String {
    guard let value else { return "" }
    return String(cString: value)
}

private func metricCallback(_ context: UnsafeMutableRawPointer?,
                            _ backend: UnsafePointer<CChar>?,
                            _ platform: UnsafePointer<CChar>?,
                            _ device: UnsafePointer<CChar>?,
                            _ driver: UnsafePointer<CChar>?,
                            _ category: UnsafePointer<CChar>?,
                            _ test: UnsafePointer<CChar>?,
                            _ display: UnsafePointer<CChar>?,
                            _ metric: UnsafePointer<CChar>?,
                            _ unit: UnsafePointer<CChar>?,
                            _ value: Float,
                            _ status: UnsafePointer<CChar>?,
                            _ reason: UnsafePointer<CChar>?) {
    guard let context else { return }
    let box = Unmanaged<CallbackBox>.fromOpaque(context).takeUnretainedValue()
    box.onMetric(ResultEntry(
        backend: nativeString(backend),
        platform: nativeString(platform),
        device: nativeString(device),
        driver: nativeString(driver),
        category: nativeString(category),
        test: nativeString(test),
        display: nativeString(display),
        metric: nativeString(metric),
        unit: nativeString(unit),
        value: value,
        status: nativeString(status),
        reason: nativeString(reason)
    ))
}

private func deviceCallback(_ context: UnsafeMutableRawPointer?,
                            _ backend: UnsafePointer<CChar>?,
                            _ platform: UnsafePointer<CChar>?,
                            _ device: UnsafePointer<CChar>?,
                            _ driver: UnsafePointer<CChar>?,
                            _ propsJson: UnsafePointer<CChar>?,
                            _ platformIndex: Int32,
                            _ deviceIndex: Int32) {
    guard let context else { return }
    let box = Unmanaged<CallbackBox>.fromOpaque(context).takeUnretainedValue()
    box.onDevice(DeviceInfo(
        backend: nativeString(backend),
        platformName: nativeString(platform),
        deviceName: nativeString(device),
        driverVersion: nativeString(driver),
        propsJson: nativeString(propsJson),
        platformIndex: Int(platformIndex),
        deviceIndex: Int(deviceIndex)
    ))
}

private enum NativeBenchmark {
    static func catalogJson() -> String {
        guard let raw = clpeak_ios_copy_backend_catalog_json() else { return "" }
        defer { clpeak_ios_free_string(raw) }
        return String(cString: raw)
    }

    static func version() -> String {
        guard let raw = clpeak_ios_version() else { return "unknown" }
        return String(cString: raw)
    }

    static func run(argv: [String],
                    onDevice: @escaping (DeviceInfo) -> Void,
                    onMetric: @escaping (ResultEntry) -> Void) -> Int32 {
        let box = CallbackBox(onDevice: onDevice, onMetric: onMetric)
        let context = Unmanaged.passUnretained(box).toOpaque()

        let callbacks = ClpeakIOSCallbacks(metric: metricCallback, device: deviceCallback)

        let cStrings = argv.map { strdup($0) }
        defer { cStrings.forEach { free($0) } }

        var pointers: [UnsafePointer<CChar>?] = cStrings.map {
            guard let value = $0 else { return nil }
            return UnsafePointer(value)
        }

        return pointers.withUnsafeMutableBufferPointer { buffer in
            clpeak_ios_launch(Int32(argv.count), buffer.baseAddress, callbacks, context)
        }
    }

}

@MainActor
final class BenchmarkViewModel: ObservableObject {
    enum Screen {
        case setup
        case results
    }

    @Published private(set) var catalog: BackendCatalog = .empty
    @Published var selection = BenchmarkSelection()
    @Published private(set) var isRunning = false
    @Published private(set) var screen: Screen = .setup
    @Published private(set) var deviceInfoByBackend: [String: DeviceInfo] = [:]
    @Published private(set) var categoriesByBackend: [String: [BenchmarkCategory]] = [:]
    @Published private(set) var backendOrder: [String] = []
    @Published var selectedBackend: String?
    @Published var errorMessage: String?

    let version = NativeBenchmark.version()

    private var accumulated: [ResultEntry] = []

    init() {
        loadCatalog()
    }

    func loadCatalog() {
        Task {
            let json = await Task.detached { NativeBenchmark.catalogJson() }.value
            let parsed = BackendCatalog.from(json: json)
            catalog = parsed
            selection = .all(of: parsed)
        }
    }

    func selectAll() {
        selection = .all(of: catalog)
    }

    func clearSelection() {
        selection = BenchmarkSelection()
    }

    func setBackend(_ backend: BackendInfo, enabled: Bool) {
        var next = selection
        next.setBackend(backend, enabled: enabled)
        selection = next
    }

    func toggleDevice(backend: BackendInfo, device: DeviceRef) {
        var next = selection
        next.toggleDevice(backend: backend.name, device: device.id)
        selection = next
    }

    func returnToSetup() {
        guard !isRunning else { return }
        screen = .setup
    }

    func runBenchmarks() {
        guard !isRunning else { return }
        guard !selection.isEmpty else {
            errorMessage = "Select at least one device to benchmark."
            return
        }

        isRunning = true
        screen = .results
        errorMessage = nil
        deviceInfoByBackend = [:]
        categoriesByBackend = [:]
        backendOrder = []
        selectedBackend = nil
        accumulated.removeAll()

        let argv = buildArgv()

        Task {
            let result = await Task.detached {
                NativeBenchmark.run(
                    argv: argv,
                    onDevice: { info in
                        Task { @MainActor [weak self] in
                            self?.deviceInfoByBackend[info.backend] = info
                            self?.ensureBackendVisible(info.backend)
                        }
                    },
                    onMetric: { entry in
                        Task { @MainActor [weak self] in
                            self?.accumulated.append(entry)
                            self?.rebuild()
                        }
                    }
                )
            }.value

            if result != 0 {
                errorMessage = "Benchmark exited with error (\(result))."
            }
            isRunning = false
            rebuild()
        }
    }

    private func buildArgv() -> [String] {
        var args = ["clpeak", "--max-time", "150"]

        for backend in catalog.backends {
            let selected = selection.devicesByBackend[backend.name] ?? []
            if selected.isEmpty {
                if backend.name == "Vulkan" { args.append("--no-vulkan") }
                if backend.name == "Metal" { args.append("--no-metal") }
                continue
            }

            if selected.count == 1, let only = selected.first {
                if backend.name == "Vulkan" {
                    args += ["--vk-device", String(only)]
                } else if backend.name == "Metal" {
                    args += ["--mtl-device", String(only)]
                }
            }
        }

        return args
    }

    private func rebuild() {
        var order: [String] = []
        var grouped: [String: [String: [ResultEntry]]] = [:]

        for entry in accumulated {
            if !order.contains(entry.backend) {
                order.append(entry.backend)
            }
            let key = "\(entry.category)|\(entry.test)"
            grouped[entry.backend, default: [:]][key, default: []].append(entry)
        }

        var rebuilt: [String: [BenchmarkCategory]] = [:]
        for backend in order {
            let cards = grouped[backend] ?? [:]
            rebuilt[backend] = cards.keys.sorted().compactMap { key in
                guard let entries = cards[key], let first = entries.first else { return nil }
                if entries.allSatisfy({ $0.status != "ok" }) { return nil }
                return BenchmarkCategory(
                    backend: backend,
                    test: first.test,
                    displayName: first.display.isEmpty ? first.test.clpeakDisplayName : first.display,
                    category: first.category,
                    unit: first.unit,
                    metrics: entries
                )
            }
        }

        backendOrder = order
        categoriesByBackend = rebuilt
        if selectedBackend == nil || !(selectedBackend.map(order.contains) ?? false) {
            selectedBackend = order.first
        }
    }

    private func ensureBackendVisible(_ backend: String) {
        if !backendOrder.contains(backend) {
            backendOrder.append(backend)
        }
        if selectedBackend == nil {
            selectedBackend = backend
        }
    }
}
