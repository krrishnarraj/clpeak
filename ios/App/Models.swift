import Foundation

struct DeviceRef: Identifiable, Hashable {
    let id: Int
    let name: String
    let type: String
    let apiVersion: String

    init(index: Int, name: String, type: String = "", apiVersion: String = "") {
        self.id = index
        self.name = name
        self.type = type
        self.apiVersion = apiVersion
    }
}

struct BackendInfo: Identifiable, Hashable {
    let id: String
    let available: Bool
    let devices: [DeviceRef]

    var name: String { id }
    var isEmpty: Bool { devices.isEmpty }
}

struct BackendCatalog {
    static let empty = BackendCatalog(backends: [])

    let backends: [BackendInfo]

    var hasAnyDevice: Bool {
        backends.contains { !$0.devices.isEmpty }
    }

    func backend(named name: String) -> BackendInfo? {
        backends.first { $0.name == name }
    }

    static func from(json: String) -> BackendCatalog {
        guard
            let data = json.data(using: .utf8),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let rawBackends = root["backends"] as? [[String: Any]]
        else {
            return .empty
        }

        let parsed = rawBackends.compactMap { backend -> BackendInfo? in
            guard let name = backend["name"] as? String else { return nil }
            let available = backend["available"] as? Bool ?? false
            let platforms = backend["platforms"] as? [[String: Any]] ?? []
            let devices = platforms.flatMap { platform -> [DeviceRef] in
                let rawDevices = platform["devices"] as? [[String: Any]] ?? []
                return rawDevices.compactMap { device in
                    guard
                        let index = device["index"] as? Int,
                        let name = device["name"] as? String
                    else {
                        return nil
                    }
                    return DeviceRef(
                        index: index,
                        name: name,
                        type: device["type"] as? String ?? "",
                        apiVersion: device["api"] as? String ?? ""
                    )
                }
            }
            return BackendInfo(id: name, available: available, devices: devices)
        }

        return BackendCatalog(backends: parsed.filter { $0.name == "Vulkan" || $0.name == "Metal" })
    }
}

struct BenchmarkSelection: Equatable {
    var devicesByBackend: [String: Set<Int>] = [:]

    var isEmpty: Bool {
        devicesByBackend.values.allSatisfy { $0.isEmpty }
    }

    func isSelected(backend: String, device: Int) -> Bool {
        devicesByBackend[backend]?.contains(device) == true
    }

    func isBackendEnabled(_ backend: BackendInfo) -> Bool {
        !(devicesByBackend[backend.name] ?? []).isEmpty
    }

    mutating func toggleDevice(backend: String, device: Int) {
        var selected = devicesByBackend[backend] ?? []
        if selected.contains(device) {
            selected.remove(device)
        } else {
            selected.insert(device)
        }
        devicesByBackend[backend] = selected
    }

    mutating func setBackend(_ backend: BackendInfo, enabled: Bool) {
        devicesByBackend[backend.name] = enabled ? Set(backend.devices.map(\.id)) : []
    }

    static func all(of catalog: BackendCatalog) -> BenchmarkSelection {
        var selection = BenchmarkSelection()
        for backend in catalog.backends {
            selection.devicesByBackend[backend.name] = Set(backend.devices.map(\.id))
        }
        return selection
    }
}

struct DeviceInfo: Identifiable, Hashable {
    var id: String { "\(backend)|\(platformIndex)|\(deviceIndex)" }

    let backend: String
    let platformName: String
    let deviceName: String
    let driverVersion: String
    let propsJson: String
    let platformIndex: Int
    let deviceIndex: Int

    func props() -> [(String, String)] {
        guard
            let data = propsJson.data(using: .utf8),
            let rows = try? JSONSerialization.jsonObject(with: data) as? [[String: String]]
        else {
            return []
        }
        return rows.compactMap { row in
            guard let key = row["k"], let value = row["v"] else { return nil }
            return (key, value)
        }
    }
}

struct ResultEntry: Identifiable, Hashable {
    var id = UUID()

    let backend: String
    let platform: String
    let device: String
    let driver: String
    let category: String
    let test: String
    let display: String
    let metric: String
    let unit: String
    let value: Float
    let status: String
    let reason: String
}

enum TestType {
    case bandwidth
    case fpCompute
    case intCompute
    case latency
    case unknown

    init(category: String) {
        switch category {
        case "bandwidth": self = .bandwidth
        case "fp_compute": self = .fpCompute
        case "int_compute": self = .intCompute
        case "latency": self = .latency
        default: self = .unknown
        }
    }
}

struct BenchmarkCategory: Identifiable, Hashable {
    var id: String { "\(backend)|\(category)|\(test)" }

    let backend: String
    let test: String
    let displayName: String
    let category: String
    let unit: String
    let metrics: [ResultEntry]

    var testType: TestType { TestType(category: category) }
    var allSkipped: Bool { metrics.allSatisfy { $0.status != "ok" } }
    var peakValue: Float { metrics.map(\.value).max() ?? 0 }
    var skipStatus: String { metrics.first?.status ?? "skipped" }
    var skipReason: String { metrics.first?.reason ?? "" }
}

extension String {
    var clpeakDisplayName: String {
        split(separator: "_")
            .map { word in word.prefix(1).uppercased() + word.dropFirst() }
            .joined(separator: " ")
    }
}
