import Flutter
import UIKit
import UserNotifications

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)
    if let registrar = engineBridge.pluginRegistry.registrar(forPlugin: "RunNotice") {
      RunNotice.register(with: registrar.messenger())
    }
    if let registrar = engineBridge.pluginRegistry.registrar(forPlugin: "HostFacts") {
      HostFacts.register(with: registrar.messenger())
    }
  }
}

// The platform half of lib/src/services/run_notice.dart: the notification a
// run posts when the app leaves the screen mid-run.  iOS suspends the app
// seconds after it leaves, so it is posted at once; a tap opens the app.
enum RunNotice {
  static let id = "run"

  static func register(with messenger: FlutterBinaryMessenger) {
    let channel = FlutterMethodChannel(name: "kr.XpuPeak/notify", binaryMessenger: messenger)
    channel.setMethodCallHandler { call, result in
      let center = UNUserNotificationCenter.current()
      switch call.method {
      case "ask":
        // Asks only the first time; after that it answers with the setting.
        center.requestAuthorization(options: [.alert]) { granted, _ in
          DispatchQueue.main.async { result(granted) }
        }
      case "show":
        let args = call.arguments as? [String: Any] ?? [:]
        let content = UNMutableNotificationContent()
        content.title = args["title"] as? String ?? ""
        content.body = args["body"] as? String ?? ""
        center.add(UNNotificationRequest(identifier: id, content: content, trigger: nil))
        result(nil)
      case "clear":
        center.removeDeliveredNotifications(withIdentifiers: [id])
        center.removePendingNotificationRequests(withIdentifiers: [id])
        result(nil)
      default:
        result(FlutterMethodNotImplemented)
      }
    }
  }
}

// The platform half of lib/src/services/host_facts.dart: the power source a
// run starts on, in the result document's words (docs/format-v4.md,
// Conditions), for xpupeak_set_host_facts.  It is UIKit's, so it is read
// here, on the main thread the channel answers on; Low Power Mode the native
// side reads itself.  A state iOS does not know -- the simulator's -- is
// left out.
enum HostFacts {
  static func register(with messenger: FlutterBinaryMessenger) {
    let channel = FlutterMethodChannel(name: "kr.XpuPeak/system", binaryMessenger: messenger)
    channel.setMethodCallHandler { call, result in
      switch call.method {
      case "hostFacts":
        var facts: [String: Any] = [:]
        // Unknown until monitoring is on.
        UIDevice.current.isBatteryMonitoringEnabled = true
        switch UIDevice.current.batteryState {
        case .charging, .full: facts["power_source"] = "ac"
        case .unplugged: facts["power_source"] = "battery"
        default: break
        }
        result(facts)
      default:
        result(FlutterMethodNotImplemented)
      }
    }
  }
}
