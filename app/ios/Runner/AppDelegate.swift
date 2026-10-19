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
  }
}

// The platform half of lib/src/services/run_notice.dart: the notification a
// run posts when the app leaves the screen mid-run.  iOS suspends the app
// seconds after it leaves, so it is posted at once; a tap opens the app.
enum RunNotice {
  static let id = "run"

  static func register(with messenger: FlutterBinaryMessenger) {
    let channel = FlutterMethodChannel(name: "kr.clpeak/notify", binaryMessenger: messenger)
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
