import Cocoa
import FlutterMacOS

class MainFlutterWindow: NSWindow {
  override func awakeFromNib() {
    let flutterViewController = FlutterViewController()
    self.contentViewController = flutterViewController

    // Roomier default than the 800x600 nib frame: results tables and the
    // wide (navigation rail) layout both want the extra width.
    self.contentMinSize = NSSize(width: 900, height: 640)
    var windowFrame = self.frame
    windowFrame.size = self.frameRect(
      forContentRect: NSRect(x: 0, y: 0, width: 1280, height: 860)).size
    self.setFrame(windowFrame, display: true)
    self.center()

    RegisterGeneratedPlugins(registry: flutterViewController)

    super.awakeFromNib()
  }
}
