# iOS
iOS app bundle for running clpeak with Vulkan-over-MoltenVK and Metal.

## Dependencies
Install [vulkan sdk](https://vulkan.lunarg.com/sdk/home) for mac os.
While installing, in the 'Select Components' windows, choose these items:
 - KosmicKrisp (Vulkan on Metal)
 - System Global Installation
 - Development libraries for iOS

## Build
### iOS
cmake step: `cmake -S ios -B ios/build -G Xcode -DCMAKE_SYSTEM_NAME=iOS`
open `ios/build` in xcode
or build in cmdline
`cmake --build ios/build --config Debug --target clpeak_ios -- -sdk iphoneos CODE_SIGNING_ALLOWED=NO`

To build without the Vulkan (MoltenVK) backend:
`cmake -S ios -B ios/build -G Xcode -DCMAKE_SYSTEM_NAME=iOS -DCLPEAK_IOS_ENABLE_VULKAN=OFF`

### iOS Simulator
cmake step: `cmake -S ios -B ios/build-sim -G Xcode -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphonesimulator`
open `ios/build-sim` in xcode
or build in cmdline
`cmake --build ios/build-sim --config Debug --target clpeak_ios_simulator -- -sdk iphonesimulator CODE_SIGNING_ALLOWED=NO`

## Quick Lookups
- Looking for the native Swift bridge? → see `Native/clpeak_ios_bridge.h`
- Looking for benchmark dispatch? → see `Native/entry_ios.mm`
- Looking for structured result callbacks? → see `Native/logger_ios.mm`
- Looking for Swift state and argv construction? → see `App/BenchmarkViewModel.swift`
- Looking for Vulkan bundle packaging? → see `CMakeLists.txt`

## Key Files
| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Xcode/iOS bundle target; links common, Vulkan, and Metal backends |
| `Info.plist.in` | App bundle plist template |
| `App/ClpeakApp.swift` | SwiftUI app entry and top-level navigation |
| `App/Models.swift` | Catalog, selection, device info, and result grouping models |
| `App/BenchmarkViewModel.swift` | Observable app state, native calls, and benchmark run lifecycle |
| `Native/clpeak_ios_bridge.h` | C ABI imported by Swift |
| `Native/entry_ios.mm` | Native backend enumeration and benchmark execution |
| `Native/logger_ios.mm` | `logger` implementation that forwards structured rows to Swift |

## When You Change This Directory
- If you add/remove app directories or major build inputs → update `../AGENTS.md`
- If native callbacks change shape → update `App/BenchmarkViewModel.swift`
- If enabled backends change → update `CMakeLists.txt` and `App/Models.swift`
