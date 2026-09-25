import 'dart:convert';
import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'clpeak_library.dart';

/// Return codes of clpeak_launch (see src/ffi/clpeak_ffi.h).
const int clpeakRunOk = 0;
const int clpeakRunBadArgs = -1;
const int clpeakRunCancelled = -2;
const int clpeakRunBusy = -3;

/// The native event callback signature:
///   void (*ClpeakEventCallback)(void *user_data, char *event_json)
typedef ClpeakEventCallbackNative = Void Function(
    Pointer<Void> userData, Pointer<Utf8> eventJson);

typedef _VersionNative = Pointer<Utf8> Function();
typedef _CatalogNative = Pointer<Utf8> Function();
typedef _FreeStringNative = Void Function(Pointer<Utf8>);
typedef _FreeString = void Function(Pointer<Utf8>);
typedef _LaunchNative = Int32 Function(Int32 argc, Pointer<Pointer<Utf8>> argv,
    Pointer<NativeFunction<ClpeakEventCallbackNative>> onEvent,
    Pointer<Void> userData);
typedef ClpeakLaunch = int Function(int argc, Pointer<Pointer<Utf8>> argv,
    Pointer<NativeFunction<ClpeakEventCallbackNative>> onEvent,
    Pointer<Void> userData);
typedef _RequestCancelNative = Void Function();
typedef ClpeakRequestCancel = void Function();
typedef _SetOnnxLibNative = Void Function(Pointer<Utf8> path);
typedef _SetOnnxLib = void Function(Pointer<Utf8> path);
typedef _SetOnnxEpLibsNative = Void Function(Pointer<Utf8> spec);
typedef _SetOnnxEpLibs = void Function(Pointer<Utf8> spec);
typedef _SetOnnxWinmlNative = Void Function(Int32 enabled, Pointer<Utf8> path);
typedef _SetOnnxWinml = void Function(int enabled, Pointer<Utf8> path);
typedef _OnnxStatusNative = Pointer<Utf8> Function();
typedef _SetLitertLibNative = Void Function(Pointer<Utf8> path);
typedef _SetLitertLib = void Function(Pointer<Utf8> path);
typedef _SetLitertStageNative = Void Function(Pointer<Utf8> dir);
typedef _SetLitertStage = void Function(Pointer<Utf8> dir);
typedef _LitertStatusNative = Pointer<Utf8> Function();

/// Thin, manual dart:ffi bindings over the 12 clpeak_* symbols.
class ClpeakBindings {
  ClpeakBindings._(DynamicLibrary lib)
      : _version = lib.lookupFunction<_VersionNative, _VersionNative>(
            'clpeak_version'),
        _catalog = lib.lookupFunction<_CatalogNative, _CatalogNative>(
            'clpeak_copy_backend_catalog_json'),
        _freeString = lib.lookupFunction<_FreeStringNative, _FreeString>(
            'clpeak_free_string'),
        launch =
            lib.lookupFunction<_LaunchNative, ClpeakLaunch>('clpeak_launch'),
        requestCancel =
            lib.lookupFunction<_RequestCancelNative, ClpeakRequestCancel>(
                'clpeak_request_cancel'),
        _setOnnxLibrary = lib.lookupFunction<_SetOnnxLibNative, _SetOnnxLib>(
            'clpeak_set_onnx_library'),
        _setOnnxEpLibraries =
            lib.lookupFunction<_SetOnnxEpLibsNative, _SetOnnxEpLibs>(
                'clpeak_set_onnx_ep_libraries'),
        _setOnnxWinml = lib.lookupFunction<_SetOnnxWinmlNative, _SetOnnxWinml>(
            'clpeak_set_onnx_winml'),
        _onnxStatus = lib.lookupFunction<_OnnxStatusNative, _OnnxStatusNative>(
            'clpeak_copy_onnx_status_json'),
        _setLitertLibrary =
            lib.lookupFunction<_SetLitertLibNative, _SetLitertLib>(
                'clpeak_set_litert_library'),
        _setLitertNpuStageDir =
            lib.lookupFunction<_SetLitertStageNative, _SetLitertStage>(
                'clpeak_set_litert_npu_stage_dir'),
        _litertStatus =
            lib.lookupFunction<_LitertStatusNative, _LitertStatusNative>(
                'clpeak_copy_litert_status_json');

  factory ClpeakBindings.open() => ClpeakBindings._(openClpeakLibrary());

  final _VersionNative _version;
  final _CatalogNative _catalog;
  final _FreeString _freeString;
  final ClpeakLaunch launch;
  final ClpeakRequestCancel requestCancel;
  final _SetOnnxLib _setOnnxLibrary;
  final _SetOnnxEpLibs _setOnnxEpLibraries;
  final _SetOnnxWinml _setOnnxWinml;
  final _OnnxStatusNative _onnxStatus;
  final _SetLitertLib _setLitertLibrary;
  final _SetLitertStage _setLitertNpuStageDir;
  final _LitertStatusNative _litertStatus;

  /// clpeak version string, e.g. "2.1.0-3-gabc1234".
  String version() => _version().toDartString();

  /// Consume a malloc'd native string: decode then free.
  String? takeString(Pointer<Utf8> ptr) {
    if (ptr == nullptr) return null;
    try {
      return ptr.toDartString();
    } finally {
      _freeString(ptr);
    }
  }

  /// Device catalog as the inventoryToJson() document.
  Map<String, dynamic> backendCatalog() {
    final json = takeString(_catalog());
    if (json == null) return const {'backends': []};
    return jsonDecode(json) as Map<String, dynamic>;
  }

  /// Choose which onnxruntime library the ONNX backend loads, ahead of the
  /// platform's conventional names; an empty path goes back to searching.
  ///
  /// Takes effect on the next [backendCatalog] (which is what loads the
  /// runtime) until a runtime has loaded; from then on a different choice is
  /// kept for the next launch and reported as [OnnxStatus.pendingRuntime].
  /// Between runs only.  A no-op where ONNX Runtime is linked in (iOS) or the
  /// backend is absent.
  void setOnnxLibrary(String path) {
    final ptr = path.toNativeUtf8();
    try {
      _setOnnxLibrary(ptr);
    } finally {
      malloc.free(ptr);
    }
  }

  /// The plugin execution-provider libraries to register on the runtime's
  /// environment (ONNX Runtime 1.22+), replacing the previous set.  Each
  /// entry is the registration name the provider expects and the library to
  /// load; an implicit entry is one the app bundled speculatively, whose
  /// failure to register is a verbose-log matter rather than a run note.
  /// Unlike the runtime itself this stays live: the next enumeration or run
  /// registers what was added and unregisters what was removed.  Between
  /// runs only.
  void setOnnxEpLibraries(List<OnnxEpLibrary> libs) {
    final spec = libs
        .map((l) => '${l.implicit ? '!' : ''}${l.name}=${l.path}')
        .join('\n');
    final ptr = spec.toNativeUtf8();
    try {
      _setOnnxEpLibraries(ptr);
    } finally {
      malloc.free(ptr);
    }
  }

  /// Windows ML's execution-provider catalog: when enabled, the vendor
  /// providers Windows 11 installs from the Microsoft Store are registered
  /// on the next enumeration or run.  [path] names
  /// Microsoft.Windows.AI.MachineLearning.dll or its directory, or is empty
  /// to search beside the loaded runtime and the executable.  Part of the
  /// runtime setup, so the same contract as [setOnnxLibrary]: once a runtime
  /// has loaded, a change waits for the next launch.
  void setOnnxWinml({required bool enabled, required String path}) {
    final ptr = path.toNativeUtf8();
    try {
      _setOnnxWinml(enabled ? 1 : 0, ptr);
    } finally {
      malloc.free(ptr);
    }
  }

  /// Which ONNX Runtime is loaded, or why none is — see [OnnxStatus].
  OnnxStatus onnxStatus() {
    final json = takeString(_onnxStatus());
    if (json == null) return const OnnxStatus.unavailable('no response');
    return OnnxStatus.fromJson(jsonDecode(json) as Map<String, dynamic>);
  }

  /// The same for LiteRT: which libLiteRt the backend loads, ahead of the
  /// platform's conventional names (on Android the one packaged in the app).
  /// Same contract as [setOnnxLibrary]: once a LiteRT has loaded, a different
  /// choice waits for the next launch ([LitertStatus.pendingPath]).
  void setLitertLibrary(String path) {
    final ptr = path.toNativeUtf8();
    try {
      _setLitertLibrary(ptr);
    } finally {
      malloc.free(ptr);
    }
  }

  /// Android: where the LiteRT backend may stage one vendor's NPU shims when
  /// the APK carries several (links under `dir/<vendor>/`, remade at every
  /// launch).  Before enumeration; a no-op elsewhere.
  void setLitertNpuStageDir(String dir) {
    final ptr = dir.toNativeUtf8();
    try {
      _setLitertNpuStageDir(ptr);
    } finally {
      malloc.free(ptr);
    }
  }

  /// Which LiteRT is loaded, or why none is — see [LitertStatus].
  LitertStatus litertStatus() {
    final json = takeString(_litertStatus());
    if (json == null) return const LitertStatus.unavailable('no response');
    return LitertStatus.fromJson(jsonDecode(json) as Map<String, dynamic>);
  }

}

/// One plugin execution-provider library, as [ClpeakBindings.setOnnxEpLibraries]
/// takes it: the registration name the provider expects and the library.
class OnnxEpLibrary {
  const OnnxEpLibrary(
      {required this.name, required this.path, this.implicit = false});

  final String name; // "QNNExecutionProvider"
  final String path; // absolute, or a bare soname on Android
  final bool implicit; // bundled on the off-chance; failures stay verbose-only

  Map<String, dynamic> toJson() =>
      {'name': name, 'path': path, if (implicit) 'implicit': true};

  factory OnnxEpLibrary.fromJson(Map<String, dynamic> m) => OnnxEpLibrary(
        name: m['name'] as String? ?? '',
        path: m['path'] as String? ?? '',
        implicit: m['implicit'] as bool? ?? false,
      );
}

/// How one plugin library fared on the runtime's environment.
class OnnxEpLibraryStatus {
  const OnnxEpLibraryStatus({
    required this.name,
    required this.path,
    required this.named,
    required this.registered,
    required this.error,
  });

  final String name;
  final String path;
  final bool named; // asked for by a person (not an implicit bundle)
  final bool registered;
  final String error; // when !registered

  factory OnnxEpLibraryStatus.fromJson(Map<String, dynamic> m) =>
      OnnxEpLibraryStatus(
        name: m['name'] as String? ?? '',
        path: m['path'] as String? ?? '',
        named: m['named'] as bool? ?? true,
        registered: m['registered'] as bool? ?? false,
        error: m['error'] as String? ?? '',
      );
}

/// A runtime setup chosen after the runtime loaded: it loads the next time
/// clpeak starts (see [OnnxStatus.pendingRuntime]).
class OnnxPendingRuntime {
  const OnnxPendingRuntime(
      {required this.path, required this.winml, required this.winmlPath});

  final String path; // empty = the default search
  final bool winml; // the Windows ML catalog on
  final String winmlPath; // its DLL or folder; empty = searched for

  factory OnnxPendingRuntime.fromJson(Map<String, dynamic> m) {
    final winml = m['winml'] as Map<String, dynamic>? ?? const {};
    return OnnxPendingRuntime(
      path: m['path'] as String? ?? '',
      winml: winml['enabled'] as bool? ?? false,
      winmlPath: winml['path'] as String? ?? '',
    );
  }
}

/// State of the ONNX Runtime, as clpeak_copy_onnx_status_json() reports it.
class OnnxStatus {
  const OnnxStatus({
    required this.available,
    required this.linkedIn,
    required this.version,
    required this.path,
    required this.error,
    this.epLibraries = const [],
    this.winmlEnabled = false,
    this.winmlPath = '',
    this.winmlError = '',
    this.pendingRuntime,
  });

  const OnnxStatus.unavailable(this.error)
      : available = false,
        linkedIn = false,
        version = '',
        path = '',
        epLibraries = const [],
        winmlEnabled = false,
        winmlPath = '',
        winmlError = '',
        pendingRuntime = null;

  final bool available;

  /// The runtime is built into the binary rather than loaded, so there is no
  /// library to choose.  True on iOS, where the official ONNX Runtime pod is
  /// a static framework and iOS will not dlopen anything else.
  final bool linkedIn;

  final String version; // "1.29.0"
  final String path; // what was loaded; the resolved file even when found
  // by name, empty only when statically linked
  final String error; // populated only when !available

  /// What the last environment registered; a library chosen since is absent
  /// until the next enumeration.
  final List<OnnxEpLibraryStatus> epLibraries;

  /// The Windows ML catalog: whether it is on, which DLL answered, and why
  /// it gave nothing. Both [winmlPath] and [winmlError] empty while enabled
  /// means nothing has resolved it yet (pending until the next enumeration
  /// or run, like [epLibraries]).
  final bool winmlEnabled;
  final String winmlPath;
  final String winmlError;

  /// The runtime setup -- library and Windows ML -- chosen since this
  /// runtime loaded.  One process keeps the runtime it loaded first
  /// (src/onnx/onnx_runtime.h), so the choice loads the next time clpeak
  /// starts.  Null when nothing waits.
  final OnnxPendingRuntime? pendingRuntime;

  factory OnnxStatus.fromJson(Map<String, dynamic> m) {
    final winml = m['winml'] as Map<String, dynamic>? ?? const {};
    final pending = m['pendingRuntime'] as Map<String, dynamic>?;
    return OnnxStatus(
      available: m['available'] as bool? ?? false,
      linkedIn: m['linkedIn'] as bool? ?? false,
      version: m['version'] as String? ?? '',
      path: m['path'] as String? ?? '',
      error: m['error'] as String? ?? '',
      epLibraries: [
        for (final e in (m['epLibraries'] as List<dynamic>? ?? const []))
          OnnxEpLibraryStatus.fromJson(e as Map<String, dynamic>),
      ],
      winmlEnabled: winml['enabled'] as bool? ?? false,
      winmlPath: winml['path'] as String? ?? '',
      winmlError: winml['error'] as String? ?? '',
      pendingRuntime:
          pending == null ? null : OnnxPendingRuntime.fromJson(pending),
    );
  }
}

/// State of the LiteRT runtime, as clpeak_copy_litert_status_json() reports
/// it.  LiteRT has no runtime version string; `version` is the ABI version
/// the app was built against.
class LitertStatus {
  const LitertStatus({
    required this.available,
    required this.version,
    required this.path,
    required this.error,
    this.pendingPath,
  });

  const LitertStatus.unavailable(this.error)
      : available = false,
        version = '',
        path = '',
        pendingPath = null;

  final bool available;
  final String version; // "ABI 1.0.0"
  final String path; // what was loaded
  final String error; // populated only when !available

  /// A library chosen since this one loaded, which loads the next time
  /// clpeak starts ('' = the default search); null when nothing waits.
  final String? pendingPath;

  factory LitertStatus.fromJson(Map<String, dynamic> m) {
    final pending = m['pendingRuntime'] as Map<String, dynamic>?;
    return LitertStatus(
      available: m['available'] as bool? ?? false,
      version: m['version'] as String? ?? '',
      path: m['path'] as String? ?? '',
      error: m['error'] as String? ?? '',
      pendingPath: pending == null ? null : pending['path'] as String? ?? '',
    );
  }
}
