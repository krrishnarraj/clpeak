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
typedef _LoadResultNative = Pointer<Utf8> Function(Pointer<Utf8> path);
typedef _LoadResult = Pointer<Utf8> Function(Pointer<Utf8> path);

/// Thin, manual dart:ffi bindings over the 7 clpeak_* symbols.
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
        _loadResultFile = lib.lookupFunction<_LoadResultNative, _LoadResult>(
            'clpeak_load_result_file_json');

  factory ClpeakBindings.open() => ClpeakBindings._(openClpeakLibrary());

  final _VersionNative _version;
  final _CatalogNative _catalog;
  final _FreeString _freeString;
  final ClpeakLaunch launch;
  final ClpeakRequestCancel requestCancel;
  final _LoadResult _loadResultFile;

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

  /// Load a saved result file; returns the saveJson document or null when
  /// the file is unreadable/rejected.
  Map<String, dynamic>? loadResultFile(String path) {
    final pathPtr = path.toNativeUtf8();
    try {
      final json = takeString(_loadResultFile(pathPtr));
      if (json == null) return null;
      return jsonDecode(json) as Map<String, dynamic>;
    } finally {
      malloc.free(pathPtr);
    }
  }
}
