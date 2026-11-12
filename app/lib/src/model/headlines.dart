/// The headline catalog (docs/format-v4.md, The headline catalog): the
/// measures worth a summary row, as data with a version of its own, outside
/// the result format -- `assets/measures.json`, the one catalog for every
/// summary of a run, so the app's and the CLI's show the same.
library;

import 'dart:convert';

import 'package:flutter/services.dart' show rootBundle;

import 'measures.dart';

/// One headline: the measure keys it selects, of which its value on a
/// processor is the best there.
class Headline {
  const Headline({
    required this.id,
    required this.title,
    required this.quantity,
    this.processors,
    this.families,
    this.dims,
  });

  final String id;
  final String title;

  /// The quantity of the keys it selects: `flops`, `ops`,
  /// `bytes_per_second`, …
  final String quantity;

  /// The processor kinds it applies to; null for every kind.
  final Set<String>? processors;

  /// The families it selects from; null for every family.
  final Set<String>? families;

  /// Identity-dim objects, of which a selected key's identity dims, with
  /// their defaults filled in, must equal one; null to select a key of any
  /// dims.
  final List<Map<String, Object>>? dims;

  bool appliesTo(String processorKind) =>
      processors == null || processors!.contains(processorKind);

  /// Whether [key] is one of this headline's measures.  A dims object is
  /// normalised as a reading's dims are -- variant keys dropped, defaults
  /// dropped for the key's family -- so `{"dtype": "fp16"}` and
  /// `{"dtype": "fp16", "acc": "fp32"}` select the same matrix keys, and
  /// neither selects an fp16-accumulate one.
  bool selects(MeasureKey key) {
    if (key.quantity != quantity) return false;
    if (families != null && !families!.contains(key.family)) return false;
    final alternatives = dims;
    if (alternatives == null) return true;
    for (final d in alternatives) {
      if (_sameDims(identityDims(d, key.family), key.identity)) return true;
    }
    return false;
  }

  static bool _sameDims(Map<String, String> a, Map<String, String> b) {
    if (a.length != b.length) return false;
    for (final e in a.entries) {
      if (b[e.key] != e.value) return false;
    }
    return true;
  }

  factory Headline.fromJson(Map<String, dynamic> m) {
    Set<String>? strings(Object? v) =>
        v is List ? {for (final s in v) if (s is String) s} : null;
    final dims = m['dims'];
    return Headline(
      id: m['id'] as String? ?? '',
      title: m['title'] as String? ?? m['id'] as String? ?? '',
      quantity: m['quantity'] as String? ?? '',
      processors: strings(m['processors']),
      families: strings(m['families']),
      dims: dims is List
          ? [
              for (final d in dims)
                if (d is Map)
                  {
                    for (final e in d.entries)
                      if (e.key is String && e.value != null)
                        e.key as String: e.value as Object,
                  },
            ]
          : null,
    );
  }
}

class HeadlineCatalog {
  const HeadlineCatalog({required this.version, required this.headlines});

  static const schema = 'xpupeak/measures';

  /// `catalog_version`: the catalog's own, not the format's.
  final int version;
  final List<Headline> headlines;

  static const empty = HeadlineCatalog(version: 0, headlines: []);

  /// From the catalog's JSON.  A document of another schema reads as no
  /// headlines rather than as wrong ones.
  factory HeadlineCatalog.fromJson(Map<String, dynamic> m) {
    if (m['schema'] != schema) return empty;
    return HeadlineCatalog(
      version: (m['catalog_version'] as num?)?.toInt() ?? 0,
      headlines: [
        for (final h in (m['headlines'] is List ? m['headlines'] as List : const []))
          if (h is Map<String, dynamic>) Headline.fromJson(h),
      ],
    );
  }

  static const assetPath = 'assets/measures.json';

  static Future<HeadlineCatalog>? _bundled;

  /// The catalog the app ships, read once.  A catalog that cannot be read
  /// shows no headlines; the results below it are unaffected.
  static Future<HeadlineCatalog> bundled() => _bundled ??= () async {
        try {
          final json = jsonDecode(await rootBundle.loadString(assetPath));
          return json is Map<String, dynamic>
              ? HeadlineCatalog.fromJson(json)
              : empty;
        } catch (_) {
          return empty;
        }
      }();
}
