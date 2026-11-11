/// The words a result document uses for what was measured and what it ran
/// on, mirrored from include/common/vocabulary.h and listed in
/// docs/format-v4.md (Vocabulary).
///
/// The file carries strings, and the model keeps them as strings: a newer
/// build's words reach this one as values it does not know, and are shown as
/// written rather than mapped onto something they are not.  The tables here
/// are for reasoning about the words this build knows -- a dim key's class
/// and default, a family's data type and accumulator -- and every list is
/// append-only, as on the native side: a value's string never changes and is
/// never reused, and a dim key's class and default never change.
library;

// ── Data types ─────────────────────────────────────────────────────────────

/// Element types, block-scaled formats as values of their own (an MX or NV
/// block carries its scales, so `mxfp4` is not `fp4_e2m1` with a dim), and
/// integer types.  A mixed format is described with dims -- `weights`,
/// `acc`, `kv`, `quant`, `sparsity` -- never with a value of its own.
const dataTypes = <String>[
  // Float
  'fp64', 'fp32', 'tf32', 'fp16', 'bf16',
  'fp8_e4m3', 'fp8_e5m2', 'fp8_e4m3fnuz', 'fp8_e5m2fnuz',
  'fp6_e3m2', 'fp6_e2m3', 'fp4_e2m1',
  // Block-scaled
  'mxfp8_e4m3', 'mxfp8_e5m2', 'mxfp6_e3m2', 'mxfp6_e2m3', 'mxfp4', 'nvfp4',
  // Integer
  'int64', 'int32', 'int24', 'int16', 'int8', 'int4',
  'uint64', 'uint16', 'uint8', 'uint4',
  // Appended after the seed list
  'uint32',
];

const _integerTypes = <String>{
  'int64', 'int32', 'int24', 'int16', 'int8', 'int4',
  'uint64', 'uint32', 'uint16', 'uint8', 'uint4',
};

final Set<String> _knownTypes = dataTypes.toSet();

bool isKnownDataType(String t) => _knownTypes.contains(t);
bool isIntegerType(String t) => _integerTypes.contains(t);
bool isFloatType(String t) => isKnownDataType(t) && !isIntegerType(t);

/// The accumulator a matrix, dot-product or library operation on [t] uses
/// unless it says otherwise: fp64 for fp64, fp32 for every other float,
/// int64 for the 64-bit integers, int32 for the rest.  Null for a type this
/// build does not know.
String? usualAccumulator(String t) {
  if (!isKnownDataType(t)) return null;
  if (t == 'fp64') return 'fp64';
  if (t == 'int64' || t == 'uint64') return 'int64';
  return isIntegerType(t) ? 'int32' : 'fp32';
}

// ── Dims ───────────────────────────────────────────────────────────────────

/// A dim key's class.  An identity key must match for two readings to be
/// comparable; a variant key is an implementation choice, over which the
/// best reading is taken.  A key this build does not know is identity, so
/// it never merges what it does not understand.
enum DimClass { identity, variant }

/// What a dim's value is.  Values are strings or whole numbers, and stay so.
enum DimKind { dataType, integer, word, text }

/// Where an identity key's default comes from.
enum DimDefault {
  /// No default: absent means not applicable, and present is always keyed.
  none,

  /// A fixed value ([DimKeyInfo.defaultValue]).
  value,

  /// The family's data type ([FamilyInfo.dtype]).
  familyDtype,

  /// The family's accumulator rule ([FamilyInfo.acc]).
  familyAcc,

  /// The reading's own data type.
  sameAsDtype,
}

class DimKeyInfo {
  const DimKeyInfo(this.key, this.cls, this.kind, this.def,
      {this.defaultValue = '', this.words = const []});

  final String key;
  final DimClass cls;
  final DimKind kind;
  final DimDefault def;

  /// [DimDefault.value] only.
  final String defaultValue;

  /// A word key's values, in list order.
  final List<String> words;
}

/// Every dim key this build knows, in the order of vocabulary.h's `Dim`.
const dimKeys = <String, DimKeyInfo>{
  'dtype': DimKeyInfo(
      'dtype', DimClass.identity, DimKind.dataType, DimDefault.familyDtype),
  'weights': DimKeyInfo(
      'weights', DimClass.identity, DimKind.dataType, DimDefault.sameAsDtype),
  'acc': DimKeyInfo(
      'acc', DimClass.identity, DimKind.dataType, DimDefault.familyAcc),
  'quant': DimKeyInfo('quant', DimClass.identity, DimKind.word,
      DimDefault.value,
      defaultValue: 'none',
      words: ['none', 'qdq', 'lut', 'block', 'channel']),
  'sparsity': DimKeyInfo('sparsity', DimClass.identity, DimKind.word,
      DimDefault.value,
      defaultValue: 'dense', words: ['dense', '2:4']),
  'scope': DimKeyInfo('scope', DimClass.identity, DimKind.word,
      DimDefault.value,
      defaultValue: 'device', words: ['core', 'device']),
  'threads_per_core': DimKeyInfo('threads_per_core', DimClass.identity,
      DimKind.integer, DimDefault.none),
  'level': DimKeyInfo(
      'level', DimClass.identity, DimKind.word, DimDefault.none,
      words: ['l1', 'l2', 'l3', 'slc', 'dram']),
  'pattern': DimKeyInfo('pattern', DimClass.identity, DimKind.word,
      DimDefault.value,
      defaultValue: 'random',
      words: ['random', 'linear', 'page_stride', 'predictable']),
  'in_flight': DimKeyInfo('in_flight', DimClass.identity, DimKind.integer,
      DimDefault.value,
      defaultValue: '1'),
  'op': DimKeyInfo('op', DimClass.identity, DimKind.word, DimDefault.none,
      words: [
        'read', 'write', 'copy', 'triad', 'fdiv', 'fsqrt', 'silu', //
        'softmax', 'layernorm', 'dispatch', 'roundtrip', 'mispredict',
      ]),
  'direction': DimKeyInfo(
      'direction', DimClass.identity, DimKind.word, DimDefault.none,
      words: ['h2d', 'd2h', 'roundtrip']),
  'memory': DimKeyInfo(
      'memory', DimClass.identity, DimKind.word, DimDefault.none,
      words: ['pinned', 'pageable']),
  'working_set': DimKeyInfo(
      'working_set', DimClass.identity, DimKind.integer, DimDefault.none),
  'layer': DimKeyInfo(
      'layer', DimClass.identity, DimKind.word, DimDefault.none,
      words: ['conv3x3s2', 'conv1x1', 'depthwise3x3']),
  'pixel_format': DimKeyInfo(
      'pixel_format', DimClass.identity, DimKind.word, DimDefault.none,
      words: ['rgba32f', 'rgba16f', 'rgba8', 'r32f']),
  'phase': DimKeyInfo(
      'phase', DimClass.identity, DimKind.word, DimDefault.none,
      words: ['prefill', 'decode']),
  'context': DimKeyInfo(
      'context', DimClass.identity, DimKind.integer, DimDefault.none),
  'contention': DimKeyInfo('contention', DimClass.identity, DimKind.word,
      DimDefault.value,
      defaultValue: 'none', words: ['none', 'all']),
  'workload': DimKeyInfo(
      'workload', DimClass.identity, DimKind.word, DimDefault.none,
      words: ['trivial_op', 'small_matmul', 'session_create', 'model_create']),
  'width': DimKeyInfo(
      'width', DimClass.variant, DimKind.integer, DimDefault.none),
  'chains': DimKeyInfo(
      'chains', DimClass.variant, DimKind.integer, DimDefault.none),
  'isa': DimKeyInfo('isa', DimClass.variant, DimKind.word, DimDefault.none,
      words: [
        'scalar', 'sse2', 'ssse3', 'sse4_2', 'avx', 'avx2', 'avx512', //
        'avx10', 'amx', 'neon', 'sve', 'sve2', 'sme', 'sme2',
      ]),
  'simd_bits': DimKeyInfo(
      'simd_bits', DimClass.variant, DimKind.integer, DimDefault.none),
  'impl': DimKeyInfo('impl', DimClass.variant, DimKind.word, DimDefault.none,
      words: [
        'cublaslt', 'hipblaslt', 'rocblas', 'rocwmma', 'onemkl', 'mps', //
        'mpsgraph', 'accelerate', 'bnns',
      ]),
  'form': DimKeyInfo('form', DimClass.variant, DimKind.text, DimDefault.none),
  'tile': DimKeyInfo('tile', DimClass.variant, DimKind.text, DimDefault.none),
  'problem_size': DimKeyInfo(
      'problem_size', DimClass.variant, DimKind.integer, DimDefault.none),
  // Appended after the seed list
  'kv': DimKeyInfo(
      'kv', DimClass.identity, DimKind.dataType, DimDefault.sameAsDtype),
};

/// Whether two readings must agree on [key] to be comparable.  A key this
/// build does not know is identity.
bool isIdentityDim(String key) =>
    (dimKeys[key]?.cls ?? DimClass.identity) == DimClass.identity;

// ── Families ───────────────────────────────────────────────────────────────

/// How a family measures (docs/format-v4.md, Families).
enum FamilyMethod { vector, matrix, library, graph, memory, latency, crypto, string }

/// How a family's accumulator defaults.
enum AccRule {
  /// [usualAccumulator] of the reading's data type.
  usual,

  /// The operation's own type: a vector fp16 FMA accumulates in fp16.
  sameAsDtype,

  /// [FamilyInfo.accType].
  fixed,
}

/// A family: a Benchmark, by its flag name.
class FamilyInfo {
  const FamilyInfo(this.flag, this.method,
      {this.dtype = '', this.acc = AccRule.usual, this.accType = ''});

  final String flag;
  final FamilyMethod method;

  /// The family's data type, for a family of one; empty otherwise.
  final String dtype;
  final AccRule acc;

  /// [AccRule.fixed] only.
  final String accType;

  bool get hasDtype => dtype.isNotEmpty;
}

/// Every family this build knows, in the order of the `Benchmark` enum.
const families = <String, FamilyInfo>{
  'single-precision-compute': FamilyInfo(
      'single-precision-compute', FamilyMethod.vector,
      dtype: 'fp32', acc: AccRule.sameAsDtype),
  'half-precision-compute': FamilyInfo(
      'half-precision-compute', FamilyMethod.vector,
      dtype: 'fp16', acc: AccRule.sameAsDtype),
  'double-precision-compute': FamilyInfo(
      'double-precision-compute', FamilyMethod.vector,
      dtype: 'fp64', acc: AccRule.sameAsDtype),
  // fp16 inputs, fp32 accumulate: what "mixed" means.
  'mixed-precision-compute': FamilyInfo(
      'mixed-precision-compute', FamilyMethod.vector,
      dtype: 'fp16', acc: AccRule.fixed, accType: 'fp32'),
  'bfloat16-compute': FamilyInfo('bfloat16-compute', FamilyMethod.vector,
      dtype: 'bf16', acc: AccRule.sameAsDtype),
  'integer-compute': FamilyInfo('integer-compute', FamilyMethod.vector,
      dtype: 'int32', acc: AccRule.sameAsDtype),
  'integer-compute-fast': FamilyInfo(
      'integer-compute-fast', FamilyMethod.vector,
      dtype: 'int24', acc: AccRule.sameAsDtype),
  'integer-compute-char': FamilyInfo(
      'integer-compute-char', FamilyMethod.vector,
      dtype: 'int8', acc: AccRule.sameAsDtype),
  'integer-compute-short': FamilyInfo(
      'integer-compute-short', FamilyMethod.vector,
      dtype: 'int16', acc: AccRule.sameAsDtype),
  // Dot products sum into a wider accumulator.
  'int8-dot-product-compute': FamilyInfo(
      'int8-dot-product-compute', FamilyMethod.vector,
      dtype: 'int8'),
  'int16-dot-product-compute': FamilyInfo(
      'int16-dot-product-compute', FamilyMethod.vector,
      dtype: 'int16'),
  // E4M3 and E5M2 both: no one data type.
  'fp8-dot-product-compute': FamilyInfo(
      'fp8-dot-product-compute', FamilyMethod.vector),
  // fp32 and fp64 rows.
  'divide-sqrt-compute': FamilyInfo(
      'divide-sqrt-compute', FamilyMethod.vector,
      acc: AccRule.sameAsDtype),
  'integer-divide-compute': FamilyInfo(
      'integer-divide-compute', FamilyMethod.vector,
      dtype: 'uint64', acc: AccRule.sameAsDtype),
  'matrix-compute': FamilyInfo('matrix-compute', FamilyMethod.matrix),
  'gemm': FamilyInfo('gemm', FamilyMethod.library),
  'attention': FamilyInfo('attention', FamilyMethod.library),
  'convolution': FamilyInfo('convolution', FamilyMethod.graph),
  'numeric-error': FamilyInfo('numeric-error', FamilyMethod.graph),
  'smt-scaling': FamilyInfo('smt-scaling', FamilyMethod.vector,
      dtype: 'fp32', acc: AccRule.sameAsDtype),
  'aes': FamilyInfo('aes', FamilyMethod.crypto),
  'sha256': FamilyInfo('sha256', FamilyMethod.crypto),
  'sha512': FamilyInfo('sha512', FamilyMethod.crypto),
  'crc32c': FamilyInfo('crc32c', FamilyMethod.crypto),
  'string-scan': FamilyInfo('string-scan', FamilyMethod.string),
  'utf8-validate': FamilyInfo('utf8-validate', FamilyMethod.string),
  'global-memory-bandwidth':
      FamilyInfo('global-memory-bandwidth', FamilyMethod.memory),
  'local-memory-bandwidth':
      FamilyInfo('local-memory-bandwidth', FamilyMethod.memory),
  'image-memory-bandwidth':
      FamilyInfo('image-memory-bandwidth', FamilyMethod.memory),
  'transfer-bandwidth': FamilyInfo('transfer-bandwidth', FamilyMethod.memory),
  'tensor-bandwidth': FamilyInfo('tensor-bandwidth', FamilyMethod.graph),
  'activation': FamilyInfo('activation', FamilyMethod.graph),
  'cache-bandwidth': FamilyInfo('cache-bandwidth', FamilyMethod.memory),
  'texture-sample': FamilyInfo('texture-sample', FamilyMethod.memory),
  'kernel-launch-latency':
      FamilyInfo('kernel-launch-latency', FamilyMethod.latency),
  'memory-latency': FamilyInfo('memory-latency', FamilyMethod.latency),
  'atomics': FamilyInfo('atomics', FamilyMethod.latency),
  'branch-penalty': FamilyInfo('branch-penalty', FamilyMethod.latency),
  'store-forward': FamilyInfo('store-forward', FamilyMethod.latency),
  'transformer-block': FamilyInfo('transformer-block', FamilyMethod.graph),
};

/// The default of dim [key] for a reading of [family] (a Benchmark's flag
/// name) whose data type is [dtype] -- its dims', or else its family's: the
/// value an absent key means, or null for a key with none.  Null too for a
/// key this build does not know, which is identity everywhere it appears.
/// [family] and [dtype] may be ones this build does not know.
///
/// The same rule as `dimDefaultFor()` in src/common/vocabulary.cpp: the two
/// must agree on every key.
String? dimDefaultFor(String key, String family, String dtype) {
  final info = dimKeys[key];
  if (info == null) return null;
  final fam = families[family];
  final type = dtype.isNotEmpty ? dtype : (fam?.dtype ?? '');
  switch (info.def) {
    case DimDefault.none:
      return null;
    case DimDefault.value:
      return info.defaultValue;
    case DimDefault.familyDtype:
      return (fam != null && fam.hasDtype) ? fam.dtype : null;
    case DimDefault.sameAsDtype:
      return type.isEmpty ? null : type;
    case DimDefault.familyAcc:
      if (fam != null && fam.acc == AccRule.fixed) return fam.accType;
      if (!isKnownDataType(type)) return null;
      if (fam != null && fam.acc == AccRule.sameAsDtype) return type;
      return usualAccumulator(type);
  }
}

// ── Hardware ───────────────────────────────────────────────────────────────

/// A processor's kind, and a device's `type` (which adds `unknown`).
const processorKinds = <String>['cpu', 'gpu', 'npu', 'accelerator'];

/// A package: an SoC, a host CPU with nothing joined to it, or a card.
const packageKinds = <String>['soc', 'cpu', 'discrete'];

/// What linked a device to its processor (docs/format-v4.md, Linking).
const linkEvidence = <String>[
  'uuid', 'luid', 'pci_bus', 'registry', 'host', 'soc', 'sole', 'own',
];

const clusterKinds = <String>['prime', 'performance', 'efficiency', 'low_power'];

/// A cache in a processor's facts -- not the `level` dim, whose `l1` is
/// `l1d` here.
const cacheLevels = <String>['l1d', 'l1i', 'l2', 'l3', 'slc'];

const engineKinds = <String>['matrix'];
const engineScopes = <String>['core', 'cluster'];
const engineReaches = <String>['instructions', 'library'];

const vendors = <String>[
  'amd', 'apple', 'arm', 'google', 'imagination', 'intel', 'mediatek',
  'nvidia', 'qualcomm', 'samsung', 'microsoft', 'broadcom', 'huawei', 'mesa',
  'pocl', 'ampere', 'hygon', 'zhaoxin', 'fujitsu', 'marvell',
];

// ── System ─────────────────────────────────────────────────────────────────

const formFactors = <String>['mobile', 'laptop', 'desktop', 'server'];
const formFactorDetails = <String>['phone', 'tablet', 'embedded'];
const formFactorEvidence = <String>[
  'model', 'os', 'characteristics', 'battery', 'platform_role', 'chassis',
  'device_tree', 'default',
];
const powerSources = <String>['ac', 'battery'];

// ── Backends ───────────────────────────────────────────────────────────────

/// What a backend reaches: the GPU compute APIs, the native CPU backend, the
/// ML runtimes.  `portable` marks Vulkan, OpenCL and ONNX Runtime.
const backendKinds = <String>['gpu', 'cpu', 'ai'];
