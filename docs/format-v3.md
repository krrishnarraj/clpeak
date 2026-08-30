---
title: Result format v3
---

# clpeak result format v3

`clpeak -o run.clpeak.json` writes one JSON document describing one run. It is
the only format clpeak writes and the only one it reads back (`--compare`, and
the GUI's run history).

The document is a tree — **run → devices → tests → metrics** — which is the
shape the CLI prints and the GUI renders, so nothing has to regroup a flat
table and guess at what belonged together.

Everything here is produced by `src/common/run_document.cpp` and modelled by
`include/common/run_document.h`; the Dart side mirrors it in
`app/lib/src/model/result_model.dart`.

---

## Example

```jsonc
{
  "schema": "clpeak/run",
  "format_version": 3,
  "clpeak_version": "3.0.0-12-gabc1234",
  "generated_at": "2026-08-29T14:03:11Z",
  "duration_s": 148.24,
  "cancelled": false,

  "host": {
    "os": "Macintosh", "os_version": "26.6.2", "arch": "arm64",
    "cpu": "Apple M1 Pro", "logical_cores": 10, "memory_bytes": 34359738368
  },

  "invocation": {
    "argv": ["clpeak", "-o", "run.clpeak.json"],
    "target_time_us": 500000, "target_time_us_cpu": 2000000, "warmup": 2,
    "categories": ["fp_compute", "bandwidth", "latency"]
  },

  "notes": [
    { "backend": "ONNX", "message": "QNN execution provider not found" }
  ],

  "devices": [{
    "backend": "CUDA", "platform": "CUDA",
    "name": "NVIDIA GeForce RTX 5060", "driver": "580.65.06",
    "type": "gpu", "device_index": 0,
    "properties": [ { "key": "Arch", "value": "sm_120" } ],

    "tests": [{
      "id": "cublas_gemm_fp",
      "title": "cuBLASLt GEMM peak",
      "category": "fp_compute",
      "shape": "heterogeneous",
      "axis": "data type",
      "direction": "higher_is_better",
      "quantity": "flops",
      "unit": "TFLOPS",
      "scale": 1e12,
      "description": "Matrix-multiply speed through NVIDIA's own tuned library…",
      "metrics": [
        { "id": "fp32", "value": 14.87 },
        { "id": "nvf4_e2m1", "value": 300.43,
          "description": "Four-bit floats with NVIDIA's own scaling…" },
        { "id": "mxf4_e2m1", "status": "unsupported",
          "reason": "FP4 tensor cores require Blackwell — unsupported on sm_120" }
      ]
    }]
  }]
}
```

---

## Run

| key | meaning |
|---|---|
| `schema` | always `"clpeak/run"` |
| `format_version` | `3`. A reader rejects any other value outright rather than half-parsing it |
| `clpeak_version` | the git-describe of the build that produced the file |
| `generated_at` | ISO-8601 UTC |
| `duration_s` | wall-clock seconds for the whole run |
| `cancelled` | present and `true` only for a run stopped part-way. **A cancelled run is a partial one** — without this flag every test it never reached would read as hardware that lacks the feature |
| `host` | the machine, not its owner: no hostname, username, serial or MAC |
| `invocation` | how clpeak was asked to run. Every number is sensitive to it — a shorter `--max-time` measures a different thing, and a selective run is not a full one even though the file is the same shape |
| `notes` | messages emitted outside any reading (a missing library, a driver warning). Usually the only record of *why* something is absent |
| `devices` | one entry per benchmarked device |

`invocation.tests` is written only when the run was narrowed to specific tests;
`invocation.iters` only when pinned with `-i` (absent means each test was
calibrated to a time budget, which is the normal and comparable mode).

## Device

`backend` / `platform` / `name` / `device_index` identify it. All three
strings are whitespace-trimmed: drivers pad them (Intel's OpenCL runtime
returns its CPU name with five trailing spaces), and padding in an identity is
two names for one device the day a driver changes how much of it there is. The index is
part of that identity because a name is not unique — MoltenVK exposes one GPU
twice, and a multi-GPU box has N identical cards; without it their readings
fold into one block and every test ends up with two of everything. `driver` is
deliberately **not** identity, so a baseline stays comparable across a driver
update. `type` is `gpu` | `cpu` | `accelerator` | `unknown`. `properties` are
free-form facts the backend chose to report (compute units, VRAM, clocks).

## Test

| key | meaning |
|---|---|
| `id` | canonical tag, stable across machines and runs |
| `title` | human-readable name |
| `variant` | runtime qualifier that is *not* part of the identity — a CPU ISA (`AVX2+FMA`), a GPU arch, a library version. Two variants of one test are two tests; their key is `id@variant` |
| `category` | `fp_compute` \| `int_compute` \| `crypto` \| `string` \| `bandwidth` \| `latency` \| `ai` \| `unknown` |
| `shape` | `homogeneous` \| `heterogeneous` — see below |
| `axis` | what varies across the readings (see below). Optional |
| `direction` | `higher_is_better` \| `lower_is_better` |
| `quantity`, `unit`, `scale` | see *Units* |
| `description` | what the test measures, in plain language |
| `metrics` | the readings |

### `shape` — the one thing that cannot be inferred

- **`homogeneous`** — the readings are interchangeable variants of one
  measurement: `float` / `float2` / `float4`, `int8_dp` chain depths, a CPU
  kernel at one thread and at all of them, or a test with a single reading. The
  best of them *is* the test's answer, so a presenter may collapse the test to
  that number.

- **`heterogeneous`** — each reading is its own measurement: cuBLASLt's nine
  datatypes, `memory_latency`'s L1/L2/L3/DRAM, transfer's h2d vs d2h,
  `smt_scaling`'s two thread counts (where the *comparison* is the result).
  There is no single answer, and picking the largest reading invents one.

Nothing else in the document determines it. `mps_attention` has one reading and
is homogeneous; `mps_gemm` has three and is not; both are TFLOPS. The same tag
even differs by backend — a GPU's `global_memory_bandwidth` is a vector-width
sweep, the CPU's is read/copy/triad. So it is authored in the backend, at the
`beginTest()` call site, next to the description.

`heterogeneous` is the default, which means an unclassified test is verbose
rather than wrong.

### `axis` — what varies

A short noun phrase, shown by the GUI as the header over a heterogeneous test's
readings and by `--describe` as "Readings vary by …". The vocabulary in use:

| | |
|---|---|
| what the instruction is fed | `data type` · `pixel format` · `convolution shape` |
| how much work is in flight | `vector width` · `chains in flight` · `threads` · `contention` |
| where the data is | `cache level` · `memory level` · `weight size` · `direction` |
| what is being done | `operation` · `operation and size` · `what is submitted` |
| how much context | `prompt length` · `phase and context length` |

It is optional, and left empty where no single noun covers the readings —
kernel-launch latency measures a one-way cost and a full round trip, and an
invented word for that pair would read worse than none.

## Metric

| key | meaning |
|---|---|
| `id` | stable slug within the test (`fp8_e4m3`, `DRAM x8`) |
| `label` | display form; **omitted when it equals `id`**, which is the usual case |
| `value` | the reading, in the test's `unit`. Present exactly when the reading succeeded |
| `status` | `unsupported` \| `skipped` \| `error`. **Omitted for a successful reading** — every row of a healthy file is `ok`, so spelling it out on each would be noise |
| `reason` | why, for a non-`ok` reading |
| `description` | what this one reading means |
| `unit`, `quantity`, `scale` | present only when this reading overrides its test's |
| `direction` | present only when this reading overrides its test's |

The unit override is what lets one heterogeneous test hold both TFLOPS and
TOPS readings, instead of being split into the `-fp` / `-int` twins that older
clpeak emitted purely so the unit string could differ.  It is set on skipped
readings too: an unsupported int8 row still has to say it would have been ops
and not flops, or it reads as its test's unit and claims something false.

## Units

A reading is stored **as measured**, with enough alongside it to normalize:

```jsonc
"quantity": "flops",   // what is measured
"unit": "TFLOPS",      // display symbol, ready to print
"scale": 1e12          // value * scale  ->  SI base unit (FLOP/s here)
```

`scale` is what makes readings comparable. clpeak reports GFLOPS in one test
and TFLOPS in the next, µs in one latency test and ns in another, so a bare
number means nothing on its own. Multiplying by `scale` puts every reading in
its quantity's SI base unit, which is how one formatter serves every test
instead of a switch over unit strings.

`quantity` is one of `flops`, `ops`, `bytes_per_second`, `seconds`,
`items_per_second`, `ratio`, `count`, `unknown`. The last three have no SI
ladder to slide along, so a presenter prints them exactly as measured.

The table mapping clpeak's internal unit tokens (`gflops`, `us`, `ppm`, …) to
these fields is `src/common/units.cpp`; a token missing from it passes through
as its own symbol with `quantity: "unknown"` and `scale: 1`, so a new unit
appears correctly in the output before anyone adds it there.

## Conventions

- **Absent means default.** Optional fields are omitted rather than written
  empty, so a file carries only facts. `status` absent ⇒ `ok`; `label` absent ⇒
  `id`; metric unit fields absent ⇒ the test's.
- **Numbers use the classic locale**, always `.` — the GUI hosts the writer
  inside toolkits that call `setlocale(LC_ALL, "")`, and a comma decimal
  separator would produce a file that is not JSON at all.
- **Seven significant digits** on every value: enough to round-trip a float's
  precision, enough to keep a six-digit GFLOPS reading whole, and — unlike the
  fixed four decimals v2 used — it does not flatten the parts-per-million
  numeric-error readings to `0.0000`.

## What changed from v2

v2 wrote three formats (XML, CSV, JSON) of one flat table of rows, and carried
no notion of whether a test's readings were comparable to each other. v3 is one
nested JSON document that says so.

There is no migration path and no compatibility shim: a v2 file is rejected
with a message naming its version. Regenerate it.

| v2 | v3 |
|---|---|
| `--xml-file` / `--json-file` / `--csv-file` | `-o, --output` |
| flat `entries[]` (JSON) / `run → category → test → metric` (XML) | `devices[] → tests[] → metrics[]` |
| `unit: "gflops"` | `unit: "GFLOPS"` + `quantity` + `scale` |
| direction inferred from the unit, in the GUI only | `direction`, per test, resolved natively |
| — | `shape`, `axis`, `variant` |
| — | `generated_at`, `duration_s`, `cancelled`, `host`, `invocation`, `notes` |
| ISA slugged into the test tag | `id` + `variant` |
| `category: ""` for unknown | `category: "unknown"` |
