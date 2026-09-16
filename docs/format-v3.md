---
title: Result format v3
---

# clpeak result format v3

`clpeak -o run.clpeak.json` writes one JSON document describing one run. It is
the only format clpeak writes and the only one it reads back (`--compare`, and
the GUI's run history).

The document is a tree — **run → devices → tests → metrics** — which is the
shape the CLI prints and the GUI renders, so nothing has to regroup a flat
table and guess at what belonged together. Beside the tree it carries the
run's **diagnostic stream** (`log`), so a file from a machine nobody can
reach explains its own numbers: with `--verbose` it holds everything the
terminal would have shown, scoped and timed, plus the device inventory.

Everything here is produced by `src/common/run_document.cpp` and modelled by
`include/common/run_document.h`; the Dart side mirrors it in
`app/lib/src/model/result_model.dart` and `run_document.dart`.

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

  "build": { "backends": ["CUDA", "Vulkan", "OpenCL", "CPU", "ONNX"] },

  "host": {
    "os": "Macintosh", "os_version": "26.6.2", "arch": "arm64",
    "cpu": "Apple M1 Pro", "logical_cores": 10, "memory_bytes": 34359738368
  },

  "invocation": {
    "argv": ["clpeak", "--verbose", "-o", "run.clpeak.json"],
    "target_time_us": 500000, "target_time_us_cpu": 2000000, "warmup": 2,
    "categories": ["compute", "bandwidth", "latency"],
    "verbose": true
  },

  "devices": [{
    "backend": "CUDA", "platform": "CUDA",
    "name": "NVIDIA GeForce RTX 5060", "driver": "580.65.06",
    "type": "gpu", "device_index": 0,
    "properties": [ { "key": "Arch", "value": "sm_120" } ],

    "tests": [{
      "id": "cublas_gemm_fp",
      "title": "cuBLASLt GEMM peak",
      "category": "compute",
      "shape": "heterogeneous",
      "axis": "data type",
      "direction": "higher_is_better",
      "quantity": "flops",
      "unit": "FLOPS",
      "description": "Matrix-multiply speed through NVIDIA's own tuned library…",
      "duration_s": 3.42,
      "metrics": [
        { "id": "fp32", "value": 14.87 },
        { "id": "nvf4_e2m1", "value": 300.43,
          "description": "Four-bit floats with NVIDIA's own scaling…" },
        { "id": "mxf4_e2m1", "status": "unsupported",
          "reason": "FP4 tensor cores require Blackwell — unsupported on sm_120" }
      ]
    }]
  }],

  "inventory": [
    { "name": "CUDA", "flag": "cuda", "available": true,
      "platforms": [{ "index": 0, "name": "CUDA",
        "devices": [{ "index": 0, "name": "NVIDIA GeForce RTX 5060", "type": "GPU",
                      "arch": "sm_120", "compute_units": 36, "global_mem_bytes": 8589934592 }] }] },
    { "name": "ONNX", "flag": "onnx", "available": false,
      "reason": "onnxruntime library not found", "platforms": [] }
  ],

  "log": [
    {"elapsed_s": 0.31, "level": "warning", "backend": "ONNX", "message": "ONNX: onnxruntime library not found"},
    {"elapsed_s": 4.2, "level": "debug", "backend": "CUDA", "device": "NVIDIA GeForce RTX 5060", "device_index": 0, "test": "global_memory_bandwidth", "message": "global_memory_bandwidth: working set 1024 MB, L2 32 MB"},
    {"elapsed_s": 9.77, "level": "warning", "source": "vulkan", "backend": "Vulkan", "device": "NVIDIA GeForce RTX 5060", "device_index": 0, "test": "kernel_latency", "message": "query timeout (VK_ERROR_DEVICE_LOST)"},
    {"elapsed_s": 12.4, "level": "error", "backend": "OpenCL", "device": "…", "device_index": 1, "message": "OpenCL: program build failed on … (clBuildProgram -11):\n<kernel>:14:3: error: …"}
  ]
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
| `build` | what the binary is made of: `backends` lists every backend compiled into it, in run order. The first question on a "backend X is missing" report is whether it was ever there |
| `host` | the machine, not its owner: no hostname, username, serial or MAC |
| `invocation` | how clpeak was asked to run. Every number is sensitive to it — a shorter `--max-time` measures a different thing, and a selective run is not a full one even though the file is the same shape |
| `devices` | one entry per benchmarked device |
| `inventory` | `--verbose` only: what `--list-devices` would have shown (see *Inventory*) |
| `log` | the diagnostic stream, in order (see *Log*). The warnings in it are usually the only record of *why* something is absent |

`invocation.tests` is written only when the run was narrowed to specific tests;
`invocation.iters` only when pinned with `-i` (absent means each test was
calibrated to a time budget, which is the normal and comparable mode);
`invocation.verbose` only when `--verbose` was on — so an empty debug log
reads as "was not asked", not "nothing happened".

`generated_at` is the moment the run **started**: `log[].elapsed_s` counts
from it.

## Device

`backend` / `platform` / `name` / `device_index` identify it. `device_index`
is the backend's own numbering -- the one `--list-devices` prints and
`--device` takes -- and runs consecutively across an OpenCL backend's
platforms. All three strings are whitespace-trimmed: drivers pad them (Intel's OpenCL runtime
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
| `category` | `compute` \| `crypto` \| `string` \| `bandwidth` \| `latency` \| `ai` \| `unknown` |
| `shape` | `homogeneous` \| `heterogeneous` — see below |
| `axis` | what varies across the readings (see below). Optional |
| `direction` | `higher_is_better` \| `lower_is_better` |
| `quantity`, `unit` | see *Units* |
| `description` | what the test measures, in plain language |
| `duration_s` | wall-clock seconds the test was open (summed over reopens). A test that took forty seconds against a half-second budget is its own diagnosis |
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
| `unit`, `quantity` | present only when this reading overrides its test's |
| `direction` | present only when this reading overrides its test's |

The unit override is what lets one heterogeneous test hold both TFLOPS and
TOPS readings, instead of being split into the `-fp` / `-int` twins that older
clpeak emitted purely so the unit string could differ.  It is set on skipped
readings too: an unsupported int8 row still has to say it would have been ops
and not flops, or it reads as its test's unit and claims something false.

## Units

A reading is stored **in SI** — `FLOP/s`, `byte/s`, `s`, `Texel/s`, `ppm` —
with the unit that explains it:

```jsonc
"quantity": "flops",   // what is measured
"unit": "FLOPS"        // display symbol, ready to print
```

Values are already SI so the presenters pick `G/T/P` or `µ/n` via the SI
ladder.

`quantity` is one of `flops`, `ops`, `bytes_per_second`, `seconds`,
`items_per_second`, `ratio`, `count`, `unknown`. The last three have no SI
ladder to slide along, so a presenter prints them exactly as measured.

The table mapping clpeak's internal unit tokens (`flops`, `bps`, `s`, `ppm`,
…) to these fields is `src/common/units.cpp`; a token missing from it passes
through as its own symbol with `quantity: "unknown"`, so a new unit
appears correctly in the output before anyone adds it there.

## Log

Everything clpeak had to say outside a reading, in emission order — the
terminal transcript, as data. One JSON object per line in the pretty-printed
file, so it reads and greps like a transcript while the rest of the document
keeps one key per line;
`jq -r '.log[] | "\(.elapsed_s) [\(.level)] \(.message)"' run.clpeak.json`
reproduces what the CLI printed.

| key | meaning |
|---|---|
| `elapsed_s` | seconds since the run started (`generated_at`). Where the run stalled is visible from the gaps |
| `level` | `error` \| `warning` \| `info` \| `debug` |
| `source` | the library a message was relayed from: `onnxruntime` (its logger), `vulkan` (a `VK_EXT_debug_utils` messenger), `opencl` (the context's error callback), `console` (output captured off stdout/stderr around a call that prints). **Absent for clpeak's own messages** |
| `backend`, `device`, `device_index`, `test` | the scope the line fired in — `test` is the test's key (`id@variant`), joinable to `tests[]`. Each is absent when no such scope was open: a message between two backends carries none |
| `message` | the text. Embedded newlines are kept — a compiler's build log is its line structure |

What each level means, and when it is in the file:

| level | what | recorded |
|---|---|---|
| `error` | something failed that was expected to work: a driver init, a kernel build (the build log is the message), a runtime call | always |
| `warning` | why something is absent or partial: a library not found, a device that would not init, a provider that declined a graph | always |
| `info` | a fact worth keeping, not worth printing | always |
| `debug` | the trace a maintainer reads when a number looks wrong: working-set sizes, calibration decisions, chain comparisons, tile choices, the runtimes' own narration | `--verbose` only |

The CLI prints its own warnings and errors inline with the results and
everything else to stderr under `--verbose`; the GUI shows warnings and errors
at the foot of a run's results and folds the rest beneath them. The file has
it all regardless.

The log is capped at 4 MiB of message text; past that, `debug` and `info`
entries are dropped and a final `warning` says how many. Console captures
record the first 200 lines of a call and count the rest.

### The run-log sidecar

With `-o`, every entry is also appended — and flushed — to a sidecar as it
happens: the output path with `.json` swapped for `.log` (`run.clpeak.json` →
`run.clpeak.log`), one JSON object per line under a header line naming the
run:

```jsonc
{"schema": "clpeak/run-log", "format_version": 3, "clpeak_version": "…", "generated_at": "…", "build": {…}, "host": {…}, "invocation": {…}}
{"elapsed_s": 0.31, "level": "warning", "backend": "ONNX", "message": "…"}
…
```

The document is written when the run ends. A native crash inside a driver —
the case `--verbose` exists for — means it never is, and the sidecar is then
the only record of the run; its last line is usually where. It is removed once
the document has been saved, so one that outlives its run is the record of a
run the process died in. The GUI adopts one it finds on its next launch and
offers it for export.

## Inventory

`--verbose` only. The `backends` array of the device catalog, exactly as the
GUI's run screen and `--list-devices` see it, taken before the run: every
backend compiled in, whether it was `available` and if not its `reason`, any
`notes` from enumeration (a provider the runtime names but nothing here can
run), and every device each backend saw — including the ones a `--device`
list left out. It is the evidence for "my NPU is not listed", which is a
debugging question; in the CLI it costs an enumeration pass, which is why it
is not on by default.

Per device: `index`, `name`, `type`, and whichever of `arch`, `driver`,
`api`, `compute_units`, `clock_mhz`, `global_mem_bytes`, `max_alloc_bytes`,
`fp16`, `fp64` the backend could answer. The keys are those of
`include/common/inventory.h`, which is the one serializer for both uses.

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
| `unit: "gflops"` | `unit: "FLOPS"` + `quantity` |
| direction inferred from the unit, in the GUI only | `direction`, per test, resolved natively |
| — | `shape`, `axis`, `variant` |
| — | `generated_at`, `duration_s`, `cancelled`, `build`, `host`, `invocation` |
| `--verbose` to a terminal, or nowhere at all on a phone | `log` (every level, scoped and timed), `inventory`, per-test `duration_s`, the run-log sidecar |
| ISA slugged into the test tag | `id` + `variant` |
| `category: ""` for unknown | `category: "unknown"` |
