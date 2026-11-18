---
title: Result format v4
---

# XpuPeak result format v4

> **Work in progress.** Format 4 stays open to change until XpuPeak 4.0.0 is
> released, and then freezes: from 4.0.0 on it changes only by adding
> (*Compatibility*). What has changed from v3 so far is listed at the end.
>
> Marks like *(V3)* name the 4.0 change that fills a field. A field so marked
> is in the format and read by every reader, but not yet written by every
> build. The marks go before 4.0.0.

`xpupeak -o run.xpupeak.json` writes one JSON document describing one run. It
is the only format XpuPeak writes and the only one it reads back (`--compare`,
`--resume`, and the GUI's run history).

The document is a tree — **run → devices → tests → metrics** — which is the
shape the CLI prints and the GUI renders, so nothing has to regroup a flat
table and guess at what belonged together. Beside the tree it describes the
**hardware** the run happened on: the system, its packages and the processors
in them, which every device names. A device is one backend's view of a
processor, so the GPU that Metal, Vulkan and OpenCL each measured is one
processor with three devices. And every test and reading says what it
measured in words all backends share — a **family** and **dims** — so that
readings from different backends, devices and runs can be compared.

It also carries the run's **diagnostic stream** (`log`), so a file from a
machine nobody can reach explains its own numbers: with `--verbose` it holds
everything the terminal would have shown, scoped and timed, plus the device
inventory.

Everything here is produced by `src/common/run_document.cpp` and modelled by
`include/common/run_document.h`; the words come from
`include/common/vocabulary.h`. The Dart side mirrors them in
`app/lib/src/model/`.

Every key below is marked with its class (*Compatibility*): **C** contract,
**I** informational, **D** diagnostic.

---

## Example

Abbreviated; the values are illustrative.

```jsonc
{
  "schema": "xpupeak/run",
  "format_version": 4,
  "format_minor": 0,
  "run_id": "6f1c2e8a-0d4b-4a59-9a77-3b1f0c2d9e10",
  "generator": { "name": "XpuPeak", "version": "4.0.0-40-gabc1234" },
  "generated_at": "2026-11-09T14:03:11Z",
  "duration_s": 148.24,

  "build": {
    "backends": [
      {"name": "Metal", "kind": "gpu"},
      {"name": "Vulkan", "kind": "gpu", "portable": true},
      {"name": "CPU", "kind": "cpu"},
      {"name": "ONNX", "kind": "ai", "portable": true}
    ],
    "config": "Release"
  },

  "invocation": {
    "argv": ["xpupeak", "-o", "run.xpupeak.json"],
    "target_time_us": 500000, "target_time_us_ml_gpu": 1000000,
    "target_time_us_cpu": 2000000, "warmup": 2,
    "categories": ["compute", "crypto", "string", "bandwidth", "latency", "ai"],
    "standard": true
  },

  "system": {
    "form_factor": "laptop", "form_factor_from": "battery",
    "manufacturer": "Apple", "model": "MacBookPro18,3",
    "name": "MacBook Pro (14-inch, 2021)", "soc": "Apple M1 Pro",
    "os": { "name": "macOS", "version": "27.2" }, "arch": "arm64",
    "memory_bytes": 34359738368
  },
  "conditions": { "power_source": "ac", "low_power_mode": false },
  "annotations": { "cooling": "stock" },

  "packages": [ { "id": "pkg0", "kind": "soc", "vendor": "apple", "name": "Apple M1 Pro" } ],
  "processors": [
    { "id": "cpu0", "kind": "cpu", "package": "pkg0", "vendor": "apple", "name": "Apple M1 Pro",
      "isa": "arm64", "cores": 10, "threads": 10,
      "clusters": [ { "kind": "performance", "cores": 8, "max_mhz": 3228 },
                    { "kind": "efficiency", "cores": 2, "max_mhz": 2064 } ],
      "engines": [ {"kind": "matrix", "name": "AMX", "scope": "cluster", "reached_by": "library"} ],
      "features": [ "neon", "fp16", "dotprod", "fhm", "aes", "sha2", "sha512", "crc32" ] },
    { "id": "gpu0", "kind": "gpu", "package": "pkg0", "vendor": "apple", "name": "Apple M1 Pro",
      "arch": "Apple7", "integrated": true, "compute_units": 16, "compute_unit": "core" },
    { "id": "npu0", "kind": "npu", "package": "pkg0", "vendor": "apple",
      "name": "Apple Neural Engine", "compute_units": 16 }
  ],

  "devices": [
    {
      "backend": "Metal", "platform": "Metal", "name": "Apple M1 Pro",
      "driver": "macOS 27.2", "type": "gpu", "device_index": 0,
      "processor": "gpu0", "link": "registry",
      "facts": { "vendor_id": "106b", "arch": "Apple7", "integrated": true, "compute_units": 16 },
      "evidence": {"registry": "9f2c41d0a7e3b5c8"},
      "properties": [ { "key": "Apple family", "value": "Apple7" } ],
      "tests": [
        {
          "id": "simdgroup_matrix", "title": "simdgroup_matrix peak",
          "benchmark": "matrix-compute", "revision": 2,
          "category": "compute", "shape": "heterogeneous", "axis": "data type",
          "direction": "higher_is_better", "quantity": "flops", "unit": "FLOPS",
          "description": "Matrix-multiply speed through the GPU's simdgroup matrices.",
          "duration_s": 3.42,
          "metrics": [
            { "id": "fp16", "dims": {"dtype": "fp16"}, "value": 4.1e12 },
            { "id": "bf16", "dims": {"dtype": "bf16"}, "status": "unsupported",
              "reason": "bf16 simdgroup matrices need M3 or newer" }
          ]
        }
      ]
    },
    {
      "backend": "CPU", "platform": "CPU", "name": "Apple M1 Pro", "type": "cpu",
      "device_index": 0, "processor": "cpu0", "link": "host", "evidence": {"host": true},
      "tests": [
        {
          "id": "single_precision_compute", "variant": "NEON",
          "benchmark": "single-precision-compute",
          "dims": {"dtype": "fp32", "isa": "neon", "simd_bits": 128},
          "…": "…",
          "metrics": [
            { "id": "float ST", "dims": {"scope": "core"}, "value": 1.03e11 },
            { "id": "float MT", "value": 8.6e11 }
          ]
        }
      ]
    },
    {
      "backend": "ONNX", "name": "Apple CoreML (Neural Engine)", "type": "npu",
      "device_index": 0, "processor": "npu0", "link": "soc",
      "runtime": { "provider": "CoreMLExecutionProvider", "version": "1.31.0" },
      "tests": [ "…" ]
    }
  ],

  "log": [
    {"elapsed_s": 0.31, "level": "warning", "backend": "ONNX", "message": "ONNX: onnxruntime library not found"},
    {"elapsed_s": 4.2, "level": "debug", "backend": "Metal", "device": "Apple M1 Pro", "device_index": 0, "test": "global_memory_bandwidth", "message": "global_memory_bandwidth: working set 1024 MB"}
  ]
}
```

---

## Run

| key | class | meaning |
|---|---|---|
| `schema` | C | always `"xpupeak/run"` |
| `format_version` | C | `4`. A reader rejects any other value outright rather than half-parsing it |
| `format_minor` | C | the additive revision of format 4 the file was written in; absent means `0`. A reader accepts every minor (*Compatibility*) |
| `run_id` | C | a random UUID per run, kept across `--resume`, so one run uploaded twice is recognised as one. It identifies the run, not who ran it |
| `generator` | C | what wrote the file: `name`, which is `XpuPeak` (under either command name, `xpupeak` or `clpeak`), and `version`, the git-describe of the build that produced it |
| `generated_at` | C | ISO-8601 UTC: the moment the run **started**; `log[].elapsed_s` counts from it |
| `duration_s` | I | seconds the run spent running, the time it was held or paused at a reading included; for a resumed run, its parts summed — the time between them is not the run's |
| `cancelled` | C | present and `true` only for a run stopped part-way. **A cancelled run is a partial one** — without this flag every test it never reached would read as hardware that lacks the feature |
| `build` | C | what the binary is made of (below) |
| `invocation` | C | how XpuPeak was asked to run (below). Every number is sensitive to it |
| `system` | C | the machine (*System*) |
| `conditions` | C | what the machine was doing (*Conditions*) |
| `annotations` | C | what only a person knows about the run (*Annotations*) |
| `resumed` | C | present only for a run continued after its process ended part-way: one entry per part after the first (*Resumed runs*) |
| `packages`, `processors` | C | the hardware (*Hardware*) |
| `devices` | C | one entry per benchmarked device |
| `ext` | — | experimental fields (*Compatibility*) |
| `inventory` | D | `--verbose` only: what `--list-devices` would have shown (*Inventory*) |
| `log` | D | the diagnostic stream, in order (*Log*). The warnings in it are usually the only record of *why* something is absent |

**`build`** lists every backend compiled into the binary, in run order, as
objects: `name` (as `devices[].backend` names it), `kind` — `gpu`, `cpu` or
`ai` (*Vocabulary*) — and `portable: true` for the layers that run on more
than one vendor's silicon (Vulkan, OpenCL, ONNX Runtime). The first question
on a "backend X is missing" report is whether it was ever there. `config` is
the CMake configuration it was built in (`Release`, `Debug`, …) when the
build stamped it: a debug build's host-side work runs unoptimised, and a slow
run from one is not the device's fault.

**`invocation`**: `argv` (I) is the command line. `target_time_us`,
`target_time_us_ml_gpu` and `target_time_us_cpu` are the `--max-time-gpu`,
`--max-time-ml-gpu` and `--max-time-cpu` budgets in microseconds, and
`warmup` the warm-up count. `categories` lists the categories enabled;
`tests` is written only when the run was narrowed to specific tests, by their
flag names; `iters` only when pinned with `-i` (absent means each test was
calibrated to a time budget, which is the normal and comparable mode);
`verbose` only when `--verbose` was on — so an empty debug log reads as "was
not asked", not "nothing happened". `standard` is `true` for a run that used
this build's default budgets, warm-up and iterations and ran every category
and test: the run a leaderboard ranks. Choosing backends and devices
(`--cuda`, `--devices`) leaves a run standard; absent means not standard.

## System

The machine, not its owner. Everything is optional: what the platform cannot
answer is absent, and what would identify the unit or its owner is never read
(*Never recorded*).

| key | class | meaning |
|---|---|---|
| `form_factor` | C | `mobile`, `laptop`, `desktop` or `server`. A reader takes an absent one as `desktop` |
| `form_factor_detail` | C | finer, when known: `phone` or `tablet` under mobile, `tablet` under laptop (a Windows slate), `embedded` under desktop |
| `form_factor_from` | C | what it was read from (below): `model`, `os`, `characteristics`, `battery`, `platform_role`, `chassis`, `device_tree`, or `default` when nothing spoke for it |
| `virtual` | C | `true` when the hardware that ran is not the system named: a virtual machine, an emulator or a simulator. The processors are then what the guest was shown |
| `translated` | C | `true` for a binary run under a translator: an x86 build under Rosetta or Prism, an ARM build under Android's native bridge |
| `manufacturer` | C | the system's maker as the OS spells it: `Apple`, `samsung`, DMI `sys_vendor` (`Dell Inc.`, `LENOVO`). Where the product name is a placeholder (below), the board's maker |
| `brand` | C | the brand it is sold under, where it is not the maker's own name: Android's `ro.product.brand` (`Redmi` and `POCO` under `Xiaomi`) |
| `model` | C | the model id: `MacBookPro18,3`, `iPhone16,1`, `SM-S921B`, DMI `product_name` (`XPS 15 9530`, or Lenovo's machine type, `21DD`), a device-tree board's model ("Raspberry Pi 5 Model B Rev 1.0") |
| `codename` | C | the maker's name for the device: Android's `ro.product.device` (`e1s`) |
| `name` | C | the marketing name, **only as the OS gives it** ("MacBook Pro (14-inch, 2021)"): Apple silicon's device tree, Android's `ro.product.marketname` where the OEM sets one, and the product version Lenovo keeps its name in ("ThinkPad X1 Carbon Gen 10"). No table in the tool maps a model id to a name. A display shows the `system.name` annotation, else `name`, else `model`, else `board`, else `soc` |
| `family` | C | DMI `product_family` / SMBIOS `SystemFamily`: `XPS`, `ThinkPad X1 Carbon Gen 10` |
| `board` | C | the board (DMI `board_name`, SMBIOS `BaseBoardProduct`), written only where the product name is a placeholder: the machine was built from parts, and the board is what names it |
| `soc` | C | the SoC: "Apple M1 Pro", Android's `ro.soc.manufacturer` + `ro.soc.model` ("QTI SM8650"); the CPU brand where there is no SoC name |
| `os` | C | `name`: `macOS`, `iOS`, `Android`, `Linux`, `Windows`, `Windows Server`. `version`, the user-visible release: `27.2`, `16`, Linux's `VERSION_ID` (`24.04`), `11 24H2`. `build`: `26B123`, `BP2A.250605.031`, `26100.2314`. `kernel`, the kernel's release where it is not the version (Darwin's, Linux's, Android's). On Linux `distribution`, os-release's `PRETTY_NAME` ("Ubuntu 24.04.1 LTS") |
| `arch` | C | the binary's architecture: `arm64`, `x86_64` |
| `memory_bytes` | C | physical memory |
| `ext` | — | experimental fields |

**Where it comes from**, per OS:

| OS | identity | `soc` | form factor, and `form_factor_from` | `virtual`, `translated` |
|---|---|---|---|---|
| macOS | `Apple`; `hw.model`; IORegistry `IODeviceTree:/product` `product-name` (Apple silicon) | `product-soc-name`, else the CPU brand | an internal battery among IOKit's power sources: laptop, else desktop (`battery`) | `kern.hv_vmm_present`; `sysctl.proc_translated` |
| iOS | `Apple`; `hw.machine`, or `SIMULATOR_MODEL_IDENTIFIER` in the simulator; no name | the CPU brand, where the OS gives one | mobile; `iPhone…` phone, `iPad…` tablet (`model`) | the simulator, whose hardware and kernel are the Mac's |
| Android | `ro.product.manufacturer`, `.brand`, `.model`, `.device`; `ro.product.marketname`, its partition copies, `ro.config.marketing_name`, `ro.vendor.oplus.market.name` | `ro.soc.manufacturer` + `ro.soc.model` | mobile; tablet where `ro.build.characteristics` says `tablet`, else phone, and no detail for a TV, a watch or a car (`characteristics`; `os` where the property is empty) | the emulator (`ro.kernel.qemu`, `ro.boot.qemu`, `ro.hardware` `ranchu` or `goldfish`); a kernel of another ISA than the binary's |
| Linux | DMI `sys_vendor`, `product_name`, `product_family`, `product_version`, `board_vendor`, `board_name`; on a board with no DMI, `/proc/device-tree/model`, made by the vendor its first `compatible` names | the CPU brand | DMI `chassis_type` (`chassis`), then ACPI's `pm_profile` (`platform_role`), then a battery under `/sys/class/power_supply` (`battery`), then a device-tree board (`device_tree`) | DMI vendor and product strings, `/sys/hypervisor`, the x86 `hypervisor` CPU flag, a `linux,dummy-virt` device tree; a kernel of another ISA than the binary's |
| Windows | SMBIOS through `HKLM\HARDWARE\DESCRIPTION\System\BIOS`: `SystemManufacturer`, `SystemProductName`, `SystemFamily`, `SystemVersion`, `BaseBoardManufacturer`, `BaseBoardProduct` | the CPU brand | `PowerDeterminePlatformRoleEx` (`platform_role`), then a battery (`GetSystemPowerStatus`, `battery`) | SMBIOS vendor and product strings — not the CPUID hypervisor bit, which virtualization-based security sets on bare metal; `IsWow64Process2` |

- **Chassis types**: 8, 9, 10, 14, 31, 32 → laptop; 17, 23, 25, 28, 29 →
  server; 3–7, 13, 15, 35, 36 → desktop; 30 → mobile, tablet; 33, 34 →
  desktop, embedded. Any other is no evidence.
- **Platform roles** (Windows', and Linux's `pm_profile`, which is the same
  ACPI field): desktop, workstation and appliance PC → desktop; mobile →
  laptop; slate → laptop, tablet; the server roles → server.
- **A device-tree board** with a battery is a laptop (an ARM Chromebook,
  Linux on a MacBook); one without is an embedded desktop (a Raspberry Pi, a
  Jetson).
- **A placeholder** is what firmware writes where the OEM wrote nothing —
  "System Product Name", "System manufacturer", "To Be Filled By O.E.M.",
  "Default string", "Not Applicable" and their kind, or a string with no
  letter or digit in it. It is never recorded as a maker, model or family.

## Conditions

What the machine was doing, which changes every number. All C; each absent
when not known.

| key | meaning |
|---|---|
| `power_source` | `ac` or `battery`, as the run starts. macOS: what IOKit says it draws from. Windows: `GetSystemPowerStatus`. Linux: `/sys/class/power_supply` — an adapter online, else the battery's charging state; a machine with no battery at all runs on `ac`, but not a VM. iOS and Android: what the app passes in (`xpupeak_set_host_facts`), which Android reads natively too where SELinux shows it the supplies |
| `low_power_mode` | `true` or `false`: Low Power Mode on macOS and iOS, energy saver on Windows, battery saver on Android (from the app) |
| `power_profile` | the OS's own name for it. Windows: the power plan (`balanced`, `high-performance`, `power-saver`, `ultimate-performance`) and, under Balanced, the power-mode overlay instead (`best-power-efficiency`, `balanced`, `better-performance`, `best-performance`); a plan a user made reads `custom`, since its GUID is the install's own. Linux: `/sys/firmware/acpi/platform_profile` (`low-power`, `balanced`, `performance`, …) |
| `cpu_governor` | Linux and Android: each cpufreq policy's `scaling_governor`, each distinct one once, in policy order and joined by `,` (`schedutil`, or `schedutil,performance` where the clusters differ) |
| `thermal_start`, `thermal_end` | the OS's thermal state at the start and the end of the run, in its own words: Apple's `nominal`, `fair`, `serious`, `critical` (`NSProcessInfo`), Android's `none`, `light`, `moderate`, `severe`, `critical`, `emergency`, `shutdown` (`AThermal`). A desktop OS keeps no such state |

A resumed run keeps the conditions its first part started in; `thermal_end`
is the last part's.

## Annotations

Facts only a person knows, as a flat map of strings (C). The tool never fills
it, and it never overwrites a measured fact: a display may prefer an
annotation, a reader keeps both. Its writers are `--annotate key=value`
*(V10)*, the app's Details editor *(V8)*, and a site that collects results,
at upload. Recommended keys:

| key | for |
|---|---|
| `system.name` | the marketing name the OS won't give, or a name for a custom desktop |
| `cooling` | `stock`, a cooler's model, `passive`, … |
| `power_limit` | a power or TDP limit set by hand |
| `overclock` | an overclock or undervolt |
| `memory` | the memory configuration: channels, speed, timings |
| `gpu_board` | the maker of a discrete GPU's board |
| `notes` | anything else |

Any other key is allowed; a reader shows what it does not know as written.

## Hardware

A **package** is one chip or card: an SoC, a CPU package, a discrete GPU. A
**processor** is one CPU, GPU or NPU in a package. Both are the linker's
answer (*Linking*), written before the file is saved. Ids are run-local and
deterministic for one machine: a processor's is its kind and its place among
that kind's — `cpu0`, `gpu1`, `npu0`, `accelerator0` — counted over the host
package's processors first, then the rest in order of first appearance in run
order, so `gpu0` is the same GPU on every run there. Packages are `pkg0`, the
host's, then one per processor outside it, in the same order. The host CPU
processor is always there, whether or not a device ran on it, and a device
that nothing links to another gets a processor of its own.

`packages[]`, all C:

| key | meaning |
|---|---|
| `id` | `pkg0`, `pkg1`, … |
| `kind` | `soc` (the host CPU and whatever shares its package: an integrated GPU, an on-die NPU), `cpu` (a host CPU with nothing joined), `discrete` (a card). Absent for a package of its own whose processor no API says is integrated or not |
| `vendor` | a vendor (*Vocabulary*), when known: the host package's from `system.soc`'s first word where it names one, else its CPU's |
| `name` | "Apple M1 Pro", "NVIDIA GeForce RTX 5060": the host package's is `system.soc` where there is one, else its CPU's; another's, its processor's |
| `ext` | experimental fields |

`processors[]`, all C. The facts are written on the processor itself, merged
from the most native backend that ran, in registry order: each field from the
first device that gives it, `vendor_id` and `device_id` together from the
first whose vendor id is a known PCI one, a count together with what it
counts. The host CPU's come from the CPU backend's device alone — another
backend's CPU driver reports its own vendor (Intel's OpenCL runtime on an AMD
CPU) — or, when the CPU backend did not run, are the host's brand, thread
count and ISA. Each device keeps its own API's (`devices[].facts`), so a
consumer can merge them differently.

| key | meaning |
|---|---|
| `id` | `cpu0`, `gpu0`, `npu0` |
| `kind` | `cpu`, `gpu`, `npu` or `accelerator` |
| `package` | its package's `id` |
| `vendor`, `vendor_name` | the vendor (*Vocabulary*), and the vendor's own string |
| `name` | its name |
| GPU and NPU facts | `vendor_id` and `device_id` (the PCI ids, lower-case hex — a model's ids, not a unit's), `arch` (`sm_120`, `gfx1201`, `Apple7`), `integrated`, `compute_units` and `compute_unit` (what one is called: `SM`, `CU`, `Xe-core`, `core`), `clock_mhz`, `memory_bytes` |
| CPU facts | `isa` (`arm64`, `x86_64`), `cores`, `threads`, `sockets` (a multi-socket CPU is one processor); `cpuid` (x86: `vendor`, `family`, `model`, `stepping`); `clusters[]` (`kind`, `cores`, `threads_per_core`, `max_mhz`, `microarch`, `midr`, `caches[]`); `caches[]` the whole processor shares (`level` — `l1d`, `l1i`, `l2`, `l3`, `slc` — `size_bytes` of one, `instances`, `line_bytes`); `engines[]`; `features[]`, one normalised list (`neon`, `dotprod`, `i8mm`, `avx512_vnni`, `amx_tile`, …) |
| `ext` | experimental fields |

**Engines.** A CPU's matrix engine is an engine of the CPU processor, not a
processor of its own: `kind: matrix`, a `name` (`AMX`, `SME`), a `scope` —
one per `core` (Intel's AMX) or one per `cluster` (Apple's AMX and M4's SME)
— and `reached_by`, `instructions` or `library` (Apple's AMX only through
Accelerate). Its readings are the CPU's `matrix-compute` and `gemm` readings;
a per-cluster engine's all-core number does not scale with cores.

**CPU facts** are the OS's and CPUID's word, for every CPU the OS has online,
and never a cache size XpuPeak measured: a level the OS lists without a size
has no `size_bytes`.

- A **cluster** is one kind of core: a MIDR and the OS's capacity and clock on
  Arm (a faster bin of one core is a cluster of its own), the PMU that claims
  the CPU on x86 (Intel's P- and E-cores), and the caches it reads; Apple's
  are its performance levels. They are listed fastest first, and `kind` goes
  by rank: `performance` down to `efficiency`, with `prime` above them on Arm
  and `low_power` below them on x86. A CPU of one kind of core has one
  cluster and no `kind`. Its `max_mhz` is the fastest of its cores', absent
  where the OS gives none (Apple silicon, Windows).
- A cache is its cluster's, counted in the cluster's `instances`, unless more
  than one kind of core, or every CPU, reads an instance of it: then it is in
  the processor's `caches[]`. So a Threadripper's four CCX L3s are its one
  cluster's, and a desktop Core's ring L3 is the processor's.
- `microarch` is what the MIDR names. x86 has none: `cpuid` is kept as it is,
  and naming a family and model is a reader's table, like a model id's name.
- `features` are the words of the CPU feature list (*Vocabulary*), as the
  running CPU and OS allow them, in the list's order.

## Device

| key | class | meaning |
|---|---|---|
| `backend`, `platform`, `name`, `device_index` | C | its identity within the run (below) |
| `driver` | C | the driver's version: metadata, not identity |
| `platform_index` | C | the platform's index, for a backend with platforms (OpenCL) |
| `type` | C | what the device is: `gpu`, `cpu`, `npu`, `accelerator` or `unknown`. An ML runtime's NPU is `npu`; `accelerator` is for every other device that is neither, Android's NNAPI among them |
| `processor` | C | the `id` of the processor it is a view of |
| `link` | C | what linked it there (*Linking*) |
| `facts` | C | what its API reported, typed: the processor facts above, as far as the API goes, with `vendor_name` the vendor's own string as the API gives it (`NVIDIA Corporation`). A vendor id is the API's: Apple's OpenCL answers `1027f00`, a CPU driver its own maker's |
| `evidence` | C | the identity evidence that links it, never an identifier itself (*Linking*) |
| `runtime` | C | an ML runtime's view of where it ran: `provider` (ONNX Runtime's execution provider), `accelerator` (Core ML's compute device — `neural_engine`, `gpu`, `cpu` —, LiteRT's — `npu`, `gpu`, `cpu` —, OpenVINO's `device_type`), `version` (ONNX Runtime's; LiteRT and Core ML report none) |
| `properties` | C / I | free-form facts the backend chose to report, `key` and `value` strings. The shape is contract; the content informational |
| `tests` | C | the tests |
| `ext` | — | experimental fields |

`backend` / `platform` / `name` / `device_index` identify a device within the
run. `device_index` is the backend's own numbering -- the one
`--list-devices` prints and `--devices` takes -- and runs consecutively
across an OpenCL backend's platforms. All three strings are
whitespace-trimmed: drivers pad them (Intel's OpenCL runtime returns its CPU
name with five trailing spaces), and padding in an identity is two names for
one device the day a driver changes how much of it there is. The index is
part of that identity because a name is not unique — MoltenVK exposes one GPU
twice, and a multi-GPU box has N identical cards; without it their readings
fold into one block and every test ends up with two of everything. `driver`
is deliberately **not** identity, so a baseline stays comparable across a
driver update.

### Linking

Each backend passes, with each device, what its API says about who the
device is, and what it says about the hardware (`facts`). Before the file is
saved, the tool merges the devices that are views of one processor, by these
rules and no others:

- **Host.** Every device that executes on the CPU is the host CPU: the CPU
  backend, llvmpipe and lavapipe, PoCL, Intel's CPU runtime, oneAPI's CPU
  device, and the CPU paths of ONNX Runtime, LiteRT and Core ML — every
  device with `host: true` or of type `cpu`.
- **Strong.** An equal UUID, LUID, PCI bus address or Metal registry ID
  merges. UUIDs are compared verbatim and only ever add a link: NVIDIA's
  CUDA, Vulkan and OpenCL report one scheme, other vendors' APIs each their
  own, so two different UUIDs say nothing. The PCI bus address is what links
  a discrete part across vendors' APIs, the LUID a Windows adapter, the
  registry ID an Apple GPU (Metal, and Core ML's GPU device, which names the
  same `MTLDevice`).
- **SoC.** An ML runtime's NPU (the Neural Engine, Hexagon, XDNA, Intel's
  NPU, LiteRT's vendor dispatch libraries) is the SoC's NPU: all of them are
  one processor, unless their vendors say otherwise. An Android GPU delegate
  is the SoC's GPU: it joins the one integrated GPU another backend sees, or
  is a processor of its own on the SoC. Android's NNAPI runs somewhere on the
  SoC and names no one unit: an `accelerator` of its own, on the SoC.
- **Sole.** The views of one vendor's GPU (NPU, accelerator) that nothing
  else links merge when the run shows only one: no backend lists two of that
  kind and vendor, and nothing says two of them are different devices — two
  vendors, two PCI device ids, one integrated and one not, two different
  LUIDs, PCI bus addresses or registry IDs, or one backend listing both. A
  GPU whose vendor is unknown and that nothing identifies — the platform's
  default GPU (LiteRT's WebGPU and Metal accelerators, which name no
  adapter), a runtime that names none — joins the run's one GPU on the same
  terms. Never one of several.
- **Nothing else merges.** A device left alone is a processor of its own.

A processor is in the host package when it is the host CPU, is on the SoC by
the rules above, or is integrated (`facts.integrated`); otherwise it is a
package of its own, `discrete` when its API says it is not integrated.

`link` says how a device came to its processor: the first of `uuid`, `luid`,
`pci_bus` and `registry` it shares with another device there; `host` or
`soc` for a device with that evidence; `sole` for one the sole rule joined,
and for the device it joined; `own` for a processor of its own.

`evidence` keeps what the linker saw, so that better rules can re-link an old
file: a token per strong identifier (`uuid`, `luid`, `pci_bus`, `registry`),
and the weaker evidence as it is — `host: true`, `soc: "npu"`, `"gpu"` or
`"accelerator"`, `default_gpu: true`. **The tokens are HMACs under a key made
per run**, which the result never holds (only the checkpoint does, for a
resume): the first 64 bits of HMAC-SHA256 over the evidence key's name, a zero
byte and the identifier in its one written form (a UUID as 32 lower-case hex
digits, a LUID and a registry ID as 16, a PCI bus address as `0000:01:00.0`),
as 16 hex digits. Two devices in one file can be seen to share an identifier;
nothing correlates across files, and the raw values never leave the process
— they are made into tokens as each device opens, so a checkpoint holds
tokens too. A key a newer build adds to `evidence` with a string value is a
token.

Every decision is a `debug` line on the log under `--verbose` — `hardware:
vulkan:0 (Apple M1 Pro) -> gpu0 by sole` — and so is every device a rule
left apart that it might have joined, with the reason (`litert:0 not linked:
its GPU is not named (Vulkan lists 2 GPUs)`). No line holds an identifier:
what a library relays has its identifiers replaced by their tokens (*Log*).

## Test

| key | class | meaning |
|---|---|---|
| `id` | C | canonical tag, stable across machines and runs; never reused for another measurement |
| `title` | I | human-readable name |
| `variant` | C | runtime qualifier that is *not* part of the measurement's identity — a CPU ISA (`AVX2+FMA`), a GPU arch, a library version. Two variants of one test are two tests; their key is `id@variant` |
| `benchmark` | C | the family the test measures, by its flag name (below) |
| `revision` | C | the test's methodology version; absent means `1` (below) |
| `dims` | C | the coordinates every reading of the test shares (below) |
| `category` | C | `compute` \| `crypto` \| `string` \| `bandwidth` \| `latency` \| `ai` \| `unknown` |
| `shape` | C | `homogeneous` \| `heterogeneous` — see below |
| `axis` | I | what varies across the readings (see below). Optional |
| `direction` | C | `higher_is_better` \| `lower_is_better` |
| `quantity`, `unit` | C | see *Units* |
| `description` | I | what the test measures, in plain language |
| `duration_s` | I | wall-clock seconds the test was open (summed over reopens). A test that took forty seconds against a half-second budget is its own diagnosis |
| `metrics` | C | the readings |
| `ext` | — | experimental fields |

### `benchmark` — the family

Test ids name an implementation: the vendor matmul is `cublas_gemm`,
`hipblaslt_gemm`, `mps_gemm`, `onnx_gemm`, `coreml_gemm`. `benchmark` names
what is measured, the same for all of them: the `--<test>` flag that selects
it, `gemm`. Every test has one. It is stamped where the backend admits the
Benchmark, with no call site saying so (`Peak::admit`,
`include/common/peak.h`), and the family table under *Vocabulary* gives each
one's method and data type.

A test can hold the readings of two families: the CPU's
`integer_dot_product` holds its int8 rows, measured under
`int8-dot-product-compute`, and its int16 rows, measured under
`int16-dot-product-compute`. The test names the family it was opened under,
and a reading of the other carries its own `benchmark`.

### `dims` — what a reading measured

A reading's coordinates, from a fixed vocabulary (*Vocabulary*):
`{"dtype": "fp16", "acc": "fp16"}`, `{"level": "l1", "scope": "core"}`. A
test's `dims` hold what all its readings share — a vector test's data type, a
CPU test's ISA — and a reading's own add to them and win over them. A value
is a string or a whole number, and stays so.

Every key has one fixed class:

- **Identity** keys must match for two readings to be comparable: the data
  type, the accumulator, a cache level, a direction.
- **Variant** keys are implementation choices, and the best reading is taken
  over them: a vector width, an ISA, a library, a race's winner.

Where one word means both it is two keys: a GEMM's `problem_size` is variant,
a bandwidth test's `working_set` identity. An identity key's default is what
its absence means (`sparsity` dense, `scope` the whole device, `acc` the
usual accumulator); a key without one is not applicable where absent, and
keyed wherever present. A key a reader does not know is identity: it never
merges what it does not understand.

**The measure key** of a reading is its family, its quantity, and its
identity dims that differ from their defaults, sorted:
`matrix-compute|flops|dtype=fp16`. Two readings with the same key — and the
same test `revision`, where both come from one backend's test — are
comparable across backends, devices and runs, and the best of a key on one
processor is that processor's peak for it. Peaks, best-of-backend views and
headlines are derived from the readings by the consumer, never stored in the
file (*Compatibility*). The `shape` still says how to present a test; the
dims say how to compare its readings.

Readings carry their dims as they are emitted.

### `revision`

A test's methodology version, absent meaning `1`. It goes up whenever a
change moves the test's numbers for a reason other than the hardware: a
kernel fix (the Metal simdgroup fold that read 4× high), a counting rule
(conv3x3 → stride 2), a different problem size, a new identity dim on its
readings. Readings of two revisions of one test are not compared, and a
ranking takes the newest.

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

| key | class | meaning |
|---|---|---|
| `id` | C | stable slug within the test (`fp8_e4m3`, `DRAM x8`); never reused for another measurement |
| `label` | I | display form; **omitted when it equals `id`**, which is the usual case |
| `benchmark` | C | the reading's family, present only when it is not its test's |
| `dims` | C | the reading's coordinates, over its test's |
| `value` | C | the reading, in the test's `unit`: always a finite number. Present exactly when the reading succeeded; a measurement that came out NaN or infinite is an `error` reading instead |
| `status` | C | `unsupported` \| `skipped` \| `error`. **Omitted for a successful reading** — every row of a healthy file is `ok`, so spelling it out on each would be noise |
| `reason` | I | why, for a non-`ok` reading |
| `description` | I | what this one reading means |
| `unit`, `quantity` | C | present only when this reading overrides its test's |
| `direction` | C | present only when this reading overrides its test's |
| `ext` | — | experimental fields |

The unit override is what lets one heterogeneous test hold both TFLOPS and
TOPS readings, instead of being split into the `-fp` / `-int` twins that v2
emitted purely so the unit string could differ.  It is set on skipped
readings too: an unsupported int8 row still has to say it would have been ops
and not flops, or it reads as its test's unit and claims something false. So
are a skipped reading's dims: they say which measurement is missing.

## Vocabulary

The words the format uses, from `include/common/vocabulary.h`. Every list is
append-only (*Compatibility*): a value's string never changes and is never
reused, and a reader shows a value it does not know as written.

### Data types

| kind | values |
|---|---|
| float | `fp64`, `fp32`, `tf32`, `fp16`, `bf16`, `fp8_e4m3`, `fp8_e5m2`, `fp8_e4m3fnuz`, `fp8_e5m2fnuz`, `fp6_e3m2`, `fp6_e2m3`, `fp4_e2m1` |
| block-scaled | `mxfp8_e4m3`, `mxfp8_e5m2`, `mxfp6_e3m2`, `mxfp6_e2m3`, `mxfp4`, `nvfp4` |
| integer | `int64`, `int32`, `int24`, `int16`, `int8`, `int4`, `uint64`, `uint16`, `uint8`, `uint4`, `uint32` |

A block-scaled format is a value of its own: an MX or NV block carries its
scales, so `mxfp4` is not `fp4_e2m1` with a dim. A mixed format is described
with dims — `weights`, `acc`, `kv`, `quant`, `sparsity` — never with a value
of its own. The **usual accumulator** of a type is `fp64` for fp64, `fp32`
for every other float, `int64` for the 64-bit integers and `int32` for the
rest.

### Dim keys

| key | class | value | default | meaning |
|---|---|---|---|---|
| `dtype` | identity | a data type | the family's (below) | the type the operation computes on: its inputs' |
| `weights` | identity | a data type | the reading's `dtype` | the stored weights' type, where it is not the activations' |
| `acc` | identity | a data type | the family's rule (below) | the accumulator's type |
| `quant` | identity | `none`, `qdq`, `lut`, `block`, `channel` | `none` | how quantized weights are scaled: quantize-dequantize pairs, a lookup table, a scale per block, a scale per output channel |
| `sparsity` | identity | `dense`, `2:4` | `dense` | structured sparsity |
| `scope` | identity | `core`, `device` | `device` | how much of the processor ran it: one thread on one core, or all of it |
| `threads_per_core` | identity | a number | — | threads per core, where SMT is what is measured |
| `level` | identity | `l1`, `l2`, `l3`, `slc`, `dram` | — | where the data lived |
| `pattern` | identity | `random`, `linear`, `page_stride`, `predictable` | `random` | the access pattern; `page_stride` puts every read on another page, `predictable` is a branch the CPU guesses right |
| `in_flight` | identity | a number | `1` | independent accesses in flight at once |
| `op` | identity | `read`, `write`, `copy`, `triad`, `fdiv`, `fsqrt`, `silu`, `softmax`, `layernorm`, `dispatch`, `roundtrip`, `mispredict` | — | the operation |
| `direction` | identity | `h2d`, `d2h`, `roundtrip` | — | which way the data moved between the host and the device |
| `memory` | identity | `pinned`, `pageable` | — | the host memory a transfer used |
| `working_set` | identity | bytes | — | bytes the reading's data occupies |
| `layer` | identity | `conv3x3s2`, `conv1x1`, `depthwise3x3` | — | a convolution layer's shape |
| `pixel_format` | identity | `rgba32f`, `rgba16f`, `rgba8`, `r32f` | — | an image's texel format |
| `phase` | identity | `prefill`, `decode` | — | a transformer block's phase |
| `context` | identity | tokens | — | a prefill's sequence length, a decode's KV-cache length |
| `contention` | identity | `none`, `all` | `none` | whether every other thread competed for the same data |
| `workload` | identity | `trivial_op`, `small_matmul`, `session_create`, `model_create` | — | what a dispatch-latency reading submits |
| `kv` | identity | a data type | the reading's `dtype` | a transformer block's KV-cache type |
| `bound` | identity | `throughput`, `latency` | `throughput` | what limits a compute kernel: enough independent chains to fill every pipe, as in every compute peak, or one chain whose every operation waits on the one before |
| `width` | variant | a number | — | vector width, in elements |
| `chains` | variant | a number | — | independent dependency chains in flight |
| `isa` | variant | `scalar`, `sse2`, `ssse3`, `sse4_2`, `avx`, `avx2`, `avx512`, `avx10`, `amx`, `neon`, `sve`, `sve2`, `sme`, `sme2` | — | the instruction set a CPU kernel was written in |
| `simd_bits` | variant | a number | — | vector register width in bits: 128, 256, 512, or the SVE or SME vector length |
| `impl` | variant | `cublaslt`, `hipblaslt`, `rocblas`, `rocwmma`, `onemkl`, `mps`, `mpsgraph`, `accelerate`, `bnns` | — | the library that ran it |
| `form` | variant | text | — | a spelling of the same work: a race's winner, a layout, a graph form |
| `tile` | variant | text | — | a matrix tile or instruction shape |
| `problem_size` | variant | a number | — | a GEMM's edge, where a test sweeps sizes |

How today's labels map, for example: `float` → `{dtype: fp32}` (on the
test); `mp` → fp16 with the family's fp32 accumulator; `bf16_2` →
`{width: 2}`; `int8_dp4` → `{chains: 4}` (independent chains, in Vulkan and
OpenCL alike), but OpenCL's integer-compute `int8` → `{width: 8}` on an
int32 test; `fp16 f16acc` → `{dtype: fp16, acc: fp16}`; `fp8_e4m3 2:4` →
`{dtype: fp8_e4m3, sparsity: "2:4"}`; `nvf4_e2m1` and `nvfp4` → `nvfp4`;
`int8_weight` → `{dtype: fp16, weights: int8, quant: block}` (ONNX's
blocks of 32; Core ML's is `quant: channel`, and LiteRT's, dynamic-range
quantization, `{dtype: fp32, weights: int8, quant: channel}`); `int4_lut` →
`{dtype: fp16, weights: int4, quant: lut}`; `int8_qdq` → `{dtype: int8,
quant: qdq}`; `int16x8` → `{dtype: int16, weights: int8, quant: qdq}`;
`fp16_acc32` → `{dtype: fp16, acc: fp32}`, against the LiteRT GPU's `fp16`
`{dtype: fp16, acc: fp16}`; `int8_qdq_conv1x1` → `{dtype: int8, quant: qdq,
form: conv1x1}`; `int8_kv` →
`{dtype: fp16, kv: int8}`; `float ST` → `{scope: core}`; `fdiv fp64 MT` →
`{dtype: fp64, op: fdiv}`; `DRAM x8` → `{level: dram, in_flight: 8}`;
`fp16_decode_kv2048` → `{dtype: fp16, phase: decode, context: 2048}`.

### Families

A family is a Benchmark, by its flag name. Its method is how it measures:
`vector` (ALU throughput in registers), `matrix` (a matrix engine through
intrinsics), `library` (a vendor's tuned library), `graph` (a model through
an ML runtime), `memory`, `latency`, `crypto` or `string`. A family of one
data type gives its readings' `dtype` default. `acc` defaults by the
family's rule: a vector family computes in its own type (an fp16 FMA
accumulates in fp16), and every other family accumulates in the type's usual
accumulator — except where the table says otherwise.

| family | method | dtype | acc |
|---|---|---|---|
| `single-precision-compute` | vector | fp32 | its own type |
| `half-precision-compute` | vector | fp16 | its own type |
| `double-precision-compute` | vector | fp64 | its own type |
| `mixed-precision-compute` | vector | fp16 | fp32 |
| `bfloat16-compute` | vector | bf16 | its own type |
| `integer-compute` | vector | int32 | its own type |
| `integer-compute-fast` | vector | int24 | its own type |
| `integer-compute-char` | vector | int8 | its own type |
| `integer-compute-short` | vector | int16 | its own type |
| `int8-dot-product-compute` | vector | int8 | usual |
| `int16-dot-product-compute` | vector | int16 | usual |
| `fp8-dot-product-compute` | vector | — | usual |
| `divide-sqrt-compute` | vector | — | its own type |
| `integer-divide-compute` | vector | uint64 | its own type |
| `matrix-compute` | matrix | — | usual |
| `gemm` | library | — | usual |
| `attention` | library | — | usual |
| `convolution` | graph | — | usual |
| `numeric-error` | graph | — | usual |
| `smt-scaling` | vector | fp32 | its own type |
| `aes`, `sha256`, `sha512`, `crc32c` | crypto | — | — |
| `string-scan`, `utf8-validate` | string | — | — |
| `global-memory-bandwidth`, `local-memory-bandwidth`, `image-memory-bandwidth`, `transfer-bandwidth`, `cache-bandwidth`, `texture-sample` | memory | — | — |
| `tensor-bandwidth`, `activation`, `transformer-block` | graph | — | usual |
| `kernel-launch-latency`, `memory-latency`, `atomics`, `branch-penalty`, `store-forward` | latency | — | — |

Each family's category is the category its flag is listed under in
`xpupeak --help`.

### Hardware and system

| list | values |
|---|---|
| processor kind, device `type` | `cpu`, `gpu`, `npu`, `accelerator` (and `unknown` for a device) |
| package kind | `soc`, `cpu`, `discrete` |
| `link` | `uuid`, `luid`, `pci_bus`, `registry`, `host`, `soc`, `sole`, `own` |
| cluster kind | `prime`, `performance`, `efficiency`, `low_power` |
| cache level | `l1d`, `l1i`, `l2`, `l3`, `slc` |
| engine | `kind`: `matrix`; `scope`: `core`, `cluster`; `reached_by`: `instructions`, `library` |
| CPU feature | x86: `sse2`, `ssse3`, `sse4_1`, `sse4_2`, `avx`, `avx2`, `fma`, `avx512f`, `avx512bw`, `avx512vl`, `avx512dq`, `avx512_vnni`, `avx512_bf16`, `avx512_fp16`, `avx_vnni`, `avx_vnni_int8`, `avx_vnni_int16`, `avx10`, `avx10_2`, `amx_tile`, `amx_int8`, `amx_bf16`, `amx_fp16`, `amx_fp8`, `vaes`; Arm: `neon`, `fp16`, `fhm`, `dotprod`, `bf16`, `i8mm`, `sve`, `sve2`, `sme`, `sme2`, `sme_f64f64`, `fp8dot4`; both: `aes`, `sha256`, `sha512`, `crc32`. One word per capability on every OS, the same word on both architectures where the capability is one; x86 words follow the Linux kernel's flags, Arm's are the FEAT_ names in lower case (`neon` for AdvSIMD) |
| form factor | `mobile`, `laptop`, `desktop`, `server` |
| form factor detail | `phone`, `tablet`, `embedded` |
| `form_factor_from` | `model`, `os`, `characteristics`, `battery`, `platform_role`, `chassis`, `device_tree`, `default` |
| power source | `ac`, `battery` |
| backend kind | `gpu` (CUDA, ROCm, Metal, oneAPI, Vulkan, OpenCL), `cpu` (CPU), `ai` (CoreML, LiteRT, ONNX); `portable`: Vulkan, OpenCL, ONNX |

**Vendors**, with the ids an API's vendor id maps from — PCI ids, and the
ids Khronos assigns above 0xFFFF to vendors with none — and a CPU's CPUID
vendor string or ARM MIDR implementer:

| vendor | PCI / Khronos ids | CPU |
|---|---|---|
| `amd` | 1002, 1022 | `AuthenticAMD` |
| `apple` | 106b | MIDR 0x61 |
| `arm` | 13b5 | MIDR 0x41 |
| `google` | 1ae0 | |
| `imagination` | 1010 | |
| `intel` | 8086, 8087 | `GenuineIntel`, MIDR 0x69 |
| `mediatek` | 14c3 | |
| `nvidia` | 10de | MIDR 0x4e |
| `qualcomm` | 5143, 17cb | MIDR 0x51 |
| `samsung` | 144d | MIDR 0x53 |
| `microsoft` | 1414 | MIDR 0x6d |
| `broadcom` | 14e4 | MIDR 0x42 |
| `huawei` | 19e5 | MIDR 0x48 |
| `mesa` | 10005 | |
| `pocl` | 10006 | |
| `ampere` | | MIDR 0xc0 |
| `hygon` | | `HygonGenuine` |
| `zhaoxin` | | `  Shanghai  `, `CentaurHauls` |
| `fujitsu` | | MIDR 0x46 |
| `marvell` | | MIDR 0x43, 0x56 |

## Units

A reading is stored **in SI** — `FLOP/s`, `byte/s`, `s`, `Texel/s`, `ppm` —
with the unit that explains it:

```jsonc
"quantity": "flops",   // what is measured
"unit": "FLOPS"        // display symbol, ready to print
```

Values are already SI so the presenters pick `G/T/P` or `µ/n` via the SI
ladder. Units never change: a value stays SI, and `quantity` names what it
counts.

`quantity` is one of `flops`, `ops`, `bytes_per_second`, `seconds`,
`items_per_second`, `ratio`, `count`, `unknown`. The last three have no SI
ladder to slide along, so a presenter prints them exactly as measured.

The table mapping XpuPeak's internal unit tokens (`flops`, `bps`, `s`, `ppm`,
…) to these fields is `src/common/units.cpp`; a token missing from it passes
through as its own symbol with `quantity: "unknown"`, so a new unit
appears correctly in the output before anyone adds it there.

## Log

Everything XpuPeak had to say outside a reading, in emission order — the
terminal transcript, as data (D). One JSON object per line in the
pretty-printed file, so it reads and greps like a transcript while the rest
of the document keeps one key per line;
`jq -r '.log[] | "\(.elapsed_s) [\(.level)] \(.message)"' run.xpupeak.json`
reproduces what the CLI printed.

| key | meaning |
|---|---|
| `elapsed_s` | seconds since the run started (`generated_at`) — across the gap of a resumed run too. Where the run stalled is visible from the gaps |
| `level` | `error` \| `warning` \| `info` \| `debug` |
| `source` | the library a message was relayed from: `onnxruntime` (its logger), `vulkan` (a `VK_EXT_debug_utils` messenger), `opencl` (the context's error callback), `console` (output captured off stdout/stderr around a call that prints), `app` (what the app running it says: that it left the screen or came back, that the run was paused or resumed -- a run holds at its next reading meanwhile), `android` (how the OS says the process ended; sidecar only, see below). **Absent for XpuPeak's own messages** |
| `backend`, `device`, `device_index`, `test` | the scope the line fired in — `test` is the test's key (`id@variant`), joinable to `tests[]`. Each is absent when no such scope was open: a message between two backends carries none |
| `message` | the text. Embedded newlines are kept — a compiler's build log is its line structure. A relayed message (one with a `source`) has every identifier in it — a PCI bus address, a UUID, a LUID or registry ID after its name — replaced by its token, `<pci_bus 7e05b9d1c2a84f36>`, the one the device's `evidence` holds (*Linking*): a runtime's own log names the devices it found |

What each level means, and when it is in the file:

| level | what | recorded |
|---|---|---|
| `error` | something failed that was expected to work: a driver init, a kernel build (the build log is the message), a runtime call | always |
| `warning` | why something is absent or partial: a library not found, a device that would not init, a provider that declined a graph | always |
| `info` | a fact worth keeping, not worth printing | always |
| `debug` | the trace a maintainer reads when a number looks wrong: working-set sizes, calibration decisions, chain comparisons, tile choices, the runtimes' own narration | `--verbose` only |

Every run also records where it is: an `info` line as each backend, device
and test starts (`Backend: …`, `Device 0: …`, the test's title). Under
`--verbose` the readings join them, one line each as the terminal showed them,
so the log is the run's transcript. (The sidecar below always has them.)

The CLI prints its own warnings and errors inline with the results and
everything else to stderr under `--verbose`; the GUI shows warnings and errors
at the foot of a run's results and folds the rest beneath them. The file has
it all regardless.

Only `debug` is capped: past 64 MiB of debug text the log drops further
`debug` entries, with a `warning` where it starts and a final one saying how
many. `info`, `warning` and `error` are never dropped. Console captures record
the first 200 lines of a call and count the rest.

### The run-log sidecar

With `-o`, every entry is also appended — and flushed — to a sidecar as it
happens (D): the output path with `.json` swapped for `.log`
(`run.xpupeak.json` → `run.xpupeak.log`), one JSON object per line under a
header line naming the run:

```jsonc
{"schema": "xpupeak/run-log", "format_version": 4, "format_minor": 0, "run_id": "…", "generator": {"name": "XpuPeak", "version": "…"}, "generated_at": "…", "build": {…}, "invocation": {…}, "system": {…}}
{"elapsed_s": 0.31, "level": "warning", "backend": "ONNX", "message": "…"}
…
```

The document is written when the run ends. A native crash inside a driver —
the case `--verbose` exists for — means it never is, and the sidecar is then
the only record of the run; its last line is usually where. It is removed once
the document has been saved, so one that outlives its run is the record of a
run the process died in. The GUI adopts one it finds on its next launch and
offers it for export, and beside its checkpoint (below) to continue the run.
A resumed run goes on writing the same sidecar.

Being that record, the sidecar is not the document's `log` line for line. It
carries every reading as an `info` line with or without `--verbose`, since a
run that dies leaves no `devices` to hold them, and it caps `debug` on its own
at 128 MiB, again with a `warning` where it starts dropping, so the minutes
before a late crash are still in it.

On Android the app adds one line of its own to such a sidecar on its next
launch: an `error` with `source: "android"`, after everything the run
recorded, saying how Android recorded the process's end — `LOW_MEMORY`,
`SIGNALED`, `CRASH_NATIVE`, `ANR` and the rest of `ApplicationExitInfo`'s
reasons, with the signal, whether the app was in the foreground and its
memory at the time — followed, for a native crash, by the tombstone's
signal, abort message, crashing thread's backtrace and the process's last
log lines. Its `elapsed_s` is when the process ended.

### The checkpoint

With `-o`, a run also keeps what it has finished beside the output (D): the
path with `.json` swapped for `.partial.json` (`run.xpupeak.json` →
`run.xpupeak.partial.json`). It is this document as far as the run has got —
the meta, the hardware, and the devices with every Benchmark finished so far
(one `--<test>` flag's worth of tests on a device) — without `log` or
`inventory`, which the sidecar holds; on one line, with `"schema":
"xpupeak/run-partial"`, a `progress` object, and `evidence_key`, the run's key
for linking evidence, which a resumed run needs so that its devices' tokens
match the part before's, and which no result holds:

```jsonc
{"schema": "xpupeak/run-partial", "format_version": 4, …, "progress": {
  "backends_done": ["Vulkan", "OpenCL"],
  "devices": [{"backend": "CPU", "device_index": 0, "benchmarks_done": ["single-precision-compute", …],
               "held_tests": ["integer_compute@NEON"]}]},
 "evidence_key": "…", "devices": [ … ]}
```

`backends_done` names the backends finished whole, and `devices` the devices
of the one in flight: each `done`, or with the Benchmarks finished on it by
their flag names. It is rewritten whole, through a rename, as each Benchmark,
device and backend finishes, and when the run is about to wait between two
readings — held while the app is off screen or paused, or cooling — since a
phone may end an app it holds. Then the devices also carry the tests of the
Benchmark in flight with the readings they have, and `held_tests` names them
by key (`id` or `id@variant`). It is removed with the sidecar once the
document is saved, so one left behind is a run that did not finish — or one
that was cancelled, which keeps it beside the document it saved. It is not a
result — `--compare` refuses it — and `--resume` continues the run from it.

## Resumed runs

A run whose process ended part-way — a phone closed the app off screen, the
user swiped it away, a driver crashed it — or that was cancelled can be
continued: `xpupeak --resume run.xpupeak.json` (or `run.xpupeak.partial.json`)
runs, with the options the run started with, only what its checkpoint does
not hold. The Benchmark the run was in when it stopped runs again
from its start, or with `--skip-crashed`, for a crash that would repeat, is
recorded in its place as an `error` reading `crashed` instead. Where the run
stopped while held, the readings its `held_tests` had are kept, and the
Benchmark run again adds only the ones they lack. Only the build
that started a run continues it, on the OS it started on: every field of
`system.os` the same, so a kernel or OS build update between parts refuses
the resume.

The parts end in one document under the same name and the same `run_id`.
`generated_at` is the first part's start, `log[].elapsed_s` goes on counting
from it across the gap, and `duration_s` is the time spent running, the parts
summed. The `log` holds the parts in order, each later one starting with an
`info` line `resumed the run started at …`. And `resumed` holds one entry per
part after the first (C):

| key | meaning |
|---|---|
| `at` | ISO-8601 UTC: when the part started |
| `elapsed_s` | the same moment on the log's clock |
| `after` | why the part before ended, when that is known: Android's record of the process's end, or `cancelled` |
| `from` | the first test the part measured: `backend`, `device`, `device_index`, and `test`, the test's key |
| `skipped` | `--skip-crashed` only: where the part before died, recorded and not run again |

Tests are in run order, so everything from `from` on was measured in that
part — at another time, and maybe at another temperature, than the tests
before it.

## Inventory

`--verbose` only (D). The `backends` array of the device catalog, exactly as
the GUI's run screen and `--list-devices` see it, taken before the run: every
backend compiled in, whether it was `available` and if not its `reason`, any
`notes` from enumeration (a provider the runtime names but nothing here can
run), and every device each backend saw — including the ones a `--devices`
list left out. It is the evidence for "my NPU is not listed", which is a
debugging question; in the CLI it costs an enumeration pass, which is why it
is not on by default.

Per device: `index`, `name`, `type`, and whichever of `arch`, `driver`,
`api`, `origin` (ONNX plugin EPs only, e.g. `EP plugin (QNNExecutionProvider)`),
`compute_units`, `clock_mhz`, `global_mem_bytes`, `max_alloc_bytes`,
`fp16`, `fp64` the backend could answer. The keys are those of
`include/common/inventory.h`, which is the one serializer for both uses.

## Reading a run as hardware

The document keeps the order the run produced it in: backend by backend,
device by device. A reader that wants the hardware — a compare, a site that
collects results — turns it around by the processor every device names.
Nothing in this section or the next is stored in the file: a reader derives
it, so better rules reach old files for free (*Compatibility*).

### By processor

- **The machine** is `system`, one per file.
- **The hardware** is `packages[]` and `processors[]`: each processor names
  its `package`, and each device its `processor`. A processor's devices are
  the backends' views of it, and its readings are all of theirs, each keeping
  the backend, device and test it came from.
- **A device with no `processor`**, or one naming a processor the file does
  not list, is a processor of its own. **A processor no device names** was
  not measured but is part of the machine: the host CPU, when the CPU backend
  did not run.
- **Ids are the run's.** `gpu0` in two runs of one machine is one GPU when
  both runs saw the same devices (*Hardware*). Across machines an id is only
  a label, and processors pair by kind and model (below).
- **`link` and `evidence`** are the linker's answer and what it saw. A reader
  with rules of its own can re-link a file from them and the devices'
  `facts`; a token means something only beside the tokens of its own file.

An M1 Pro's eleven devices make one package, `pkg0` (`soc`), and three
processors: `cpu0` (the CPU backend, and the CPU paths of Core ML, LiteRT and
ONNX Runtime), `gpu0` (Metal, Vulkan, OpenCL, and Core ML's and LiteRT's GPU
paths) and `npu0` (Core ML's Neural Engine and ONNX Runtime's CoreML
provider). This lists each processor's devices:

```console
jq '.devices | group_by(.processor) | map({processor: .[0].processor, devices: map("\(.backend):\(.device_index)")})' run.xpupeak.json
```

A table of readings flattens the same tree: one row per reading, with the
run's `run_id`, the processor and its package, the processor's model
(below), the backend, the device's `name` and `device_index`, the test's key
(`id@variant`) and `revision`, and the reading's family, quantity, unit,
direction, effective dims (*Comparing runs*), and value or status.

### Models

Processors of two machines are one part when their models match: a site
groups runs by the model, and a compare pairs processors by kind and then by
model. The file holds the facts a model is read from; the model, like a peak,
is the reader's to derive. The recommended reading:

| processor | its model | because |
|---|---|---|
| a CPU | `vendor`, `name`, and each cluster's `cores` (or `cores`); where the name is generic, `cpuid`'s `family` and `model` (x86) or the clusters' `midr` (ARM) | one name covers parts binned to fewer cores: an Apple M1 Pro has 8+2 or 6+2 |
| an integrated GPU (`integrated: true`) | its package's `name`, and its `compute_units` | the SoC names its GPU, binned by cores too (an M1 Pro's has 14 or 16), and an API's ids for it need not name a part: on an M1 Pro, Vulkan's device id is `64` |
| a discrete GPU | `vendor_id` and `device_id`, else `vendor` and `name` | the PCI device id names the part |
| an NPU, an accelerator | `vendor`, `name`, and its package's `name` | the SoC names it: the Neural Engine of an M1 Pro |

`vendor` is a word of the vocabulary; `name`, `vendor_name` and the other
facts are as the APIs spell them, so a reader that keys on a name compares it
as written. A system's model is its `manufacturer` and `model` (`Apple`,
`MacBookPro18,3`; `samsung`, `SM-S921B`), or its `board` where it was built
from parts. A display calls it by the `system.name` annotation, else `name`,
else `model`, `board` or `soc` (*System*). Naming a model id — `SM-S921B` is
a Galaxy S24 — is the reader's job, never the tool's.

## Comparing runs

What compares in a file is its readings, by what they measured. Peaks, the
best of backends, headlines and rankings are all derived from the readings by
the rules below, which this document fixes so that every reader — the app,
the CLI's `--compare`, a site — derives the same.

### Measure keys

A reading's **measure key** is what it measured, in words every backend
shares. For each reading that has a `value`:

1. **Family**: the reading's `benchmark`, else its test's.
2. **Quantity**: the reading's `quantity`, else its test's, as written.
3. **Dims**: the test's `dims` with the reading's over them — a key in both
   takes the reading's value. These are its **effective dims**.
4. **Data type**: the effective `dtype`, else the family's (*Families*), else
   none.
5. **Identity dims**: each effective dim whose key is identity (*Dim keys*),
   and each whose key the reader does not know, except where the value is
   the key's default for this family and data type. Variant dims (`width`,
   `chains`, `isa`, `simd_bits`, `impl`, `form`, `tile`, `problem_size`)
   never count: the best is taken over them.
6. **Text**: a string as written, a whole number in decimal (`2048`, also
   where the file says `2048.0`).
7. **Key**: the family, the quantity, then `key=value` for each identity dim
   in order of key by code unit, joined by `|`.

Each key's default is in the *Dim keys* table: `dtype` is the family's own
data type, for a family of one; `acc` follows the family's rule; `weights`
and `kv` are the reading's data type; the others that have one are a fixed
word or number (`sparsity: dense`, `scope: device`, `in_flight: 1`). A key
with no default — `level`, `op`, `direction`, `working_set`, … — counts
wherever it is present. A family or a
data type the reader does not know gives what it can: no `dtype` default; for
`acc`, the usual accumulator of a known data type, and no default for an
unknown one.

- Metal's `simdgroup_matrix` → `fp16` `{dtype: fp16}`:
  `matrix-compute|flops|dtype=fp16`.
- `fp16 f16acc` `{dtype: fp16, acc: fp16}`:
  `matrix-compute|flops|acc=fp16|dtype=fp16`. With `acc: fp32`, the usual
  accumulator, the same reading is `matrix-compute|flops|dtype=fp16`.
- `fp8_e4m3 2:4` `{dtype: fp8_e4m3, sparsity: "2:4"}`:
  `matrix-compute|flops|dtype=fp8_e4m3|sparsity=2:4`.
- The CPU's `single_precision_compute@NEON`, whose test has `{dtype: fp32,
  isa: neon, simd_bits: 128}`: `float ST` `{scope: core}` is
  `single-precision-compute|flops|scope=core`, and `float MT` is
  `single-precision-compute|flops`. The ISA is variant, so a CPU's NEON and
  SVE variants reach the same keys, and the better is the CPU's.
- `mp` with no dims: `mixed-precision-compute|flops` (its fp16 data type and
  fp32 accumulator are the family's).
- The CPU's `memory_latency`, a pointer chase on one core whose test has
  `{scope: core}`: `DRAM x8` `{level: dram, in_flight: 8}` is
  `memory-latency|seconds|in_flight=8|level=dram|scope=core`.
- `int8_qdq`, in OPS, `{dtype: int8, quant: qdq}`:
  `gemm|ops|dtype=int8|quant=qdq`.
- `fp16_decode_kv2048` `{dtype: fp16, phase: decode, context: 2048}`:
  `transformer-block|bytes_per_second|context=2048|dtype=fp16|phase=decode`.

Two readings with one key measured the same thing, whatever backend, device
or run they come from — unless they are readings of one backend's test at
two revisions (below).

### The best of a key

On a processor, the **best** of a key is its best reading over every device
of that processor, by the reading's `direction`: the largest value where
higher is better, the smallest where lower is. It is that processor's peak
for the key, and the best of each backend beside it is that backend's. The
headline catalog (below) selects among these bests.

- **Ties** go to the first reading in this order: backend name, device index,
  test key (`id@variant`), reading id; names and ids by code unit, the index
  as a number. Never the order the file lists them in.
- **Provenance.** A best is a reading: it names its backend, device, test and
  reading, and a view shows them with the value.
- **Contradictions.** Readings of one key that disagree on `direction` — only
  a file that contradicts itself has them — take the direction of the first
  in the order above.

### Two runs side by side

A compare pairs two runs' processors by kind, then by model (*Models*), and a
person may re-pair them. For each pair it sets the keys both sides have side
by side: each side's best, with the backend that reached it, and their ratio,
turned by the key's direction so that above 1 is better. The keys of one side
alone follow. Where both bests are readings of one backend's test at two
revisions, it says "method changed" instead of a ratio. One device in two
runs of one machine, before and after a driver update, compares reading by
reading instead: by backend, device name and the occurrence of that name,
test key and reading id.

### Revisions

A test's `revision` (absent, `1`) is its method's version. Readings of one
backend's test — the same `backend` and test `id`, whatever the `variant` —
at two revisions are not compared: a compare says "method changed" instead of
a ratio. A ranking keeps, for each backend's test, the newest revision it has
seen, and ranks the readings at it alone: the others were measured another
way, before a kernel fix or under another counting rule, and may be off by a
factor (the Metal simdgroup fold read 4× high). The best of a key on one
processor can still draw on several tests, each at its own revision.

### Which runs rank

A file says how it was made. A ranking that lists runs from many machines
takes only runs made the same way:

| | what it says | in a ranking |
|---|---|---|
| `invocation.standard` absent | the budgets, warm-up or iterations were changed (`iters` is set), or the tests narrowed (`tests`) | out |
| `cancelled: true` | stopped part-way: a test it never reached is not missing from the hardware | out |
| `build.config` other than `Release` | a `Debug` build's host-side work runs unoptimised; absent, the build did not say | out |
| `system.translated: true` | an x86 binary under Rosetta or Prism, an ARM one under Android's native bridge | out, at least for its CPU's readings |
| `system.virtual: true` | the processors are what a VM, an emulator or a simulator showed: features may be masked, a topology made up | apart |
| `resumed` | complete, but measured in parts, at other times and maybe other temperatures; a part with `skipped` recorded a crash instead of a reading | in, marked |
| `conditions` | battery power, low-power mode, a thermal state that rose | in, shown; low-power runs may be set apart |

A ranking may also want a released `generator.version` (`4.0.0`): a
development build (`4.0.0-40-gabc1234`, `-dirty`, `unknown`) runs code no
release has.

### One run, one record

`run_id` names the run: files with one `run_id` are one run, and a site keeps
one record of it. A run's file changes after it is first saved in two ways
only — `--resume` continues it (more readings, another `resumed` part,
`cancelled` gone), and a person edits its `annotations` — so a site keeps the
latest state: the file that holds every reading of the other and more, and
the annotations of the latest upload. Two files of one `run_id` whose
readings differ where both have one are not two states of one run: one of
them was edited (*Trust*, below). A `run_id` is random per run and names no
machine and no person: two runs of one machine have two, and nothing in a
file joins them, by design.

### Runs from many people

**Annotations** are what a person said about a run (*Annotations*), never a
measurement. A display may call a system by its `system.name` annotation, but
grouping, filtering and ranking go by what the tool read. Annotations are
text someone typed: a site escapes them, bounds their length, and shows them
as that person's words. Its upload form may write the recommended keys and
keys of its own; a reader shows a key it does not know as written.

**Privacy.** What a file never holds is the *Never recorded* list. A path is
where a person's name can still reach a file: a home directory is named after
its user, and some paths are written whole — the program's own in
`invocation.argv`, an output or library path given on the command line, a
cache or temporary folder in a log line. So a site publishes less than it
keeps:

- `log` and `inventory` are for maintainers (D): a site drops them, or keeps
  them for the uploader alone.
- `invocation.argv` (I): a site keeps the flags and drops the paths.
- `evidence` tokens join devices within one file, and nothing else.
- `annotations` hold whatever a person typed, a name included.

**Trust.** A file is JSON anyone can edit. Nothing in it is signed, and the
schema checks only its shape, so a site checks what it receives:

- **Against its peers.** A reading far above the same key's on the same model
  is suspect, and so is any reading above what the hardware can do: the tool
  counts only arithmetic that ran, so a rate above the hardware's peak is a
  bug or an edit.
- **Against itself.** Each backend's best of a key on one processor checks
  the others: one far above the rest is suspect, and the best of backends
  would carry it to the top. A file whose readings repeat another `run_id`'s
  is a copy.
- **Provenance, always.** Every number a site shows names the run, backend,
  test and reading it came from, and the `generator` version that measured
  it; and what the tool read stays apart from what a person typed.

## The headline catalog

Headlines are the measures worth a summary row or a leaderboard: a
processor's peak floating-point and integer compute, its memory bandwidth.
They are not in the run document. They are data of their own,
`app/assets/measures.json` (`"schema": "xpupeak/measures"`), with a
`catalog_version` of its own, read by the app and embedded in the CLI
*(V10 embeds it)*. Each headline has an `id`, a `title` and the `quantity`
it selects, and may narrow that by the processor kinds it applies to
(`processors`), the `families` it selects from, and `dims`: a list of
identity-dim objects, defaults filled in, of which a selected key must equal
one. Each of the three, absent, means any. Its value on a processor is the
best reading of the selected keys there, over every backend, named with the
backend and the reading it came from: a raw measurement, never a score. The
catalog the app ships holds the flagship numbers its results summary shows
for every processor: `peak-float` (`flops`, any family and dims, so a 2:4
sparse NVFP4 reading counts), `peak-integer` (`ops`) and
`peak-memory-bandwidth` (`global-memory-bandwidth`).

## Compatibility

The format will keep changing as needs arrive. The aim is that it changes by
adding, and that the parts most likely to change are not stored at all.

**The promise, from the 4.0.0 release on.** A file any 4.x build wrote stays
readable by every later 4.x reader — the app, the CLI, a site that collects
results — and means the same thing. Until 4.0.0 anything may change, and
files written by development builds are not migrated.

**Versioning.**

- `format_version` stays `4` for the life of the promise. `format_minor`
  counts additive revisions (4.0, 4.1, …); absent means 0.
- **A minor may add**: optional keys anywhere, enum values, dim keys,
  families, quantities, sections.
- **A minor may not**: remove, rename or retype a key; change a key's or a
  value's meaning; change a default; or make an optional key required.
- **Words keep their form.** Every value of a list here — a data type, a dim
  key or word, a family, a kind, a vendor, a quantity, a status, a feature —
  is lowercase ASCII letters and digits joined by `_`, `-`, `.` or `:`
  (`fp8_e4m3`, `2:4`, `single-precision-compute`), and so is every word a
  minor adds. A dim's value is a string or a whole number. So a reader, and
  the schema, can take a word they have never seen.
- **Readers** accept any minor of their major, and say when a file comes from
  a newer one ("written by a newer XpuPeak; some details aren't shown"). They
  treat what they don't know as opaque and never fail on it: an unknown key
  is ignored; an unknown enum value is shown as written; an unknown dim key is
  treated as identity, so it never merges what it doesn't understand.
- **A major bump (5)** would get a new schema string and a converter, never a
  silent change. The rules here exist so that it never has to happen.

**Three classes of field**, marked on every key above.

- **Contract (C)**: identity, structure, measurements and their coordinates
  — everything compare, history and a site rely on. Stable under the
  promise.
- **Informational (I)**: prose and display aids — titles, descriptions,
  reasons, `axis`, the content of `properties`, log messages. The shape is
  stable; the words may change in any release.
- **Diagnostic (D)**: `log`, `inventory`, and the internal files (the
  sidecar, the checkpoint, the app's history index). For maintainers. They
  may change in any release, and a site may drop them on ingest.

**Meaning stays put** — the part JSON can't enforce.

- **Ids.** Test and reading ids never change meaning and are never reused.
- **`revision`** on a test goes up whenever a change moves its numbers for a
  reason other than the hardware. It is authored at the test's
  `beginTest()`, beside the code that changed, and the commit says why.
- **Dims.** A dim key's class and default never change; a correction
  becomes a new key. Giving an existing test's readings a new identity dim
  bumps its revision.
- **Units** never change.

**Store facts, derive views.**

- **Stored:** measurements; each API's own device facts; the linking
  evidence, salted per run; and the linking result (`processors`,
  `packages`), since only the tool sees the evidence. A consumer with better
  rules can re-link old files from the facts and evidence.
- **Derived, never stored:** peaks, best of backends, headlines, scores and
  rankings. Improving them never touches the format, and old files get the
  new rules for free. The headline catalog is data with a version of its
  own, outside the format; the measure key is fixed by this document and
  changes only by gaining dim keys.

**Room to grow without a minor.**

- **`ext`.** An `ext` object may sit on any contract object — the run,
  `system`, `conditions`, a package, a processor, a device, a test, a
  reading — and holds experimental data. Readers ignore it, and a tool that
  rewrites a document keeps it. Keys inside may change in any release; a
  field graduates out of `ext` in a minor.
- **New schemas.** A new kind of document gets a schema string of its own
  (`xpupeak/…`) rather than changing the run document: a collection of
  repeated runs, say, or a container that holds this document beside large
  optional data.

**What every file has.** `schema`, `format_version`, `run_id`, `generator`
(its `name` and `version`), `generated_at` and `devices`; every device its
`backend`, `name`, `type` and `tests`; every test its `id`, `benchmark`,
`quantity`, `unit`, `direction` and `metrics`; every reading its `id`, and a
`value` or else a `status`, never both; every package its `id`, and every
processor its `id` and `kind`. Everything else is optional, and stays so.

**Checked by machine.** [`format-v4.schema.json`](format-v4.schema.json),
beside this document, is the run document's JSON Schema (draft 2020-12). It
checks the keys' types, what every file has, and the form of a word, a
timestamp, a `run_id`, a hex id and a token. It is open: a key it does not
know passes anywhere, and so does any word of the right form, so every file
of format 4 validates against the schema of every minor, and each minor
revises it only by adding. A file that fails it was written wrong, not by a
newer build. What JSON cannot say stays in this document's words: that a
device's `processor` is in `processors[]`, what a reading means. It leaves
`log` and `inventory` unchecked (D). **Golden files** *(V11, from V12's
runs)*: one real document from each released minor, read by both loaders,
with its expected measure keys and best of backends.

**Rewriting keeps the unknown.** A tool that edits a document — the app
adding annotations — edits the JSON in place and keeps every key it does not
know; otherwise an older app would strip a newer file.

**Privacy is part of the contract.** No minor may add anything on the
never-recorded list below.

## Never recorded

A result file is made to be shared, so nothing in one may identify the unit
it ran on or the person who ran it, and no minor may add:

- serial numbers, DMI `product_uuid`, IORegistry `product-id`;
- hostnames, user names, MAC addresses;
- names the user gave the device (Android's `device_name`, iOS's
  `UIDevice.name`) — a name the user chooses to give goes in `annotations`;
- a GPU's UUID or LUID, a PCI bus address, a Metal registry ID — only their
  per-run tokens (*Linking*);
- the key those tokens are made under, which only the checkpoint holds.
- a power plan's GUID, which is unique to the install, and who and where an
  OS build was made (Android's `ro.build.user` and `ro.build.host`, a local
  build's `eng.<user>` id).

The log is held to the same rule, a vendor runtime's relayed words included
(*Log*).

## Conventions

- **Absent means default.** Optional fields are omitted rather than written
  empty, so a file carries only facts. `status` absent ⇒ `ok`; `label` absent ⇒
  `id`; metric unit fields absent ⇒ the test's; `revision` absent ⇒ 1; a dim
  absent ⇒ its default.
- **Numbers use the classic locale**, always `.` — the GUI hosts the writer
  inside toolkits that call `setlocale(LC_ALL, "")`, and a comma decimal
  separator would produce a file that is not JSON at all.
- **Seven significant digits** on every value: enough to round-trip a float's
  precision, enough to keep a six-digit GFLOPS reading whole, and — unlike the
  fixed four decimals v2 used — it does not flatten the parts-per-million
  numeric-error readings to `0.0000`. Counts, bytes and dim numbers are whole
  numbers, written as such.

## What changed from v3

v4 is v3 under the program's new name, XpuPeak, and the hardware and
vocabulary that make results comparable. The rest of v4 is still being
worked out, and this table grows with it until the release.

There is no migration path and no compatibility shim: 4.0 reads no file that
clpeak 3.x wrote. `--compare`, `--resume` and the app's *Open file* refuse one
by its schema, saying it "was written by clpeak 3.x, which this version does
not read", and the app does not look for 3.x history. Run it again with
XpuPeak.

| v3 | v4 |
|---|---|
| `"schema": "clpeak/run"`, `"clpeak/run-log"`, `"clpeak/run-partial"` | `"xpupeak/run"`, `"xpupeak/run-log"`, `"xpupeak/run-partial"` |
| `"format_version": 3` | `"format_version": 4`, and `format_minor` |
| `"clpeak_version": "…"` | `"generator": {"name": "XpuPeak", "version": "…"}` |
| `.clpeak.json` for a result (the app's files and these examples), with its `.clpeak.log` sidecar and `.clpeak.partial.json` checkpoint | `.xpupeak.json`, `.xpupeak.log`, `.xpupeak.partial.json` |
| the app's history in `~/.clpeak` (`%USERPROFILE%\.clpeak`), or `<app documents>/clpeak` on a phone | `~/.xpupeak` (`%USERPROFILE%\.xpupeak`), or `<app documents>/xpupeak` |
| `host`: `os`, `os_version`, `arch`, `cpu`, `logical_cores`, `memory_bytes` | `system`: the maker, model and marketing name, the SoC, the form factor, `virtual` and `translated`, the OS in detail, `arch`, `memory_bytes`. The CPU's cores go to its processor |
| `build.backends`: names | objects with `kind` and `portable` |
| — | `run_id`, `conditions`, `annotations`, `invocation.standard` |
| — | `packages[]`, `processors[]`; on devices `processor`, `link`, `facts`, `evidence`, `runtime` |
| device `type`: `gpu`, `cpu`, `accelerator`, `unknown` | `npu` too |
| — | on tests `benchmark`, `dims`, `revision`; on readings `dims` and a `benchmark` of their own |
| — | `ext` on the contract objects |
| — | the checkpoint's `evidence_key` |
