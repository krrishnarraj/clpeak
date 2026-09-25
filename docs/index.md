---
layout: default
title: clpeak — compute latency peak
---

clpeak measures the **peak compute throughput, memory bandwidth and latency**
that CPUs, GPUs and NPUs actually reach. It uses small, tight kernels alongside
each vendor's own tuned libraries, reaches a device through every API that
exposes it, and runs the same tests on each, so the numbers from one machine
can be compared side by side.
{: .lede }

<figure>
  <picture>
    <source media="(prefers-color-scheme: dark)"
            srcset="{{ '/assets/img/results-npu-dark.png' | relative_url }}">
    <img src="{{ '/assets/img/results-npu-light.png' | relative_url }}"
         alt="clpeak app showing Apple M1 Pro results: eleven devices across seven backends, with the Neural Engine selected and its matmul and convolution rates per weight format.">
  </picture>
  <figcaption>
    One run on an Apple M1 Pro. Every device the machine exposes gets its own
    results, here the Neural Engine through Core ML.
  </figcaption>
</figure>

## <a id="what"></a>What it measures

- **Compute:** single, half, double, mixed and bf16 precision, integer and
  dot-product throughput, and divide and square-root rates.
- **Matrix engines:** tensor cores and their equivalents, driven directly:
  CUDA `mma.sync`, AMD MFMA and WMMA, Apple `simdgroup_matrix`, Intel XMX
  `joint_matrix`, Vulkan cooperative matrices, and CPU AMX, SME and i8mm.
  Dense and 2:4 sparse where the hardware has both.
- **Vendor libraries:** cuBLASLt, hipBLASLt, MPS, oneMKL, Accelerate and BNNS,
  so a hand-written kernel can be checked against the tuned path.
- **Bandwidth:** device, local/shared, image and host-transfer bandwidth; CPU
  cache and DRAM bandwidth per level; resident-weight and transfer bandwidth
  through the AI runtimes.
- **Latency:** kernel-launch round trip, memory latency per cache level,
  atomics, branch mispredicts and store-to-load forwarding; dispatch cost
  through the AI runtimes.
- **AI workloads:** matmul and convolution peaks per data type, activations,
  and one transformer decoder block (prefill, decode and per-token latency)
  across weight formats from fp32 down to int4. Each rate comes with its
  numeric error against an fp32 reference, so a faster format shows what it
  costs in accuracy.
- **CPU extras:** AES, SHA and CRC32-C rates, string scanning and UTF-8
  validation, and fp32 scaling from one thread per core to every SMT thread.

Every test and every reading carries a short description of what it measures
and how to read it: the info glyph beside it in the app, and
`clpeak --describe` on the command line.

### Backends

<div class="table-scroll" markdown="1">

| Backend | Runs on |
|---|---|
| CUDA | NVIDIA GPUs |
| ROCm/HIP | AMD GPUs |
| Metal | Apple silicon and Intel Macs |
| oneAPI/SYCL | Intel GPUs |
| Vulkan | any Vulkan 1.1+ GPU, including cooperative-matrix paths |
| OpenCL | any conformant GPU, CPU or accelerator |
| CPU | x86-64 and AArch64, with kernels picked per instruction set at run time |
| ONNX Runtime | NPUs, GPUs and CPUs through its execution providers (Core ML, QNN, OpenVINO, VitisAI, NNAPI, CUDA, TensorRT, DirectML, …), each provider as a device |
| Core ML | Apple's Neural Engine, GPU and CPU, through the system framework |
| LiteRT | NPUs through the vendor dispatch libraries (Qualcomm, MediaTek, Google Tensor, Samsung, Intel), its GPU accelerator and XNNPACK on the CPU |

</div>

The ONNX Runtime and LiteRT libraries are loaded when clpeak runs rather than
linked, so neither needs installing to build. `--onnx-lib` and `--litert-lib`
point at a specific build, and `--onnx-ep` adds a plugin execution provider
such as Qualcomm's QNN.

### What a number means

A reading is only published for the device it names. When an AI runtime
cannot place a whole graph on the requested NPU or GPU, the row reports
`unsupported` with the reason instead of quietly measuring a CPU fallback. On
Apple hardware clpeak reads Core ML's compute plan for each timed model to
check, per operation, which unit ran it. A reading that cannot be taken on a
device is listed with the reason, never left blank.

Saved runs from known hardware live in
[`results/`](https://github.com/{{ site.repo }}/tree/master/results).
They are the baselines to check a surprising number against.

## The app

The app and the CLI are the same benchmark engine. One Flutter app runs on
Windows, macOS, Linux and Android. It lists every device it finds, streams
results in as they are measured, and keeps a history of runs you can rename and
export as JSON.

<div class="shots" markdown="1">

<figure>
  <picture>
    <source media="(prefers-color-scheme: dark)"
            srcset="{{ '/assets/img/results-gpu-dark.png' | relative_url }}">
    <img src="{{ '/assets/img/results-gpu-light.png' | relative_url }}"
         alt="CUDA results on an NVIDIA GeForce RTX 5060: tensor-core rates per data type, from fp64 up to 2:4-sparse fp4, followed by cuBLASLt GEMM rates.">
  </picture>
  <figcaption>Tensor cores on an RTX 5060, one reading per data type.</figcaption>
</figure>

<figure>
  <picture>
    <source media="(prefers-color-scheme: dark)"
            srcset="{{ '/assets/img/results-cpu-dark.png' | relative_url }}">
    <img src="{{ '/assets/img/results-cpu-light.png' | relative_url }}"
         alt="CPU results on an AMD Threadripper PRO 3955WX: detected ISA, core count and cache sizes, then per-ISA compute and divide and square-root rates.">
  </picture>
  <figcaption>The native CPU backend, with the detected ISA and caches.</figcaption>
</figure>

<figure>
  <picture>
    <source media="(prefers-color-scheme: dark)"
            srcset="{{ '/assets/img/dashboard-dark.png' | relative_url }}">
    <img src="{{ '/assets/img/dashboard-light.png' | relative_url }}"
         alt="The Benchmark screen: a Run button and the devices found on this machine, grouped by backend.">
  </picture>
  <figcaption>Every device on the machine, grouped by backend.</figcaption>
</figure>

<figure>
  <picture>
    <source media="(prefers-color-scheme: dark)"
            srcset="{{ '/assets/img/custom-run-dark.png' | relative_url }}">
    <img src="{{ '/assets/img/custom-run-light.png' | relative_url }}"
         alt="The Custom run screen: per-device checkboxes, test-category chips and time-budget sliders for GPU and CPU backends.">
  </picture>
  <figcaption>A custom run narrows the devices, categories and time budgets.</figcaption>
</figure>

</div>

## <a id="download"></a>Download

Each release has a zip per platform with the CLI and the app. The `cuda`,
`rocm` and `oneapi` variants add the backends that need a vendor SDK at build
time. On macOS the `.dmg` holds the app.

<div id="release-list">
  <p class="rel-note">
    See the <a href="https://github.com/{{ site.repo }}/releases/latest">latest
    release</a> for downloads.
  </p>
</div>

<p class="rel-note">
  Older versions are on the
  <a href="https://github.com/{{ site.repo }}/releases">releases page</a>.
</p>

<div class="store-links">
  <a href="https://snapcraft.io/clpeak">
    <svg viewBox="0 0 24 24" aria-hidden="true" width="22" height="22">
      <path d="M12 2.6 21 7v10l-9 4.4L3 17V7z"/>
      <path d="M3 7l9 4.4L21 7M12 11.4v10"/>
    </svg>
    <span class="store-name">Snap Store</span>
    <span class="store-sub">Linux</span>
  </a>
  <a href="https://play.google.com/store/apps/details?id=kr.clpeak">
    <svg viewBox="0 0 24 24" aria-hidden="true" width="22" height="22">
      <rect x="6" y="2.5" width="12" height="19" rx="2.5"/>
      <path d="M10.5 18.5h3"/>
    </svg>
    <span class="store-name">Google Play</span>
    <span class="store-sub">Android</span>
  </a>
</div>

The snap uses classic confinement so it can reach the GPU drivers and device
nodes the benchmarks need:

```console
sudo snap install clpeak --classic
```

### <a id="unsigned"></a>Running the release binaries

The release binaries aren't signed with a developer certificate, so Windows and
macOS ask before running them the first time.

**Windows:** before extracting the zip, right-click it → **Properties** → tick
**Unblock** → **OK**. Otherwise every file extracted from it carries the
downloaded-from-the-internet mark. If SmartScreen still says "Windows protected
your PC", choose **More info** → **Run anyway**. From PowerShell, this unblocks
a folder that was already extracted:

```powershell
Get-ChildItem -Recurse .\clpeak-* | Unblock-File
```

**macOS:** open the `.dmg` and drag clpeak to Applications. macOS refuses the
first launch; open **System Settings** → **Privacy & Security** and click
**Open Anyway** next to clpeak. Right-click → *Open* no longer does this on
macOS 15 and later. Or clear the quarantine flag from a terminal, which is also
the way to run the CLI from the zip:

```console
xattr -dr com.apple.quarantine /Applications/clpeak-gui.app
xattr -dr com.apple.quarantine ~/Downloads/clpeak-*-macos-arm64
```

## <a id="build"></a>Build from source

```console
git clone --recursive https://github.com/krrishnarraj/clpeak
cd clpeak
cmake -S . -B build
cmake --build build -j
./build/clpeak
```

A backend is built when its SDK is found; each can be left out at configure
time. oneAPI needs the DPC++ compiler (`-DCMAKE_CXX_COMPILER=icpx`).

<div class="table-scroll" markdown="1">

| CMake option | Default | When `OFF` |
|---|---|---|
| `CLPEAK_ENABLE_CUDA` | `ON` | no CUDA backend, even with the toolkit installed |
| `CLPEAK_ENABLE_ROCM` | `ON` | no ROCm/HIP backend, even with the SDK installed |
| `CLPEAK_ENABLE_METAL` | `ON` | no Metal backend on Apple platforms |
| `CLPEAK_ENABLE_ONEAPI` | `ON` | no oneAPI/SYCL backend |
| `CLPEAK_ENABLE_VULKAN` | `ON` | no Vulkan backend, even with the SDK installed |
| `CLPEAK_ENABLE_OPENCL` | `ON` | no OpenCL backend |
| `CLPEAK_ENABLE_CPU` | `ON` | no native CPU backend |
| `CLPEAK_ENABLE_ONNX` | `ON` | no ONNX Runtime backend (otherwise always built; the runtime is loaded when clpeak runs) |
| `CLPEAK_ENABLE_COREML` | `ON` | no Core ML backend (Apple only) |
| `CLPEAK_ENABLE_LITERT` | `ON` | no LiteRT backend (otherwise always built; the runtime is loaded when clpeak runs) |
| `CLPEAK_ENABLE_GUI` | `ON` | no desktop app (also skipped when no Flutter SDK is found) |

</div>

With the Flutter SDK on `PATH`, the desktop app is built alongside the CLI into
`build/clpeak-gui/` (`cmake --build build --target clpeak-gui`). Android and
iOS builds are described in
[`app/AGENTS.md`](https://github.com/{{ site.repo }}/blob/master/app/AGENTS.md).

## Command line

Backends say *where* a test runs and test flags say *what* runs. A test flag
applies to every backend that has the test, and `--no-<x>` takes anything
away.

```console
clpeak                            # every test on every device
clpeak --list-devices             # every device, named as --devices takes it
clpeak --devices cuda:0,vulkan:1  # only these devices
clpeak --devices coreml:0         # Core ML on the Neural Engine
clpeak --metal --cpu              # only these backends
clpeak --no-opencl                # everything except OpenCL
clpeak --compute                  # one category: --compute, --bandwidth, --latency, --ai, …
clpeak --single-precision-compute # one test, on every backend that has it
clpeak --gemm                     # the vendor's tuned matmul, everywhere
clpeak --onnx --transformer-block # the decoder block on every ONNX provider
clpeak --describe                 # what each test and each reading measures
clpeak -o run.clpeak.json         # save the run as JSON
clpeak --compare run.clpeak.json  # run again and compare with a saved run
clpeak --max-time 200             # a shorter time budget per test
```

`clpeak --help` lists every flag. Every flag parses in every build, so a
script can say `--no-cuda` on a Mac. Runs saved with `-o` use the
[result format](format-v3.html) that the app's history and `--compare` read
back.

## <a id="bugs"></a>Reporting a bug

Numbers that look wrong, missing devices and crashes are all worth reporting.
Attach a verbose run, which records every diagnostic and the device inventory,
so the report can be debugged without access to your machine.

- **CLI:** run `clpeak --verbose -o report.clpeak.json` and attach
  `report.clpeak.json`. If clpeak crashed before finishing, attach
  `report.clpeak.log` from the same folder instead: it is written as the run
  goes, and its last lines are usually where it stopped.
- **App:** turn on **Settings** → **Diagnostics** → **Verbose diagnostics**,
  run again, then use **Export JSON** on the results page. A run the app did
  not survive is listed in **History** under "Runs that did not finish", with
  its own export.

Then [open an issue](https://github.com/{{ site.repo }}/issues/new/choose) and
add the file to it.

## Contributing

Issues and pull requests are welcome. The repository is documented for people
and coding agents alike: `AGENTS.md` files map the tree level by level,
starting at the
[root one](https://github.com/{{ site.repo }}/blob/master/AGENTS.md), with the
architecture, the directory map and the conventions for adding a benchmark or
a backend.
