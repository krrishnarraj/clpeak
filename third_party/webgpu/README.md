# WebGPU C API header

Copied verbatim from `webgpu-native/webgpu-headers`, BSD-3-Clause licensed
(copyright notice retained in the file).

- **Upstream:** https://github.com/webgpu-native/webgpu-headers
- **Commit:** `706853a9da45b8e89b7ea005aa267294d115f8ce`
- **Path:** `webgpu.h`

Header only — the WebGPU backend dlopens a conforming `webgpu.h` runtime at
run time (`src/webgpu/webgpu_runtime.cpp`), so building clpeak needs no WebGPU
SDK, and the shipped binary has no link-time dependency on any implementation.

## Why this is checked in, not a submodule

clpeak's git submodules (`libopencl-stub/`, `Vulkan-Headers/`) are only used
by the Android build (`android.yml`). Desktop CI workflows, Snap/Flatpak
builds, GitHub release source tarballs, and plain `git clone` builds do not
initialize submodules. Checking in the single ~250 KB `webgpu.h` header keeps
the WebGPU backend built out of the box everywhere (matching `onnxruntime/`
and `litert/`), while `tools/update_webgpu_headers.sh` keeps updating cheap
and records the exact upstream commit pinned above.

## Updating

```sh
tools/update_webgpu_headers.sh --check                                     # pinned commit vs upstream HEAD
tools/update_webgpu_headers.sh 706853a9da45b8e89b7ea005aa267294d115f8ce    # refetch webgpu.h, rewrite the pin
tools/update_webgpu_headers.sh main                                        # refetch upstream main, record its SHA
```

Because `webgpu.h` is pre-1.0, the pinned revision must stay ABI-compatible
with the runtime libraries dlopen'd at run time. The script fetches `webgpu.h`
before touching the tree and rewrites the **Commit** line above from the
resolved SHA so the pin cannot drift from the file contents. After updating,
rebuild and verify against a conforming WebGPU runtime.
