# ONNX Runtime C API headers

Copied verbatim from the ONNX Runtime source tree, MIT licensed
(copyright notices retained in each file).

- **Upstream:** https://github.com/microsoft/onnxruntime
- **Tag:** `v1.29.0`  (`ORT_API_VERSION` 29)
- **Path:** `include/onnxruntime/core/session/`

| File | |
|------|--|
| `onnxruntime_c_api.h` | the C API |
| `onnxruntime_error_code.h` | included by the above |
| `onnxruntime_ep_c_api.h` | included by the above |

Headers only — the ONNX backend dlopens the runtime at run time
(`src/onnx/onnx_runtime.cpp`), so building clpeak needs no ONNX Runtime
installation, and the shipped binary has no link-time dependency on it.

## Why these are checked in, not a submodule

`microsoft/onnxruntime` is a whole runtime, not a header package: a
`--depth 1` clone measures ~814 MB (243 MB pack + 10,866 files) to deliver
the three files above, and every clone and CI job that inits submodules
would pay it. There is no upstream headers-only repo to point at. Checked-in
copies cost ~400 KB and `tool/update_onnx_headers.sh` keeps updating cheap.

## Updating

```sh
tool/update_onnx_headers.sh --check      # pinned tag vs latest upstream
tool/update_onnx_headers.sh v1.30.0      # refetch all three, rewrite the pin
```

The script fetches every file before writing any, so a bad tag cannot leave
a mix of two releases behind, and it rewrites the **Tag** line above from
the header it actually fetched — the pin cannot drift from the contents.
After updating, rebuild and check that `kMinApiVersion` in
`src/onnx/onnx_runtime.cpp` still names the oldest runtime worth supporting.
