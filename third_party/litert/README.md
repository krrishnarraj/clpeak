# LiteRT C API headers

Copied verbatim from the `litert_cc_sdk.zip` asset of one LiteRT release,
Apache-2.0 licensed (copyright notices retained in each file).

- **Upstream:** https://github.com/google-ai-edge/LiteRT
- **Tag:** `v2.2.0`  (`LITERT_RUNTIME_ABI_VERSION` 1.0.0)
- **Path:** `litert_cc_sdk/litert/c/` inside the release's `litert_cc_sdk.zip`

Only the C API is vendored (`litert/c/`, 35 headers): the transitive include
closure of the entry points `src/litert/litert_runtime.cpp` resolves by name.
The C++ wrapper in the same zip (`litert/cc/`) is deliberately left out — it
pulls in Abseil and FlatBuffers, neither of which clpeak builds against.

`litert/build_common/build_config.h` is not upstream's: LiteRT's own CMake
generates it at configure time from `build_config.h.in`, and the copy here
is that template with both feature toggles off.

Headers only — the LiteRT backend dlopens the runtime at run time
(`src/litert/litert_runtime.cpp`), so building clpeak needs no LiteRT
installation, and the shipped binary has no link-time dependency on it.

## Why these are checked in, not a submodule

`google-ai-edge/LiteRT` is the whole runtime plus the TFLite converter: a
`--depth 1` clone is well over a gigabyte to deliver 33 files. The release
zip is 280 KB and `tool/update_litert_headers.sh` refetches it.

## Updating

```sh
tool/update_litert_headers.sh --check    # pinned tag vs latest upstream
tool/update_litert_headers.sh v2.3.0     # refetch the closure, rewrite the pin
```

The script downloads the release's `litert_cc_sdk.zip`, recomputes the
include closure from the same root headers, replaces the vendored set in one
step and rewrites the **Tag** line above from the release it actually
fetched — the pin cannot drift from the contents. After updating, rebuild
and check that every entry point `src/litert/litert_runtime.cpp` resolves
still exists in `litert_runtime_c_api_so_symbols.txt` upstream.
