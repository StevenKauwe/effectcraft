# Effect plug-ins

EffectCraft loads third-party effects through a small, versioned plug-in API (**API version 1**).
A plug-in effect behaves exactly like a built-in one: it is listed in Effects & Presets (and the
Effect menu) under its category, applied by id (`effect.apply {"effect": "org.example.posterize"}`)
or display name, its parameters are ordinary properties (keyframes, expressions, Effect Controls,
`prop.set`), and it is saved in projects by id.

There are two ways to write one:

| | WebAssembly plug-in | Rust plug-in |
|---|---|---|
| What | a `.wasm` (or `.wat`) module | a type implementing `effectcraft_effects::plugin::EffectPlugin` |
| Loaded by | Effect ▸ Load Effect Plug-in…, `effect.plugins.load {path \| folder}`, or the `Plug-ins` folder next to the settings at start-up | `register_plugin(Arc::new(MyEffect))` in an app that links EffectCraft's crates |
| Sandbox | yes: no imports (no files, clock or network), fuel-limited, at most 1 GiB of memory, deterministic floats | none (it is your code) |
| Platforms | desktop and CLI (the wasmi interpreter, crate feature `effectcraft-plugin/wasm`); not the web build | everywhere |

`effect.plugins.list` lists what is loaded: `{api, wasm, plugins: [{id, name, category, version,
author, source, params}]}`.

## The contract

* **Pixels** are premultiplied RGBA `f32` (0–1; values may exceed 1 in 32 bpc projects), row
  major, processed **in place**. The buffer is the layer's (after masks and the effects above),
  possibly at preview resolution (`scale` = buffer pixels per layer pixel) and padded.
* **Parameters** arrive flattened to `f64`s in the order they are declared: slider, angle,
  checkbox (0 / 1) and popup (0-based index) take one value, point two (already in **buffer
  pixels**), colour four (RGBA 0–1).
* **Time** is layer time in seconds.
* Rendering must be **deterministic** and may run on several threads at once (a WebAssembly
  plug-in gets one instance per thread).
* A failed render (an error code, a trap, running out of fuel) leaves the frame unchanged.

## The manifest

Every plug-in describes itself with a JSON manifest:

```json
{
  "api": 1,
  "id": "org.effectcraft.example.posterize-bands",
  "name": "Posterize Bands",
  "category": "Stylize",
  "version": "1.0.0",
  "author": "EffectCraft contributors",
  "description": "Quantizes luminance into tinted bands.",
  "params": [
    {"id": "levels", "name": "Levels", "type": "slider", "default": 4, "min": 2, "max": 64, "sliderMax": 16, "decimals": 0},
    {"id": "mix", "name": "Tint Amount", "type": "slider", "default": 50, "min": 0, "max": 100},
    {"id": "tint", "name": "Tint", "type": "color", "default": [1.0, 0.55, 0.2, 1.0]},
    {"id": "invert", "name": "Invert Bands", "type": "checkbox", "default": false}
  ]
}
```

* `id`: stable, reverse-domain; letters, digits, `.`, `_`, `-`. `ec.` is reserved for built-ins.
  Ids must be unique (a second plug-in with a taken id is refused).
* `category`: one of the Effects & Presets categories (`Blur & Sharpen`, `Stylize`, …) or a new
  one, which appears after the built-in categories.
* Parameter types: `slider` (`default`, `min`, `max`, optional `sliderMin`, `sliderMax`,
  `decimals`), `angle` (`default` degrees), `checkbox` (`default` bool), `popup` (`options`,
  `default` index), `point` (`default` as fractions of the layer size, `[0.5, 0.5]` = centre),
  `color` (`default` RGBA 0–1). Parameter ids are letters, digits and `_`, unique; at most 256
  parameters; a slider needs `min ≤ default ≤ max` (and `sliderMin ≤ sliderMax`); the manifest
  is at most 1 MiB.

## WebAssembly ABI (v1)

A module with **no imports** that exports:

| export | signature | |
|---|---|---|
| `memory` | memory | linear memory shared with the host |
| `ec_api_version` | `() -> i32` | must return `1` |
| `ec_manifest_ptr`, `ec_manifest_len` | `() -> i32` | the UTF-8 JSON manifest in memory |
| `ec_alloc` | `(bytes: i32) -> i32` | a buffer of at least `bytes` bytes, 8-byte aligned; it may reuse the previous one |
| `ec_render` | `(pixels: i32, width: i32, height: i32, params: i32, nparams: i32, time: f64, scale: f64) -> i32` | process the pixels in place; return 0 on success |

For each frame the host calls `ec_alloc` once for a buffer holding the parameters (`nparams`
little-endian `f64`s) followed by the pixels (`width × height × 4` little-endian `f32`s), fills it,
calls `ec_render`, and reads the pixels back. Each call gets a fuel budget proportional to the
frame size; a module that loops forever stops with an error instead of hanging the app.

### Example

[`examples/plugins/posterize-bands`](../examples/plugins/posterize-bands) is a complete plug-in in
plain Rust (no dependencies):

```sh
cd examples/plugins/posterize-bands
cargo build --release --target wasm32-unknown-unknown
# then, in EffectCraft: Effect ▸ Load Effect Plug-in… → target/wasm32-unknown-unknown/release/posterize_bands.wasm
# or headless:
effectcraft-cli exec effect.plugins.load '{"path": "target/wasm32-unknown-unknown/release/posterize_bands.wasm"}'
```

Any language that compiles to WebAssembly works (C, Zig, AssemblyScript…) as long as the module
has no imports. `crates/plugin/src/tests.rs` has a plug-in written directly in WebAssembly text.

## Rust plug-ins

```rust
use std::sync::Arc;
use effectcraft_effects::plugin::*;

struct Swap(PluginManifest);

impl EffectPlugin for Swap {
    fn manifest(&self) -> &PluginManifest { &self.0 }
    fn render(&self, frame: &mut PluginFrame, params: &PluginParams, _time: f64) -> Result<(), String> {
        if params.b("on") {
            for p in frame.pixels.iter_mut() { p.swap(0, 2); }
        }
        Ok(())
    }
}

register_plugin(Arc::new(Swap(manifest)))?; // before the UI builds its menus
```

`PluginParams` reads values by parameter id (`f`, `b`, `point`, `color`). At most 64 plug-in
effects can be registered per process.

### GPU kernels

With GPU acceleration on, a plug-in effect normally makes the GPU compositor read the frame back,
run `render` on the CPU and upload the result. A Rust plug-in can avoid that by also providing a
GPU kernel: WGSL compute passes with `render`'s semantics. `render` stays the reference, the
kernel must match it, and the effect gets the GPU badge.

```rust
impl EffectPlugin for Swap {
    // manifest, render as above
    fn gpu_kernel(&self) -> Option<&GpuKernel> {
        Some(&self.kernel) // built once: the same kernel every call
    }
}

let kernel = GpuKernel {
    wgsl: r"
        @compute @workgroup_size(16, 16)
        fn swap(@builtin(global_invocation_id) id: vec3<u32>) {
            if (id.x >= frame.size.x || id.y >= frame.size.y) { return; }
            let c = textureLoad(src, vec2<i32>(id.xy), 0);
            textureStore(dst, vec2<i32>(id.xy), select(c, c.zyxw, params[0] != 0.0));
        }".into(),
    passes: vec!["swap".into()],
    table: None,
};
```

* The compositor puts `effectcraft_effects::plugin::GPU_KERNEL_PRELUDE` in front of the source.
  It declares `frame` (size, pass index, parameter count, layer time, scale, table length),
  `src` (the previous pass's output; for the first pass, the effect's input), `orig` (the
  effect's input), `dst` (the output, `rgba32float`), `params` and `table` (read-only `f32`
  arrays). Those names are taken.
* `passes` are entry points declared `@workgroup_size(16, 16)`, run in order, one invocation
  per pixel. Each must write every pixel of `dst`, and the last pass's `dst` is the result.
  Several passes cover effects that need neighbourhoods of an intermediate result, such as a
  displacement followed by a blur of the displaced image.
* `params` is `EffectPlugin::gpu_params`. By default it holds the flattened parameters as `f32`
  (points in buffer pixels). Override it to derive values on the CPU in `f64`, such as frame or
  drawing numbers from the time, scaled radii or seeds.
* `table` is constant data (noise textures, lookup tables), uploaded once per device.
* The output keeps the input's size, offset and scale: kernels cannot pad or resize.
* A kernel that fails to compile on a device is logged once, and the effect then renders on the
  CPU there. On the web build, plug-in kernels always render on the CPU, because compile errors
  cannot be waited for.

This is an optional Rust-side addition: the manifest, the WebAssembly ABI and the API version
are unchanged, and plug-ins without a kernel behave as before.

## Compatibility

The API version is checked when a plug-in loads: a manifest or module for another version is
refused with a message. Within a version the contract above does not change; additions (new
parameter types, new optional exports) bump the version.
