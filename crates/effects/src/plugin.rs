//! The effect plug-in API, version [`PLUGIN_API_VERSION`].
//!
//! A plug-in is anything implementing [`EffectPlugin`]: a [`PluginManifest`] (stable id,
//! display name, Effects & Presets category, typed parameters) and a `render` that processes a
//! [`PluginFrame`] in place. [`register_plugin`] turns it into an ordinary [`EffectSpec`]:
//! instances are property groups like every built-in effect, so plug-in parameters animate,
//! take expressions, show in Effect Controls and are addressable by agents, and the effect is
//! applied by id (`effect.apply {"effect": "org.example.posterize"}`) like a built-in.
//!
//! Plug-ins come from Rust code (implement the trait and register it at start-up) or from
//! sandboxed WebAssembly modules (`effectcraft-plugin`, see `docs/plugins.md`), which implement
//! the same contract through a small C-like ABI.
//!
//! The contract (stable within an API version):
//! * pixels are premultiplied RGBA `f32` (0–1, may exceed 1 in 32 bpc projects), row-major,
//!   in the layer's buffer, which may be scaled (preview resolution) and padded;
//! * parameters are flattened to `f64`s in declaration order: slider / angle / checkbox (0 or 1)
//!   / popup (0-based index) take one value, point two (**frame pixels**, already mapped to the
//!   buffer), colour four (RGBA 0–1);
//! * `time` is layer time in seconds; `scale` is the buffer's pixels per layer pixel (scale
//!   pixel distances like blur radii by it);
//! * rendering must be deterministic (same inputs, same output) and may run on several threads
//!   at once.
//!
//! A Rust plug-in may also provide a GPU implementation ([`EffectPlugin::gpu_kernel`]): WGSL
//! compute passes that the GPU compositor (`effectcraft-gpu`) runs on the GPU-resident image
//! instead of reading it back for `render`. `render` stays the reference the kernel must match.

use std::sync::{Arc, OnceLock, RwLock};

use effectcraft_keyframe::Value;
use effectcraft_project::ParamUi;
use serde::{Deserialize, Serialize};

use crate::{Buf, EffectCtx, EffectSpec, ParamSpec, RenderFn};

/// The plug-in API version this build implements.
pub const PLUGIN_API_VERSION: u32 = 1;

/// The most plug-in effects one process can register.
pub const MAX_PLUGINS: usize = 64;

/// A plug-in parameter's kind, default and range.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum PluginParamKind {
    Slider {
        default: f64,
        #[serde(default)]
        min: f64,
        #[serde(default = "hundred")]
        max: f64,
        /// Slider range (defaults to min..max).
        #[serde(default, rename = "sliderMin")]
        slider_min: Option<f64>,
        #[serde(default, rename = "sliderMax")]
        slider_max: Option<f64>,
        #[serde(default = "two")]
        decimals: u8,
    },
    Angle {
        #[serde(default)]
        default: f64,
    },
    Checkbox {
        #[serde(default)]
        default: bool,
    },
    Popup {
        options: Vec<String>,
        /// 0-based index.
        #[serde(default)]
        default: u32,
    },
    /// Default as fractions of the layer size (`[0.5, 0.5]` = centre).
    Point {
        #[serde(default = "centre")]
        default: [f64; 2],
    },
    /// RGBA, 0–1.
    Color {
        #[serde(default = "white")]
        default: [f64; 4],
    },
}

fn hundred() -> f64 {
    100.0
}
fn two() -> u8 {
    2
}
fn centre() -> [f64; 2] {
    [0.5, 0.5]
}
fn white() -> [f64; 4] {
    [1.0; 4]
}

impl PluginParamKind {
    /// How many `f64`s the parameter flattens to.
    pub fn width(&self) -> usize {
        match self {
            PluginParamKind::Point { .. } => 2,
            PluginParamKind::Color { .. } => 4,
            _ => 1,
        }
    }
}

/// One parameter of a plug-in effect.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PluginParam {
    /// Stable id (the property's match name; letters, digits, `_`).
    pub id: String,
    pub name: String,
    #[serde(flatten)]
    pub kind: PluginParamKind,
}

/// What a plug-in declares about itself.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PluginManifest {
    /// The plug-in API version it was written for (must be [`PLUGIN_API_VERSION`]).
    pub api: u32,
    /// Stable, reverse-domain effect id (`org.example.posterize`); `ec.` is reserved.
    pub id: String,
    pub name: String,
    /// Effects & Presets category: one of [`crate::CATEGORIES`] or a new one.
    pub category: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub author: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub params: Vec<PluginParam>,
}

impl PluginManifest {
    /// Check the manifest against this API version.
    pub fn validate(&self) -> Result<(), String> {
        const MAX_PARAMS: usize = 256;
        if self.api != PLUGIN_API_VERSION {
            return Err(format!("plug-in `{}` targets plug-in API {} but this build implements API {PLUGIN_API_VERSION}", self.id, self.api));
        }
        let ok_id = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
        if !ok_id(&self.id) || self.id.starts_with("ec.") {
            return Err(format!("bad plug-in id `{}` (reverse-domain, letters, digits, `.`, `_`, `-`; `ec.` is reserved)", self.id));
        }
        if self.name.trim().is_empty() || self.category.trim().is_empty() {
            return Err(format!("plug-in `{}` needs a name and a category", self.id));
        }
        if self.params.len() > MAX_PARAMS {
            return Err(format!("plug-in `{}` has {} parameters (at most {MAX_PARAMS})", self.id, self.params.len()));
        }
        let mut seen = std::collections::HashSet::new();
        for p in &self.params {
            if p.id.is_empty() || !p.id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') || !seen.insert(p.id.as_str()) {
                return Err(format!("plug-in `{}`: bad or duplicate parameter id `{}`", self.id, p.id));
            }
            if let PluginParamKind::Popup { options, default } = &p.kind
                && (options.is_empty() || *default as usize >= options.len())
            {
                return Err(format!("plug-in `{}`: popup `{}` needs options and a default index in range", self.id, p.id));
            }
            if let PluginParamKind::Slider { default, min, max, slider_min, slider_max, .. } = &p.kind {
                let (lo, hi) = (slider_min.unwrap_or(*min), slider_max.unwrap_or(*max));
                if [*default, *min, *max, lo, hi].iter().any(|v| !v.is_finite()) || min > max || lo > hi || default < min || default > max {
                    return Err(format!("plug-in `{}`: slider `{}` needs min ≤ default ≤ max (and sliderMin ≤ sliderMax)", self.id, p.id));
                }
            }
        }
        Ok(())
    }

    /// Total flattened parameter count.
    pub fn flat_len(&self) -> usize {
        self.params.iter().map(|p| p.kind.width()).sum()
    }
}

/// The pixels a plug-in renders, in place.
pub struct PluginFrame<'a> {
    pub width: u32,
    pub height: u32,
    /// Premultiplied RGBA, row-major, `width * height` pixels.
    pub pixels: &'a mut [[f32; 4]],
    /// Buffer pixels per layer pixel (preview resolution).
    pub scale: f64,
}

/// Parameter values for one render, flattened (see the module docs) with typed accessors.
pub struct PluginParams<'a> {
    pub manifest: &'a PluginManifest,
    pub flat: &'a [f64],
}

impl PluginParams<'_> {
    fn slot(&self, id: &str) -> Option<&[f64]> {
        let mut at = 0;
        for p in &self.manifest.params {
            let w = p.kind.width();
            if p.id == id {
                return self.flat.get(at..at + w);
            }
            at += w;
        }
        None
    }
    /// Slider / angle / popup index / checkbox (0 or 1).
    pub fn f(&self, id: &str) -> f64 {
        self.slot(id).and_then(|s| s.first().copied()).unwrap_or(0.0)
    }
    pub fn b(&self, id: &str) -> bool {
        self.f(id) != 0.0
    }
    /// Point in frame pixels.
    pub fn point(&self, id: &str) -> [f64; 2] {
        self.slot(id).map(|s| [s[0], s[1]]).unwrap_or([0.0; 2])
    }
    pub fn color(&self, id: &str) -> [f64; 4] {
        self.slot(id).map(|s| [s[0], s[1], s[2], s[3]]).unwrap_or([1.0; 4])
    }
}

/// The bindings every plug-in GPU kernel sees: the GPU compositor puts this in front of
/// [`GpuKernel::wgsl`]. Names declared here (`PluginFrame`, `frame`, `src`, `orig`, `dst`,
/// `params`, `table`) are taken.
pub const GPU_KERNEL_PRELUDE: &str = r"// effectcraft plug-in kernel bindings
struct PluginFrame {
    size: vec2<u32>,   // buffer width and height in pixels
    pass_index: u32,   // index of this pass in GpuKernel::passes
    param_count: u32,  // floats in `params`
    time: f32,         // layer time in seconds
    scale: f32,        // buffer pixels per layer pixel
    table_len: u32,    // floats in `table` (0 without a table)
    _pad: u32,
}
@group(0) @binding(0) var<uniform> frame: PluginFrame;
// the previous pass's output (the first pass: the effect's input)
@group(0) @binding(1) var src: texture_2d<f32>;
// the effect's input, in every pass
@group(0) @binding(2) var orig: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba32float, write>;
// EffectPlugin::gpu_params
@group(0) @binding(4) var<storage, read> params: array<f32>;
// GpuKernel::table (one zero when there is none)
@group(0) @binding(5) var<storage, read> table: array<f32>;
";

/// A plug-in's GPU implementation ([`EffectPlugin::gpu_kernel`]): WGSL compute passes with
/// [`EffectPlugin::render`]'s semantics.
///
/// Each pass is a compute entry point declared `@compute @workgroup_size(16, 16)`, dispatched
/// once over the buffer (one invocation per pixel, `global_invocation_id.xy`; invocations past
/// `frame.size` must return). It reads the bindings of [`GPU_KERNEL_PRELUDE`]: `src` (the
/// previous pass's output; the effect's input for the first pass), `orig` (the effect's input),
/// `params` ([`EffectPlugin::gpu_params`]) and `table`, and writes every pixel of `dst` with
/// `textureStore`. The last pass's `dst` is the effect's output, at the input's size, offset and
/// scale (kernels cannot pad or resize the buffer). Pixels are premultiplied RGBA `f32`, as for
/// `render`. If the WGSL does not compile on a device, the effect renders on the CPU there.
#[derive(Clone)]
pub struct GpuKernel {
    /// The passes' entry points and their helpers (without the prelude).
    pub wgsl: String,
    /// Entry points, run in order.
    pub passes: Vec<String>,
    /// Read-only data that is the same for every frame (noise textures, lookup tables):
    /// uploaded once per device, `table` in WGSL.
    pub table: Option<Arc<[f32]>>,
}

impl std::fmt::Debug for GpuKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuKernel")
            .field("wgsl_bytes", &self.wgsl.len())
            .field("passes", &self.passes)
            .field("table_len", &self.table.as_ref().map(|t| t.len()))
            .finish()
    }
}

/// An effect plug-in.
pub trait EffectPlugin: Send + Sync {
    fn manifest(&self) -> &PluginManifest;
    /// Process `frame` in place. `time` is layer time in seconds. An error leaves the frame as
    /// it was (the error is logged once).
    fn render(&self, frame: &mut PluginFrame, params: &PluginParams, time: f64) -> Result<(), String>;
    /// Where the plug-in came from (a file path, `built-in`…), for listings.
    fn source(&self) -> String {
        "built-in".into()
    }
    /// A GPU implementation of [`EffectPlugin::render`] (see [`GpuKernel`]): with GPU
    /// acceleration on, the GPU compositor runs it instead of reading the frame back for
    /// `render`. It must produce what `render` produces. `None` (the default): the effect
    /// always renders on the CPU. Asked once, at registration, and again for each GPU frame, so
    /// return the same kernel every time.
    fn gpu_kernel(&self) -> Option<&GpuKernel> {
        None
    }
    /// The kernel's `params` for one frame of a buffer at `scale` (buffer pixels per layer
    /// pixel). The default is the flattened parameters as `f32`, in declaration order (see the
    /// module docs). Override it to derive values in `f64` on the CPU first, such as integer
    /// frame or drawing numbers from `time`, which `f32` time on the GPU cannot give exactly.
    fn gpu_params(&self, params: &PluginParams, time: f64, scale: f64) -> Vec<f32> {
        let _ = (time, scale);
        params.flat.iter().map(|v| *v as f32).collect()
    }
}

static SLOTS: [OnceLock<Arc<dyn EffectPlugin>>; MAX_PLUGINS] = [const { OnceLock::new() }; MAX_PLUGINS];

/// Registered plug-in effects, in registration order.
fn plugin_specs() -> &'static RwLock<Vec<&'static EffectSpec>> {
    static P: OnceLock<RwLock<Vec<&'static EffectSpec>>> = OnceLock::new();
    P.get_or_init(|| RwLock::new(vec![]))
}

/// The registered plug-in effects.
pub fn plugins() -> Vec<&'static EffectSpec> {
    plugin_specs().read().map(|v| v.clone()).unwrap_or_default()
}

/// The plug-in behind a registered effect id.
pub fn plugin(id: &str) -> Option<Arc<dyn EffectPlugin>> {
    SLOTS.iter().filter_map(|s| s.get()).find(|p| p.manifest().id == id).cloned()
}

fn leak(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

fn param_spec(p: &PluginParam) -> ParamSpec {
    let (default, ui) = match &p.kind {
        PluginParamKind::Slider { default, min, max, slider_min, slider_max, decimals } => (
            Value::Scalar(*default),
            ParamUi::Slider { min: *min, max: *max, slider_min: slider_min.unwrap_or(*min), slider_max: slider_max.unwrap_or(*max), decimals: *decimals },
        ),
        PluginParamKind::Angle { default } => (Value::Scalar(*default), ParamUi::Angle),
        PluginParamKind::Checkbox { default } => (Value::Bool(*default), ParamUi::Checkbox),
        PluginParamKind::Popup { options, default } => (Value::Enum(*default), ParamUi::Popup { options: options.clone() }),
        PluginParamKind::Point { default } => (Value::Vec2(*default), ParamUi::Point),
        PluginParamKind::Color { default } => (Value::Color(*default), ParamUi::Color),
    };
    ParamSpec { id: leak(&p.id), name: leak(&p.name), default, ui }
}

/// Register a plug-in effect: it appears in Effects & Presets under its category and is
/// applied by id like a built-in. Fails for a bad manifest, a taken id or when
/// [`MAX_PLUGINS`] are registered. A plug-in with a [`GpuKernel`] is a GPU effect
/// ([`EffectSpec::gpu`]).
pub fn register_plugin(plugin: Arc<dyn EffectPlugin>) -> Result<&'static EffectSpec, String> {
    let m = plugin.manifest().clone();
    m.validate()?;
    let gpu = match plugin.gpu_kernel() {
        Some(k) if k.passes.is_empty() => return Err(format!("plug-in `{}`: a GPU kernel needs at least one pass", m.id)),
        Some(_) => true,
        None => false,
    };
    let mut specs = plugin_specs().write().map_err(|_| "plug-in registry poisoned".to_string())?;
    if crate::registry().iter().any(|s| s.id == m.id) || specs.iter().any(|s| s.id == m.id) {
        return Err(format!("an effect with id `{}` is already registered", m.id));
    }
    let slot = SLOTS.iter().position(|s| s.get().is_none()).ok_or_else(|| format!("too many plug-ins (at most {MAX_PLUGINS})"))?;
    SLOTS[slot].set(plugin).map_err(|_| "plug-in slot taken".to_string())?;
    let spec: &'static EffectSpec = Box::leak(Box::new(EffectSpec {
        id: leak(&m.id),
        name: leak(&m.name),
        category: leak(&m.category),
        params: m.params.iter().map(param_spec).collect(),
        render: TRAMPOLINES[slot],
        gpu,
        float: true,
    }));
    specs.push(spec);
    Ok(spec)
}

/// Each slot needs its own `fn` (effect render functions are plain function pointers).
fn trampoline<const I: usize>(ctx: &EffectCtx, buf: Buf) -> Buf {
    run_slot(I, ctx, buf)
}

macro_rules! trampolines {
    ($($i:literal)*) => { [$(trampoline::<$i> as RenderFn),*] };
}

const TRAMPOLINES: [RenderFn; MAX_PLUGINS] = trampolines!(
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
    32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
);

/// Flatten an instance's evaluated parameters for the plug-in (points mapped to frame pixels).
pub fn flatten(m: &PluginManifest, ctx: &EffectCtx, buf: &Buf) -> Vec<f64> {
    flatten_at(m, ctx, buf.offset, buf.scale)
}

/// [`flatten`] for a buffer with this offset and scale ([`Buf::to_px`]) whose pixels live
/// elsewhere (on the GPU).
pub fn flatten_at(m: &PluginManifest, ctx: &EffectCtx, offset: [f64; 2], scale: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(m.flat_len());
    for p in &m.params {
        let v = ctx.params.get(&p.id);
        match &p.kind {
            PluginParamKind::Point { .. } => {
                let q = v.map(Value::as_vec2).unwrap_or([0.0; 2]);
                out.extend([q[0] * scale + offset[0], q[1] * scale + offset[1]]);
            }
            PluginParamKind::Color { .. } => {
                let c = v.map(Value::as_color).unwrap_or([1.0; 4]);
                out.extend(c.iter().map(|x| *x as f64));
            }
            PluginParamKind::Checkbox { .. } => out.push(if v.map(Value::as_bool).unwrap_or(false) { 1.0 } else { 0.0 }),
            PluginParamKind::Popup { .. } => out.push(v.map(Value::as_enum).unwrap_or(0) as f64),
            _ => out.push(v.map(Value::as_f64).unwrap_or(0.0)),
        }
    }
    out
}

fn run_slot(slot: usize, ctx: &EffectCtx, mut buf: Buf) -> Buf {
    let Some(plugin) = SLOTS[slot].get() else { return buf };
    let m = plugin.manifest();
    let flat = flatten(m, ctx, &buf);
    let params = PluginParams { manifest: m, flat: &flat };
    let (width, height) = (buf.img.width, buf.img.height);
    let before = buf.img.data.clone();
    let mut frame = PluginFrame { width, height, pixels: &mut buf.img.data, scale: buf.scale };
    if let Err(e) = plugin.render(&mut frame, &params, ctx.time) {
        log::warn!("plug-in `{}`: {e}", m.id);
        buf.img.data = before;
    }
    buf
}
