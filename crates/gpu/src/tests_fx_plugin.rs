//! Plug-in effects with a GPU kernel vs their CPU `render` (the oracle): see `fx_plugin.rs`.

use std::sync::{Arc, OnceLock};

use effectcraft_effects::plugin::{EffectPlugin, GpuKernel, PluginFrame, PluginManifest, PluginParam, PluginParamKind, PluginParams, register_plugin};
use effectcraft_keyframe::Value;
use effectcraft_project::BitDepth;
use effectcraft_render::Accelerator;
use effectcraft_time::Tick;

use crate::tests::{Scene, check, compare_at, effect_case, gpu, n, opts, pattern};

/// A two-pass kernel exercising every binding: pass 1 tints (`params`: slider, colour, a step
/// derived from time in `gpu_params`) and multiplies by `table`; pass 2 blurs pass 1's output
/// (`src`) 3×3 and mixes it with the effect's input (`orig`) by the distance to a point param.
const WGSL: &str = r"
fn tinted(x: i32, y: i32) -> vec4<f32> {
    let c = textureLoad(src, vec2<i32>(x, y), 0);
    let tint = vec3<f32>(params[1], params[2], params[3]) * (1.0 + 0.1 * params[8]);
    let rgb = mix(c.xyz, tint * c.w, params[0]) * table[u32(x + 3 * y) % frame.table_len];
    return vec4<f32>(rgb, c.w);
}

@compute @workgroup_size(16, 16)
fn test_tint(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= frame.size.x || id.y >= frame.size.y) {
        return;
    }
    textureStore(dst, vec2<i32>(id.xy), tinted(i32(id.x), i32(id.y)));
}

@compute @workgroup_size(16, 16)
fn test_blur(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= frame.size.x || id.y >= frame.size.y) {
        return;
    }
    let p = vec2<i32>(id.xy);
    let size = vec2<i32>(frame.size);
    var s = vec4<f32>(0.0);
    var n = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = p + vec2<i32>(dx, dy);
            if (all(q >= vec2<i32>(0)) && all(q < size)) {
                s += textureLoad(src, q, 0);
                n += 1.0;
            }
        }
    }
    let w = clamp(distance(vec2<f32>(p), vec2<f32>(params[5], params[6])) / max(params[7], 1e-6), 0.0, 1.0);
    textureStore(dst, p, s / n * w + textureLoad(orig, p, 0) * (1.0 - w));
}
";

fn table() -> Arc<[f32]> {
    (0..16).map(|i| 0.8 + 0.4 * ((i * 7) % 16) as f32 / 16.0).collect()
}

fn manifest(id: &str) -> PluginManifest {
    let slider = |id: &str, default: f64, max: f64| PluginParam {
        id: id.into(),
        name: id.into(),
        kind: PluginParamKind::Slider { default, min: 0.0, max, slider_min: None, slider_max: None, decimals: 2 },
    };
    PluginManifest {
        api: effectcraft_effects::plugin::PLUGIN_API_VERSION,
        id: id.into(),
        name: id.into(),
        category: "Stylize".into(),
        version: "1".into(),
        author: "EffectCraft tests".into(),
        description: String::new(),
        params: vec![
            slider("amount", 0.5, 1.0),
            PluginParam { id: "tint".into(), name: "Tint".into(), kind: PluginParamKind::Color { default: [1.0, 0.55, 0.2, 1.0] } },
            PluginParam { id: "center".into(), name: "Center".into(), kind: PluginParamKind::Point { default: [0.4, 0.6] } },
            slider("radius", 20.0, 200.0),
        ],
    }
}

/// The CPU reference of `WGSL`.
struct TintBlur {
    m: PluginManifest,
    kernel: Option<GpuKernel>,
}

/// Animation step from time, computed in f64 (what `gpu_params` is for).
fn step(time: f64) -> f32 {
    ((time * 24.0).floor() % 5.0) as f32
}

impl EffectPlugin for TintBlur {
    fn manifest(&self) -> &PluginManifest {
        &self.m
    }
    fn render(&self, frame: &mut PluginFrame, p: &PluginParams, time: f64) -> Result<(), String> {
        let (w, h) = (frame.width as usize, frame.height as usize);
        let (amount, c) = (p.f("amount") as f32, p.color("tint"));
        let k = 1.0 + 0.1 * step(time);
        let tint = [c[0] as f32 * k, c[1] as f32 * k, c[2] as f32 * k];
        let table = table();
        let orig = frame.pixels.to_vec();
        let a: Vec<[f32; 4]> = orig
            .iter()
            .enumerate()
            .map(|(i, px)| {
                let t = table[(i % w + 3 * (i / w)) % 16];
                let m = |ch: usize| (px[ch] * (1.0 - amount) + tint[ch] * px[3] * amount) * t;
                [m(0), m(1), m(2), px[3]]
            })
            .collect();
        let ctr = p.point("center");
        let r = (p.f("radius") * frame.scale) as f32;
        for y in 0..h {
            for x in 0..w {
                let (mut s, mut n) = ([0.0f32; 4], 0.0f32);
                for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                    for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                        for ch in 0..4 {
                            s[ch] += a[yy * w + xx][ch];
                        }
                        n += 1.0;
                    }
                }
                let d = ((x as f32 - ctr[0] as f32).powi(2) + (y as f32 - ctr[1] as f32).powi(2)).sqrt();
                let wt = (d / r.max(1e-6)).clamp(0.0, 1.0);
                let o = orig[y * w + x];
                frame.pixels[y * w + x] = std::array::from_fn(|ch| s[ch] / n * wt + o[ch] * (1.0 - wt));
            }
        }
        Ok(())
    }
    fn gpu_kernel(&self) -> Option<&GpuKernel> {
        self.kernel.as_ref()
    }
    fn gpu_params(&self, p: &PluginParams, time: f64, scale: f64) -> Vec<f32> {
        let c = p.color("tint");
        let ctr = p.point("center");
        [p.f("amount"), c[0], c[1], c[2], c[3], ctr[0], ctr[1], p.f("radius") * scale].iter().map(|v| *v as f32).chain([step(time)]).collect()
    }
}

const KERNEL_ID: &str = "org.effectcraft.test.tint-blur";
const CPU_ONLY_ID: &str = "org.effectcraft.test.tint-blur-cpu";
const BROKEN_ID: &str = "org.effectcraft.test.tint-blur-broken";

/// The test plug-ins, registered once per process: with the kernel, without one, and with a
/// kernel that does not compile.
fn register() {
    static DONE: OnceLock<()> = OnceLock::new();
    DONE.get_or_init(|| {
        let kernel = |wgsl: &str| Some(GpuKernel { wgsl: wgsl.into(), passes: vec!["test_tint".into(), "test_blur".into()], table: Some(table()) });
        register_plugin(Arc::new(TintBlur { m: manifest(KERNEL_ID), kernel: kernel(WGSL) })).unwrap();
        register_plugin(Arc::new(TintBlur { m: manifest(CPU_ONLY_ID), kernel: None })).unwrap();
        register_plugin(Arc::new(TintBlur { m: manifest(BROKEN_ID), kernel: kernel("fn test_tint( {") })).unwrap();
    });
}

#[test]
fn plugin_kernels_match_their_cpu_render() {
    register();
    // On a buffer (scale 1 and 0.5, as a layer and an adjustment) and composited at 8 and 32 bpc.
    effect_case(KERNEL_ID, &[]);
    effect_case(KERNEL_ID, &[("amount", n(0.9)), ("radius", n(7.0)), ("tint", Value::Color([0.2, 0.4, 1.0, 1.0])), ("center", Value::Vec2([10.0, 30.0]))]);
    // On an adjustment layer over a footage layer, at another time (another step).
    for depth in [BitDepth::Bpc8, BitDepth::Bpc32] {
        let mut s = Scene::new(depth);
        let bg = s.footage(97, 61);
        s.push(bg);
        let mut adj = s.solid([1.0; 3], 97, 61);
        adj.switches.adjustment = true;
        s.effect(&mut adj, KERNEL_ID, &[("amount", n(0.3))]);
        s.push(adj);
        check(&format!("plug-in kernel on an adjustment layer {depth:?}"), compare_at(&s, opts(), Tick::from_seconds_f64(0.5)), 0.0);
    }
}

#[test]
fn plugin_kernels_are_gpu_effects() {
    register();
    let spec = |id| effectcraft_effects::find(id).unwrap();
    assert!(spec(KERNEL_ID).gpu, "a plug-in with a kernel is a GPU effect");
    assert!(!spec(CPU_ONLY_ID).gpu, "a plug-in without one is not");
    assert!(crate::effects::supports(KERNEL_ID));
    assert!(!crate::effects::supports(CPU_ONLY_ID));
    let Some(g) = gpu() else { return };
    assert!(g.supports_effect(KERNEL_ID));
    assert!(!g.supports_effect(CPU_ONLY_ID));
}

#[test]
fn a_kernel_that_does_not_compile_renders_on_the_cpu() {
    register();
    let Some(g) = gpu() else { return };
    let spec = effectcraft_effects::find(BROKEN_ID).unwrap();
    let params = effectcraft_effects::Params {
        values: spec.params.iter().map(|ps| (ps.id.to_string(), effectcraft_effects::default_value(ps, [40.0, 30.0]))).collect(),
    };
    let ctx = || effectcraft_effects::EffectCtx { params: &params, time: 0.0, layer_size: [40.0, 30.0], seed: 1, adjustment: false, env: Default::default() };
    let buf = effectcraft_effects::Buf { img: pattern(3, 40, 30), offset: [0.0; 2], scale: 1.0 };
    // Declined twice: the failure is remembered, not retried.
    for _ in 0..2 {
        assert!(g.effects(&[effectcraft_render::FxStep { spec, ctx: ctx() }], &buf, None).is_none());
    }
    // A frame still renders, with the effect on the CPU.
    let mut s = Scene::new(BitDepth::Bpc32);
    let mut l = s.footage(70, 44);
    s.effect(&mut l, BROKEN_ID, &[]);
    s.push(l);
    check("broken plug-in kernel falls back to the CPU", compare_at(&s, opts(), Tick::ZERO), 0.0);
}

#[test]
fn a_kernel_needs_a_pass() {
    let mut m = manifest("org.effectcraft.test.no-passes");
    m.params.clear();
    let p = TintBlur { m, kernel: Some(GpuKernel { wgsl: WGSL.into(), passes: vec![], table: None }) };
    let err = register_plugin(Arc::new(p)).err().unwrap();
    assert!(err.contains("at least one pass"), "{err}");
}
