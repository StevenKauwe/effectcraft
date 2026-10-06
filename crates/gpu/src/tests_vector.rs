//! Shape layers rasterised on the GPU (`vector`, vello) against the CPU's tiny-skia buffers.
//! The two rasterisers anti-alias differently, so edges differ slightly and the per-pixel
//! tolerance of the other GPU tests does not apply: the frame must agree on average (mean
//! difference ≤ 0.5 / 255 per channel) and nearly everywhere (≤ 0.5 % of pixels more than
//! 16 / 255 apart). Interiors, colours, gradients, strokes, effects and placement are the CPU's.
//! Measured on an M2 Max: mean 0.20–0.26, 0.09–0.25 % over 16 levels, all on edges (tiny-skia
//! samples coverage in quarter steps vertically, so near-horizontal edges differ most).

use effectcraft_keyframe::{Gradient, ShapePath, Value};
use effectcraft_project::build::{self, Ids};
use effectcraft_project::{BitDepth, Layer, LayerSource, PropGroup};
use effectcraft_raster::Image;
use effectcraft_render::{Backend, RenderOpts, Renderer};
use effectcraft_time::Tick;

use crate::tests::{Pattern, Scene, gpu, set, v3};

fn shape_layer(s: &mut Scene, name: &str, groups: impl FnOnce(&mut Ids) -> Vec<PropGroup>) -> Layer {
    let mut l = build::layer(&mut s.p, &s.comp, name, LayerSource::Shape, (s.comp.width, s.comp.height), None);
    let mut next = s.p.next_id;
    let gs = groups(&mut Ids(&mut next));
    s.p.next_id = next;
    let c = l.props.group_mut("contents").expect("shape contents");
    for g in gs {
        c.children.push(g.into());
    }
    set(&mut l, "transform/position", v3(0.0, 0.0));
    set(&mut l, "transform/anchor", v3(0.0, 0.0));
    l
}

fn group(ids: &mut Ids, name: &str, items: Vec<PropGroup>, rot: f64, pos: [f64; 2]) -> PropGroup {
    let mut g = build::shape_group(ids, name, items);
    if let Some(p) = g.prop_mut("transform/rotation") {
        p.value = Value::Scalar(rot);
    }
    if let Some(p) = g.prop_mut("transform/position") {
        p.value = Value::Vec2(pos);
    }
    g
}

/// Paper, a wash with an ink outline, an even-odd star, a tapered open line, a radial glow
/// (Add) and a watercolour fill with Turbulent Displace and Gaussian Blur; continuously
/// rasterised under a scaled, rotated parent transform where `collapse` is set.
fn painted(depth: BitDepth, collapse: bool) -> Scene {
    let mut s = Scene::new(depth);
    s.comp.width = 320;
    s.comp.height = 200;
    if let Some(c) = s.p.comp_mut(s.cid) {
        c.width = 320;
        c.height = 200;
    }
    let bg = s.solid([0.95, 0.92, 0.86], 320, 200);
    s.push(bg);

    let mut wc = shape_layer(&mut s, "watercolour", |ids| {
        let blob = ShapePath::polygon(&[[60.0, 40.0], [190.0, 30.0], [230.0, 120.0], [120.0, 170.0], [40.0, 120.0]], true);
        let items = vec![build::shape_path(ids, blob), build::shape_fill(ids, [0.55, 0.75, 0.45, 1.0])];
        vec![group(ids, "blob", items, 0.0, [0.0, 0.0])]
    });
    s.effect(&mut wc, "ec.distort.turbulentdisplace", &[("amount", Value::Scalar(12.0)), ("size", Value::Scalar(31.0)), ("complexity", Value::Scalar(2.0))]);
    s.effect(&mut wc, "ec.blur.gaussian", &[("blurriness", Value::Scalar(6.0))]);
    s.push(wc);

    let mut ink = shape_layer(&mut s, "ink", |ids| {
        let mut stroke = build::shape_stroke(ids, [0.2, 0.15, 0.12, 1.0], 3.5);
        for (k, v) in [("join", Value::Enum(1)), ("cap", Value::Enum(1))] {
            if let Some(p) = stroke.prop_mut(k) {
                p.value = v;
            }
        }
        let wash = build::shape_fill(ids, [0.85, 0.45, 0.35, 1.0]);
        let body = vec![build::shape_ellipse(ids, [120.0, 80.0], [0.0, 0.0]), stroke, wash];
        let mut star = build::shape_star(ids, true, 5.0, [0.0, 0.0], 44.0, 18.0);
        if let Some(p) = star.prop_mut("direction") {
            p.value = Value::Enum(0);
        }
        let mut eo = build::shape_fill(ids, [0.3, 0.35, 0.7, 1.0]);
        if let Some(p) = eo.prop_mut("rule") {
            p.value = Value::Enum(1);
        }
        let mut line = build::shape_stroke(ids, [0.1, 0.1, 0.1, 1.0], 6.0);
        for (k, v) in
            [("cap", Value::Enum(1)), ("taper/endLength", Value::Scalar(45.0)), ("taper/endWidth", Value::Scalar(20.0)), ("taper/units", Value::Enum(1))]
        {
            if let Some(p) = line.prop_mut(k) {
                p.value = v;
            }
        }
        let curve = ShapePath::polygon(&[[20.0, 180.0], [80.0, 150.0], [150.0, 175.0], [240.0, 140.0], [300.0, 160.0]], false);
        let curve = build::shape_path(ids, curve);
        vec![
            group(ids, "line", vec![curve, line], 0.0, [0.0, 0.0]),
            group(ids, "star", vec![star, eo], 12.0, [250.0, 60.0]),
            group(ids, "body", body, -20.0, [150.0, 95.0]),
        ]
    });
    if collapse {
        ink.switches.collapse = true;
        set(&mut ink, "transform/scale", Value::Vec3([130.0, 130.0, 100.0]));
        set(&mut ink, "transform/rotation", Value::Scalar(7.0));
        set(&mut ink, "transform/anchor", v3(160.0, 100.0));
        set(&mut ink, "transform/position", v3(161.3, 99.6));
    }
    s.push(ink);

    let mut glow = shape_layer(&mut s, "glow", |ids| {
        let g = Gradient {
            colors: vec![(0.0, [1.0, 0.9, 0.6, 1.0]), (1.0, [1.0, 0.9, 0.6, 1.0])],
            opacities: vec![(0.0, 1.0), (0.18, 0.8), (0.45, 0.32), (0.75, 0.08), (1.0, 0.0)],
        };
        let items = vec![build::shape_ellipse(ids, [140.0, 140.0], [0.0, 0.0]), build::shape_gradient_fill(ids, true, [0.0, 0.0], [70.0, 0.0], g)];
        vec![group(ids, "glow", items, 0.0, [90.0, 70.0])]
    });
    glow.blend_mode = effectcraft_color::BlendMode::Add;
    set(&mut glow, "transform/opacity", Value::Scalar(60.0));
    s.push(glow);
    s
}

/// (mean |difference| per channel, fraction of pixels with a channel more than 16 levels
/// apart), on 8-bit straight colour over the comp's pixels.
fn perceptual(cpu: &Image, gpu: &Image) -> (f64, f64) {
    assert_eq!((cpu.width, cpu.height), (gpu.width, gpu.height), "size");
    let (mut sum, mut over) = (0.0f64, 0usize);
    for (a, b) in cpu.data.iter().zip(&gpu.data) {
        let mut px = 0.0f32;
        for c in 0..4 {
            let d = (a[c].clamp(0.0, 1.0) - b[c].clamp(0.0, 1.0)).abs() * 255.0;
            sum += d as f64;
            px = px.max(d);
        }
        over += (px > 16.0) as usize;
    }
    let n = cpu.data.len().max(1) as f64;
    (sum / (n * 4.0), over as f64 / n)
}

#[test]
fn shape_layers_rasterise_on_the_gpu_like_the_cpu() {
    for depth in [BitDepth::Bpc8, BitDepth::Bpc32] {
        for collapse in [false, true] {
            let s = painted(depth, collapse);
            let Some(g) = gpu() else { return };
            g.context().clear_uploads();
            let opts = RenderOpts::default();
            let cpu = Renderer::new(&s.p, &Pattern, RenderOpts { backend: Backend::Cpu, ..opts }).comp_frame_cpu(s.cid, Tick::ZERO);
            let mut r = Renderer::new(&s.p, &Pattern, RenderOpts { backend: Backend::Gpu, ..opts });
            r.accel = Some(g);
            let before = g.context().shape_layers_drawn();
            let img = g.render(&r, s.cid, Tick::ZERO).expect("the GPU renders the frame");
            assert_eq!(g.context().shape_layers_drawn() - before, 3, "{depth:?} collapse {collapse}: every shape layer rasterised on the GPU");
            let (mean, over16) = perceptual(&cpu, &img);
            eprintln!("{depth:?} collapse {collapse}: mean {mean:.3} levels, {:.3} % of pixels over 16 levels", over16 * 100.0);
            assert!(mean <= 0.5 && over16 <= 0.005, "{depth:?} collapse {collapse}: mean {mean:.3}, over 16 levels {:.3} %", over16 * 100.0);
            // The same frame again comes from the GPU's cache of processed shape layers.
            let again = g.render(&r, s.cid, Tick::ZERO).expect("the GPU renders the frame");
            assert_eq!(g.context().shape_layers_drawn() - before, 3, "{depth:?} collapse {collapse}: cached layers are not rasterised again");
            assert_eq!(again.data, img.data, "{depth:?} collapse {collapse}: cached layers draw the same frame");
        }
    }
}
