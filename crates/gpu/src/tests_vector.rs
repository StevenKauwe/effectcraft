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
            let renders = g.context().vector.renders.load(std::sync::atomic::Ordering::Relaxed);
            let img = g.render(&r, s.cid, Tick::ZERO).expect("the GPU renders the frame");
            assert_eq!(g.context().shape_layers_drawn() - before, 3, "{depth:?} collapse {collapse}: every shape layer rasterised on the GPU");
            let renders = g.context().vector.renders.load(std::sync::atomic::Ordering::Relaxed) - renders;
            assert_eq!(renders, 1, "{depth:?} collapse {collapse}: the frame's shape layers share one vello render");
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

/// A shape layer larger than the frame (a floor seen through a window), with Turbulent
/// Displace and Gaussian Blur, at the 8-bit depth and a fractional position.
fn floor_through_a_window() -> Scene {
    let mut s = Scene::new(BitDepth::Bpc8);
    let mut floor = shape_layer(&mut s, "floor", |ids| {
        let mut gs = Vec::new();
        for k in 0..24 {
            let (x, y) = ((k % 6) as f64 * 40.0 + 20.0, (k / 6) as f64 * 45.0 + 22.0);
            let c = [0.2 + 0.03 * k as f64, 0.5, 0.9 - 0.03 * k as f64, 1.0];
            let items = vec![build::shape_ellipse(ids, [30.0, 26.0], [0.0, 0.0]), build::shape_fill(ids, c)];
            gs.push(group(ids, "dab", items, k as f64 * 7.0, [x, y]));
        }
        gs
    });
    s.effect(&mut floor, "ec.distort.turbulentdisplace", &[("amount", Value::Scalar(12.0)), ("size", Value::Scalar(31.0)), ("complexity", Value::Scalar(2.0))]);
    s.effect(&mut floor, "ec.blur.gaussian", &[("blurriness", Value::Scalar(6.0))]);
    set(&mut floor, "transform/position", v3(-71.3, -58.6));
    s.push(floor);
    s
}

/// A large shape layer is drawn only where the frame can see it, widened by its effects' reach:
/// fewer pixels, the same frame. Its third miss draws it whole (static content under a moving
/// camera), and that frame is the same too.
#[test]
fn large_shape_layers_draw_only_what_shows() {
    use std::sync::atomic::Ordering::Relaxed;
    let s = floor_through_a_window();
    let Some(g) = gpu() else { return };
    let mut r = Renderer::new(&s.p, &Pattern, RenderOpts { backend: Backend::Gpu, ..RenderOpts::default() });
    r.accel = Some(g);
    let v = &g.context().vector;
    let mut frames = Vec::new();
    for _ in 0..3 {
        g.context().clear_uploads();
        let before = v.drawn_px.load(Relaxed);
        let img = g.render(&r, s.cid, Tick::ZERO).expect("the GPU renders the frame");
        frames.push((img, v.drawn_px.load(Relaxed) - before));
    }
    let (cropped, cropped_px) = &frames[0];
    let (whole, whole_px) = &frames[2];
    assert_eq!(frames[1].1, *cropped_px, "the second miss is cropped too");
    assert!(*cropped_px * 5 < *whole_px * 4, "cropped {cropped_px} px, whole {whole_px} px");
    assert_eq!(cropped.data, whole.data, "the crop draws the same frame as the whole layer");
    let cpu = Renderer::new(&s.p, &Pattern, RenderOpts { backend: Backend::Cpu, ..RenderOpts::default() }).comp_frame_cpu(s.cid, Tick::ZERO);
    let (mean, over16) = perceptual(&cpu, cropped);
    assert!(mean <= 0.5 && over16 <= 0.005, "against the CPU: mean {mean:.3}, over 16 levels {:.3} %", over16 * 100.0);
}

/// vello targets come back to their pool once the work reading them is submitted, and not
/// before: a target still to be read is never drawn over. The layers an encoder draws share
/// one target (its atlas).
#[test]
fn vello_targets_are_reused_after_submit() {
    use std::sync::atomic::Ordering::Relaxed;
    let painted = painted(BitDepth::Bpc32, false);
    let floor = floor_through_a_window();
    let Some(g) = gpu() else { return };
    g.context().clear_uploads();
    let scene_of = |s: &Scene, name: &str| {
        let r = Renderer::new(&s.p, &Pattern, RenderOpts::default());
        let ctx = r.eval_ctx(s.cid, Tick::ZERO).expect("comp");
        let layer = ctx.comp.layers.iter().find(|l| l.name == name).expect("layer");
        r.shape_scene(&ctx, layer).expect("a shape scene")
    };
    let (a, b) = (scene_of(&painted, "ink"), scene_of(&floor, "floor"));
    let v = &g.context().vector;
    let made = || v.targets_made.load(Relaxed);
    let before = made();
    let mut e = crate::context::Enc::new(g.context());
    let first_a = crate::vector::rasterise(&mut e, &a, None).expect("vello");
    let first_b = crate::vector::rasterise(&mut e, &b, None).expect("vello");
    assert_eq!(made() - before, 1, "an encoder's layers share one atlas");
    let mut other = crate::context::Enc::new(g.context());
    let other_b = crate::vector::rasterise(&mut other, &b, None).expect("vello");
    assert_eq!(made() - before, 2, "an atlas is not reused before its reader is submitted");
    e.submit();
    let again_a = crate::vector::rasterise(&mut e, &a, None).expect("vello");
    let again_b = crate::vector::rasterise(&mut e, &b, None).expect("vello");
    assert_eq!(made() - before, 2, "a submitted atlas is reused");
    let read = |e: &mut crate::context::Enc, i| e.download(i).expect("readback").data;
    let (fa, fb) = (read(&mut e, &first_a), read(&mut e, &first_b));
    assert_eq!(read(&mut e, &again_a), fa);
    assert_eq!(read(&mut e, &again_b), fb);
    assert_eq!(read(&mut other, &other_b), fb, "a layer draws the same pixels anywhere in an atlas");
    assert!(!e.failed && !other.failed);
}

/// A processed shape layer reaches the GPU cache only once the encoder that draws it has
/// submitted that work: another thread's frame that took it earlier could be submitted first
/// and read the texture before it was drawn (an empty layer, or a reused texture's old pixels).
/// Until then the drawing encoder still finds it itself.
#[test]
fn cached_shape_layers_are_shared_only_once_drawn() {
    let painted = painted(BitDepth::Bpc32, false);
    let Some(g) = gpu() else { return };
    g.context().clear_uploads();
    let r = Renderer::new(&painted.p, &Pattern, RenderOpts::default());
    let ctx = r.eval_ctx(painted.cid, Tick::ZERO).expect("comp");
    let layer = ctx.comp.layers.iter().find(|l| l.name == "ink").expect("layer");
    let scene = r.shape_scene(&ctx, layer).expect("a shape scene");
    let key = 0x5eed_cafe_u64;
    let mut a = crate::context::Enc::new(g.context());
    let img = crate::vector::rasterise(&mut a, &scene, None).expect("vello");
    crate::vector::publish(&mut a, key, (img.clone(), scene.offset, scene.scale));
    assert!(g.context().vector.get(key).is_none(), "not shared before its encoder submits");
    assert!(crate::vector::lookup(&a, key).is_some(), "its own encoder finds it");
    let want = a.download(&img).expect("readback");
    let shared = g.context().vector.get(key).expect("shared once submitted");
    let mut b = crate::context::Enc::new(g.context());
    assert_eq!(b.download(&shared.0).expect("readback").data, want.data);
    assert!(want.data.iter().any(|p| p[3] > 0.0), "the layer drew something");
}
