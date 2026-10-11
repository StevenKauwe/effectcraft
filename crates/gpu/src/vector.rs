//! Shape layers rasterised on the GPU with vello (linebender's compute 2D renderer, on this
//! device). The CPU renderer describes a shape layer's source as vector draws
//! ([`Renderer::shape_scene`](effectcraft_render::Renderer::shape_scene): the buffer `shapes::render`
//! would fill, strokes already outlined), vello fills them into an RGBA8 texture of that size,
//! and a kernel premultiplies it into a working image, clamped and quantised to the bit depth as
//! the CPU source is. Only the anti-aliasing differs from the CPU (vello's analytic area
//! coverage against tiny-skia's), by a level or two on edges.
//!
//! Processed shape layers (source and effects) are kept on the GPU under the CPU layer cache's
//! key, so a layer that does not change between frames is drawn once.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use effectcraft_geom::Mat3;
use effectcraft_path::FillRule;
use effectcraft_render::shapes::{ShapePaint, ShapeScene};
use vello::kurbo::{Affine, BezPath, PathEl, Point};
use vello::peniko::{self, Color, Fill};

use crate::context::{Enc, GpuImage, Params};

/// Processed shape layers kept on the GPU (bytes of RGBA f32).
const CACHE_BUDGET: usize = 1024 << 20;

/// Largest distance (buffer pixels) between a curve and the lines it is flattened into.
const FLATTEN_TOLERANCE: f64 = 0.02;

/// Gradient ramps are handed to vello as this many stops (plus the gradient's own).
const RAMP_STOPS: usize = 64;

/// A processed layer on the GPU: image, buffer offset and scale (as [`effectcraft_effects::Buf`]).
pub(crate) type Content = (GpuImage, [f64; 2], f64);

#[derive(Default)]
struct Cache {
    map: HashMap<u64, (Content, u64)>,
    bytes: usize,
    clock: u64,
}

/// vello targets are pooled by size rounded up to this many pixels.
const TARGET_STEP: u32 = 64;
/// RGBA8 targets kept for reuse (bytes).
const TARGET_BUDGET: usize = 256 << 20;

#[derive(Default)]
struct Targets {
    free: HashMap<(u32, u32), Vec<wgpu::Texture>>,
    bytes: usize,
}

/// The vello renderers, the processed-layer cache and the pool of vello targets.
#[derive(Default)]
pub(crate) struct State {
    /// Whether vello renders on this device (the first renderer is built to find out).
    usable: OnceLock<bool>,
    /// Renderers nobody is using. A render takes one and builds another when they are all in use,
    /// so frames rendering on several threads never queue for one renderer.
    renderers: Mutex<Vec<vello::Renderer>>,
    cache: Mutex<Cache>,
    targets: Mutex<Targets>,
    /// How often each full content key missed the cache (see [`State::recurring`]).
    misses: Mutex<HashMap<u64, u32>>,
    /// Shape layer sources rasterised here (diagnostics, tests).
    pub(crate) drawn: AtomicU64,
    /// Pixels of those sources (tests).
    pub(crate) drawn_px: AtomicU64,
    /// vello targets created, not taken from the pool (tests).
    pub(crate) targets_made: AtomicU64,
    /// vello renders (tests).
    pub(crate) renders: AtomicU64,
}

impl State {
    pub(crate) fn get(&self, key: u64) -> Option<Content> {
        let mut c = self.cache.lock().ok()?;
        c.clock += 1;
        let now = c.clock;
        let (content, used) = c.map.get_mut(&key)?;
        *used = now;
        Some(content.clone())
    }

    pub(crate) fn insert(&self, key: u64, content: Content) {
        let Ok(mut c) = self.cache.lock() else { return };
        let bytes = content.0.width as usize * content.0.height as usize * 16;
        if bytes > CACHE_BUDGET / 4 {
            return;
        }
        c.clock += 1;
        let now = c.clock;
        if let Some(((old, _, _), _)) = c.map.insert(key, (content, now)) {
            c.bytes -= old.width as usize * old.height as usize * 16;
        }
        c.bytes += bytes;
        while c.bytes > CACHE_BUDGET {
            let Some(k) = c.map.iter().min_by_key(|(_, (_, used))| *used).map(|(k, _)| *k) else { break };
            if let Some(((old, _, _), _)) = c.map.remove(&k) {
                c.bytes -= old.width as usize * old.height as usize * 16;
            }
        }
    }

    /// Count a miss of a layer's full content key: true from the third miss on, when the same
    /// content keeps being wanted (it is static; only what shows of it changed), so drawing
    /// all of it once is cheaper than drawing each crop.
    pub(crate) fn recurring(&self, key: u64) -> bool {
        let Ok(mut m) = self.misses.lock() else { return false };
        if m.len() > 1 << 16 {
            m.clear();
        }
        let n = m.entry(key).or_insert(0);
        *n += 1;
        *n >= 3
    }

    pub(crate) fn clear(&self) {
        if let Ok(mut c) = self.cache.lock() {
            c.map.clear();
            c.bytes = 0;
        }
        if let Ok(mut t) = self.targets.lock() {
            t.free.clear();
            t.bytes = 0;
        }
    }

    /// An RGBA8 storage texture at least `w × h` (pooled; contents undefined).
    fn target(&self, device: &wgpu::Device, w: u32, h: u32) -> wgpu::Texture {
        let size = (w.div_ceil(TARGET_STEP) * TARGET_STEP, h.div_ceil(TARGET_STEP) * TARGET_STEP);
        if let Ok(mut t) = self.targets.lock()
            && let Some(tex) = t.free.get_mut(&size).and_then(Vec::pop)
        {
            t.bytes -= size.0 as usize * size.1 as usize * 4;
            return tex;
        }
        self.targets_made.fetch_add(1, Ordering::Relaxed);
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vello shape layer"),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    /// Return targets whose last reader has been submitted (or discarded): work vello submits
    /// later runs after it on the queue.
    pub(crate) fn recycle(&self, textures: Vec<wgpu::Texture>) {
        let Ok(mut t) = self.targets.lock() else { return };
        for tex in textures {
            let bytes = tex.width() as usize * tex.height() as usize * 4;
            if t.bytes + bytes > TARGET_BUDGET {
                continue;
            }
            t.bytes += bytes;
            t.free.entry((tex.width(), tex.height())).or_default().push(tex);
        }
    }

    fn build(device: &wgpu::Device) -> Option<vello::Renderer> {
        let opts = vello::RendererOptions {
            use_cpu: false,
            antialiasing_support: vello::AaSupport::area_only(),
            num_init_threads: std::num::NonZeroUsize::new(1),
            pipeline_cache: None,
        };
        match vello::Renderer::new(device, opts) {
            Ok(r) => Some(r),
            Err(e) => {
                log::info!("gpu: shape layers rasterise on the CPU (vello: {e})");
                None
            }
        }
    }

    fn idle(&self) -> std::sync::MutexGuard<'_, Vec<vello::Renderer>> {
        self.renderers.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Whether shape layers can rasterise on this device.
    fn usable(&self, device: &wgpu::Device) -> bool {
        *self.usable.get_or_init(|| {
            let r = Self::build(device);
            let ok = r.is_some();
            self.idle().extend(r);
            ok
        })
    }

    /// One vello render (`None` without a renderer, or when the render failed). Renders on
    /// different threads each use a renderer of their own.
    fn render(&self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &vello::Scene, view: &wgpu::TextureView, params: &vello::RenderParams) -> Option<()> {
        if !self.usable(device) {
            return None;
        }
        let idle = self.idle().pop();
        let mut r = idle.or_else(|| Self::build(device))?;
        self.renders.fetch_add(1, Ordering::Relaxed);
        let done = r.render_to_texture(device, queue, scene, view, params);
        self.idle().push(r);
        done.map_err(|err| log::warn!("gpu: vello: {err}")).ok()
    }
}

fn affine(m: &Mat3) -> Affine {
    let a = &m.0;
    Affine::new([a[0][0], a[1][0], a[0][1], a[1][1], a[0][2], a[1][2]])
}

fn color(c: [f32; 4]) -> Color {
    Color::new(c)
}

/// The brush of a paint at `opacity`, and its transform (gradient space → layer space).
fn brush(paint: &ShapePaint, opacity: f32) -> (peniko::Brush, Option<Affine>) {
    match paint {
        ShapePaint::Solid(c) => (peniko::Brush::Solid(color([c[0], c[1], c[2], c[3] * opacity])), None),
        ShapePaint::Gradient { g, radial, start, end, focal, to_layer } => {
            // The ramp is piecewise linear in straight colour between the colour and opacity
            // stops: sample it there and evenly in between.
            let mut ts: Vec<f64> = (0..=RAMP_STOPS).map(|i| i as f64 / RAMP_STOPS as f64).collect();
            ts.extend(g.colors.iter().map(|s| s.0).chain(g.opacities.iter().map(|s| s.0)).filter(|t| (0.0..=1.0).contains(t)));
            ts.sort_by(f64::total_cmp);
            ts.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
            let stops: Vec<(f32, Color)> = ts
                .iter()
                .map(|&t| {
                    let c = g.sample(t);
                    (t as f32, color([c[0], c[1], c[2], c[3] * opacity]))
                })
                .collect();
            let (s, e) = (Point::new(start[0], start[1]), Point::new(end[0], end[1]));
            let mut gr = if *radial {
                let r = s.distance(e) as f32;
                match focal {
                    // Ray from the focal point to the circle: a two-point gradient from a point.
                    Some(f) => peniko::Gradient::new_two_point_radial(Point::new(f[0], f[1]), 0.0, s, r),
                    None => peniko::Gradient::new_radial(s, r),
                }
            } else {
                peniko::Gradient::new_linear(s, e)
            };
            gr.interpolation_alpha_space = peniko::InterpolationAlphaSpace::Unpremultiplied;
            (peniko::Brush::Gradient(gr.with_stops(stops.as_slice())), Some(affine(to_layer)))
        }
    }
}

/// Offer processed content `e` drew to other frames under `key`. It goes into the cache when
/// `e` submits the work that draws it ([`Enc::submit`]): another encoder that took it sooner could
/// be submitted first and read the texture before it was drawn. `e` finds it at once
/// ([`lookup`]).
pub(crate) fn publish(e: &mut Enc, key: u64, content: Content) {
    // Other frames' encoders will read it.
    content.0.share();
    e.unpublished.push((key, content));
}

/// Processed content under `key`: drawn by `e` itself, or cached (drawn and submitted).
pub(crate) fn lookup(e: &Enc, key: u64) -> Option<Content> {
    if let Some((_, c)) = e.unpublished.iter().find(|(k, _)| *k == key) {
        return Some(c.clone());
    }
    e.g.vector.get(key)
}

/// Shape layers drawn by one encoder share vello renders: each goes into its own region of an
/// atlas this many pixels square, and the atlas renders once, just before the encoder submits
/// the work that reads it ([`flush`]). Each vello render creates and destroys textures, which
/// takes a device-wide lock every other thread's submissions wait on; one render per layer
/// made that lock the bottleneck of a shape-heavy frame.
const ATLAS: u32 = 4096;
/// Regions start on vello's 16-pixel tile grid, so a layer draws the same pixels anywhere.
const TILE: u32 = 16;
/// An atlas render keeps its estimated line segments and tiles under these: vello's buffers
/// hold 2^21 of each, and an overflow would drop geometry silently.
const ATLAS_SEGMENTS: usize = 1 << 20;
const ATLAS_TILES: usize = 1 << 20;

/// The atlas an encoder is filling ([`ATLAS`]).
#[derive(Default)]
pub(crate) struct Atlas(Option<Open>);

struct Open {
    scene: vello::Scene,
    target: wgpu::Texture,
    /// Shelf packing: the next region's x, the current row's top and height, and the extent used.
    x: u32,
    row_y: u32,
    row_h: u32,
    used: (u32, u32),
    segments: usize,
    tiles: usize,
}

impl Atlas {
    /// Drop an atlas that was never rendered (its encoder was discarded unsubmitted).
    pub(crate) fn discard(&mut self) -> Option<wgpu::Texture> {
        self.0.take().map(|o| o.target)
    }
}

/// A shape scene's draws as vello paths in buffer pixels, with estimates of the line
/// segments and tiles vello will allocate for them.
struct Paths {
    fills: Vec<(Fill, peniko::Brush, Option<Affine>, BezPath)>,
    segments: usize,
    tiles: usize,
}

/// A line's points on a 1/256-pixel grid: exact in f32 at any atlas position (below 2^15
/// pixels), so a layer's coverage does not depend on where its region is.
fn snap(el: PathEl) -> PathEl {
    // `round` is a call into libm on targets without SSE4.1, and this runs for every point of
    // every shape of every frame. Coordinates are far inside the range an i64 holds, where
    // rounding half away from zero is truncating and stepping one further when a half or more
    // was cut off; anything else (and NaN) takes the call.
    let round = |v: f64| {
        if v.is_nan() || v.abs() >= 1e15 {
            return v.round();
        }
        let t = v as i64 as f64;
        let cut = v - t;
        // No branch: which half a coordinate falls in is as good as random.
        t + (i8::from(cut >= 0.5) - i8::from(cut <= -0.5)) as f64
    };
    let q = |p: Point| Point::new(round(p.x * 256.0) / 256.0, round(p.y * 256.0) / 256.0);
    match el {
        PathEl::MoveTo(p) => PathEl::MoveTo(q(p)),
        PathEl::LineTo(p) => PathEl::LineTo(q(p)),
        other => other,
    }
}

fn paths(sc: &ShapeScene) -> Paths {
    let (mut fills, mut segments, mut tiles) = (Vec::with_capacity(sc.draws.len()), 0usize, 0usize);
    for d in &sc.draws {
        // All of a draw's paths fill as one (their windings add), as on the CPU. Curves are
        // flattened here, in buffer pixels, finer than vello's own flattening: tiny-skia follows
        // curves to a small fraction of a pixel, and ink outlines are mostly curves.
        let to_px = affine(&d.to_px);
        // Room for the lines up front (a curve flattens to several): growing the path as it
        // filled was a tenth of the time spent here.
        let mut path = BezPath::with_capacity(d.paths.iter().map(|p| p.elements().len()).sum::<usize>() * 4);
        for p in &d.paths {
            if p.elements().iter().all(|el| matches!(el, PathEl::MoveTo(_) | PathEl::LineTo(_) | PathEl::ClosePath)) {
                path.extend(p.elements().iter().map(|el| snap(to_px * *el)));
            } else {
                vello::kurbo::flatten(p.elements().iter().map(|el| to_px * *el), FLATTEN_TOLERANCE, |el| path.push(snap(el)));
            }
        }
        // A line crosses about (|dx| + |dy|) / 16 tiles; a path allocates the tiles of its
        // bounding box within the buffer.
        let (mut last, mut lo, mut hi) = (Point::ZERO, Point::new(f64::MAX, f64::MAX), Point::new(f64::MIN, f64::MIN));
        for el in path.elements() {
            let p = match *el {
                PathEl::MoveTo(p) => p,
                PathEl::LineTo(p) => {
                    segments += 1 + (((p.x - last.x).abs() + (p.y - last.y).abs()) / TILE as f64) as usize;
                    p
                }
                _ => {
                    segments += 1;
                    continue;
                }
            };
            (last, lo, hi) = (p, Point::new(lo.x.min(p.x), lo.y.min(p.y)), Point::new(hi.x.max(p.x), hi.y.max(p.y)));
        }
        let (bw, bh) = (hi.x.min(sc.width as f64) - lo.x.max(0.0), hi.y.min(sc.height as f64) - lo.y.max(0.0));
        if bw > 0.0 && bh > 0.0 {
            tiles += ((bw / TILE as f64).ceil() as usize + 1) * ((bh / TILE as f64).ceil() as usize + 1);
        }
        let rule = match d.rule {
            FillRule::NonZero => Fill::NonZero,
            FillRule::EvenOdd => Fill::EvenOdd,
        };
        let (b, bt) = brush(&d.paint, d.opacity);
        fills.push((rule, b, bt.map(|t| to_px * t), path));
    }
    Paths { fills, segments, tiles }
}

/// Render the encoder's open atlas now (vello submits its own commands, ahead of the encoder's
/// that read the atlas). A failed render marks the encoder failed: its frame is then not used.
pub(crate) fn flush(e: &mut Enc) {
    let Some(o) = e.atlas.0.take() else { return };
    let view = o.target.create_view(&Default::default());
    let params = vello::RenderParams { base_color: Color::TRANSPARENT, width: o.used.0, height: o.used.1, antialiasing_method: vello::AaConfig::Area };
    let rendered = e.g.vector.render(&e.g.device, &e.g.queue, &o.scene, &view, &params).is_some();
    if !rendered {
        e.failed = true;
    }
    e.recycle_after_submit(o.target);
}

/// A `w` × `h` region (top-left corner) for a layer in the encoder's atlas, flushing a full
/// atlas and opening a new one as needed.
fn place(e: &mut Enc, w: u32, h: u32, segments: usize, tiles: usize) -> (u32, u32) {
    let fits = |o: &Open| -> Option<(u32, u32, bool)> {
        if o.segments + segments > ATLAS_SEGMENTS || o.tiles + tiles > ATLAS_TILES {
            return None;
        }
        if o.x + w <= ATLAS && o.row_y + h <= ATLAS {
            return Some((o.x, o.row_y, false));
        }
        let y = o.row_y + o.row_h;
        (w <= ATLAS && y + h <= ATLAS).then_some((0, y, true))
    };
    let spot = e.atlas.0.as_ref().and_then(fits);
    let (x, y, new_row) = match spot {
        Some(s) => s,
        None => {
            flush(e);
            let target = e.g.vector.target(&e.g.device, ATLAS, ATLAS);
            e.atlas.0 = Some(Open { scene: vello::Scene::new(), target, x: 0, row_y: 0, row_h: 0, used: (0, 0), segments: 0, tiles: 0 });
            (0, 0, false)
        }
    };
    if let Some(o) = e.atlas.0.as_mut() {
        if new_row {
            o.row_y = y;
            o.row_h = 0;
        }
        o.x = (x + w).next_multiple_of(TILE);
        o.row_h = o.row_h.max(h.next_multiple_of(TILE));
        o.used = (o.used.0.max(x + w), o.used.1.max(y + h));
        o.segments += segments;
        o.tiles += tiles;
    }
    (x, y)
}

/// Rasterise a shape scene into a premultiplied working image, clamped and quantised to
/// `levels` when set. `None` = not handled here (no vello on this device, too large). The
/// pixels are drawn when the encoder's atlas renders ([`flush`]).
pub(crate) fn rasterise(e: &mut Enc, sc: &ShapeScene, levels: Option<f32>) -> Option<GpuImage> {
    let (w, h) = (sc.width, sc.height);
    if !e.g.fits(w, h) {
        return None;
    }
    if sc.draws.is_empty() {
        return Some(e.image(w, h));
    }
    if !e.g.vector.usable(&e.g.device) {
        return None;
    }
    let ps = paths(sc);
    let (src, offset) = if w <= ATLAS && h <= ATLAS && ps.segments <= ATLAS_SEGMENTS && ps.tiles <= ATLAS_TILES {
        let (x, y) = place(e, w, h, ps.segments, ps.tiles);
        let o = e.atlas.0.as_mut()?;
        let at = Affine::translate((x as f64, y as f64));
        // Clipped to its region: a cropped layer's paths reach past it.
        o.scene.push_clip_layer(Fill::NonZero, at, &vello::kurbo::Rect::new(0.0, 0.0, w as f64, h as f64));
        for (rule, b, bt, path) in &ps.fills {
            o.scene.fill(*rule, at, b, *bt, path);
        }
        o.scene.pop_layer();
        (o.target.clone(), [x, y])
    } else {
        // Larger than an atlas: a render of its own, now.
        let mut scene = vello::Scene::new();
        for (rule, b, bt, path) in &ps.fills {
            scene.fill(*rule, Affine::IDENTITY, b, *bt, path);
        }
        let target = e.g.vector.target(&e.g.device, w, h);
        let view = target.create_view(&Default::default());
        let params = vello::RenderParams { base_color: Color::TRANSPARENT, width: w, height: h, antialiasing_method: vello::AaConfig::Area };
        e.g.vector.render(&e.g.device, &e.g.queue, &scene, &view, &params)?;
        e.recycle_after_submit(target.clone());
        (target, [0, 0])
    };
    let mut p = Params::default();
    let l = levels.unwrap_or(0.0);
    p.f[0] = [l, if l > 0.0 { 1.0 / l } else { 0.0 }, 0.0, 0.0];
    p.u[0] = [offset[0], offset[1], 0, 0];
    let mut out = e.scratch(w, h);
    e.pixels("premultiply8", &p, &GpuImage::new(src, w, h), None, &out, None);
    out.levels = levels;
    e.g.vector.drawn.fetch_add(1, Ordering::Relaxed);
    e.g.vector.drawn_px.fetch_add(w as u64 * h as u64, Ordering::Relaxed);
    Some(out)
}
