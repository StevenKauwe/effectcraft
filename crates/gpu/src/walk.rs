//! The per-frame compositing walk on the GPU: the same steps as `Renderer::draw_comp` (layer
//! order, 3D runs, collapsed precomps, track mattes, Preserve Transparency, layer styles, motion
//! blur, bit-depth quantisation, colour conversions), with layer pixels taken from the CPU
//! renderer and its layer cache. Classic and Advanced 3D runs (environment skies included) and
//! adjustment layers run on the GPU; steps without a GPU implementation (anything a kernel
//! declines) read the canvas back, run on the CPU and upload the result.

use std::sync::Arc;

use effectcraft_color::BlendMode;
use effectcraft_effects::Buf;
use effectcraft_project::{ItemId, Layer, LayerSource, Quality};
use effectcraft_raster::Image;
use effectcraft_render::{EvalCtx, Placement, Renderer, styles};
use effectcraft_time::Tick;

use crate::context::{Enc, GpuImage};
use crate::ops;

/// Render a comp frame into a GPU image (the canvas in the output space, as
/// `Renderer::comp_frame_cpu` returns it). `None` when something cannot run here (the caller
/// renders on the CPU instead).
pub(crate) fn render<'g>(e: &mut Enc<'g>, r: &Renderer, comp_id: ItemId, t: Tick) -> Option<GpuImage> {
    // Region-of-interest frames are cropped on the CPU path.
    if r.opts.roi.is_some() {
        return None;
    }
    let ctx = r.eval_ctx(comp_id, t)?;
    let s = r.opts.scale;
    let w = ((ctx.comp.width as f64 * s).round() as u32).max(1);
    let h = ((ctx.comp.height as f64 * s).round() as u32).max(1);
    if r.depth() > 16 || !e.g.fits(w, h) {
        return None;
    }
    let mut canvas = e.zeros(w, h);
    draw_comp(e, r, &ctx, &mut canvas)?;
    let pipe = r.pipe();
    if let Some(c) = pipe.from_blend() {
        canvas = ops::convert(e, &canvas, &c);
    }
    if let Some(l) = pipe.levels {
        canvas = ops::quantize(e, &canvas, l);
    }
    if r.depth() == 0
        && let Some(c) = pipe.output()
    {
        canvas = ops::convert(e, &canvas, &c);
        if let Some(l) = pipe.levels {
            canvas = ops::quantize(e, &canvas, l);
        }
    }
    if r.depth() == 0
        && let Some((lin, mode, enc)) = pipe.output_hdr()
    {
        on_cpu(e, &mut canvas, |img| {
            effectcraft_render::color::output_hdr(img, lin, mode, enc);
            pipe.quantize(img);
        })?;
    }
    Some(canvas)
}

/// Run a CPU drawing step on the canvas (read back, draw, upload).
fn on_cpu(e: &mut Enc, canvas: &mut GpuImage, f: impl FnOnce(&mut Image)) -> Option<()> {
    let mut img = e.download(canvas)?;
    f(&mut img);
    *canvas = e.g.upload_image(&img)?;
    Some(())
}

fn draw_comp<'a>(e: &mut Enc, r: &Renderer<'a>, ctx: &EvalCtx<'a>, canvas: &mut GpuImage) -> Option<()> {
    let visible = r.visible_layers(ctx);
    let levels = r.pipe().levels;
    let mut i = 0;
    while i < visible.len() {
        if visible[i].is_3d() {
            let mut j = i;
            while j < visible.len() && visible[j].is_3d() {
                j += 1;
            }
            draw_3d(e, r, ctx, &visible[i..j], canvas)?;
            i = j;
        } else {
            match r.collapsed(ctx, visible[i]) {
                Some(item) => draw_collapsed(e, r, ctx, visible[i], item, canvas)?,
                None => draw_layer(e, r, ctx, visible[i], canvas)?,
            }
            i += 1;
        }
        // 8/16 bpc: the comp is an integer buffer after every layer.
        if let Some(l) = levels {
            *canvas = ops::quantize(e, canvas, l);
        }
    }
    Some(())
}

/// A run of consecutive 3D layers: Classic 3D on the GPU (`classic3d`), with 3D adjustment
/// layers splitting the run as on the CPU; Advanced 3D, wireframes and runs the kernel cannot
/// hold draw on the CPU.
fn draw_3d<'a>(e: &mut Enc, r: &Renderer<'a>, ctx: &EvalCtx<'a>, run: &[&'a Layer], canvas: &mut GpuImage) -> Option<()> {
    // Environment Light Background layers: the sky behind the rest of the run (drawn first, as
    // `three_d::compose::draw_run` does).
    if run.iter().any(|l| l.environment_background) {
        for l in run.iter().filter(|l| l.environment_background) {
            if let Some(sky) = r.sky_draw(ctx, l) {
                *canvas = crate::adv3d::sky(e, &sky, canvas)?;
            }
        }
        let rest: Vec<&'a Layer> = run.iter().copied().filter(|l| !l.environment_background).collect();
        return draw_3d(e, r, ctx, &rest, canvas);
    }
    if let Some(k) = run.iter().position(|l| l.switches.adjustment) {
        draw_3d(e, r, ctx, &run[..k], canvas)?;
        draw_layer(e, r, ctx, run[k], canvas)?;
        return draw_3d(e, r, ctx, &run[k + 1..], canvas);
    }
    if run.is_empty() {
        return Some(());
    }
    // Advanced 3D: rendered and composited on the GPU (`adv3d`).
    if let Some(prep) = r.prepare_adv_run(ctx, run, (canvas.width, canvas.height)) {
        match crate::adv3d::draw_run(e, &prep, canvas) {
            Some(img) => *canvas = img,
            None => on_cpu(e, canvas, |img| r.draw_3d_run(ctx, run, img))?,
        }
        return Some(());
    }
    // Advanced 3D with layers on the 2D compositing path (blend modes, track mattes, Preserve
    // Transparency).
    if let Some(split) = r.split_adv_run(ctx, run, (canvas.width, canvas.height)) {
        match draw_split(e, r, ctx, &split, canvas) {
            Some(img) => *canvas = img,
            None => on_cpu(e, canvas, |img| r.draw_3d_run(ctx, run, img))?,
        }
        return Some(());
    }
    match r.prepare_3d_run(ctx, run, (canvas.width, canvas.height)).and_then(|prep| crate::classic3d::draw_run(e, &prep, canvas)) {
        Some(img) => *canvas = img,
        None => on_cpu(e, canvas, |img| r.draw_3d_run(ctx, run, img))?,
    }
    Some(())
}

/// `three_d::adv::draw_run`'s split path on the GPU canvas: the plain layers as one scene over
/// the canvas, then each special layer (far to near) rendered alone, hidden where the main
/// scene is nearer, matted (a 3D matte through the camera, else the 2D placement), Preserve
/// Transparency and blended. `None` = render the run on the CPU.
fn draw_split(e: &mut Enc, r: &Renderer, ctx: &EvalCtx, split: &effectcraft_render::three_d::adv::SplitRun, canvas: &GpuImage) -> Option<GpuImage> {
    let (w, h) = (canvas.width, canvas.height);
    let main_prep = split.main();
    if main_prep.out != (w, h) {
        return None;
    }
    let encode = !main_prep.linear;
    let main = crate::adv3d::resolve_run(e, &main_prep)?;
    let mut cv = match &main {
        Some(m) => crate::adv3d::finish(e, m, encode, Some(canvas)),
        None => canvas.clone(),
    };
    for i in 0..split.len() {
        let Some(iso) = crate::adv3d::resolve_run(e, &split.special(i))? else { continue };
        let iso = match &main {
            Some(m) => crate::adv3d::occluded(e, &iso, m, encode),
            None => crate::adv3d::finish(e, &iso, encode, None),
        };
        let matte3d = split.with_matte3d(i, |p| match p {
            None => Some(e.zeros(w, h)),
            Some(p) => Some(match crate::adv3d::resolve_run(e, p)? {
                Some(m) => crate::adv3d::finish(e, &m, encode, None),
                None => e.zeros(w, h),
            }),
        });
        let matte3d = match matte3d {
            Some(m) => Some(m?),
            None => None,
        };
        composite_iso_with(e, r, ctx, split.layer(i), iso, &mut cv, 1.0, matte3d, None)?;
    }
    Some(cv)
}

fn draw_collapsed<'a>(e: &mut Enc, r: &Renderer<'a>, ctx: &EvalCtx<'a>, layer: &Layer, item: ItemId, canvas: &mut GpuImage) -> Option<()> {
    let opacity = r.layer_opacity(ctx, layer);
    if opacity <= 0.0 {
        return Some(());
    }
    let matte = r.track_matte(ctx, layer).is_some();
    if layer.blend_mode == BlendMode::Normal && !matte && !layer.preserve_transparency {
        if let Some((sub, nctx)) = r.collapse_into(ctx, layer, item, opacity) {
            draw_comp(e, &sub, &nctx, canvas)?;
        }
        return Some(());
    }
    let Some((sub, nctx)) = r.collapse_into(ctx, layer, item, 1.0) else { return Some(()) };
    let mut iso = e.zeros(canvas.width, canvas.height);
    draw_comp(e, &sub, &nctx, &mut iso)?;
    composite_iso(e, r, ctx, layer, iso, canvas, opacity)
}

fn draw_layer(e: &mut Enc, r: &Renderer, ctx: &EvalCtx, layer: &Layer, canvas: &mut GpuImage) -> Option<()> {
    if layer.switches.adjustment {
        if draw_adjustment(e, r, ctx, layer, canvas).is_some() {
            return Some(());
        }
        return on_cpu(e, canvas, |img| r.draw_layer_cpu(ctx, layer, img));
    }
    if r.quality(layer) == Quality::Wireframe {
        // The layer's bounds as a one-pixel outline (the CPU's pixels).
        *canvas = crate::adv3d::wireframe(e, canvas, &r.wireframe_pixels(ctx, layer, (canvas.width, canvas.height)));
        return Some(());
    }
    let opacity = r.layer_opacity(ctx, layer);
    if opacity <= 0.0 {
        return Some(());
    }
    if draw_shape_layer(e, r, ctx, layer, canvas, opacity).is_some() {
        return Some(());
    }
    let Some(st) = r.styled_layer(ctx, layer) else { return Some(()) };
    if st.plain {
        return composite_layer(e, r, ctx, layer, &st.body, canvas, opacity);
    }
    // Layer styles: exterior passes with their own modes, then the body; the layer's opacity
    // fades the whole stack; knockout and channel switches as on the CPU.
    let bl = styles::blending(ctx, layer);
    let (w, h) = (canvas.width, canvas.height);
    if r.track_matte(ctx, layer).is_some() || layer.preserve_transparency {
        let mut iso = e.zeros(w, h);
        for (b, m) in &st.passes {
            iso = place(e, r, ctx, layer, b, &iso, *m, 1.0, None)?;
        }
        iso = place(e, r, ctx, layer, &st.body, &iso, BlendMode::Normal, 1.0, None)?;
        return composite_iso(e, r, ctx, layer, iso, canvas, opacity);
    }
    let mut tmp = canvas.clone();
    if bl.knockout > 0 {
        let k = place(e, r, ctx, layer, &st.content, &e.zeros(w, h), BlendMode::Normal, 1.0, None)?;
        tmp = ops::knockout(e, &tmp, &k);
    }
    for (b, m) in &st.passes {
        tmp = place(e, r, ctx, layer, b, &tmp, *m, 1.0, None)?;
    }
    tmp = place(e, r, ctx, layer, &st.body, &tmp, layer.blend_mode, 1.0, None)?;
    *canvas = ops::channel_mix(e, canvas, &tmp, bl.channels, opacity);
    Some(())
}

/// A plain shape layer (no layer styles, track matte or Preserve Transparency) without leaving
/// the GPU: its source rasterised here (`vector`), its effects run on the device, the result
/// placed as `place` places a layer buffer. `None` = not handled (nothing was drawn); the
/// layer then takes the CPU layer-buffer path.
fn draw_shape_layer(e: &mut Enc, r: &Renderer, ctx: &EvalCtx, layer: &Layer, canvas: &mut GpuImage, opacity: f32) -> Option<()> {
    if !matches!(layer.source, LayerSource::Shape) || styles::active(ctx, layer) || r.track_matte(ctx, layer).is_some() || layer.preserve_transparency {
        return None;
    }
    let (mut img, offset, scale) = shape_content(e, r, ctx, layer, (canvas.width, canvas.height))?;
    if let Some(c) = r.pipe().to_blend() {
        img = ops::convert(e, &img, &c);
    }
    let pl = r.placement(ctx, layer, &shell(img.width, img.height, offset, scale));
    *canvas = place_image(e, &img, &pl, canvas, layer.blend_mode, opacity, r.pipe().levels);
    Some(())
}

/// A layer buffer without pixels: placement reads only its size, offset and scale.
fn shell(width: u32, height: u32, offset: [f64; 2], scale: f64) -> Buf {
    Buf { img: Image { width, height, data: Vec::new() }, offset, scale }
}

/// A shape layer's processed pixels on the GPU (source → effects), from the vector cache when
/// its content key matches.
///
/// Shape layers are drawn at their content's full extent, often several times the frame (a
/// world-space floor, a sky behind a camera move), and most layers change every drawing (the
/// boil). So a layer that misses the cache is drawn only where the `view` (output pixels) can
/// see it, widened by its effects' reach: the same buffer, cropped. A layer whose full key
/// keeps coming back while its crop changes (static content under a moving camera) is drawn
/// whole once and reused, unless the whole is much larger than what shows.
fn shape_content(e: &mut Enc, r: &Renderer, ctx: &EvalCtx, layer: &Layer, view: (u32, u32)) -> Option<crate::vector::Content> {
    let full_key = r.content_key_of(ctx, layer);
    if let Some(hit) = full_key.and_then(|k| e.g.vector.get(k)) {
        return Some(hit);
    }
    let mut scene = r.shape_scene(ctx, layer)?;
    let mut key = full_key;
    if let Some(crop) = visible_crop(r, ctx, layer, &scene, view) {
        let crop_key = full_key.map(|k| effectcraft_render::cache::derive(k, crop_bits(crop)));
        if let Some(hit) = crop_key.and_then(|k| e.g.vector.get(k)) {
            return Some(hit);
        }
        // Drawing all of a static layer pays off when it is not much larger than what shows.
        let whole = (scene.width as u64 * scene.height as u64) <= 4 * crop[2] as u64 * crop[3] as u64;
        if !(whole && full_key.is_some_and(|k| e.g.vector.recurring(k))) {
            crop_scene(&mut scene, crop);
            key = crop_key;
        }
    }
    let levels = r.pipe().levels;
    let img = crate::vector::rasterise(e, &scene, levels)?;
    let mut fx = crate::effects::GpuFx::new(e, img, scene.offset, scene.scale, levels);
    r.run_effects_on(ctx, layer, false, &mut fx);
    let out = fx.finish()?;
    if let Some(k) = key {
        e.g.vector.insert(k, out.clone());
    }
    Some(out)
}

/// The part of a shape scene's buffer (x, y, width, height in buffer pixels) that can show in
/// a `view`-sized output once placed, widened by the effects' reach and the resampling
/// footprint. `None` when the whole buffer is needed or about as cheap (most of it shows,
/// motion blur, an effect of unknown reach).
fn visible_crop(r: &Renderer, ctx: &EvalCtx, layer: &Layer, sc: &effectcraft_render::shapes::ShapeScene, view: (u32, u32)) -> Option<[u32; 4]> {
    let reach = r.effects_reach(ctx, layer)? * sc.scale + 3.0;
    let pl = r.placement(ctx, layer, &shell(sc.width, sc.height, sc.offset, sc.scale));
    let [m] = pl.matrices.as_slice() else { return None };
    let inv = m.inverse()?;
    let (vw, vh) = (view.0 as f64, view.1 as f64);
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (x, y) in [(0.0, 0.0), (vw, 0.0), (0.0, vh), (vw, vh)] {
        let p = inv.apply(effectcraft_geom::vec2(x, y));
        if !p.x.is_finite() || !p.y.is_finite() {
            return None;
        }
        (x0, y0, x1, y1) = (x0.min(p.x), y0.min(p.y), x1.max(p.x), y1.max(p.y));
    }
    let (w, h) = (sc.width as f64, sc.height as f64);
    // The crop starts on vello's 16-pixel tile grid, so every tile sees the same coordinates
    // as in the whole buffer and draws the same pixels (off the grid, f32 rounding flips a
    // level here and there).
    let cx0 = ((x0 - reach) / 16.0).floor().clamp(0.0, w) * 16.0;
    let cy0 = ((y0 - reach) / 16.0).floor().clamp(0.0, h) * 16.0;
    let cx1 = (x1 + reach).ceil().clamp(0.0, w);
    let cy1 = (y1 + reach).ceil().clamp(0.0, h);
    let (cw, ch) = ((cx1 - cx0).max(1.0), (cy1 - cy0).max(1.0));
    if cw * ch > 0.8 * w * h {
        return None;
    }
    Some([cx0 as u32, cy0 as u32, cw as u32, ch as u32])
}

fn crop_bits(c: [u32; 4]) -> u64 {
    (c[0] as u64) | (c[1] as u64) << 16 | (c[2] as u64) << 32 | (c[3] as u64) << 48
}

/// Restrict a scene to a sub-rectangle of its buffer: the same pixels there, with the buffer's
/// origin moved to the rectangle's corner.
fn crop_scene(sc: &mut effectcraft_render::shapes::ShapeScene, c: [u32; 4]) {
    let shift = effectcraft_geom::Mat3::translate(effectcraft_geom::vec2(-(c[0] as f64), -(c[1] as f64)));
    sc.offset = [sc.offset[0] - c[0] as f64, sc.offset[1] - c[1] as f64];
    sc.width = c[2];
    sc.height = c[3];
    for d in &mut sc.draws {
        d.to_px = shift * d.to_px;
    }
}

/// Adjustment layer (`Renderer::draw_adjustment`): the effect stack runs on the comp below,
/// resident on the GPU (non-GPU effects read back, run on the CPU and upload), and the result
/// replaces the canvas inside the layer's footprint × opacity. `None` = draw it on the CPU.
fn draw_adjustment(e: &mut Enc, r: &Renderer, ctx: &EvalCtx, layer: &Layer, canvas: &mut GpuImage) -> Option<()> {
    let opacity = r.layer_opacity(ctx, layer);
    let Some(foot) = r.adjustment_footprint(ctx, layer) else { return Some(()) };
    let (w, h) = (canvas.width, canvas.height);
    let matte = place(e, r, ctx, layer, &foot, &e.zeros(w, h), BlendMode::Normal, 1.0, None)?;
    let pipe = r.pipe();
    // Effects run in the working space, not the linear blending space.
    let below = match pipe.from_blend() {
        Some(c) => ops::convert(e, canvas, &c),
        None => canvas.clone(),
    };
    let mut fx = crate::effects::GpuFx::new(e, below, [0.0; 2], r.opts.scale, pipe.levels);
    r.run_effects_on(ctx, layer, true, &mut fx);
    let (mut adjusted, _, _) = fx.finish()?;
    if adjusted.width != w || adjusted.height != h {
        return Some(());
    }
    if let Some(c) = pipe.to_blend() {
        adjusted = ops::convert(e, &adjusted, &c);
    }
    *canvas = ops::adjust_mix(e, canvas, &adjusted, &matte, opacity);
    Some(())
}

fn composite_layer(e: &mut Enc, r: &Renderer, ctx: &EvalCtx, layer: &Layer, buf: &Arc<Buf>, canvas: &mut GpuImage, opacity: f32) -> Option<()> {
    let matte = r.track_matte(ctx, layer).is_some();
    if !matte && !layer.preserve_transparency {
        // The comp is quantised after the layer (8/16 bpc) in the same pass.
        *canvas = place(e, r, ctx, layer, buf, canvas, layer.blend_mode, opacity, r.pipe().levels)?;
        return Some(());
    }
    // Render in isolation, then matte / preserve transparency, then blend.
    let iso = place(e, r, ctx, layer, buf, &e.zeros(canvas.width, canvas.height), BlendMode::Normal, 1.0, None)?;
    composite_iso(e, r, ctx, layer, iso, canvas, opacity)
}

/// Track matte / Preserve Transparency on an isolated layer render, then blend.
fn composite_iso(e: &mut Enc, r: &Renderer, ctx: &EvalCtx, layer: &Layer, iso: GpuImage, canvas: &mut GpuImage, opacity: f32) -> Option<()> {
    composite_iso_with(e, r, ctx, layer, iso, canvas, opacity, None, r.pipe().levels)
}

/// `Renderer::composite_iso_with`: the matte from `matte_img` when given (a 3D matte drawn
/// through the camera), else the matte layer placed in 2D; the blend quantised to `levels`.
#[allow(clippy::too_many_arguments)]
fn composite_iso_with(
    e: &mut Enc,
    r: &Renderer,
    ctx: &EvalCtx,
    layer: &Layer,
    mut iso: GpuImage,
    canvas: &mut GpuImage,
    opacity: f32,
    matte_img: Option<GpuImage>,
    levels: Option<f32>,
) -> Option<()> {
    if let Some((m, kind)) = r.track_matte(ctx, layer) {
        let mimg = match matte_img.filter(|i| (i.width, i.height) == (canvas.width, canvas.height)) {
            Some(i) => i,
            None => {
                let mut mimg = e.zeros(canvas.width, canvas.height);
                if m.is_active_at(ctx.time)
                    && let Some(mb) = r.layer_buf(ctx, m)
                {
                    let mo = ctx.opacity(m) as f32;
                    mimg = place(e, r, ctx, m, &mb, &mimg, BlendMode::Normal, mo, None)?;
                }
                mimg
            }
        };
        iso = ops::matte(e, &iso, &mimg, kind);
    }
    if layer.preserve_transparency {
        iso = ops::preserve(e, &iso, canvas);
    }
    *canvas = ops::blend_full(e, canvas, &iso, layer.blend_mode, opacity, layer.id.0 as u32, levels);
    Some(())
}

/// Draw a processed layer buffer onto `target` (transform, motion blur sub-samples), quantising
/// the result to `levels` when set.
#[allow(clippy::too_many_arguments)]
fn place(
    e: &mut Enc,
    r: &Renderer,
    ctx: &EvalCtx,
    layer: &Layer,
    buf: &Arc<Buf>,
    target: &GpuImage,
    mode: BlendMode,
    opacity: f32,
    levels: Option<f32>,
) -> Option<GpuImage> {
    if buf.img.is_empty() || opacity <= 0.0 {
        return Some(target.clone());
    }
    let src = e.g.upload_buf(buf)?;
    let pl = r.placement(ctx, layer, buf);
    Some(place_image(e, &src, &pl, target, mode, opacity, levels))
}

/// Draw a layer image already on the GPU onto `target` with its placement.
fn place_image(e: &mut Enc, src: &GpuImage, pl: &Placement, target: &GpuImage, mode: BlendMode, opacity: f32, levels: Option<f32>) -> GpuImage {
    if let [m] = pl.matrices.as_slice() {
        return ops::warp_q(e, target, src, m, pl.sampling, mode, opacity, pl.seed, None, levels);
    }
    let mut acc = e.zeros(target.width, target.height);
    let k = 1.0 / pl.matrices.len() as f32;
    for m in &pl.matrices {
        acc = ops::warp(e, &acc, src, m, pl.sampling, BlendMode::Normal, 1.0, 0, Some(k));
    }
    ops::blend_full(e, target, &acc, mode, opacity, pl.seed, levels)
}
