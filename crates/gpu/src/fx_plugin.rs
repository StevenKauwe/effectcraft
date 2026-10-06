//! Plug-in effects with a GPU kernel ([`effectcraft_effects::plugin::GpuKernel`]): the
//! plug-in's WGSL passes behind [`effectcraft_effects::plugin::GPU_KERNEL_PRELUDE`], compiled
//! once per device on first use, run on the GPU-resident buffer. A kernel that does not compile
//! on this device is logged once and the effect renders on the CPU (`apply` returns `None`), as
//! does every plug-in kernel on the web build, where compile errors cannot be waited for.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use effectcraft_effects::EffectCtx;
use effectcraft_effects::plugin::{EffectPlugin, GpuKernel, PluginParams};
use wgpu::util::DeviceExt;

use crate::context::{Enc, GpuContext};
use crate::effects::GBuf;
#[cfg(not(target_arch = "wasm32"))]
use native::compile;

/// A registered plug-in effect with a GPU kernel.
pub(crate) fn has_kernel(id: &str) -> bool {
    !id.starts_with("ec.") && effectcraft_effects::plugin::plugin(id).is_some_and(|p| p.gpu_kernel().is_some())
}

/// One plug-in's kernel compiled for a device.
pub(crate) struct Compiled {
    bgl: wgpu::BindGroupLayout,
    passes: Vec<(String, wgpu::ComputePipeline)>,
    table: wgpu::Buffer,
    table_len: u32,
}

/// Compiled kernels by plug-in id; `None` = the kernel failed on this device (CPU fallback).
#[derive(Default)]
pub(crate) struct Kernels(Mutex<HashMap<String, Option<Arc<Compiled>>>>);

impl Kernels {
    fn get(&self, g: &GpuContext, id: &str, k: &GpuKernel) -> Option<Arc<Compiled>> {
        // Held while compiling, so concurrent frames compile a kernel once.
        let mut m = self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(c) = m.get(id) {
            return c.clone();
        }
        let c = match compile(g, k) {
            Ok(c) => Some(Arc::new(c)),
            Err(e) => {
                log::warn!("gpu: plug-in `{id}`: its GPU kernel is unusable on {} ({e}); it renders on the CPU", g.name);
                None
            }
        };
        m.insert(id.to_string(), c.clone());
        c
    }
}

#[cfg(target_arch = "wasm32")]
fn compile(_g: &GpuContext, _k: &GpuKernel) -> Result<Compiled, String> {
    Err("plug-in kernels run natively only (compile errors cannot be waited for on the web)".into())
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use effectcraft_effects::plugin::{GPU_KERNEL_PRELUDE, GpuKernel};
    use wgpu::util::DeviceExt;

    use super::Compiled;
    use crate::context::{FORMAT, GpuContext, storage_tex_entry, tex_entry};

    fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }
    }

    /// Bytes of `v`, at least 16 (storage bindings may not be empty).
    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        let mut b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        b.resize(b.len().max(16), 0);
        b
    }

    pub(super) fn compile(g: &GpuContext, k: &GpuKernel) -> Result<Compiled, String> {
        let d = &g.device;
        let table: &[f32] = k.table.as_deref().unwrap_or(&[0.0]);
        let table_bytes = (table.len() as u64).saturating_mul(4);
        let limit = d.limits().max_storage_buffer_binding_size;
        if table_bytes > limit {
            return Err(format!("table of {table_bytes} bytes; the device binds at most {limit}"));
        }
        let table_len = u32::try_from(table.len()).map_err(|_| "table too long".to_string())?;
        // Compile errors are captured here instead of reaching the device's uncaptured-error handler.
        let validation = d.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = d.push_error_scope(wgpu::ErrorFilter::Internal);
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("effectcraft plug-in kernel"),
            source: wgpu::ShaderSource::Wgsl(format!("{GPU_KERNEL_PRELUDE}\n{}", k.wgsl).into()),
        });
        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("effectcraft plug-in kernel"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                tex_entry(1),
                tex_entry(2),
                storage_tex_entry(3, FORMAT),
                storage_entry(4),
                storage_entry(5),
            ],
        });
        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("effectcraft plug-in kernel"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let passes = k
            .passes
            .iter()
            .map(|entry| {
                let p = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                });
                (entry.clone(), p)
            })
            .collect();
        let table = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("plug-in table"),
            contents: &f32_bytes(table),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let errors = [pollster::block_on(internal.pop()), pollster::block_on(validation.pop())];
        if let Some(e) = errors.into_iter().flatten().next() {
            return Err(e.to_string());
        }
        Ok(Compiled { bgl, passes, table, table_len: if k.table.is_some() { table_len } else { 0 } })
    }
}

/// Run plug-in `id`'s kernel on `b`. `None` = render it on the CPU.
pub(crate) fn apply(e: &mut Enc, id: &str, ctx: &EffectCtx, b: GBuf) -> Option<GBuf> {
    let plugin: Arc<dyn EffectPlugin> = effectcraft_effects::plugin::plugin(id)?;
    let kernel = plugin.gpu_kernel()?;
    let k = e.g.plugin_kernels.get(e.g, id, kernel)?;
    let m = plugin.manifest();
    let flat = effectcraft_effects::plugin::flatten_at(m, ctx, b.offset, b.scale);
    let params = plugin.gpu_params(&PluginParams { manifest: m, flat: &flat }, ctx.time, b.scale);
    let param_count = u32::try_from(params.len()).ok()?;
    let pbuf = e.data(&params);
    let (w, h) = (b.img.width, b.img.height);
    let orig = b.img.texture.create_view(&Default::default());
    let mut cur = b.img.clone();
    for (i, (entry, pipe)) in k.passes.iter().enumerate() {
        let mut header = Vec::with_capacity(32);
        for u in [w, h, i as u32, param_count] {
            header.extend_from_slice(&u.to_le_bytes());
        }
        for f in [ctx.time as f32, b.scale as f32] {
            header.extend_from_slice(&f.to_le_bytes());
        }
        for u in [k.table_len, 0] {
            header.extend_from_slice(&u.to_le_bytes());
        }
        let ub = e.g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &header, usage: wgpu::BufferUsages::UNIFORM });
        let out = e.scratch(w, h);
        let sv = cur.texture.create_view(&Default::default());
        let ov = out.texture.create_view(&Default::default());
        let bg = e.g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &k.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&sv) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&orig) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&ov) },
                wgpu::BindGroupEntry { binding: 4, resource: pbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: k.table.as_entire_binding() },
            ],
        });
        e.dispatch_with(entry, pipe, &bg, (w.div_ceil(16), h.div_ceil(16)));
        cur = out;
    }
    Some(GBuf { img: cur, ..b })
}
