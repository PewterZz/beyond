//! wgpu render pipeline for rectangles (the primitive for all block UI).

use wgpu::util::DeviceExt;

/// A single rectangle draw command.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RectInstance {
    /// [x, y, width, height] in screen pixels.
    pub rect: [f32; 4],
    /// RGBA fill color.
    pub color: [f32; 4],
    /// Corner radius in pixels (0 = sharp).
    pub corner_radius: f32,
    /// Border width in pixels (0 = no border).
    pub border_width: f32,
    /// RGBA border color.
    pub border_color: [f32; 4],
}

impl RectInstance {
    pub fn filled(x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) -> Self {
        Self {
            rect: [x, y, w, h],
            color,
            corner_radius: 0.0,
            border_width: 0.0,
            border_color: [0.0; 4],
        }
    }

    pub fn with_border(mut self, width: f32, color: [f32; 4]) -> Self {
        self.border_width = width;
        self.border_color = color;
        self
    }

    pub fn with_radius(mut self, radius: f32) -> Self {
        self.corner_radius = radius;
        self
    }
}

/// Global push constants for the rect pipeline (screen size for NDC conversion).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RectUniforms {
    /// Screen dimensions [width, height].
    pub screen_size: [f32; 2],
    pub _pad: [f32; 2],
}

pub struct RectPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
    pub instance_buffer: wgpu::Buffer,
    pub instance_capacity: u32,
}

const MAX_RECTS: u32 = 8192;

impl RectPipeline {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rect_shader"),
            source: wgpu::ShaderSource::Wgsl(RECT_SHADER.into()),
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rect_uniforms"),
            contents: bytemuck::bytes_of(&RectUniforms {
                screen_size: [800.0, 600.0],
                _pad: [0.0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rect_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rect_bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rect_instances"),
            size: (std::mem::size_of::<RectInstance>() as u64) * MAX_RECTS as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rect_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rect_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<RectInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x4,  // rect
                        1 => Float32x4,  // color
                        2 => Float32,    // corner_radius
                        3 => Float32,    // border_width
                        4 => Float32x4,  // border_color
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            uniform_buffer,
            uniform_bind_group,
            instance_buffer,
            instance_capacity: MAX_RECTS,
        }
    }

    pub fn update_screen_size(&self, queue: &wgpu::Queue, width: f32, height: f32) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&RectUniforms {
                screen_size: [width, height],
                _pad: [0.0; 2],
            }),
        );
    }

    pub fn upload_instances(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[RectInstance],
    ) {
        if instances.is_empty() {
            return;
        }
        let needed = instances.len() as u32;
        if needed > self.instance_capacity {
            // Grow to the next power of two so we don't reallocate every frame
            // when the scene's rect count climbs (e.g. long block streams,
            // TUI modes with many cell rects).
            let new_cap = needed.next_power_of_two().max(self.instance_capacity * 2);
            self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rect_instances"),
                size: (std::mem::size_of::<RectInstance>() as u64) * new_cap as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_cap;
        }
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));
    }

    pub fn draw<'rp>(&'rp self, pass: &mut wgpu::RenderPass<'rp>, count: u32) {
        if count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
        // 4 vertices per quad (triangle strip), 1 instance per rect
        pass.draw(0..4, 0..count);
    }
}

const RECT_SHADER: &str = r#"
struct Uniforms {
    screen_size: vec2<f32>,
    _pad: vec2<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct Instance {
    @location(0) rect: vec4<f32>,          // x, y, w, h in pixels
    @location(1) color: vec4<f32>,
    @location(2) corner_radius: f32,
    @location(3) border_width: f32,
    @location(4) border_color: vec4<f32>,
};

struct Vs {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) size: vec2<f32>,
    @location(3) corner_radius: f32,
    @location(4) border_width: f32,
    @location(5) border_color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, inst: Instance) -> Vs {
    // Triangle strip: 4 vertices of a quad
    let uv = vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u));
    let px = inst.rect.x + uv.x * inst.rect.z;
    let py = inst.rect.y + uv.y * inst.rect.w;
    // Convert pixel coords to NDC (Y flipped)
    let ndc = vec2<f32>(
        (px / u.screen_size.x) * 2.0 - 1.0,
        1.0 - (py / u.screen_size.y) * 2.0,
    );
    var v: Vs;
    v.pos = vec4<f32>(ndc, 0.0, 1.0);
    v.uv = uv;
    v.color = inst.color;
    v.size = inst.rect.zw;
    v.corner_radius = inst.corner_radius;
    v.border_width = inst.border_width;
    v.border_color = inst.border_color;
    return v;
}

@fragment
fn fs_main(v: Vs) -> @location(0) vec4<f32> {
    // Rounded corners via SDF
    let half = v.size * 0.5;
    let p = v.uv * v.size - half;
    let r = v.corner_radius;
    let q = abs(p) - half + vec2<f32>(r, r);
    let dist = length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - r;
    if dist > 1.0 { discard; }

    // Border
    if v.border_width > 0.0 && dist > -v.border_width {
        return v.border_color;
    }

    return v.color;
}
"#;
