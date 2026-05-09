//! GPU voxel renderer (wgpu / Metal on macOS).
//!
//! Renders the world as instanced unit cubes in a single draw call.
//! No depth attachment is available inside an egui paint callback, so we
//! rely on back-face culling + a CPU back-to-front sort of the instance
//! buffer (painter's algorithm). For up to ~50k cells this comfortably
//! beats the CPU-painter cube renderer and stays sub-millisecond.

use eframe::egui_wgpu;
use eframe::wgpu;
use eframe::wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CubeVertex {
    pos: [f32; 3],
    normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    vp: [[f32; 4]; 4],
}

pub struct Resources {
    pipeline: wgpu::RenderPipeline,
    cube_buf: wgpu::Buffer,
    instance_buf: wgpu::Buffer,
    instance_capacity: usize,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl Resources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let cube = make_cube_vertices();
        let cube_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("morpheus.cube"),
            contents: bytemuck::cast_slice(&cube),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let initial_capacity = 4096;
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("morpheus.instances"),
            size: (initial_capacity * std::mem::size_of::<Instance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("morpheus.uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("morpheus.bgl"),
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("morpheus.bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("morpheus.shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("morpheus.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let cube_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CubeVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
            ],
        };
        let inst_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 12,
                    shader_location: 3,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("morpheus.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[cube_layout, inst_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            cube_buf,
            instance_buf,
            instance_capacity: initial_capacity,
            uniform_buf,
            bind_group,
            bind_group_layout,
        }
    }

    fn ensure_instance_capacity(&mut self, device: &wgpu::Device, n: usize) {
        if n <= self.instance_capacity {
            return;
        }
        let new_cap = (n * 2).max(self.instance_capacity * 2);
        self.instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("morpheus.instances"),
            size: (new_cap * std::mem::size_of::<Instance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.instance_capacity = new_cap;
        // bind_group doesn't reference instance_buf, so no rebuild needed.
        let _ = &self.bind_group_layout;
    }
}

pub struct VoxelCallback {
    pub instances: Vec<Instance>,
    pub vp: [f32; 16],
}

impl egui_wgpu::CallbackTrait for VoxelCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let res = match callback_resources.get_mut::<Resources>() {
            Some(r) => r,
            None => return Vec::new(),
        };
        let mut vp4: [[f32; 4]; 4] = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                vp4[i][j] = self.vp[i * 4 + j];
            }
        }
        queue.write_buffer(
            &res.uniform_buf,
            0,
            bytemuck::cast_slice(&[Uniforms { vp: vp4 }]),
        );
        if !self.instances.is_empty() {
            res.ensure_instance_capacity(device, self.instances.len());
            queue.write_buffer(
                &res.instance_buf,
                0,
                bytemuck::cast_slice(&self.instances),
            );
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: eframe::egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let res = match callback_resources.get::<Resources>() {
            Some(r) => r,
            None => return,
        };
        if self.instances.is_empty() {
            return;
        }
        render_pass.set_pipeline(&res.pipeline);
        render_pass.set_bind_group(0, &res.bind_group, &[]);
        render_pass.set_vertex_buffer(0, res.cube_buf.slice(..));
        render_pass.set_vertex_buffer(
            1,
            res.instance_buf
                .slice(..(self.instances.len() * std::mem::size_of::<Instance>()) as u64),
        );
        render_pass.draw(0..36, 0..self.instances.len() as u32);
    }
}

fn make_cube_vertices() -> [CubeVertex; 36] {
    // Unit cube centered at origin, edge length 1. CCW outward winding.
    // Each face: 2 triangles × 3 vertices.
    let h = 0.5_f32;
    let faces: [(([f32; 3], [f32; 3], [f32; 3], [f32; 3]), [f32; 3]); 6] = [
        // +X (normal +x)
        (
            (
                [h, -h, -h],
                [h,  h, -h],
                [h,  h,  h],
                [h, -h,  h],
            ),
            [1.0, 0.0, 0.0],
        ),
        // -X
        (
            (
                [-h, -h,  h],
                [-h,  h,  h],
                [-h,  h, -h],
                [-h, -h, -h],
            ),
            [-1.0, 0.0, 0.0],
        ),
        // +Y
        (
            (
                [ h,  h, -h],
                [-h,  h, -h],
                [-h,  h,  h],
                [ h,  h,  h],
            ),
            [0.0, 1.0, 0.0],
        ),
        // -Y
        (
            (
                [-h, -h, -h],
                [ h, -h, -h],
                [ h, -h,  h],
                [-h, -h,  h],
            ),
            [0.0, -1.0, 0.0],
        ),
        // +Z (top)
        (
            (
                [-h, -h, h],
                [ h, -h, h],
                [ h,  h, h],
                [-h,  h, h],
            ),
            [0.0, 0.0, 1.0],
        ),
        // -Z (bottom)
        (
            (
                [-h,  h, -h],
                [ h,  h, -h],
                [ h, -h, -h],
                [-h, -h, -h],
            ),
            [0.0, 0.0, -1.0],
        ),
    ];
    let mut out: [CubeVertex; 36] = [CubeVertex {
        pos: [0.0; 3],
        normal: [0.0; 3],
    }; 36];
    let mut k = 0;
    for ((a, b, c, d), n) in &faces {
        // tri 1: a, b, c
        out[k] = CubeVertex { pos: *a, normal: *n }; k += 1;
        out[k] = CubeVertex { pos: *b, normal: *n }; k += 1;
        out[k] = CubeVertex { pos: *c, normal: *n }; k += 1;
        // tri 2: a, c, d
        out[k] = CubeVertex { pos: *a, normal: *n }; k += 1;
        out[k] = CubeVertex { pos: *c, normal: *n }; k += 1;
        out[k] = CubeVertex { pos: *d, normal: *n }; k += 1;
    }
    out
}

const SHADER: &str = r#"
struct Uniforms {
    vp: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsIn {
    @location(0) cube_pos: vec3<f32>,
    @location(1) cube_normal: vec3<f32>,
    @location(2) inst_pos: vec3<f32>,
    @location(3) inst_color: vec4<f32>,
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    var out: VsOut;
    let world = in.cube_pos + in.inst_pos;
    out.pos = u.vp * vec4<f32>(world, 1.0);
    out.color = in.inst_color;
    out.normal = in.cube_normal;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let light = normalize(vec3<f32>(0.55, 0.65, 1.0));
    let n = normalize(in.normal);
    let diff = max(dot(n, light), 0.0);
    let shade = 0.40 + 0.60 * diff;
    return vec4<f32>(in.color.rgb * shade, in.color.a);
}
"#;
