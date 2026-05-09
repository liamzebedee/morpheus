//! GPU simulator: per-cell tree-walking Lisp interpreter via wgpu compute.
//!
//! Architecture (see also: design notes in conversation):
//!   per tick we dispatch 6 kernels:
//!     1. clear rep_count (encoder.clear_buffer)
//!     2. build_occupancy   (zero bitmap, then OR each cell's bit)
//!     3. compute_field     (per cell: min Euclidean dist to any source per gradient)
//!     4. clear source_count
//!     5. eval              (per cell: tree-walk the AST, write state/sources/reps)
//!     6. apply_replications (atomic-CAS on occupancy bitmap, allocate cell slots)
//!
//! This first cut wires the full pipeline; the WGSL eval kernel implements
//! arithmetic / control-flow / state ops, and stubs the four substrate builtins
//! (emit-gradient, read-gradient, neighbor-exists, replicate-toward) as no-ops.
//! The other kernels are present-but-empty; the next iteration fills them in.

use crate::ast::{self, FlatProgram};
use eframe::wgpu;
use eframe::wgpu::util::DeviceExt;
use std::sync::Arc;

pub const MAX_CELLS: u32 = 200_000;
pub const MAX_SOURCES_PER_GRAD: u32 = 32_768;
pub const MAX_REPS: u32 = 100_000;
pub const NUM_DIRS: u32 = 6;

pub const WORLD_HALF_XY: i32 = 64;
pub const WORLD_Z_MIN: i32 = 0;
pub const WORLD_Z_MAX: i32 = 127;

// Bitmap dimensions: x in [-64,64] → 129 vals; we round x_off into 7 bits (128).
// For bit packing we map (x+64) | ((y+64) << 7) | (z << 14), so x range capped at
// [-64, 63]. WORLD_HALF_XY=64 means valid range is [-64, 64] inclusive in CPU sim
// but the bitmap drops x=64 / y=64. The CPU sim already lives within ±64; on GPU
// we treat the valid range as [-64, 63] for the bitmap. Cells at x=64 etc. exist
// in the buffer but their `neighbor-exists` won't see them. Acceptable for the
// shapes we have (none reach the wall in xy). If it becomes a problem we widen
// to 256³ with 22 bits.
pub const OCCUPANCY_WORDS: u32 = (128 * 128 * 128) / 32;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CellGpu {
    pub pos: [i32; 3],
    pub state_init: u32,
    pub state: [f32; 32],
    pub field: [f32; 16],
}

impl CellGpu {
    pub fn empty(pos: [i32; 3]) -> Self {
        Self {
            pos,
            state_init: 0,
            state: [0.0; 32],
            field: [0.0; 16],
        }
    }
    pub fn seed(pos: [i32; 3], is_seed_slot: u32) -> Self {
        let mut c = Self::empty(pos);
        c.state[is_seed_slot as usize] = 1.0;
        c.state_init = 1u32 << is_seed_slot;
        c
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    n_top: u32,
    num_grads: u32,
    max_cells: u32,
    max_sources_per_grad: u32,
    max_reps: u32,
    world_min_x: i32,
    world_min_y: i32,
    world_min_z: i32,
    world_max_x: i32,
    world_max_y: i32,
    world_max_z: i32,
    _pad: u32,
}

pub struct GpuSim {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pub program: FlatProgram,
    pub tick: u64,

    // dynamic buffers
    cells_buf: wgpu::Buffer,
    cell_count_buf: wgpu::Buffer,
    occupancy_buf: wgpu::Buffer,
    sources_buf: wgpu::Buffer,
    source_count_buf: wgpu::Buffer,
    replications_buf: wgpu::Buffer,
    rep_count_buf: wgpu::Buffer,

    // immutable per-program buffers
    ast_nodes_buf: wgpu::Buffer,
    ast_children_buf: wgpu::Buffer,
    replicate_sites_buf: wgpu::Buffer,
    top_level_buf: wgpu::Buffer,
    uniforms_buf: wgpu::Buffer,
    /// 6 × u32 = (cells_wg_x, 1, 1, reps_wg_x, 1, 1). Updated by tiny GPU
    /// kernels each tick, consumed by indirect-dispatch of the heavy kernels.
    dispatch_args_buf: wgpu::Buffer,
    /// Atomic u32. Set to 1 by any kernel that mutates state this tick.
    /// `update_cell_dispatch_args` reads it to decide whether to dispatch
    /// the heavy kernels on the next tick. Lets fixed-point ticks become
    /// no-ops on the GPU.
    dirty_buf: wgpu::Buffer,

    // pipelines
    bgl: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    pipeline_build_occupancy: wgpu::ComputePipeline,
    pipeline_compute_field: wgpu::ComputePipeline,
    pipeline_eval: wgpu::ComputePipeline,
    pipeline_apply_priority: wgpu::ComputePipeline,
    pipeline_apply_default: wgpu::ComputePipeline,
    pipeline_update_cell_args: wgpu::ComputePipeline,
    pipeline_update_rep_args: wgpu::ComputePipeline,
}

impl GpuSim {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        program: FlatProgram,
        seed_pos: [i32; 3],
    ) -> Result<Self, String> {
        let is_seed_slot = *program
            .slot_table
            .get("is-seed")
            .ok_or_else(|| "program has no 'is-seed' variable".to_string())?;

        let cell_size = std::mem::size_of::<CellGpu>() as u64;
        let cells_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cells"),
            size: cell_size * MAX_CELLS as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // seed cell at slot 0
        let seed = CellGpu::seed(seed_pos, is_seed_slot);
        queue.write_buffer(&cells_buf, 0, bytemuck::bytes_of(&seed));

        let cell_count_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cell_count"),
            contents: bytemuck::bytes_of(&1u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        let occupancy_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occupancy"),
            size: 4 * OCCUPANCY_WORDS as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let source_size = 16u64; // vec3<f32> + f32 + pad? actually vec3<f32> aligns to 16, so 16 already.
        let sources_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sources"),
            size: source_size * MAX_SOURCES_PER_GRAD as u64 * ast::MAX_GRADS as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let source_count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("source_count"),
            size: 4 * ast::MAX_GRADS as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Replication WGSL layout:
        //   pos: vec3<i32>  offset 0..12  (vec3 size 12 align 16)
        //   n_overrides: u32 offset 12..16
        //   overrides: array<{slot:u32,value:f32}, 4>  offset 16..48
        // total 48 bytes, struct align 16.
        let rep_size = 48u64;
        let replications_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("replications"),
            size: rep_size * MAX_REPS as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let rep_count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rep_count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ast_nodes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ast_nodes"),
            contents: bytemuck::cast_slice(&program.nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let ast_children_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ast_children"),
            contents: bytemuck::cast_slice(&program.children),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let replicate_sites_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("replicate_sites"),
            contents: bytemuck::cast_slice(&program.replicate_sites),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let top_level_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("top_level"),
            contents: bytemuck::cast_slice(&program.top_level),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let uniforms = Uniforms {
            n_top: program.top_level.len() as u32,
            num_grads: program.num_grads,
            max_cells: MAX_CELLS,
            max_sources_per_grad: MAX_SOURCES_PER_GRAD,
            max_reps: MAX_REPS,
            world_min_x: -WORLD_HALF_XY,
            world_min_y: -WORLD_HALF_XY,
            world_min_z: WORLD_Z_MIN,
            world_max_x: WORLD_HALF_XY,
            world_max_y: WORLD_HALF_XY,
            world_max_z: WORLD_Z_MAX,
            _pad: 0,
        };
        let uniforms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // initial dispatch args: 1 workgroup of cells, 0 of reps (will be
        // overwritten by update_*_args kernels every tick).
        let initial_args: [u32; 6] = [1, 1, 1, 0, 1, 1];
        let dispatch_args_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dispatch_args"),
            contents: bytemuck::cast_slice(&initial_args),
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });
        // dirty flag starts at 1 to force the first tick to actually run.
        let dirty_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dirty"),
            contents: bytemuck::bytes_of(&1u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("morpheus.bgl"),
            entries: &[
                storage_entry(0, false),  // cells
                storage_entry(1, false),  // cell_count
                storage_entry(2, false),  // occupancy
                storage_entry(3, true),   // ast_nodes (read)
                storage_entry(4, true),   // ast_children (read)
                storage_entry(5, true),   // replicate_sites (read)
                storage_entry(6, true),   // top_level (read)
                storage_entry(7, false),  // sources
                storage_entry(8, false),  // source_count
                storage_entry(9, false),  // replications
                storage_entry(10, false), // rep_count
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(12, false), // dispatch_args (read+write)
                storage_entry(13, false), // dirty flag
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("morpheus.bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: cells_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cell_count_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: occupancy_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: ast_nodes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: ast_children_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: replicate_sites_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: top_level_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: sources_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: source_count_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: replications_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: rep_count_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: uniforms_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: dispatch_args_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: dirty_buf.as_entire_binding() },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("morpheus.shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("morpheus.pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let mk = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipeline_build_occupancy = mk("build_occupancy");
        let pipeline_compute_field = mk("compute_field");
        let pipeline_eval = mk("eval_main");
        let pipeline_apply_priority = mk("apply_replications_priority");
        let pipeline_apply_default = mk("apply_replications_default");
        let pipeline_update_cell_args = mk("update_cell_dispatch_args");
        let pipeline_update_rep_args = mk("update_rep_dispatch_args");

        Ok(Self {
            device,
            queue,
            program,
            tick: 0,
            cells_buf,
            cell_count_buf,
            occupancy_buf,
            sources_buf,
            source_count_buf,
            replications_buf,
            rep_count_buf,
            ast_nodes_buf,
            ast_children_buf,
            replicate_sites_buf,
            top_level_buf,
            uniforms_buf,
            dispatch_args_buf,
            dirty_buf,
            bgl,
            bind_group,
            pipeline_build_occupancy,
            pipeline_compute_field,
            pipeline_eval,
            pipeline_apply_priority,
            pipeline_apply_default,
            pipeline_update_cell_args,
            pipeline_update_rep_args,
        })
    }

    /// Record one tick's worth of dispatches. Uses indirect dispatch so the
    /// per-cell and per-rep kernels only schedule enough workgroups to cover
    /// the current live counts (no waste when the world is small).
    fn encode_tick(&self, enc: &mut wgpu::CommandEncoder) {
        // dispatch_args[0..3] = per-cell wg count; written by update_cell_dispatch_args
        // dispatch_args[3..6] = per-rep  wg count; written by update_rep_dispatch_args
        const CELL_OFFSET: u64 = 0;
        const REP_OFFSET: u64 = 12;
        enc.clear_buffer(&self.rep_count_buf, 0, None);
        enc.clear_buffer(&self.occupancy_buf, 0, None);

        // Tiny kernel: read cell_count, write per-cell workgroup count.
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline_update_cell_args);
            p.set_bind_group(0, &self.bind_group, &[]);
            p.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline_build_occupancy);
            p.set_bind_group(0, &self.bind_group, &[]);
            p.dispatch_workgroups_indirect(&self.dispatch_args_buf, CELL_OFFSET);
        }
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline_compute_field);
            p.set_bind_group(0, &self.bind_group, &[]);
            p.dispatch_workgroups_indirect(&self.dispatch_args_buf, CELL_OFFSET);
        }
        enc.clear_buffer(&self.source_count_buf, 0, None);
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline_eval);
            p.set_bind_group(0, &self.bind_group, &[]);
            p.dispatch_workgroups_indirect(&self.dispatch_args_buf, CELL_OFFSET);
        }
        // After eval, rep_count is set; write rep dispatch args.
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline_update_rep_args);
            p.set_bind_group(0, &self.bind_group, &[]);
            p.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline_apply_priority);
            p.set_bind_group(0, &self.bind_group, &[]);
            p.dispatch_workgroups_indirect(&self.dispatch_args_buf, REP_OFFSET);
        }
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline_apply_default);
            p.set_bind_group(0, &self.bind_group, &[]);
            p.dispatch_workgroups_indirect(&self.dispatch_args_buf, REP_OFFSET);
        }
    }

    pub fn step(&mut self) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tick") });
        self.encode_tick(&mut enc);
        self.queue.submit([enc.finish()]);
        self.tick += 1;
    }

    /// Batch `n` ticks into a single command encoder + submit. Avoids the
    /// per-tick submit overhead which dominates for small workloads.
    pub fn step_many_gpu(&mut self, n: usize) {
        if n == 0 { return; }
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tick.batch") },
        );
        for _ in 0..n {
            self.encode_tick(&mut enc);
        }
        self.queue.submit([enc.finish()]);
        self.tick += n as u64;
    }

    pub fn read_cells(&self) -> Vec<CellGpu> {
        let count = self.read_cell_count().min(MAX_CELLS);
        if count == 0 {
            return Vec::new();
        }
        let cell_size = std::mem::size_of::<CellGpu>() as u64;
        let bytes = (count as u64) * cell_size;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cells_staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.cells_buf, 0, &staging, 0, bytes);
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().expect("map_async failed");
        let data = slice.get_mapped_range();
        let cells: Vec<CellGpu> = bytemuck::cast_slice::<u8, CellGpu>(&data).to_vec();
        drop(data);
        staging.unmap();
        cells
    }

    pub fn read_cell_count(&self) -> u32 {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("count_staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.cell_count_buf, 0, &staging, 0, 4);
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().expect("map_async failed");
        let data = slice.get_mapped_range();
        let v = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        staging.unmap();
        v
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl crate::sim::Sim for GpuSim {
    fn step(&mut self) -> Result<(), String> {
        GpuSim::step(self);
        Ok(())
    }
    fn step_many(&mut self, n: usize) -> Result<(), String> {
        // No fixed-point detection on GPU yet — runs all n ticks.
        GpuSim::step_many_gpu(self, n);
        Ok(())
    }
    fn tick(&self) -> u64 {
        self.tick
    }
    fn cell_count(&self) -> usize {
        // Note: requires GPU readback. Cache or pair with snapshot() in hot loops.
        self.read_cell_count() as usize
    }
    fn fixed_point(&self) -> bool {
        // Not yet detected on GPU — always false until we wire a flag.
        false
    }
    fn snapshot(&mut self) -> Vec<crate::sim::CellSnapshot> {
        let cells = self.read_cells();
        let is_seed = self.program.slot_table.get("is-seed").copied();
        let is_axis = self.program.slot_table.get("is-axis").copied();
        let inside = self.program.slot_table.get("inside").copied();
        cells
            .into_iter()
            .map(|c| {
                let role = role_from_slots(&c, is_seed, is_axis, inside);
                crate::sim::CellSnapshot {
                    pos: (c.pos[0], c.pos[1], c.pos[2]),
                    role,
                }
            })
            .collect()
    }
    fn read_state(&mut self, pos: (i32, i32, i32), var: &str) -> Option<f32> {
        let slot = *self.program.slot_table.get(var)? as usize;
        let cells = self.read_cells();
        let c = cells
            .iter()
            .find(|c| (c.pos[0], c.pos[1], c.pos[2]) == pos)?;
        Some(c.state[slot])
    }
}

fn role_from_slots(
    c: &CellGpu,
    is_seed: Option<u32>,
    is_axis: Option<u32>,
    inside: Option<u32>,
) -> crate::world::CellRole {
    let truthy = |slot: Option<u32>| -> bool {
        slot.map(|s| c.state[s as usize] != 0.0).unwrap_or(false)
    };
    if truthy(is_seed) {
        crate::world::CellRole::Seed
    } else if truthy(is_axis) {
        crate::world::CellRole::Axis
    } else if truthy(inside) {
        crate::world::CellRole::Inside
    } else {
        crate::world::CellRole::Other
    }
}

/// Standalone wgpu device+queue for headless use (no surface).
pub fn create_headless_device() -> Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>, wgpu::AdapterInfo), String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL | wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|e| format!("no adapter: {e}"))?;
    let info = adapter.get_info();
    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffers_per_shader_stage = 16;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("morpheus.device"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::default(),
    }))
    .map_err(|e| format!("device: {e}"))?;
    Ok((Arc::new(device), Arc::new(queue), info))
}

// -----------------------------------------------------------------------------
// WGSL: shared header + 4 entry points
// -----------------------------------------------------------------------------

const SHADER: &str = r#"
// ---- tags (must match ast.rs) ----
const TAG_NUM: u32 = 0u;
const TAG_BOOL: u32 = 1u;
const TAG_SYM_VAR: u32 = 2u;
const TAG_SPECIAL_IF: u32 = 3u;
const TAG_SPECIAL_BEGIN: u32 = 4u;
const TAG_SPECIAL_AND: u32 = 5u;
const TAG_SPECIAL_OR: u32 = 6u;
const TAG_SPECIAL_NOT: u32 = 7u;
const TAG_SPECIAL_DEFINE: u32 = 8u;
const TAG_SPECIAL_SET: u32 = 9u;
const TAG_OP_ADD: u32 = 10u;
const TAG_OP_SUB: u32 = 11u;
const TAG_OP_MUL: u32 = 12u;
const TAG_OP_DIV: u32 = 13u;
const TAG_OP_LT: u32 = 14u;
const TAG_OP_GT: u32 = 15u;
const TAG_OP_LE: u32 = 16u;
const TAG_OP_GE: u32 = 17u;
const TAG_OP_EQ: u32 = 18u;
const TAG_BUILTIN_EMIT: u32 = 19u;
const TAG_BUILTIN_READ: u32 = 20u;
const TAG_BUILTIN_NEIGHBOR: u32 = 21u;
const TAG_BUILTIN_REPLICATE: u32 = 22u;

const NIL_INDEX: u32 = 0xFFFFFFFFu;
const STACK_SIZE: u32 = 64u;

struct Cell {
    pos: vec3<i32>,
    state_init: u32,
    state: array<f32, 32>,
    field: array<f32, 16>,
};

struct Node { tag: u32, a: u32, b: u32, c: u32 };
struct AstOverride { slot: u32, expr_node: u32 };
struct ReplicateSite {
    dir: u32, n_overrides: u32, _p0: u32, _p1: u32,
    overrides: array<AstOverride, 4>,
};

struct Source { pos: vec3<f32>, value: f32 };
struct ReplOverride { slot: u32, value: f32 };
struct Replication {
    pos: vec3<i32>, n_overrides: u32,
    overrides: array<ReplOverride, 4>,
};

struct Uniforms {
    n_top: u32,
    num_grads: u32,
    max_cells: u32,
    max_sources_per_grad: u32,
    max_reps: u32,
    world_min_x: i32, world_min_y: i32, world_min_z: i32,
    world_max_x: i32, world_max_y: i32, world_max_z: i32,
    _pad: u32,
};

@group(0) @binding(0)  var<storage, read_write> cells: array<Cell>;
@group(0) @binding(1)  var<storage, read_write> cell_count: atomic<u32>;
@group(0) @binding(2)  var<storage, read_write> occupancy: array<atomic<u32>>;
@group(0) @binding(3)  var<storage, read>       ast_nodes: array<Node>;
@group(0) @binding(4)  var<storage, read>       ast_children: array<u32>;
@group(0) @binding(5)  var<storage, read>       replicate_sites: array<ReplicateSite>;
@group(0) @binding(6)  var<storage, read>       top_level: array<u32>;
@group(0) @binding(7)  var<storage, read_write> sources: array<Source>;
@group(0) @binding(8)  var<storage, read_write> source_count: array<atomic<u32>>;
@group(0) @binding(9)  var<storage, read_write> replications: array<Replication>;
@group(0) @binding(10) var<storage, read_write> rep_count: atomic<u32>;
@group(0) @binding(11) var<uniform>             uni: Uniforms;

struct DispatchArgs {
    cell_x: u32, cell_y: u32, cell_z: u32,
    rep_x:  u32, rep_y:  u32, rep_z:  u32,
};
@group(0) @binding(12) var<storage, read_write> dispatch_args: DispatchArgs;
@group(0) @binding(13) var<storage, read_write> dirty: atomic<u32>;

// Reads the dirty flag set by the previous tick. If it's 0 (no state changed,
// no replications, no new cells), the per-cell kernels for THIS tick are
// skipped via cell_x = 0. Resets dirty so this tick's kernels can re-set it.
@compute @workgroup_size(1)
fn update_cell_dispatch_args() {
    let was_dirty = atomicExchange(&dirty, 0u);
    if was_dirty == 0u {
        dispatch_args.cell_x = 0u;
    } else {
        let cc = atomicLoad(&cell_count);
        let live = min(cc, uni.max_cells);
        dispatch_args.cell_x = (live + 63u) / 64u;
    }
    dispatch_args.cell_y = 1u;
    dispatch_args.cell_z = 1u;
}

@compute @workgroup_size(1)
fn update_rep_dispatch_args() {
    let rc = atomicLoad(&rep_count);
    let live = min(rc, uni.max_reps);
    dispatch_args.rep_x = (live + 63u) / 64u;
    dispatch_args.rep_y = 1u;
    dispatch_args.rep_z = 1u;
}

// ---- helpers ----

fn occ_bit_index(p: vec3<i32>) -> i32 {
    if p.x < -64 || p.x > 63 || p.y < -64 || p.y > 63 || p.z < 0 || p.z > 127 { return -1; }
    return i32(u32(p.x + 64)) | (i32(u32(p.y + 64)) << 7u) | (i32(u32(p.z)) << 14u);
}

fn truthy(v: f32) -> bool { return v != 0.0; }

fn dir_offset(d: u32) -> vec3<i32> {
    switch d {
        case 0u: { return vec3<i32>(1, 0, 0); }
        case 1u: { return vec3<i32>(-1, 0, 0); }
        case 2u: { return vec3<i32>(0, 1, 0); }
        case 3u: { return vec3<i32>(0, -1, 0); }
        case 4u: { return vec3<i32>(0, 0, 1); }
        case 5u: { return vec3<i32>(0, 0, -1); }
        default: { return vec3<i32>(0, 0, 0); }
    }
}

// ---- 1. build_occupancy ----
// Bitmap was zeroed by encoder.clear_buffer on the CPU side.
@compute @workgroup_size(64)
fn build_occupancy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cid = gid.x;
    if cid >= uni.max_cells { return; }
    let live = atomicLoad(&cell_count);
    if cid >= live { return; }
    let p = cells[cid].pos;
    let bi = occ_bit_index(p);
    if bi < 0 { return; }
    let word = u32(bi) >> 5u;
    let mask = 1u << (u32(bi) & 31u);
    atomicOr(&occupancy[word], mask);
}

// ---- 2. compute_field ----
// Per cell, per gradient: min Euclidean distance to nearest source. Returns 0
// if the gradient has no sources (matches CPU sim: `read_gradient` of an empty
// gradient is 0, so `(< g X)` style checks are trivially satisfied).
@compute @workgroup_size(64)
fn compute_field(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cid = gid.x;
    if cid >= uni.max_cells { return; }
    let live = atomicLoad(&cell_count);
    if cid >= live { return; }

    let p = cells[cid].pos;
    let pf = vec3<f32>(f32(p.x), f32(p.y), f32(p.z));

    for (var g: u32 = 0u; g < uni.num_grads; g = g + 1u) {
        let raw_n = atomicLoad(&source_count[g]);
        let n = min(raw_n, uni.max_sources_per_grad);
        if n == 0u {
            cells[cid].field[g] = 0.0;
            continue;
        }
        var min_d: f32 = 1e30;
        let base = g * uni.max_sources_per_grad;
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            let s = sources[base + i];
            let d = distance(pf, s.pos);
            if d < min_d { min_d = d; }
        }
        cells[cid].field[g] = min_d;
    }
}

// ---- 3. eval ----
// Per-cell tree-walking interpreter with explicit stack.

struct Frame { node: u32, phase: u32, saved: f32, saved_int: u32 };

@compute @workgroup_size(64)
fn eval_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cid = gid.x;
    if cid >= uni.max_cells { return; }
    let live = atomicLoad(&cell_count);
    if cid >= live { return; }

    var stack: array<Frame, 64>;
    var sp: u32;
    var last: f32;

    for (var t: u32 = 0u; t < uni.n_top; t = t + 1u) {
        stack[0] = Frame(top_level[t], 0u, 0.0, 0u);
        sp = 1u;
        last = 0.0;

        // Bound iterations defensively to prevent infinite loops in malformed programs.
        var iters: u32 = 0u;
        loop {
            if sp == 0u { break; }
            iters = iters + 1u;
            if iters > 200000u { break; }

            let fi = sp - 1u;
            let f = stack[fi];
            let node = ast_nodes[f.node];

            switch node.tag {
                case 0u: { // NUM
                    last = bitcast<f32>(node.a);
                    sp = sp - 1u;
                }
                case 1u: { // BOOL
                    last = f32(node.a);
                    sp = sp - 1u;
                }
                case 2u: { // SYM_VAR
                    last = cells[cid].state[node.a];
                    sp = sp - 1u;
                }
                case 3u: { // IF
                    if f.phase == 0u {
                        stack[fi].phase = 1u;
                        stack[sp] = Frame(node.a, 0u, 0.0, 0u); sp = sp + 1u;
                    } else if f.phase == 1u {
                        if truthy(last) {
                            stack[fi].phase = 2u;
                            stack[sp] = Frame(node.b, 0u, 0.0, 0u); sp = sp + 1u;
                        } else if node.c != NIL_INDEX {
                            stack[fi].phase = 2u;
                            stack[sp] = Frame(node.c, 0u, 0.0, 0u); sp = sp + 1u;
                        } else {
                            last = 0.0;
                            sp = sp - 1u;
                        }
                    } else {
                        sp = sp - 1u;
                    }
                }
                case 4u: { // BEGIN
                    let n = node.b;
                    if n == 0u { last = 0.0; sp = sp - 1u; }
                    else if f.phase < n {
                        stack[fi].phase = f.phase + 1u;
                        let ci = ast_children[node.a + f.phase];
                        stack[sp] = Frame(ci, 0u, 0.0, 0u); sp = sp + 1u;
                    } else {
                        sp = sp - 1u;
                    }
                }
                case 5u: { // AND
                    let n = node.b;
                    if n == 0u { last = 1.0; sp = sp - 1u; }
                    else if f.phase == 0u {
                        stack[fi].phase = 1u;
                        let ci = ast_children[node.a];
                        stack[sp] = Frame(ci, 0u, 0.0, 0u); sp = sp + 1u;
                    } else {
                        if !truthy(last) { last = 0.0; sp = sp - 1u; }
                        else if f.phase >= n { sp = sp - 1u; }
                        else {
                            stack[fi].phase = f.phase + 1u;
                            let ci = ast_children[node.a + f.phase];
                            stack[sp] = Frame(ci, 0u, 0.0, 0u); sp = sp + 1u;
                        }
                    }
                }
                case 6u: { // OR
                    let n = node.b;
                    if n == 0u { last = 0.0; sp = sp - 1u; }
                    else if f.phase == 0u {
                        stack[fi].phase = 1u;
                        let ci = ast_children[node.a];
                        stack[sp] = Frame(ci, 0u, 0.0, 0u); sp = sp + 1u;
                    } else {
                        if truthy(last) { sp = sp - 1u; }
                        else if f.phase >= n { last = 0.0; sp = sp - 1u; }
                        else {
                            stack[fi].phase = f.phase + 1u;
                            let ci = ast_children[node.a + f.phase];
                            stack[sp] = Frame(ci, 0u, 0.0, 0u); sp = sp + 1u;
                        }
                    }
                }
                case 7u: { // NOT
                    if f.phase == 0u {
                        stack[fi].phase = 1u;
                        stack[sp] = Frame(node.a, 0u, 0.0, 0u); sp = sp + 1u;
                    } else {
                        last = select(1.0, 0.0, truthy(last));
                        sp = sp - 1u;
                    }
                }
                case 8u: { // DEFINE (define-once)
                    let mask = 1u << node.a;
                    if (cells[cid].state_init & mask) != 0u {
                        last = 0.0; sp = sp - 1u;
                    } else if f.phase == 0u {
                        stack[fi].phase = 1u;
                        stack[sp] = Frame(node.b, 0u, 0.0, 0u); sp = sp + 1u;
                    } else {
                        cells[cid].state[node.a] = last;
                        cells[cid].state_init = cells[cid].state_init | mask;
                        last = 0.0;
                        sp = sp - 1u;
                    }
                }
                case 9u: { // SET!
                    if f.phase == 0u {
                        stack[fi].phase = 1u;
                        stack[sp] = Frame(node.b, 0u, 0.0, 0u); sp = sp + 1u;
                    } else {
                        let old = cells[cid].state[node.a];
                        if old != last { atomicOr(&dirty, 1u); }
                        cells[cid].state[node.a] = last;
                        cells[cid].state_init = cells[cid].state_init | (1u << node.a);
                        sp = sp - 1u;
                    }
                }
                case 10u: { // ADD (variadic)
                    let n = node.b;
                    if n == 0u { last = 0.0; sp = sp - 1u; }
                    else if f.phase == 0u {
                        stack[fi] = Frame(f.node, 1u, 0.0, 0u);
                        let ci = ast_children[node.a];
                        stack[sp] = Frame(ci, 0u, 0.0, 0u); sp = sp + 1u;
                    } else {
                        let acc = f.saved + last;
                        if f.phase >= n { last = acc; sp = sp - 1u; }
                        else {
                            stack[fi] = Frame(f.node, f.phase + 1u, acc, 0u);
                            let ci = ast_children[node.a + f.phase];
                            stack[sp] = Frame(ci, 0u, 0.0, 0u); sp = sp + 1u;
                        }
                    }
                }
                case 11u: { // SUB
                    let n = node.b;
                    if n == 0u { last = 0.0; sp = sp - 1u; }
                    else if f.phase == 0u {
                        stack[fi] = Frame(f.node, 1u, 0.0, 0u);
                        let ci = ast_children[node.a];
                        stack[sp] = Frame(ci, 0u, 0.0, 0u); sp = sp + 1u;
                    } else if f.phase == 1u {
                        if n == 1u { last = -last; sp = sp - 1u; }
                        else {
                            stack[fi] = Frame(f.node, 2u, last, 0u);
                            let ci = ast_children[node.a + 1u];
                            stack[sp] = Frame(ci, 0u, 0.0, 0u); sp = sp + 1u;
                        }
                    } else {
                        let acc = f.saved - last;
                        if f.phase >= n { last = acc; sp = sp - 1u; }
                        else {
                            stack[fi] = Frame(f.node, f.phase + 1u, acc, 0u);
                            let ci = ast_children[node.a + f.phase];
                            stack[sp] = Frame(ci, 0u, 0.0, 0u); sp = sp + 1u;
                        }
                    }
                }
                case 12u: { // MUL
                    let n = node.b;
                    if n == 0u { last = 1.0; sp = sp - 1u; }
                    else if f.phase == 0u {
                        stack[fi] = Frame(f.node, 1u, 1.0, 0u);
                        let ci = ast_children[node.a];
                        stack[sp] = Frame(ci, 0u, 0.0, 0u); sp = sp + 1u;
                    } else {
                        let acc = f.saved * last;
                        if f.phase >= n { last = acc; sp = sp - 1u; }
                        else {
                            stack[fi] = Frame(f.node, f.phase + 1u, acc, 0u);
                            let ci = ast_children[node.a + f.phase];
                            stack[sp] = Frame(ci, 0u, 0.0, 0u); sp = sp + 1u;
                        }
                    }
                }
                case 13u: { // DIV
                    let n = node.b;
                    if n == 0u { last = 1.0; sp = sp - 1u; }
                    else if f.phase == 0u {
                        stack[fi] = Frame(f.node, 1u, 0.0, 0u);
                        let ci = ast_children[node.a];
                        stack[sp] = Frame(ci, 0u, 0.0, 0u); sp = sp + 1u;
                    } else if f.phase == 1u {
                        if n == 1u { last = 1.0 / last; sp = sp - 1u; }
                        else {
                            stack[fi] = Frame(f.node, 2u, last, 0u);
                            let ci = ast_children[node.a + 1u];
                            stack[sp] = Frame(ci, 0u, 0.0, 0u); sp = sp + 1u;
                        }
                    } else {
                        let acc = f.saved / last;
                        if f.phase >= n { last = acc; sp = sp - 1u; }
                        else {
                            stack[fi] = Frame(f.node, f.phase + 1u, acc, 0u);
                            let ci = ast_children[node.a + f.phase];
                            stack[sp] = Frame(ci, 0u, 0.0, 0u); sp = sp + 1u;
                        }
                    }
                }
                case 14u, 15u, 16u, 17u, 18u: { // comparators (LT/GT/LE/GE/EQ)
                    if f.phase == 0u {
                        stack[fi].phase = 1u;
                        stack[sp] = Frame(node.a, 0u, 0.0, 0u); sp = sp + 1u;
                    } else if f.phase == 1u {
                        stack[fi] = Frame(f.node, 2u, last, 0u);
                        stack[sp] = Frame(node.b, 0u, 0.0, 0u); sp = sp + 1u;
                    } else {
                        var ok: bool;
                        switch node.tag {
                            case 14u: { ok = f.saved < last; }
                            case 15u: { ok = f.saved > last; }
                            case 16u: { ok = f.saved <= last; }
                            case 17u: { ok = f.saved >= last; }
                            case 18u: { ok = abs(f.saved - last) < 1e-9; }
                            default:  { ok = false; }
                        }
                        last = select(0.0, 1.0, ok);
                        sp = sp - 1u;
                    }
                }
                case 19u: { // EMIT-GRADIENT
                    if f.phase == 0u {
                        stack[fi].phase = 1u;
                        stack[sp] = Frame(node.b, 0u, 0.0, 0u); sp = sp + 1u;
                    } else {
                        let g = node.a;
                        let i = atomicAdd(&source_count[g], 1u);
                        if i < uni.max_sources_per_grad {
                            let base = g * uni.max_sources_per_grad + i;
                            let p = cells[cid].pos;
                            sources[base].pos = vec3<f32>(f32(p.x), f32(p.y), f32(p.z));
                            sources[base].value = last;
                        }
                        last = 0.0;
                        sp = sp - 1u;
                    }
                }
                case 20u: { // READ-GRADIENT
                    last = cells[cid].field[node.a];
                    sp = sp - 1u;
                }
                case 21u: { // NEIGHBOR-EXISTS
                    let p = cells[cid].pos + dir_offset(node.a);
                    let bi = occ_bit_index(p);
                    if bi < 0 {
                        last = 0.0;
                    } else {
                        let word = u32(bi) >> 5u;
                        let mask = 1u << (u32(bi) & 31u);
                        let v = atomicLoad(&occupancy[word]);
                        last = select(0.0, 1.0, (v & mask) != 0u);
                    }
                    sp = sp - 1u;
                }
                case 22u: { // REPLICATE-TOWARD
                    // Multi-phase: phase 0 claims a rep slot and (if any overrides)
                    // pushes the first override expr; phase 1 captures override results
                    // and either pushes the next or pops.
                    let site = replicate_sites[node.b];
                    if f.phase == 0u {
                        let rep_idx = atomicAdd(&rep_count, 1u);
                        if rep_idx >= uni.max_reps {
                            last = 0.0; sp = sp - 1u;
                        } else {
                            atomicOr(&dirty, 1u);
                            let tgt = cells[cid].pos + dir_offset(node.a);
                            replications[rep_idx].pos = tgt;
                            replications[rep_idx].n_overrides = site.n_overrides;
                            if site.n_overrides == 0u {
                                last = 0.0; sp = sp - 1u;
                            } else {
                                stack[fi] = Frame(f.node, 1u, bitcast<f32>(rep_idx), 0u);
                                stack[sp] = Frame(site.overrides[0].expr_node, 0u, 0.0, 0u);
                                sp = sp + 1u;
                            }
                        }
                    } else {
                        let rep_idx = bitcast<u32>(f.saved);
                        let i = f.saved_int;
                        replications[rep_idx].overrides[i].slot = site.overrides[i].slot;
                        replications[rep_idx].overrides[i].value = last;
                        let next_i = i + 1u;
                        if next_i >= site.n_overrides {
                            last = 0.0; sp = sp - 1u;
                        } else {
                            stack[fi] = Frame(f.node, 1u, f.saved, next_i);
                            stack[sp] = Frame(site.overrides[next_i].expr_node, 0u, 0.0, 0u);
                            sp = sp + 1u;
                        }
                    }
                }
                default: {
                    last = 0.0;
                    sp = sp - 1u;
                }
            }
        }
    }
}

// ---- 6. apply_replications ----
// Two passes (separate dispatches). Pass 1 processes reps with overrides
// (priority); pass 2 processes plain reps. This matches CPU determinism:
// when an override-rep and a plain-rep target the same voxel, the override
// rep wins, regardless of arrival order on the GPU.
fn apply_one(rid: u32, want_overrides: bool) {
    let live_reps = atomicLoad(&rep_count);
    let n = min(live_reps, uni.max_reps);
    if rid >= n { return; }
    let has_overrides = replications[rid].n_overrides > 0u;
    if want_overrides != has_overrides { return; }

    let p = replications[rid].pos;
    let bi = occ_bit_index(p);
    if bi < 0 { return; }
    let word = u32(bi) >> 5u;
    let mask = 1u << (u32(bi) & 31u);
    let prev = atomicOr(&occupancy[word], mask);
    if (prev & mask) != 0u { return; }   // lost the race or position already occupied

    let slot = atomicAdd(&cell_count, 1u);
    if slot >= uni.max_cells { return; }
    atomicOr(&dirty, 1u);

    cells[slot].pos = p;
    cells[slot].state_init = 0u;
    for (var i: u32 = 0u; i < 32u; i = i + 1u) { cells[slot].state[i] = 0.0; }
    for (var i: u32 = 0u; i < 16u; i = i + 1u) { cells[slot].field[i] = 0.0; }

    let no = replications[rid].n_overrides;
    for (var i: u32 = 0u; i < no; i = i + 1u) {
        let o = replications[rid].overrides[i];
        cells[slot].state[o.slot] = o.value;
        cells[slot].state_init = cells[slot].state_init | (1u << o.slot);
    }
}

@compute @workgroup_size(64)
fn apply_replications_priority(@builtin(global_invocation_id) gid: vec3<u32>) {
    apply_one(gid.x, true);
}

@compute @workgroup_size(64)
fn apply_replications_default(@builtin(global_invocation_id) gid: vec3<u32>) {
    apply_one(gid.x, false);
}
"#;
