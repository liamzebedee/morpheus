//! Engine-agnostic simulator interface.
//!
//! Both the CPU `World` and the GPU `GpuSim` implement this trait. Renderers
//! and headless drivers consume snapshots, not engine internals — that keeps
//! the simengine and the rendering paths fully decoupled.

use crate::world::CellRole;

#[derive(Clone, Copy, Debug)]
pub struct CellSnapshot {
    pub pos: (i32, i32, i32),
    pub role: CellRole,
}

pub trait Sim {
    fn step(&mut self) -> Result<(), String>;
    /// Run up to `n` ticks. Default impl loops `step()` and short-circuits
    /// at fixed-point. Engines that benefit from batching (GPU) override
    /// this to amortize submit/dispatch overhead.
    fn step_many(&mut self, n: usize) -> Result<(), String> {
        for _ in 0..n {
            if self.fixed_point() { break; }
            self.step()?;
        }
        Ok(())
    }
    fn tick(&self) -> u64;
    fn cell_count(&self) -> usize;
    fn fixed_point(&self) -> bool;
    /// Snapshot the current world. May be expensive on engines that store
    /// state on the GPU (readback) — callers should cache when running
    /// many ticks between renders.
    fn snapshot(&mut self) -> Vec<CellSnapshot>;
    /// Read a named state variable from the cell at the given world position.
    /// Used for CPU/GPU equivalence checks.
    fn read_state(&mut self, pos: (i32, i32, i32), var: &str) -> Option<f32>;
}
