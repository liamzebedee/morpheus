use crate::lisp::*;
use std::collections::{HashMap, HashSet};

pub const WORLD_HALF: i32 = 64;
pub const WORLD_Z_MAX: i32 = 127;
pub const WORLD_Z_MIN: i32 = 0;

pub type Pos = (i32, i32, i32);

pub struct Cell {
    pub env: Env,
}

pub struct World {
    pub cells: HashMap<Pos, Cell>,
    pub gradients: HashMap<String, Vec<(Pos, f64)>>,
    pub program: Vec<Val>,
    pub tick: u64,
    pub last_changed: bool,
    pub fixed_point: bool,
}

struct CellSub<'a> {
    pos: Pos,
    grads: &'a HashMap<String, Vec<(Pos, f64)>>,
    occupied: &'a HashSet<Pos>,
    emissions: Vec<(String, f64)>,
    replications: Vec<(Dir, Vec<(String, Val)>)>,
}

impl Substrate for CellSub<'_> {
    fn read_gradient(&self, name: &str) -> f64 {
        let sources = match self.grads.get(name) {
            Some(s) if !s.is_empty() => s,
            _ => return 0.0,
        };
        let mut min_d = f64::INFINITY;
        for &((sx, sy, sz), _v) in sources {
            let dx = (self.pos.0 - sx) as f64;
            let dy = (self.pos.1 - sy) as f64;
            let dz = (self.pos.2 - sz) as f64;
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d < min_d {
                min_d = d;
            }
        }
        min_d
    }

    fn emit_gradient(&mut self, name: &str, value: f64) {
        self.emissions.push((name.to_string(), value));
    }

    fn replicate_toward(&mut self, dir: Dir, child_state: Vec<(String, Val)>) {
        self.replications.push((dir, child_state));
    }

    fn neighbor_exists(&self, dir: Dir) -> bool {
        let off = dir.offset();
        let np = (self.pos.0 + off.0, self.pos.1 + off.1, self.pos.2 + off.2);
        self.occupied.contains(&np)
    }
}

impl World {
    pub fn new(program: Vec<Val>) -> Self {
        let mut w = Self {
            cells: HashMap::new(),
            gradients: HashMap::new(),
            program,
            tick: 0,
            last_changed: false,
            fixed_point: false,
        };
        w.spawn_seed();
        w
    }

    fn spawn_seed(&mut self) {
        let env = new_env();
        env.borrow_mut().insert("is-seed".into(), Val::Bool(true));
        self.cells.insert((0, 0, 0), Cell { env });
    }

    /// Remove every cell whose center lies within `radius` voxels of `center`.
    /// Returns the number of cells destroyed.
    pub fn blast(&mut self, center: (f32, f32, f32), radius: f32) -> usize {
        let r2 = radius * radius;
        let before = self.cells.len();
        self.cells.retain(|p, _| {
            let dx = p.0 as f32 - center.0;
            let dy = p.1 as f32 - center.1;
            let dz = p.2 as f32 - center.2;
            dx * dx + dy * dy + dz * dz > r2
        });
        // Drop gradient sources at destroyed positions so next tick reads
        // a clean field.
        self.gradients.retain(|_, sources| {
            sources.retain(|(pos, _)| self.cells.contains_key(pos));
            !sources.is_empty()
        });
        self.fixed_point = false;
        before - self.cells.len()
    }

    pub fn reset(&mut self, program: Vec<Val>) {
        self.cells.clear();
        self.gradients.clear();
        self.tick = 0;
        self.last_changed = false;
        self.fixed_point = false;
        self.program = program;
        self.spawn_seed();
    }

    pub fn step(&mut self) -> Result<(), String> {
        if self.fixed_point {
            return Ok(());
        }
        let prev_state = self.snapshot_state();
        let prev_grads = std::mem::take(&mut self.gradients);

        let mut positions: Vec<Pos> = self.cells.keys().copied().collect();
        positions.sort();
        let occupied: HashSet<Pos> = positions.iter().copied().collect();

        let mut new_grads: HashMap<String, Vec<(Pos, f64)>> = HashMap::new();
        let mut replications: Vec<(Pos, Vec<(String, Val)>)> = Vec::new();

        for &pos in &positions {
            let env = self.cells[&pos].env.clone();
            let mut sub = CellSub {
                pos,
                grads: &prev_grads,
                occupied: &occupied,
                emissions: Vec::new(),
                replications: Vec::new(),
            };
            for expr in &self.program {
                eval(expr, &env, &mut sub)?;
            }
            for (name, val) in sub.emissions {
                new_grads.entry(name).or_default().push((pos, val));
            }
            for (d, child_state) in sub.replications {
                let off = d.offset();
                let np = (pos.0 + off.0, pos.1 + off.1, pos.2 + off.2);
                if np.0.abs() <= WORLD_HALF
                    && np.1.abs() <= WORLD_HALF
                    && np.2 >= WORLD_Z_MIN
                    && np.2 <= WORLD_Z_MAX
                {
                    replications.push((np, child_state));
                }
            }
        }

        let mut grew = false;
        for (np, child_state) in replications {
            if !self.cells.contains_key(&np) {
                let env = new_env();
                {
                    let mut e = env.borrow_mut();
                    for (k, v) in child_state {
                        e.insert(k, v);
                    }
                }
                self.cells.insert(np, Cell { env });
                grew = true;
            }
        }

        let new_state = self.snapshot_state();
        let state_changed = new_state != prev_state;
        let changed = grew || state_changed;

        self.gradients = new_grads;
        self.tick += 1;
        self.last_changed = changed;
        if !changed {
            self.fixed_point = true;
        }
        Ok(())
    }

    fn snapshot_state(&self) -> Vec<(Pos, Vec<(String, Val)>)> {
        let mut out = Vec::with_capacity(self.cells.len());
        for (p, c) in &self.cells {
            let mut s: Vec<(String, Val)> = c
                .env
                .borrow()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            s.sort_by(|a, b| a.0.cmp(&b.0));
            out.push((*p, s));
        }
        out.sort_by_key(|x| x.0);
        out
    }

    pub fn cell_role(&self, p: Pos) -> CellRole {
        let cell = match self.cells.get(&p) {
            Some(c) => c,
            None => return CellRole::Empty,
        };
        let env = cell.env.borrow();
        let is_seed = env.get("is-seed").map(|v| v.truthy()).unwrap_or(false);
        let is_axis = env.get("is-axis").map(|v| v.truthy()).unwrap_or(false);
        let inside = env.get("inside").map(|v| v.truthy()).unwrap_or(false);
        if is_seed {
            CellRole::Seed
        } else if is_axis {
            CellRole::Axis
        } else if inside {
            CellRole::Inside
        } else {
            CellRole::Other
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CellRole {
    Empty,
    Seed,
    Axis,
    Inside,
    Other,
}

impl crate::sim::Sim for World {
    fn step(&mut self) -> Result<(), String> {
        World::step(self)
    }
    fn tick(&self) -> u64 {
        self.tick
    }
    fn cell_count(&self) -> usize {
        self.cells.len()
    }
    fn fixed_point(&self) -> bool {
        self.fixed_point
    }
    fn snapshot(&mut self) -> Vec<crate::sim::CellSnapshot> {
        self.cells
            .keys()
            .map(|&p| crate::sim::CellSnapshot {
                pos: p,
                role: self.cell_role(p),
            })
            .collect()
    }
    fn read_state(&mut self, pos: (i32, i32, i32), var: &str) -> Option<f32> {
        let cell = self.cells.get(&pos)?;
        let env = cell.env.borrow();
        let v = env.get(var)?;
        match v {
            crate::lisp::Val::Num(n) => Some(*n as f32),
            crate::lisp::Val::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
}
