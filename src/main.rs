mod lisp;
mod world;

use eframe::egui;
use glam::{Mat4, Vec3, Vec4};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use world::{CellRole, World};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let headless = args.iter().any(|a| a == "--headless");
    let max_steps: usize = args
        .windows(2)
        .find(|w| w[0] == "--steps")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(800);
    let save_every: usize = args
        .windows(2)
        .find(|w| w[0] == "--save-every")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(0);
    let out_path: Option<String> = args
        .windows(2)
        .find(|w| w[0] == "--out")
        .map(|w| w[1].clone());
    let frames_dir: Option<String> = args
        .windows(2)
        .find(|w| w[0] == "--frames-dir")
        .map(|w| w[1].clone());
    let skip_flags = ["--steps", "--save-every", "--out", "--frames-dir"];
    let path_arg = {
        let mut iter = args.iter().skip(1).peekable();
        let mut found: Option<String> = None;
        while let Some(a) = iter.next() {
            if skip_flags.contains(&a.as_str()) {
                iter.next();
                continue;
            }
            if a.starts_with("--") || a.parse::<usize>().is_ok() {
                continue;
            }
            found = Some(a.clone());
            break;
        }
        found.unwrap_or_else(|| "cylinder.local".to_string())
    };
    let code_path = PathBuf::from(&path_arg);
    let code_path = if code_path.is_absolute() {
        code_path
    } else {
        std::env::current_dir().unwrap_or_default().join(code_path)
    };

    if headless {
        if let Err(e) = run_headless(&code_path, max_steps, save_every, out_path.as_deref(), frames_dir.as_deref()) {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
        return;
    }

    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1500.0, 950.0]),
        ..Default::default()
    };
    if let Err(e) = eframe::run_native(
        "morpheus",
        opts,
        Box::new(move |_cc| Ok(Box::new(App::new(code_path)))),
    ) {
        eprintln!("gui error: {}", e);
        std::process::exit(1);
    }
}

fn run_headless(
    code_path: &PathBuf,
    max_steps: usize,
    save_every: usize,
    out_path: Option<&str>,
    frames_dir: Option<&str>,
) -> Result<(), String> {
    let code = fs::read_to_string(code_path).map_err(|e| format!("read: {}", e))?;
    let program = lisp::parse_program(&code).map_err(|e| format!("parse: {}", e))?;
    let mut world = World::new(program);

    if let Some(dir) = frames_dir {
        fs::create_dir_all(dir).map_err(|e| format!("mkdir: {}", e))?;
    }

    let mut frame_idx: usize = 0;
    let save_frame = |world: &World, dir: &str, idx: usize| -> Result<(), String> {
        let img = render_perspective(world, 1600, 1000);
        let p = PathBuf::from(dir).join(format!("frame_{:05}.png", idx));
        img.save(&p).map_err(|e| format!("save: {}", e))
    };

    if save_every > 0 {
        if let Some(dir) = frames_dir {
            save_frame(&world, dir, frame_idx)?;
            frame_idx += 1;
        }
    }

    for step in 0..max_steps {
        if world.fixed_point {
            break;
        }
        world.step().map_err(|e| format!("step: {}", e))?;
        if save_every > 0 {
            if let Some(dir) = frames_dir {
                if (step + 1) % save_every == 0 {
                    save_frame(&world, dir, frame_idx)?;
                    frame_idx += 1;
                }
            }
        }
    }

    // Always save final frame
    if let Some(dir) = frames_dir {
        save_frame(&world, dir, frame_idx)?;
    }

    let out = out_path
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("morpheus.png"));
    let img = render_perspective(&world, 1600, 1000);
    img.save(&out).map_err(|e| format!("save: {}", e))?;
    println!(
        "ticks={} cells={} fixed_point={}",
        world.tick,
        world.cells.len(),
        world.fixed_point
    );
    println!("{}", out.display());
    Ok(())
}

fn render_orthographic(
    world: &World,
) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    use image::{ImageBuffer, Rgb};
    use std::collections::HashMap;

    const VOXEL_PX: u32 = 4;
    const PAD: u32 = 12;
    let xy_w = (world::WORLD_HALF as u32 * 2 + 1) * VOXEL_PX;
    let xy_h = xy_w;
    let xz_w = xy_w;
    let xz_h = (world::WORLD_Z_MAX - world::WORLD_Z_MIN + 1) as u32 * VOXEL_PX;
    let yz_w = xy_w;
    let yz_h = xz_h;

    let total_w = xy_w + xz_w + yz_w + PAD * 4;
    let total_h = xy_h.max(xz_h).max(yz_h) + PAD * 2;
    let bg = Rgb([18u8, 20, 26]);
    let grid_color = Rgb([40u8, 44, 52]);
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(total_w, total_h, bg);

    // panel origins
    let p_xy = (PAD, PAD);
    let p_xz = (PAD * 2 + xy_w, PAD);
    let p_yz = (PAD * 3 + xy_w + xz_w, PAD);

    // draw panel grid outlines
    for &((ox, oy), (pw, ph)) in &[
        (p_xy, (xy_w, xy_h)),
        (p_xz, (xz_w, xz_h)),
        (p_yz, (yz_w, yz_h)),
    ] {
        for x in 0..pw {
            img.put_pixel(ox + x, oy, grid_color);
            img.put_pixel(ox + x, oy + ph - 1, grid_color);
        }
        for y in 0..ph {
            img.put_pixel(ox, oy + y, grid_color);
            img.put_pixel(ox + pw - 1, oy + y, grid_color);
        }
    }

    let role_rgb = |r: CellRole| -> [u8; 3] {
        match r {
            CellRole::Seed => [255, 220, 100],
            CellRole::Axis => [230, 130, 90],
            CellRole::Inside => [110, 180, 220],
            CellRole::Other => [160, 160, 170],
            CellRole::Empty => [60, 60, 60],
        }
    };

    // For each panel, project cells by dropping one axis.
    // Pick the cell closest to the viewer along the dropped axis.
    let mut top: HashMap<(i32, i32), (i32, CellRole)> = HashMap::new();   // xy: drop z, viewer at +z (max z = closest)
    let mut front: HashMap<(i32, i32), (i32, CellRole)> = HashMap::new(); // xz: drop y, viewer at -y (min y = closest)
    let mut side: HashMap<(i32, i32), (i32, CellRole)> = HashMap::new();  // yz: drop x, viewer at +x (max x = closest)

    for (&pos, _) in &world.cells {
        let role = world.cell_role(pos);
        // top: keep cell with largest z (top-down view)
        let k = (pos.0, pos.1);
        match top.get(&k) {
            Some(&(z, _)) if z >= pos.2 => {}
            _ => {
                top.insert(k, (pos.2, role));
            }
        }
        // front: keep cell with smallest y (looking +y from -y)
        let k = (pos.0, pos.2);
        match front.get(&k) {
            Some(&(y, _)) if y <= pos.1 => {}
            _ => {
                front.insert(k, (pos.1, role));
            }
        }
        // side: keep cell with largest x (looking -x from +x)
        let k = (pos.1, pos.2);
        match side.get(&k) {
            Some(&(x, _)) if x >= pos.0 => {}
            _ => {
                side.insert(k, (pos.0, role));
            }
        }
    }

    let mut paint = |ox: u32, oy: u32, px: i32, py: i32, color: [u8; 3]| {
        if px < 0 || py < 0 {
            return;
        }
        let bx = ox + px as u32 * VOXEL_PX;
        let by = oy + py as u32 * VOXEL_PX;
        for dy in 0..VOXEL_PX {
            for dx in 0..VOXEL_PX {
                let x = bx + dx;
                let y = by + dy;
                if x < total_w && y < total_h {
                    img.put_pixel(x, y, Rgb(color));
                }
            }
        }
    };

    // top-down (xy): x → column, y → row, +y up means we flip
    for (&(x, y), &(_, role)) in &top {
        let px = x + world::WORLD_HALF;
        let py = world::WORLD_HALF - y;
        paint(p_xy.0, p_xy.1, px, py, role_rgb(role));
    }

    // front (xz): x → column, z → row, +z up means flip
    for (&(x, z), &(_, role)) in &front {
        let px = x + world::WORLD_HALF;
        let py = world::WORLD_Z_MAX - z;
        paint(p_xz.0, p_xz.1, px, py, role_rgb(role));
    }

    // side (yz)
    for (&(y, z), &(_, role)) in &side {
        let px = y + world::WORLD_HALF;
        let py = world::WORLD_Z_MAX - z;
        paint(p_yz.0, p_yz.1, px, py, role_rgb(role));
    }

    img
}

fn render_perspective(
    world: &World,
    width: u32,
    height: u32,
) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    use image::{ImageBuffer, Rgb};
    use tiny_skia::{Color, FillRule, Paint, PathBuilder, Pixmap, Stroke, Transform};

    let mut pixmap = Pixmap::new(width, height).unwrap();
    pixmap.fill(Color::from_rgba8(18, 20, 26, 255));

    if world.cells.is_empty() {
        return rgba_to_rgb(&pixmap);
    }

    // bounding sphere around all cells (cube corners ±0.5)
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for (&pos, _) in &world.cells {
        let p = Vec3::new(pos.0 as f32, pos.1 as f32, pos.2 as f32);
        min = min.min(p - Vec3::splat(0.5));
        max = max.max(p + Vec3::splat(0.5));
    }
    let center = (min + max) * 0.5;
    let target = center;

    // editor defaults
    let yaw: f32 = 0.9;
    let pitch: f32 = 0.5;
    let fov_v_deg: f32 = 50.0;
    let aspect = width as f32 / height as f32;
    let fov_v = fov_v_deg.to_radians();
    let fov_h = 2.0 * ((fov_v * 0.5).tan() * aspect).atan();

    let cam_dir = Vec3::new(
        pitch.cos() * yaw.cos(),
        pitch.cos() * yaw.sin(),
        pitch.sin(),
    );

    // pick distance so geometry occupies ~75% of viewbox.
    // project corners onto plane perpendicular to cam_dir, find max perpendicular extent.
    let up_world = Vec3::Z;
    let right = cam_dir.cross(up_world).normalize();
    let up = right.cross(cam_dir).normalize();
    let mut max_h: f32 = 0.0;
    let mut max_v: f32 = 0.0;
    let mut max_along: f32 = 0.0;
    for cx in [min.x, max.x] {
        for cy in [min.y, max.y] {
            for cz in [min.z, max.z] {
                let d = Vec3::new(cx, cy, cz) - target;
                max_h = max_h.max(d.dot(right).abs());
                max_v = max_v.max(d.dot(up).abs());
                max_along = max_along.max(d.dot(cam_dir).abs());
            }
        }
    }
    let target_frac = 0.75;
    // need: max_h <= dist * tan(fov_h/2) * target_frac → dist >= max_h / (tan * target_frac)
    let need_h = max_h / ((fov_h * 0.5).tan() * target_frac);
    let need_v = max_v / ((fov_v * 0.5).tan() * target_frac);
    let dist = (need_h.max(need_v) + max_along).max(8.0);

    let cam_pos = target + cam_dir * dist;
    let view = Mat4::look_at_rh(cam_pos, target, Vec3::Z);
    let proj = Mat4::perspective_rh(fov_v, aspect, 0.1, 4000.0);
    let vp = proj * view;

    let project = |p: Vec3| -> Option<(f32, f32)> {
        let v = vp * Vec4::new(p.x, p.y, p.z, 1.0);
        if v.w <= 0.05 {
            return None;
        }
        let ndc_x = v.x / v.w;
        let ndc_y = v.y / v.w;
        Some((
            width as f32 * 0.5 + ndc_x * width as f32 * 0.5,
            height as f32 * 0.5 - ndc_y * height as f32 * 0.5,
        ))
    };

    // Ground grid (subtle, on z=0 plane)
    {
        let mut paint = Paint::default();
        paint.set_color(Color::from_rgba8(40, 44, 52, 255));
        paint.anti_alias = true;
        let stroke = Stroke {
            width: 0.6,
            ..Default::default()
        };
        let g = world::WORLD_HALF;
        for i in (-g..=g).step_by(8) {
            for &(a3, b3) in &[
                (Vec3::new(i as f32, -g as f32, 0.0), Vec3::new(i as f32, g as f32, 0.0)),
                (Vec3::new(-g as f32, i as f32, 0.0), Vec3::new(g as f32, i as f32, 0.0)),
            ] {
                if let (Some(a), Some(b)) = (project(a3), project(b3)) {
                    let mut pb = PathBuilder::new();
                    pb.move_to(a.0, a.1);
                    pb.line_to(b.0, b.1);
                    if let Some(path) = pb.finish() {
                        pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
                    }
                }
            }
        }
    }

    // Collect, depth-sort, paint quads back-to-front
    struct Q {
        pts: [(f32, f32); 4],
        depth: f32,
        color: [u8; 3],
    }
    let mut quads: Vec<Q> = Vec::with_capacity(world.cells.len() * 3);
    const FACES: [(usize, f32); 6] = [
        (0, 1.0),
        (0, -1.0),
        (1, 1.0),
        (1, -1.0),
        (2, 1.0),
        (2, -1.0),
    ];
    for (&pos, _) in &world.cells {
        let center_c = Vec3::new(pos.0 as f32, pos.1 as f32, pos.2 as f32);
        let to_cam = (cam_pos - center_c).normalize_or_zero();
        let role = world.cell_role(pos);
        let base = match role {
            CellRole::Seed => [255, 220, 100],
            CellRole::Axis => [230, 130, 90],
            CellRole::Inside => [110, 180, 220],
            CellRole::Other => [160, 160, 170],
            CellRole::Empty => [60, 60, 60],
        };
        for &(axis, sign) in &FACES {
            let mut n = Vec3::ZERO;
            match axis {
                0 => n.x = sign,
                1 => n.y = sign,
                _ => n.z = sign,
            }
            if n.dot(to_cam) <= 0.01 {
                continue;
            }
            let half = 0.5_f32;
            let face_center = center_c + n * half;
            let (uu, vv) = match axis {
                0 => (Vec3::Y, Vec3::Z),
                1 => (Vec3::X, Vec3::Z),
                _ => (Vec3::X, Vec3::Y),
            };
            let corners3 = [
                face_center - uu * half - vv * half,
                face_center + uu * half - vv * half,
                face_center + uu * half + vv * half,
                face_center - uu * half + vv * half,
            ];
            let mut corners2 = [(0.0_f32, 0.0_f32); 4];
            let mut all_in = true;
            for (i, c) in corners3.iter().enumerate() {
                match project(*c) {
                    Some(p) => corners2[i] = p,
                    None => {
                        all_in = false;
                        break;
                    }
                }
            }
            if !all_in {
                continue;
            }
            let depth = (face_center - cam_pos).length_squared();
            let shade = match (axis, sign as i32) {
                (2, 1) => 1.0,
                (2, -1) => 0.45,
                (0, 1) => 0.85,
                (0, -1) => 0.65,
                (1, 1) => 0.75,
                (1, -1) => 0.55,
                _ => 0.7,
            };
            let color = [
                (base[0] as f32 * shade).clamp(0.0, 255.0) as u8,
                (base[1] as f32 * shade).clamp(0.0, 255.0) as u8,
                (base[2] as f32 * shade).clamp(0.0, 255.0) as u8,
            ];
            quads.push(Q {
                pts: corners2,
                depth,
                color,
            });
        }
    }
    quads.sort_by(|a, b| {
        b.depth
            .partial_cmp(&a.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut fill_paint = Paint::default();
    fill_paint.anti_alias = true;
    let mut edge_paint = Paint::default();
    edge_paint.set_color(Color::from_rgba8(0, 0, 0, 120));
    edge_paint.anti_alias = true;
    let edge_stroke = Stroke {
        width: 0.5,
        ..Default::default()
    };

    for q in &quads {
        let mut pb = PathBuilder::new();
        pb.move_to(q.pts[0].0, q.pts[0].1);
        pb.line_to(q.pts[1].0, q.pts[1].1);
        pb.line_to(q.pts[2].0, q.pts[2].1);
        pb.line_to(q.pts[3].0, q.pts[3].1);
        pb.close();
        if let Some(path) = pb.finish() {
            fill_paint.set_color(Color::from_rgba8(q.color[0], q.color[1], q.color[2], 255));
            pixmap.fill_path(
                &path,
                &fill_paint,
                FillRule::Winding,
                Transform::identity(),
                None,
            );
            pixmap.stroke_path(&path, &edge_paint, &edge_stroke, Transform::identity(), None);
        }
    }

    rgba_to_rgb(&pixmap)
}

fn rgba_to_rgb(pm: &tiny_skia::Pixmap) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let w = pm.width();
    let h = pm.height();
    let data = pm.data();
    let mut out = image::ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            out.put_pixel(x, y, image::Rgb([data[i], data[i + 1], data[i + 2]]));
        }
    }
    out
}

struct App {
    world: World,
    code: String,
    code_path: PathBuf,
    running: bool,
    speed_hz: f32,
    last_tick_at: Instant,
    yaw: f32,
    pitch: f32,
    dist: f32,
    target: Vec3,
    error: Option<String>,
}

impl App {
    fn new(code_path: PathBuf) -> Self {
        let code = fs::read_to_string(&code_path).unwrap_or_default();
        let program = lisp::parse_program(&code).unwrap_or_default();
        Self {
            world: World::new(program),
            code,
            code_path,
            running: false,
            speed_hz: 4.0,
            last_tick_at: Instant::now(),
            yaw: 0.9,
            pitch: 0.5,
            dist: 60.0,
            target: Vec3::new(0.0, 0.0, 4.0),
            error: None,
        }
    }

    fn reload(&mut self) {
        match lisp::parse_program(&self.code) {
            Ok(p) => {
                self.world.reset(p);
                self.error = None;
            }
            Err(e) => self.error = Some(e),
        }
    }

    fn save(&mut self) {
        if let Err(e) = fs::write(&self.code_path, &self.code) {
            self.error = Some(format!("save: {}", e));
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.running && self.last_tick_at.elapsed().as_secs_f32() > 1.0 / self.speed_hz {
            if let Err(e) = self.world.step() {
                self.error = Some(e);
                self.running = false;
            }
            self.last_tick_at = Instant::now();
        }
        if self.running {
            ctx.request_repaint_after(std::time::Duration::from_millis(
                ((1000.0 / self.speed_hz) as u64).max(16),
            ));
        }

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let label = if self.running { "pause" } else { "play" };
                if ui.button(label).clicked() {
                    self.running = !self.running;
                    self.last_tick_at = Instant::now();
                }
                if ui.button("step").clicked() {
                    if let Err(e) = self.world.step() {
                        self.error = Some(e);
                    }
                }
                if ui.button("reset").clicked() {
                    self.reload();
                }
                ui.add(egui::Slider::new(&mut self.speed_hz, 0.5..=60.0).text("Hz"));
                ui.separator();
                ui.label(format!(
                    "tick {} · cells {}{}",
                    self.world.tick,
                    self.world.cells.len(),
                    if self.world.fixed_point {
                        " · fixed-point"
                    } else {
                        ""
                    }
                ));
                if let Some(e) = self.error.clone() {
                    ui.colored_label(egui::Color32::from_rgb(220, 80, 80), e);
                }
            });
        });

        egui::SidePanel::left("code")
            .default_width(440.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let name = self
                        .code_path
                        .file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "(no file)".into());
                    ui.heading(name);
                    if ui.button("save & reload").clicked() {
                        self.save();
                        self.reload();
                    }
                    if ui.button("reload from disk").clicked() {
                        match fs::read_to_string(&self.code_path) {
                            Ok(s) => {
                                self.code = s;
                                self.reload();
                            }
                            Err(e) => self.error = Some(format!("read: {}", e)),
                        }
                    }
                });
                ui.label(self.code_path.display().to_string());
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let mut layouter = |ui: &egui::Ui, text: &dyn egui::TextBuffer, wrap_width: f32| {
                        let mut job = lisp_highlight(text.as_str(), ui.style());
                        job.wrap.max_width = wrap_width;
                        ui.fonts_mut(|f| f.layout_job(job))
                    };
                    ui.add(
                        egui::TextEdit::multiline(&mut self.code)
                            .code_editor()
                            .desired_width(f32::INFINITY)
                            .desired_rows(40)
                            .layouter(&mut layouter),
                    );
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_3d(ui);
        });
    }
}

impl App {
    fn draw_3d(&mut self, ui: &mut egui::Ui) {
        let avail = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(avail, egui::Sense::click_and_drag());
        let painter = ui.painter_at(rect);

        if response.dragged() {
            let d = response.drag_delta();
            self.yaw -= d.x * 0.008;
            self.pitch = (self.pitch + d.y * 0.008).clamp(-1.45, 1.45);
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.0 {
                self.dist = (self.dist * (1.0 - scroll * 0.001)).clamp(8.0, 400.0);
            }
        }

        let cam_offset = Vec3::new(
            self.dist * self.pitch.cos() * self.yaw.cos(),
            self.dist * self.pitch.cos() * self.yaw.sin(),
            self.dist * self.pitch.sin(),
        );
        let cam_pos = self.target + cam_offset;
        let view = Mat4::look_at_rh(cam_pos, self.target, Vec3::Z);
        let aspect = (rect.width() / rect.height().max(1.0)).max(0.01);
        let proj = Mat4::perspective_rh(50.0_f32.to_radians(), aspect, 0.1, 2000.0);
        let vp = proj * view;

        painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(18, 20, 26));

        let project = |p: Vec3| -> Option<egui::Pos2> {
            let v = vp * Vec4::new(p.x, p.y, p.z, 1.0);
            if v.w <= 0.05 {
                return None;
            }
            let ndc_x = v.x / v.w;
            let ndc_y = v.y / v.w;
            Some(egui::pos2(
                rect.center().x + ndc_x * rect.width() * 0.5,
                rect.center().y - ndc_y * rect.height() * 0.5,
            ))
        };

        // ground grid
        let grid_color = egui::Color32::from_rgb(40, 44, 52);
        let g = world::WORLD_HALF;
        for i in (-g..=g).step_by(8) {
            if let (Some(a), Some(b)) = (
                project(Vec3::new(i as f32, -g as f32, 0.0)),
                project(Vec3::new(i as f32, g as f32, 0.0)),
            ) {
                painter.line_segment([a, b], egui::Stroke::new(0.5, grid_color));
            }
            if let (Some(a), Some(b)) = (
                project(Vec3::new(-g as f32, i as f32, 0.0)),
                project(Vec3::new(g as f32, i as f32, 0.0)),
            ) {
                painter.line_segment([a, b], egui::Stroke::new(0.5, grid_color));
            }
        }

        // axes
        if let (Some(o), Some(x), Some(y), Some(z)) = (
            project(Vec3::ZERO),
            project(Vec3::new(8.0, 0.0, 0.0)),
            project(Vec3::new(0.0, 8.0, 0.0)),
            project(Vec3::new(0.0, 0.0, 8.0)),
        ) {
            painter.line_segment(
                [o, x],
                egui::Stroke::new(2.0, egui::Color32::from_rgb(220, 80, 80)),
            );
            painter.line_segment(
                [o, y],
                egui::Stroke::new(2.0, egui::Color32::from_rgb(80, 200, 100)),
            );
            painter.line_segment(
                [o, z],
                egui::Stroke::new(2.0, egui::Color32::from_rgb(90, 130, 230)),
            );
        }

        // collect visible faces, depth sort, then paint back-to-front
        let mut quads: Vec<Quad> = Vec::with_capacity(self.world.cells.len() * 3);

        const FACES: [(usize, f32); 6] = [
            (0, 1.0),
            (0, -1.0),
            (1, 1.0),
            (1, -1.0),
            (2, 1.0),
            (2, -1.0),
        ];

        for (&pos, _cell) in self.world.cells.iter() {
            let center = Vec3::new(pos.0 as f32, pos.1 as f32, pos.2 as f32);
            let to_cam = (cam_pos - center).normalize_or_zero();
            let role = self.world.cell_role(pos);
            let base = role_color(role);

            for &(axis, sign) in &FACES {
                let mut n = Vec3::ZERO;
                match axis {
                    0 => n.x = sign,
                    1 => n.y = sign,
                    _ => n.z = sign,
                }
                if n.dot(to_cam) <= 0.01 {
                    continue;
                }
                let half = 0.5_f32;
                let face_center = center + n * half;
                let (u, v) = match axis {
                    0 => (Vec3::Y, Vec3::Z),
                    1 => (Vec3::X, Vec3::Z),
                    _ => (Vec3::X, Vec3::Y),
                };
                let corners3 = [
                    face_center - u * half - v * half,
                    face_center + u * half - v * half,
                    face_center + u * half + v * half,
                    face_center - u * half + v * half,
                ];
                let mut corners2 = [egui::Pos2::ZERO; 4];
                let mut all_in = true;
                for (i, c) in corners3.iter().enumerate() {
                    match project(*c) {
                        Some(p) => corners2[i] = p,
                        None => {
                            all_in = false;
                            break;
                        }
                    }
                }
                if !all_in {
                    continue;
                }
                let depth = (face_center - cam_pos).length_squared();
                let shade = match (axis, sign as i32) {
                    (2, 1) => 1.0,
                    (2, -1) => 0.45,
                    (0, 1) => 0.85,
                    (0, -1) => 0.65,
                    (1, 1) => 0.75,
                    (1, -1) => 0.55,
                    _ => 0.7,
                };
                let color = shade_color(base, shade);
                quads.push(Quad {
                    pts: corners2,
                    depth,
                    color,
                });
            }
        }

        quads.sort_by(|a, b| {
            b.depth
                .partial_cmp(&a.depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let edge = egui::Stroke::new(0.5, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 120));
        for q in &quads {
            let pts = vec![q.pts[0], q.pts[1], q.pts[2], q.pts[3]];
            painter.add(egui::Shape::convex_polygon(pts, q.color, edge));
        }

        painter.text(
            rect.left_bottom() + egui::vec2(8.0, -8.0),
            egui::Align2::LEFT_BOTTOM,
            "drag to orbit · scroll to zoom",
            egui::FontId::proportional(11.0),
            egui::Color32::from_rgb(140, 145, 155),
        );
    }
}

struct Quad {
    pts: [egui::Pos2; 4],
    depth: f32,
    color: egui::Color32,
}

fn lisp_highlight(src: &str, style: &egui::Style) -> egui::text::LayoutJob {
    use egui::text::{LayoutJob, TextFormat};

    let font = egui::FontId::monospace(13.0);
    let c_default = style.visuals.text_color();
    let c_comment = egui::Color32::from_rgb(110, 120, 130);
    let c_string = egui::Color32::from_rgb(180, 210, 130);
    let c_number = egui::Color32::from_rgb(220, 170, 110);
    let c_paren = egui::Color32::from_rgb(150, 155, 170);
    let c_keyword = egui::Color32::from_rgb(200, 130, 200);
    let c_builtin = egui::Color32::from_rgb(110, 190, 230);
    let c_bool = egui::Color32::from_rgb(220, 170, 110);
    let c_kwarg = egui::Color32::from_rgb(230, 200, 120);

    const KEYWORDS: &[&str] = &[
        "define", "set!", "if", "begin", "and", "or", "not", "lambda", "let", "cond", "else",
        "quote",
    ];
    const BUILTINS: &[&str] = &[
        "emit-gradient",
        "read-gradient",
        "replicate-toward",
        "neighbor-exists",
    ];

    let mut job = LayoutJob::default();
    let bytes = src.as_bytes();
    let mut i = 0;
    let push = |job: &mut LayoutJob, s: &str, color: egui::Color32, font: &egui::FontId| {
        job.append(
            s,
            0.0,
            TextFormat {
                font_id: font.clone(),
                color,
                ..Default::default()
            },
        );
    };

    while i < bytes.len() {
        let b = bytes[i];
        if b == b';' {
            let start = i;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            push(&mut job, &src[start..i], c_comment, &font);
        } else if b == b'"' {
            let start = i;
            i += 1;
            while i < bytes.len() && bytes[i] != b'"' {
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if i < bytes.len() {
                i += 1;
            }
            push(&mut job, &src[start..i], c_string, &font);
        } else if b == b'(' || b == b')' {
            push(&mut job, &src[i..i + 1], c_paren, &font);
            i += 1;
        } else if b.is_ascii_whitespace() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            push(&mut job, &src[start..i], c_default, &font);
        } else {
            let start = i;
            while i < bytes.len() {
                let c = bytes[i];
                if c.is_ascii_whitespace() || c == b'(' || c == b')' || c == b';' || c == b'"' {
                    break;
                }
                i += 1;
            }
            let tok = &src[start..i];
            let first = tok.as_bytes()[0];
            let color = if tok == "#t" || tok == "#f" {
                c_bool
            } else if first == b':' {
                c_kwarg
            } else if first == b'-' && tok.len() > 1 && tok.as_bytes()[1].is_ascii_digit()
                || first.is_ascii_digit()
                || (first == b'+' && tok.len() > 1 && tok.as_bytes()[1].is_ascii_digit())
            {
                c_number
            } else if KEYWORDS.contains(&tok) {
                c_keyword
            } else if BUILTINS.contains(&tok) {
                c_builtin
            } else {
                c_default
            };
            push(&mut job, tok, color, &font);
        }
    }
    job
}

fn role_color(r: CellRole) -> egui::Color32 {
    match r {
        CellRole::Seed => egui::Color32::from_rgb(255, 220, 100),
        CellRole::Axis => egui::Color32::from_rgb(230, 130, 90),
        CellRole::Inside => egui::Color32::from_rgb(110, 180, 220),
        CellRole::Other => egui::Color32::from_rgb(160, 160, 170),
        CellRole::Empty => egui::Color32::from_rgb(60, 60, 60),
    }
}

fn shade_color(c: egui::Color32, k: f32) -> egui::Color32 {
    let k = k.clamp(0.0, 1.5);
    egui::Color32::from_rgb(
        ((c.r() as f32 * k).clamp(0.0, 255.0)) as u8,
        ((c.g() as f32 * k).clamp(0.0, 255.0)) as u8,
        ((c.b() as f32 * k).clamp(0.0, 255.0)) as u8,
    )
}
