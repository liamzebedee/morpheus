mod lisp;
mod world;

use eframe::egui;
use glam::{Mat4, Vec3, Vec4};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use world::{CellRole, World};

fn main() -> eframe::Result<()> {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1500.0, 950.0]),
        ..Default::default()
    };
    eframe::run_native(
        "morpheus",
        opts,
        Box::new(|_cc| Ok(Box::new(App::new()))),
    )
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
    fn new() -> Self {
        let code_path = std::env::current_dir()
            .unwrap_or_default()
            .join("cylinder.local");
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
                    ui.heading("cylinder.local");
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
