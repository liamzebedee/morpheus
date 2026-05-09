//! Flattens the parsed `lisp::Val` AST into a GPU-friendly typed-AST form.
//!
//! Each node is a fixed 16-byte record with a tag and three u32 payload words.
//! Symbol names are pre-resolved at flatten time:
//!   - User variables       -> slot index 0..MAX_SLOTS
//!   - Gradient names       -> grad id   0..MAX_GRADS
//!   - Direction tokens     -> dir id    0..5
//!   - Special forms / ops  -> their own tags (no symbol lookup at runtime)
//!
//! After flattening, the GPU shader never sees strings; it dispatches on
//! `node.tag` and indexes pre-baked tables.

use crate::lisp;
use std::collections::HashMap;

// --- tags ---
pub const TAG_NUM: u32 = 0;
pub const TAG_BOOL: u32 = 1;
pub const TAG_SYM_VAR: u32 = 2;
pub const TAG_SPECIAL_IF: u32 = 3;
pub const TAG_SPECIAL_BEGIN: u32 = 4;
pub const TAG_SPECIAL_AND: u32 = 5;
pub const TAG_SPECIAL_OR: u32 = 6;
pub const TAG_SPECIAL_NOT: u32 = 7;
pub const TAG_SPECIAL_DEFINE: u32 = 8;
pub const TAG_SPECIAL_SET: u32 = 9;
pub const TAG_OP_ADD: u32 = 10;
pub const TAG_OP_SUB: u32 = 11;
pub const TAG_OP_MUL: u32 = 12;
pub const TAG_OP_DIV: u32 = 13;
pub const TAG_OP_LT: u32 = 14;
pub const TAG_OP_GT: u32 = 15;
pub const TAG_OP_LE: u32 = 16;
pub const TAG_OP_GE: u32 = 17;
pub const TAG_OP_EQ: u32 = 18;
pub const TAG_BUILTIN_EMIT: u32 = 19;
pub const TAG_BUILTIN_READ: u32 = 20;
pub const TAG_BUILTIN_NEIGHBOR: u32 = 21;
pub const TAG_BUILTIN_REPLICATE: u32 = 22;

pub const NIL_INDEX: u32 = 0xFFFF_FFFF;

pub const MAX_SLOTS: u32 = 32;
pub const MAX_GRADS: u32 = 16;
pub const MAX_OVERRIDES: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Node {
    pub tag: u32,
    pub a: u32,
    pub b: u32,
    pub c: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AstOverride {
    pub slot: u32,
    pub expr_node: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ReplicateSite {
    pub dir: u32,
    pub n_overrides: u32,
    pub _pad: [u32; 2],
    pub overrides: [AstOverride; MAX_OVERRIDES],
}

#[derive(Debug)]
pub struct FlatProgram {
    pub nodes: Vec<Node>,
    pub children: Vec<u32>,
    pub replicate_sites: Vec<ReplicateSite>,
    pub top_level: Vec<u32>,
    pub slot_table: HashMap<String, u32>,
    pub grad_table: HashMap<String, u32>,
    pub num_slots: u32,
    pub num_grads: u32,
}

pub fn flatten(program: &[lisp::Val]) -> Result<FlatProgram, String> {
    let mut f = FlatProgram {
        nodes: Vec::new(),
        children: Vec::new(),
        replicate_sites: Vec::new(),
        top_level: Vec::new(),
        slot_table: HashMap::new(),
        grad_table: HashMap::new(),
        num_slots: 0,
        num_grads: 0,
    };
    for expr in program {
        let idx = flatten_expr(&mut f, expr)?;
        f.top_level.push(idx);
    }
    // Ensure non-empty buffers (wgpu rejects zero-size storage)
    if f.nodes.is_empty() {
        f.nodes.push(Node { tag: TAG_NUM, a: 0, b: 0, c: 0 });
    }
    if f.children.is_empty() {
        f.children.push(0);
    }
    if f.replicate_sites.is_empty() {
        f.replicate_sites.push(ReplicateSite {
            dir: 0,
            n_overrides: 0,
            _pad: [0; 2],
            overrides: [AstOverride { slot: 0, expr_node: 0 }; MAX_OVERRIDES],
        });
    }
    if f.top_level.is_empty() {
        let n = emit(&mut f, Node { tag: TAG_NUM, a: 0, b: 0, c: 0 });
        f.top_level.push(n);
    }
    Ok(f)
}

fn emit(f: &mut FlatProgram, n: Node) -> u32 {
    let idx = f.nodes.len() as u32;
    f.nodes.push(n);
    idx
}

fn ensure_slot(f: &mut FlatProgram, name: &str) -> Result<u32, String> {
    if let Some(&s) = f.slot_table.get(name) {
        return Ok(s);
    }
    let s = f.num_slots;
    if s >= MAX_SLOTS {
        return Err(format!(
            "too many state variables (max {}): adding '{}'",
            MAX_SLOTS, name
        ));
    }
    f.slot_table.insert(name.to_string(), s);
    f.num_slots += 1;
    Ok(s)
}

fn ensure_grad(f: &mut FlatProgram, name: &str) -> Result<u32, String> {
    if let Some(&g) = f.grad_table.get(name) {
        return Ok(g);
    }
    let g = f.num_grads;
    if g >= MAX_GRADS {
        return Err(format!(
            "too many gradient names (max {}): adding '{}'",
            MAX_GRADS, name
        ));
    }
    f.grad_table.insert(name.to_string(), g);
    f.num_grads += 1;
    Ok(g)
}

fn parse_dir(s: &str) -> Option<u32> {
    match s {
        "+x" => Some(0),
        "-x" => Some(1),
        "+y" => Some(2),
        "-y" => Some(3),
        "+z" => Some(4),
        "-z" => Some(5),
        _ => None,
    }
}

fn flatten_expr(f: &mut FlatProgram, expr: &lisp::Val) -> Result<u32, String> {
    match expr {
        lisp::Val::Nil => Ok(emit(f, Node { tag: TAG_NUM, a: 0, b: 0, c: 0 })),
        lisp::Val::Bool(b) => Ok(emit(
            f,
            Node {
                tag: TAG_BOOL,
                a: if *b { 1 } else { 0 },
                b: 0,
                c: 0,
            },
        )),
        lisp::Val::Num(n) => Ok(emit(
            f,
            Node {
                tag: TAG_NUM,
                a: f32::to_bits(*n as f32),
                b: 0,
                c: 0,
            },
        )),
        lisp::Val::Sym(s) => {
            if parse_dir(s).is_some() {
                return Err(format!("direction symbol '{}' used as expression", s));
            }
            let slot = ensure_slot(f, s)?;
            Ok(emit(
                f,
                Node {
                    tag: TAG_SYM_VAR,
                    a: slot,
                    b: 0,
                    c: 0,
                },
            ))
        }
        lisp::Val::List(items) => {
            if items.is_empty() {
                return Ok(emit(f, Node { tag: TAG_NUM, a: 0, b: 0, c: 0 }));
            }
            let head = match &items[0] {
                lisp::Val::Sym(s) => s.clone(),
                _ => return Err("non-symbol head in list".into()),
            };
            let args = &items[1..];
            match head.as_str() {
                "define" => flatten_define(f, args),
                "set!" => flatten_set(f, args),
                "if" => flatten_if(f, args),
                "begin" => flatten_variadic(f, args, TAG_SPECIAL_BEGIN),
                "and" => flatten_variadic(f, args, TAG_SPECIAL_AND),
                "or" => flatten_variadic(f, args, TAG_SPECIAL_OR),
                "not" => flatten_unary(f, args, TAG_SPECIAL_NOT),
                "+" => flatten_variadic(f, args, TAG_OP_ADD),
                "*" => flatten_variadic(f, args, TAG_OP_MUL),
                "-" => flatten_variadic(f, args, TAG_OP_SUB),
                "/" => flatten_variadic(f, args, TAG_OP_DIV),
                "<" => flatten_binary(f, args, TAG_OP_LT),
                ">" => flatten_binary(f, args, TAG_OP_GT),
                "<=" => flatten_binary(f, args, TAG_OP_LE),
                ">=" => flatten_binary(f, args, TAG_OP_GE),
                "=" => flatten_binary(f, args, TAG_OP_EQ),
                "emit-gradient" => flatten_emit(f, args),
                "read-gradient" => flatten_read(f, args),
                "neighbor-exists" => flatten_neighbor(f, args),
                "replicate-toward" => flatten_replicate(f, args),
                _ => Err(format!("unknown form: {}", head)),
            }
        }
    }
}

fn flatten_define(f: &mut FlatProgram, args: &[lisp::Val]) -> Result<u32, String> {
    if args.len() != 2 {
        return Err("define needs 2 args".into());
    }
    let name = match &args[0] {
        lisp::Val::Sym(s) => s.clone(),
        _ => return Err("define name must be symbol".into()),
    };
    let slot = ensure_slot(f, &name)?;
    let expr = flatten_expr(f, &args[1])?;
    Ok(emit(
        f,
        Node {
            tag: TAG_SPECIAL_DEFINE,
            a: slot,
            b: expr,
            c: 0,
        },
    ))
}

fn flatten_set(f: &mut FlatProgram, args: &[lisp::Val]) -> Result<u32, String> {
    if args.len() != 2 {
        return Err("set! needs 2 args".into());
    }
    let name = match &args[0] {
        lisp::Val::Sym(s) => s.clone(),
        _ => return Err("set! name must be symbol".into()),
    };
    let slot = ensure_slot(f, &name)?;
    let expr = flatten_expr(f, &args[1])?;
    Ok(emit(
        f,
        Node {
            tag: TAG_SPECIAL_SET,
            a: slot,
            b: expr,
            c: 0,
        },
    ))
}

fn flatten_if(f: &mut FlatProgram, args: &[lisp::Val]) -> Result<u32, String> {
    if args.len() < 2 || args.len() > 3 {
        return Err("if needs 2 or 3 args".into());
    }
    let cond = flatten_expr(f, &args[0])?;
    let then = flatten_expr(f, &args[1])?;
    let else_ = if args.len() == 3 {
        flatten_expr(f, &args[2])?
    } else {
        NIL_INDEX
    };
    Ok(emit(
        f,
        Node {
            tag: TAG_SPECIAL_IF,
            a: cond,
            b: then,
            c: else_,
        },
    ))
}

fn flatten_variadic(f: &mut FlatProgram, args: &[lisp::Val], tag: u32) -> Result<u32, String> {
    let mut child_idxs = Vec::with_capacity(args.len());
    for arg in args {
        child_idxs.push(flatten_expr(f, arg)?);
    }
    let start = f.children.len() as u32;
    f.children.extend(child_idxs);
    Ok(emit(
        f,
        Node {
            tag,
            a: start,
            b: args.len() as u32,
            c: 0,
        },
    ))
}

fn flatten_unary(f: &mut FlatProgram, args: &[lisp::Val], tag: u32) -> Result<u32, String> {
    if args.len() != 1 {
        return Err("unary op needs 1 arg".into());
    }
    let arg = flatten_expr(f, &args[0])?;
    Ok(emit(
        f,
        Node {
            tag,
            a: arg,
            b: 0,
            c: 0,
        },
    ))
}

fn flatten_binary(f: &mut FlatProgram, args: &[lisp::Val], tag: u32) -> Result<u32, String> {
    if args.len() != 2 {
        return Err("binary op needs 2 args".into());
    }
    let lhs = flatten_expr(f, &args[0])?;
    let rhs = flatten_expr(f, &args[1])?;
    Ok(emit(
        f,
        Node {
            tag,
            a: lhs,
            b: rhs,
            c: 0,
        },
    ))
}

fn flatten_emit(f: &mut FlatProgram, args: &[lisp::Val]) -> Result<u32, String> {
    if args.is_empty() {
        return Err("emit-gradient needs at least a name".into());
    }
    let name = match &args[0] {
        lisp::Val::Sym(s) => s.clone(),
        _ => return Err("emit-gradient name must be symbol".into()),
    };
    let gid = ensure_grad(f, &name)?;
    let value_expr = if args.len() >= 2 {
        flatten_expr(f, &args[1])?
    } else {
        emit(
            f,
            Node {
                tag: TAG_NUM,
                a: f32::to_bits(1.0),
                b: 0,
                c: 0,
            },
        )
    };
    Ok(emit(
        f,
        Node {
            tag: TAG_BUILTIN_EMIT,
            a: gid,
            b: value_expr,
            c: 0,
        },
    ))
}

fn flatten_read(f: &mut FlatProgram, args: &[lisp::Val]) -> Result<u32, String> {
    if args.len() != 1 {
        return Err("read-gradient needs 1 arg".into());
    }
    let name = match &args[0] {
        lisp::Val::Sym(s) => s.clone(),
        _ => return Err("read-gradient name must be symbol".into()),
    };
    let gid = ensure_grad(f, &name)?;
    Ok(emit(
        f,
        Node {
            tag: TAG_BUILTIN_READ,
            a: gid,
            b: 0,
            c: 0,
        },
    ))
}

fn flatten_neighbor(f: &mut FlatProgram, args: &[lisp::Val]) -> Result<u32, String> {
    if args.len() != 1 {
        return Err("neighbor-exists needs 1 arg".into());
    }
    let dir = match &args[0] {
        lisp::Val::Sym(s) => parse_dir(s).ok_or_else(|| format!("bad direction: {}", s))?,
        _ => return Err("neighbor-exists arg must be direction symbol".into()),
    };
    Ok(emit(
        f,
        Node {
            tag: TAG_BUILTIN_NEIGHBOR,
            a: dir,
            b: 0,
            c: 0,
        },
    ))
}

fn flatten_replicate(f: &mut FlatProgram, args: &[lisp::Val]) -> Result<u32, String> {
    // (replicate-toward DIR)
    // (replicate-toward DIR ((NAME EXPR) ...))
    if args.is_empty() || args.len() > 2 {
        return Err("replicate-toward: (DIR) or (DIR OVERRIDES)".into());
    }
    let dir = match &args[0] {
        lisp::Val::Sym(s) => parse_dir(s).ok_or_else(|| format!("bad direction: {}", s))?,
        _ => return Err("replicate-toward dir must be direction symbol".into()),
    };
    let mut overrides = [AstOverride { slot: 0, expr_node: 0 }; MAX_OVERRIDES];
    let mut n_overrides: usize = 0;

    if args.len() == 2 {
        let pairs = match &args[1] {
            lisp::Val::List(l) => l.clone(),
            _ => return Err("replicate-toward overrides must be a list".into()),
        };
        for pair in pairs {
            let p = match pair {
                lisp::Val::List(l) if l.len() == 2 => l,
                _ => return Err("override must be (name expr)".into()),
            };
            let var_name = match &p[0] {
                lisp::Val::Sym(s) => s.clone(),
                _ => return Err("override name must be symbol".into()),
            };
            let slot = ensure_slot(f, &var_name)?;
            let expr_idx = flatten_expr(f, &p[1])?;
            if n_overrides >= MAX_OVERRIDES {
                return Err(format!("too many overrides (max {})", MAX_OVERRIDES));
            }
            overrides[n_overrides] = AstOverride {
                slot,
                expr_node: expr_idx,
            };
            n_overrides += 1;
        }
    }

    let site_idx = f.replicate_sites.len() as u32;
    f.replicate_sites.push(ReplicateSite {
        dir,
        n_overrides: n_overrides as u32,
        _pad: [0; 2],
        overrides,
    });
    Ok(emit(
        f,
        Node {
            tag: TAG_BUILTIN_REPLICATE,
            a: dir,
            b: site_idx,
            c: 0,
        },
    ))
}
