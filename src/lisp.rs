use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Nil,
    Bool(bool),
    Num(f64),
    Sym(String),
    List(Vec<Val>),
}

impl Val {
    pub fn truthy(&self) -> bool {
        !matches!(self, Val::Bool(false) | Val::Nil)
    }
    pub fn as_num(&self) -> Option<f64> {
        match self {
            Val::Num(n) => Some(*n),
            Val::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Dir {
    PX,
    NX,
    PY,
    NY,
    PZ,
    NZ,
}

impl Dir {
    pub fn offset(self) -> (i32, i32, i32) {
        match self {
            Dir::PX => (1, 0, 0),
            Dir::NX => (-1, 0, 0),
            Dir::PY => (0, 1, 0),
            Dir::NY => (0, -1, 0),
            Dir::PZ => (0, 0, 1),
            Dir::NZ => (0, 0, -1),
        }
    }
}

fn parse_dir(s: &str) -> Option<Dir> {
    match s {
        "+x" => Some(Dir::PX),
        "-x" => Some(Dir::NX),
        "+y" => Some(Dir::PY),
        "-y" => Some(Dir::NY),
        "+z" => Some(Dir::PZ),
        "-z" => Some(Dir::NZ),
        _ => None,
    }
}

pub trait Substrate {
    fn read_gradient(&self, name: &str) -> f64;
    fn emit_gradient(&mut self, name: &str, value: f64);
    fn replicate_toward(&mut self, dir: Dir, child_state: Vec<(String, Val)>);
    fn neighbor_exists(&self, dir: Dir) -> bool;
}

fn tokenize(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = src.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            ';' => {
                while let Some(&c) = chars.peek() {
                    if c == '\n' {
                        break;
                    }
                    chars.next();
                }
            }
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            '(' | ')' => {
                out.push(c.to_string());
                chars.next();
            }
            _ => {
                let mut s = String::new();
                while let Some(&c) = chars.peek() {
                    if matches!(c, '(' | ')' | ' ' | '\t' | '\n' | '\r' | ';') {
                        break;
                    }
                    s.push(c);
                    chars.next();
                }
                out.push(s);
            }
        }
    }
    out
}

fn parse_atom(tok: &str) -> Val {
    if tok == "#t" {
        return Val::Bool(true);
    }
    if tok == "#f" {
        return Val::Bool(false);
    }
    if let Ok(n) = tok.parse::<f64>() {
        return Val::Num(n);
    }
    Val::Sym(tok.to_string())
}

fn parse_expr(tokens: &[String], i: &mut usize) -> Result<Val, String> {
    if *i >= tokens.len() {
        return Err("unexpected EOF".into());
    }
    let t = tokens[*i].clone();
    *i += 1;
    if t == "(" {
        let mut items = Vec::new();
        while *i < tokens.len() && tokens[*i] != ")" {
            items.push(parse_expr(tokens, i)?);
        }
        if *i >= tokens.len() {
            return Err("missing )".into());
        }
        *i += 1;
        Ok(Val::List(items))
    } else if t == ")" {
        Err("unexpected )".into())
    } else {
        Ok(parse_atom(&t))
    }
}

pub fn parse_program(src: &str) -> Result<Vec<Val>, String> {
    let toks = tokenize(src);
    let mut i = 0;
    let mut exprs = Vec::new();
    while i < toks.len() {
        exprs.push(parse_expr(&toks, &mut i)?);
    }
    Ok(exprs)
}

pub type Env = Rc<RefCell<HashMap<String, Val>>>;

pub fn new_env() -> Env {
    Rc::new(RefCell::new(HashMap::new()))
}

pub fn eval(expr: &Val, env: &Env, sub: &mut dyn Substrate) -> Result<Val, String> {
    match expr {
        Val::Nil | Val::Bool(_) | Val::Num(_) => Ok(expr.clone()),
        Val::Sym(s) => env
            .borrow()
            .get(s)
            .cloned()
            .ok_or_else(|| format!("unbound: {}", s)),
        Val::List(items) => {
            if items.is_empty() {
                return Ok(Val::Nil);
            }
            if let Val::Sym(head) = &items[0] {
                let h = head.as_str();
                match h {
                    "define" => {
                        if items.len() != 3 {
                            return Err("define: expected (define name expr)".into());
                        }
                        let name = match &items[1] {
                            Val::Sym(n) => n.clone(),
                            _ => return Err("define: name must be a symbol".into()),
                        };
                        // define-once: leave existing bindings alone (this lets the substrate
                        // pre-seed cells with e.g. is-seed=#t, and lets state persist across ticks).
                        if env.borrow().contains_key(&name) {
                            return Ok(Val::Nil);
                        }
                        let v = eval(&items[2], env, sub)?;
                        env.borrow_mut().insert(name, v);
                        return Ok(Val::Nil);
                    }
                    "set!" => {
                        if items.len() != 3 {
                            return Err("set!: expected (set! name expr)".into());
                        }
                        let name = match &items[1] {
                            Val::Sym(n) => n.clone(),
                            _ => return Err("set!: name must be a symbol".into()),
                        };
                        let v = eval(&items[2], env, sub)?;
                        env.borrow_mut().insert(name, v);
                        return Ok(Val::Nil);
                    }
                    "if" => {
                        if items.len() < 3 || items.len() > 4 {
                            return Err("if: expected (if c t [e])".into());
                        }
                        let c = eval(&items[1], env, sub)?;
                        if c.truthy() {
                            return eval(&items[2], env, sub);
                        } else if items.len() == 4 {
                            return eval(&items[3], env, sub);
                        }
                        return Ok(Val::Nil);
                    }
                    "begin" => {
                        let mut last = Val::Nil;
                        for e in &items[1..] {
                            last = eval(e, env, sub)?;
                        }
                        return Ok(last);
                    }
                    "and" => {
                        let mut last = Val::Bool(true);
                        for e in &items[1..] {
                            last = eval(e, env, sub)?;
                            if !last.truthy() {
                                return Ok(Val::Bool(false));
                            }
                        }
                        return Ok(last);
                    }
                    "or" => {
                        for e in &items[1..] {
                            let v = eval(e, env, sub)?;
                            if v.truthy() {
                                return Ok(v);
                            }
                        }
                        return Ok(Val::Bool(false));
                    }
                    "not" => {
                        if items.len() != 2 {
                            return Err("not: 1 arg".into());
                        }
                        let v = eval(&items[1], env, sub)?;
                        return Ok(Val::Bool(!v.truthy()));
                    }
                    "emit-gradient" => {
                        if items.len() < 2 {
                            return Err("emit-gradient: need name".into());
                        }
                        let name = match &items[1] {
                            Val::Sym(n) => n.clone(),
                            _ => return Err("emit-gradient: name must be a symbol".into()),
                        };
                        let value = if items.len() >= 3 {
                            eval(&items[2], env, sub)?.as_num().unwrap_or(1.0)
                        } else {
                            1.0
                        };
                        sub.emit_gradient(&name, value);
                        return Ok(Val::Nil);
                    }
                    "read-gradient" => {
                        if items.len() != 2 {
                            return Err("read-gradient: need name".into());
                        }
                        let name = match &items[1] {
                            Val::Sym(n) => n.clone(),
                            _ => return Err("read-gradient: name must be a symbol".into()),
                        };
                        return Ok(Val::Num(sub.read_gradient(&name)));
                    }
                    "replicate-toward" => {
                        // (replicate-toward DIR)
                        // (replicate-toward DIR ((NAME EXPR) ...))
                        if items.len() < 2 || items.len() > 3 {
                            return Err("replicate-toward: (DIR) or (DIR OVERRIDES)".into());
                        }
                        let dir = match &items[1] {
                            Val::Sym(n) => parse_dir(n),
                            _ => None,
                        }
                        .ok_or_else(|| format!("bad direction: {:?}", items[1]))?;
                        let mut child_state: Vec<(String, Val)> = Vec::new();
                        if items.len() == 3 {
                            let pairs = match &items[2] {
                                Val::List(l) => l.clone(),
                                _ => return Err("replicate-toward overrides must be a list".into()),
                            };
                            for pair in pairs {
                                let p = match pair {
                                    Val::List(l) if l.len() == 2 => l,
                                    _ => return Err("override must be (name expr)".into()),
                                };
                                let name = match &p[0] {
                                    Val::Sym(n) => n.clone(),
                                    _ => return Err("override name must be symbol".into()),
                                };
                                let v = eval(&p[1], env, sub)?;
                                child_state.push((name, v));
                            }
                        }
                        sub.replicate_toward(dir, child_state);
                        return Ok(Val::Nil);
                    }
                    "neighbor-exists" => {
                        if items.len() != 2 {
                            return Err("neighbor-exists: need direction".into());
                        }
                        let d = match &items[1] {
                            Val::Sym(n) => parse_dir(n),
                            _ => None,
                        };
                        let dir = match d {
                            Some(d) => d,
                            None => return Err(format!("bad direction: {:?}", items[1])),
                        };
                        return Ok(Val::Bool(sub.neighbor_exists(dir)));
                    }
                    "+" | "-" | "*" | "/" | "<" | ">" | "<=" | ">=" | "=" => {
                        let mut args = Vec::with_capacity(items.len() - 1);
                        for e in &items[1..] {
                            args.push(eval(e, env, sub)?);
                        }
                        return apply_op(h, &args);
                    }
                    _ => {}
                }
            }
            Err(format!("unknown form: {:?}", items[0]))
        }
    }
}

fn apply_op(op: &str, args: &[Val]) -> Result<Val, String> {
    let nums: Vec<f64> = args
        .iter()
        .map(|a| a.as_num().ok_or_else(|| format!("{} needs numbers, got {:?}", op, a)))
        .collect::<Result<Vec<_>, _>>()?;
    match op {
        "+" => Ok(Val::Num(nums.iter().sum())),
        "-" => {
            if nums.is_empty() {
                Err("- needs args".into())
            } else if nums.len() == 1 {
                Ok(Val::Num(-nums[0]))
            } else {
                Ok(Val::Num(nums[0] - nums[1..].iter().sum::<f64>()))
            }
        }
        "*" => Ok(Val::Num(nums.iter().product())),
        "/" => {
            if nums.is_empty() {
                Err("/ needs args".into())
            } else if nums.len() == 1 {
                Ok(Val::Num(1.0 / nums[0]))
            } else {
                Ok(Val::Num(nums[0] / nums[1..].iter().product::<f64>()))
            }
        }
        "<" => Ok(Val::Bool(nums.windows(2).all(|w| w[0] < w[1]))),
        ">" => Ok(Val::Bool(nums.windows(2).all(|w| w[0] > w[1]))),
        "<=" => Ok(Val::Bool(nums.windows(2).all(|w| w[0] <= w[1]))),
        ">=" => Ok(Val::Bool(nums.windows(2).all(|w| w[0] >= w[1]))),
        "=" => Ok(Val::Bool(
            nums.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9),
        )),
        _ => Err(format!("unknown op: {}", op)),
    }
}
