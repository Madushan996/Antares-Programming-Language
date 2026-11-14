use crate::ast::*;
use crate::llm;
use crate::tensor::{self, BinK};
use crate::value::{pack_list, wrap, Closure, Value};
use std::cell::RefCell;
use std::rc::Rc;

pub struct Env {
    pub vars: RefCell<Vec<Value>>,
    pub parent: Option<Rc<Env>>,
}

pub enum Ctrl {
    Err(String, u32),
    Give(Value),
    Stop(u32),
    Skip(u32),
}

type R<T> = Result<T, Ctrl>;

pub(crate) const MAX_DEPTH: usize = 10000;

pub struct Interp {
    pub globals: Rc<Env>,
    pub(crate) depth: usize,
    /// Working values of the fast engine.
    pub(crate) stack: Vec<Value>,
    pub(crate) frames: Vec<crate::vm::Frame>,
    pub(crate) handlers: Vec<crate::vm::Handler>,
    /// Finished call frames, kept for reuse so a call needs no allocation.
    pool: Vec<Rc<Env>>,
}

pub enum Ix {
    At(Value),
    Span(Option<Value>, Option<Value>),
}

pub enum Key {
    Idx(Vec<Value>),
    Field(Rc<str>),
}

#[cold]
fn fail<T>(msg: impl Into<String>, line: u32) -> R<T> {
    Err(Ctrl::Err(msg.into(), line))
}

fn op_symbol(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Mod => "%",
        BinOp::Pow => "^",
        BinOp::MatMul => "@",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::Le => "<=",
        BinOp::Ge => ">=",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::BitXor => "xor",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
    }
}

/// Shifts treat the number as 64 plain bits; shifting by 64 or more gives 0.
#[inline]
pub fn shl(x: i64, n: i64) -> i64 {
    if (0..64).contains(&n) {
        ((x as u64) << n) as i64
    } else {
        0
    }
}

#[inline]
pub fn shr(x: i64, n: i64) -> i64 {
    if (0..64).contains(&n) {
        ((x as u64) >> n) as i64
    } else {
        0
    }
}

/// Arithmetic where at least one side is an integer, and all bit operations.
/// Gives None when the combination is not an integer matter (text, tensors).
fn int_binop(op: BinOp, a: &Value, b: &Value) -> Result<Option<Value>, String> {
    use Value::*;
    let bit = matches!(op, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr);
    let simple = |v: &Value| matches!(v, Int(_) | Num(_)) || matches!(v, Tensor(t) if t.shape.is_empty());
    if !simple(a) || !simple(b) {
        if bit {
            return Err(format!("{} works on whole numbers, not on a {} and a {}", op_symbol(op), a.kind(), b.kind()));
        }
        return Ok(None);
    }
    if let (Int(x), Int(y)) = (a, b) {
        let ord = match op {
            BinOp::Lt => Some(x < y),
            BinOp::Gt => Some(x > y),
            BinOp::Le => Some(x <= y),
            BinOp::Ge => Some(x >= y),
            _ => None,
        };
        if let Some(r) = ord {
            return Ok(Some(Bool(r)));
        }
    }
    match op {
        BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge | BinOp::Eq | BinOp::Ne | BinOp::MatMul => return Ok(None),
        BinOp::Div => return Ok(Some(Num(a.scalar().unwrap() / b.scalar().unwrap()))),
        BinOp::Pow => return Ok(Some(Num(a.scalar().unwrap().powf(b.scalar().unwrap())))),
        _ => {}
    }
    let (x, y) = match (a.as_int(), b.as_int()) {
        (Some(x), Some(y)) => (x, y),
        _ => {
            return Err(format!(
                "{} can't combine an integer with a number that has a fraction ({} and {}); use number(x) to work with fractions",
                op_symbol(op),
                a.show(),
                b.show()
            ))
        }
    };
    Ok(Some(Int(match op {
        BinOp::Add => x.wrapping_add(y),
        BinOp::Sub => x.wrapping_sub(y),
        BinOp::Mul => x.wrapping_mul(y),
        BinOp::Mod => {
            if y == 0 {
                return Err("can't take the remainder of dividing by 0".into());
            }
            x.wrapping_rem_euclid(y)
        }
        BinOp::BitAnd => x & y,
        BinOp::BitOr => x | y,
        BinOp::BitXor => x ^ y,
        BinOp::Shl => shl(x, y),
        BinOp::Shr => shr(x, y),
        _ => unreachable!(),
    })))
}

pub fn binop(op: BinOp, a: &Value, b: &Value) -> Result<Value, String> {
    use Value::*;
    if matches!(a, Int(_))
        || matches!(b, Int(_))
        || matches!(op, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr)
    {
        if let Some(v) = int_binop(op, a, b)? {
            return Ok(v);
        }
    }
    match op {
        BinOp::Eq => return Ok(Bool(a.equals(b))),
        BinOp::Ne => return Ok(Bool(!a.equals(b))),
        BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
            let ord = match (a, b) {
                (Str(x), Str(y)) => x.partial_cmp(y),
                _ => match (a.scalar(), b.scalar()) {
                    (Some(x), Some(y)) => x.partial_cmp(&y),
                    _ => {
                        return Err(format!(
                            "can't compare a {} with a {} using {}",
                            a.kind(),
                            b.kind(),
                            op_symbol(op)
                        ))
                    }
                },
            };
            use std::cmp::Ordering::*;
            return Ok(Bool(match (op, ord) {
                (_, None) => false,
                (BinOp::Lt, Some(o)) => o == Less,
                (BinOp::Gt, Some(o)) => o == Greater,
                (BinOp::Le, Some(o)) => o != Greater,
                (_, Some(o)) => o != Less,
            }));
        }
        _ => {}
    }
    if let (Num(x), Num(y)) = (a, b) {
        return Ok(Num(match op {
            BinOp::Add => x + y,
            BinOp::Sub => x - y,
            BinOp::Mul => x * y,
            BinOp::Div => x / y,
            BinOp::Mod => x.rem_euclid(*y),
            BinOp::Pow => x.powf(*y),
            BinOp::MatMul => return Err("@ joins vectors and matrices; use * to multiply numbers".into()),
            _ => unreachable!(),
        }));
    }
    if op == BinOp::Add {
        match (a, b) {
            (Str(_), _) | (_, Str(_)) => {
                return Ok(Str(format!("{}{}", a.show(), b.show()).into()))
            }
            (List(x), List(y)) => {
                let mut v = (**x).clone();
                v.extend(y.iter().cloned());
                return Ok(List(Rc::new(v)));
            }
            _ => {}
        }
    }
    let (ta, tb) = match (a.as_tensor(), b.as_tensor()) {
        (Some(x), Some(y)) => (x, y),
        _ => {
            return Err(format!(
                "can't use {} between a {} and a {}",
                op_symbol(op),
                a.kind(),
                b.kind()
            ))
        }
    };
    let t = match op {
        BinOp::Add => tensor::binary(BinK::Add, &ta, &tb)?,
        BinOp::Sub => tensor::binary(BinK::Sub, &ta, &tb)?,
        BinOp::Mul => tensor::binary(BinK::Mul, &ta, &tb)?,
        BinOp::Div => tensor::binary(BinK::Div, &ta, &tb)?,
        BinOp::Mod => tensor::binary(BinK::Mod, &ta, &tb)?,
        BinOp::Pow => tensor::binary(BinK::Pow, &ta, &tb)?,
        BinOp::MatMul => tensor::matmul(&ta, &tb)?,
        _ => return Err(format!("{} works on whole numbers", op_symbol(op))),
    };
    Ok(wrap(t))
}

pub fn to_index(v: &Value, len: usize) -> Result<usize, String> {
    let x = v
        .scalar()
        .ok_or_else(|| format!("an index must be a whole number, not a {}", v.kind()))?;
    if x != x.trunc() {
        return Err(format!("an index must be a whole number, not {}", x));
    }
    let i = x as i64;
    let j = if i < 0 { i + len as i64 } else { i };
    if j < 0 || j >= len as i64 {
        if len == 0 {
            return Err(format!("index {} is out of range: there is nothing in here", i));
        }
        return Err(format!("index {} is out of range (valid: 0 to {})", i, len - 1));
    }
    Ok(j as usize)
}

fn map_key(idx: &[Value]) -> Result<Rc<str>, String> {
    match idx {
        [Value::Str(s)] => Ok(s.clone()),
        _ => Err("a map is looked up by one piece of text, like m[\"name\"]".into()),
    }
}

pub fn index_value(base: &Value, idx: &[Value]) -> Result<Value, String> {
    match base {
        Value::Tensor(t) => {
            let mut ix = Vec::with_capacity(idx.len());
            for v in idx {
                let x = v
                    .scalar()
                    .filter(|x| *x == x.trunc())
                    .ok_or_else(|| "an index must be a whole number".to_string())?;
                ix.push(x as i64);
            }
            Ok(wrap(tensor::index(t, &ix)?))
        }
        Value::List(l) => {
            let mut cur = l[to_index(&idx[0], l.len())?].clone();
            if idx.len() > 1 {
                cur = index_value(&cur, &idx[1..])?;
            }
            Ok(cur)
        }
        Value::Str(s) => {
            let chars: Vec<char> = s.chars().collect();
            let i = to_index(&idx[0], chars.len())?;
            Ok(Value::Str(chars[i].to_string().into()))
        }
        Value::Ints(l) => {
            if idx.len() != 1 {
                return Err("an integer row takes one index".into());
            }
            Ok(Value::Int(l[to_index(&idx[0], l.len())?]))
        }
        Value::Map(m) => {
            let key = map_key(idx)?;
            m.iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| v.clone())
                .ok_or_else(|| format!("this map has no '{}'", key))
        }
        other => Err(format!("a {} can't be indexed with [ ]", other.kind())),
    }
}

fn span(a: &Option<Value>, z: &Option<Value>, n: usize) -> Result<(usize, usize), String> {
    let fix = |v: &Option<Value>, default: i64| -> Result<i64, String> {
        match v {
            None => Ok(default),
            Some(v) => {
                let x = v
                    .scalar()
                    .filter(|x| *x == x.trunc())
                    .ok_or("a range needs whole numbers, like 1:3")?;
                let i = x as i64;
                Ok((if i < 0 { i + n as i64 } else { i }).clamp(0, n as i64))
            }
        }
    };
    let s = fix(a, 0)?;
    let e = fix(z, n as i64)?.max(s);
    Ok((s as usize, e as usize))
}

/// Indexing where at least one position is a range like `1:3`.
pub fn slice_value(base: &Value, ix: &[Ix]) -> Result<Value, String> {
    match (base, ix) {
        (Value::Tensor(t), _) => {
            if ix.len() > t.shape.len() {
                return Err(format!("too many indexes: this value has {} dimension(s)", t.shape.len()));
            }
            let mut sel: Vec<Vec<usize>> = vec![];
            let mut shape = vec![];
            for (d, i) in ix.iter().enumerate() {
                let n = t.shape[d];
                match i {
                    Ix::At(v) => sel.push(vec![to_index(v, n)?]),
                    Ix::Span(a, z) => {
                        let (s, e) = span(a, z, n)?;
                        shape.push(e - s);
                        sel.push((s..e).collect());
                    }
                }
            }
            for d in ix.len()..t.shape.len() {
                shape.push(t.shape[d]);
                sel.push((0..t.shape[d]).collect());
            }
            Ok(Value::Tensor(tensor::gather(t, shape, &sel)))
        }
        (Value::List(l), [Ix::Span(a, z)]) => {
            let (s, e) = span(a, z, l.len())?;
            Ok(pack_list(l[s..e].to_vec()))
        }
        (Value::Str(st), [Ix::Span(a, z)]) => {
            let chars: Vec<char> = st.chars().collect();
            let (s, e) = span(a, z, chars.len())?;
            Ok(Value::Str(chars[s..e].iter().collect::<String>().into()))
        }
        (other, _) => Err(format!("a {} can't be sliced with a range like this", other.kind())),
    }
}

pub fn set_in(target: &mut Value, path: &[Key], v: Value) -> Result<(), String> {
    let (k, rest) = path.split_first().unwrap();
    match target {
        Value::List(rc) => {
            let idx = match k {
                Key::Idx(i) if i.len() == 1 => to_index(&i[0], rc.len())?,
                _ => return Err("a list is changed by position, like xs[0] = 5".into()),
            };
            let l = Rc::make_mut(rc);
            if rest.is_empty() {
                l[idx] = v;
                Ok(())
            } else {
                set_in(&mut l[idx], rest, v)
            }
        }
        Value::Map(rc) => {
            let key = match k {
                Key::Field(f) => f.clone(),
                Key::Idx(i) => map_key(i)?,
            };
            let m = Rc::make_mut(rc);
            match m.iter().position(|(k2, _)| *k2 == key) {
                Some(p) => {
                    if rest.is_empty() {
                        m[p].1 = v;
                        Ok(())
                    } else {
                        set_in(&mut m[p].1, rest, v)
                    }
                }
                None if rest.is_empty() => {
                    m.push((key, v));
                    Ok(())
                }
                None => Err(format!("this map has no '{}'", key)),
            }
        }
        Value::Ints(rc) => {
            let idx = match k {
                Key::Idx(i) if i.len() == 1 && rest.is_empty() => to_index(&i[0], rc.len())?,
                _ => return Err("an integer row is changed by position, like xs[0] = 5".into()),
            };
            let x = v.as_int().ok_or_else(|| format!("an integer row can only hold whole numbers, not a {}", v.kind()))?;
            Rc::make_mut(rc)[idx] = x;
            Ok(())
        }
        Value::Tensor(t) => {
            let idx = match k {
                Key::Idx(i) if rest.is_empty() && i.len() == t.shape.len() => i,
                _ => return Err("to change a tensor, give one index per dimension, like m[0, 1] = 5".into()),
            };
            if t.requires {
                return Err("a `learn` variable changes through `descend`, not by hand".into());
            }
            let x = v.scalar().ok_or("a tensor can only hold numbers")?;
            let mut off = 0;
            let mut size = t.len();
            for (d, iv) in idx.iter().enumerate() {
                let j = to_index(iv, t.shape[d])?;
                size /= t.shape[d];
                off += j * size;
            }
            match Rc::get_mut(t) {
                Some(node) => node.data.get_mut()[off] = x,
                None => {
                    let mut data = t.data.borrow().clone();
                    data[off] = x;
                    *target = Value::Tensor(tensor::new(t.shape.clone(), data));
                }
            }
            Ok(())
        }
        other => Err(format!("a {} has no parts to change", other.kind())),
    }
}

/// `target[idx] = v` for the fast engine.
pub(crate) fn set_index(target: &mut Value, idx: Value, v: Value) -> Result<(), String> {
    set_in(target, &[Key::Idx(vec![idx])], v)
}

/// Asks a model and turns the reply into the kind of value wanted
/// (used by compiled programs; the interpreter has the same logic inline).
pub fn ask_value(model: &Value, prompt: &str, kind: AskKind) -> Result<Value, String> {
    let m = match model {
        Value::Model(m) => m,
        other => {
            return Err(format!(
                "`ask` needs a model first (made with `model name = \"...\"`), not a {}",
                other.kind()
            ))
        }
    };
    let suffix = match kind {
        AskKind::Text => "",
        AskKind::Number => "\n\nReply with only a number and nothing else.",
        AskKind::YesNo => "\n\nReply with only the word yes or the word no.",
        AskKind::List => "\n\nReply with only the items, one per line, with no numbering, bullets or extra text.",
    };
    let reply = llm::ask(m, prompt, suffix)?;
    Ok(match kind {
        AskKind::Text => Value::Str(reply.trim().into()),
        AskKind::Number => match first_number(&reply) {
            Some(n) => Value::Num(n),
            None => return Err(format!("I asked for a number but the model said: {}", reply.trim())),
        },
        AskKind::YesNo => {
            let r = reply.trim().to_lowercase();
            if r.starts_with("yes") {
                Value::Bool(true)
            } else if r.starts_with("no") {
                Value::Bool(false)
            } else {
                return Err(format!("I asked for yes or no but the model said: {}", reply.trim()));
            }
        }
        AskKind::List => Value::List(Rc::new(
            reply
                .lines()
                .map(|l| l.trim().trim_start_matches(|c: char| c == '-' || c == '*' || c == '\u{2022}').trim())
                .filter(|l| !l.is_empty())
                .map(|l| Value::Str(l.into()))
                .collect(),
        )),
    })
}

pub fn iter_items(v: &Value) -> Result<Vec<Value>, String> {
    match v {
        Value::List(l) => Ok((**l).clone()),
        Value::Tensor(t) if !t.shape.is_empty() => {
            let mut out = Vec::with_capacity(t.shape[0]);
            for i in 0..t.shape[0] {
                out.push(wrap(tensor::index(t, &[i as i64])?));
            }
            Ok(out)
        }
        Value::Str(s) => Ok(s.chars().map(|c| Value::Str(c.to_string().into())).collect()),
        Value::Ints(l) => Ok(l.iter().map(|x| Value::Int(*x)).collect()),
        Value::Map(m) => Ok(m.iter().map(|(k, _)| Value::Str(k.clone())).collect()),
        other => Err(format!("can't go through a {} item by item", other.kind())),
    }
}

fn first_number(s: &str) -> Option<f64> {
    let c: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < c.len() {
        if c[i].is_ascii_digit() {
            let mut start = i;
            if start > 0 && c[start - 1] == '-' {
                start -= 1;
            }
            let mut j = i;
            let mut text = String::new();
            if start < i {
                text.push('-');
            }
            while j < c.len() && (c[j].is_ascii_digit() || c[j] == '.' || c[j] == ',') {
                if c[j] != ',' {
                    text.push(c[j]);
                }
                j += 1;
            }
            let text = text.trim_end_matches('.');
            if let Ok(n) = text.parse::<f64>() {
                return Some(n);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    None
}

impl Interp {
    pub fn new(globals: Vec<Value>) -> Self {
        Interp {
            globals: Rc::new(Env { vars: RefCell::new(globals), parent: None }),
            depth: 0,
            stack: Vec::new(),
            frames: Vec::new(),
            handlers: Vec::new(),
            pool: Vec::new(),
        }
    }

    pub fn ensure_globals(&mut self, n: usize) {
        let mut v = self.globals.vars.borrow_mut();
        if v.len() < n {
            v.resize(n, Value::Unset);
        }
    }

    pub fn run(&mut self, stmts: &[Stmt]) -> Result<Value, (String, u32)> {
        let env = self.globals.clone();
        self.depth = 0;
        // Functions defined with `to` exist from the start of the program,
        // so they can be used above the place they are written.
        for s in stmts {
            if let Stmt::Assign { idx, value: Expr::Lambda(def) } = s {
                env.vars.borrow_mut()[*idx as usize] =
                    Value::Fn(Rc::new(Closure { def: def.clone(), env: env.clone() }));
            }
        }
        self.stack.clear();
        self.frames.clear();
        self.handlers.clear();
        let code = crate::vm::compile(stmts);
        match self.exec_code(&code, &env) {
            Ok(v) => Ok(v),
            Err(Ctrl::Give(v)) => Ok(v),
            Err(Ctrl::Err(m, l)) => Err((m, l)),
            Err(Ctrl::Stop(l)) => Err(("`stop` only works inside a loop".into(), l)),
            Err(Ctrl::Skip(l)) => Err(("`skip` only works inside a loop".into(), l)),
        }
    }

    fn exec_block(&mut self, stmts: &[Stmt], env: &Rc<Env>) -> R<Value> {
        let mut last = Value::Nothing;
        for s in stmts {
            last = self.exec(s, env)?;
        }
        Ok(last)
    }

    fn truth(&mut self, e: &Expr, env: &Rc<Env>, line: u32) -> R<bool> {
        match self.eval(e, env)? {
            Value::Bool(b) => Ok(b),
            other => Self::not_yes_no(other, line),
        }
    }

    #[cold]
    pub(crate) fn not_yes_no(v: Value, line: u32) -> R<bool> {
        match v {
            Value::Nothing => Ok(false),
            other => fail(
                format!(
                    "I need a yes or no here, but got a {}. Compare it with something, like x > 0",
                    other.kind()
                ),
                line,
            ),
        }
    }

    /// Runs a loop body. Returns false when the loop should end.
    fn loop_body(&mut self, body: &[Stmt], env: &Rc<Env>) -> R<bool> {
        match self.exec_block(body, env) {
            Ok(_) | Err(Ctrl::Skip(_)) => Ok(true),
            Err(Ctrl::Stop(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    pub(crate) fn exec(&mut self, s: &Stmt, env: &Rc<Env>) -> R<Value> {
        match s {
            Stmt::Expr(e) => self.eval(e, env),
            Stmt::Assign { idx, value } => {
                let v = self.eval(value, env)?;
                env.vars.borrow_mut()[*idx as usize] = v;
                Ok(Value::Nothing)
            }
            Stmt::When { arms, otherwise } => {
                for (cond, body, line) in arms {
                    if self.truth(cond, env, *line)? {
                        return self.exec_block(body, env);
                    }
                }
                match otherwise {
                    Some(body) => self.exec_block(body, env),
                    None => Ok(Value::Nothing),
                }
            }
            _ => self.exec_other(s, env),
        }
    }

    #[inline(never)]
    fn exec_other(&mut self, s: &Stmt, env: &Rc<Env>) -> R<Value> {
        match s {
            Stmt::Expr(e) => return self.eval(e, env),
            Stmt::Assign { idx, value } => {
                let v = self.eval(value, env)?;
                env.vars.borrow_mut()[*idx as usize] = v;
            }
            Stmt::Learn { .. } | Stmt::SetPath { .. } | Stmt::Descend { .. } => return self.exec_rare(s, env),
            Stmt::Show(items) => {
                let mut parts = Vec::with_capacity(items.len());
                for e in items {
                    parts.push(self.eval(e, env)?.show());
                }
                println!("{}", parts.join(" "));
            }
            Stmt::Give(e) => {
                let v = match e {
                    Some(e) => self.eval(e, env)?,
                    None => Value::Nothing,
                };
                return Err(Ctrl::Give(v));
            }
            Stmt::Repeat { count, var, body, line } => {
                let c = self.eval(count, env)?;
                let n = match c.scalar() {
                    Some(n) if n >= 0.0 => n as u64,
                    _ => return fail("`repeat` needs a count that is zero or more", *line),
                };
                for i in 1..=n {
                    if let Some(v) = var {
                        env.vars.borrow_mut()[*v as usize] = Value::Num(i as f64);
                    }
                    if !self.loop_body(body, env)? {
                        break;
                    }
                }
            }
            Stmt::Each { var, iter, body, line } => {
                let it = self.eval(iter, env)?;
                let items = iter_items(&it).map_err(|m| Ctrl::Err(m, *line))?;
                for item in items {
                    env.vars.borrow_mut()[*var as usize] = item;
                    if !self.loop_body(body, env)? {
                        break;
                    }
                }
            }
            Stmt::While { cond, body, line } => {
                while self.truth(cond, env, *line)? {
                    if !self.loop_body(body, env)? {
                        break;
                    }
                }
            }
            Stmt::When { arms, otherwise } => {
                for (cond, body, line) in arms {
                    if self.truth(cond, env, *line)? {
                        return self.exec_block(body, env);
                    }
                }
                if let Some(body) = otherwise {
                    return self.exec_block(body, env);
                }
            }
            Stmt::AssignOuter { depth, idx, value } => {
                let v = self.eval(value, env)?;
                let mut home: &Env = env;
                for _ in 0..*depth {
                    home = home.parent.as_ref().unwrap();
                }
                home.vars.borrow_mut()[*idx as usize] = v;
            }
            Stmt::Try { body, var, handler } => {
                let depth = self.depth;
                return match self.exec_block(body, env) {
                    Err(Ctrl::Err(msg, _)) => {
                        self.depth = depth;
                        if let Some(v) = var {
                            env.vars.borrow_mut()[*v as usize] = Value::Str(msg.into());
                        }
                        self.exec_block(handler, env)
                    }
                    other => other,
                };
            }
            Stmt::Stop(l) => return Err(Ctrl::Stop(*l)),
            Stmt::Skip(l) => return Err(Ctrl::Skip(*l)),
        }
        Ok(Value::Nothing)
    }

    #[inline(never)]
    fn exec_rare(&mut self, s: &Stmt, env: &Rc<Env>) -> R<Value> {
        match s {
            Stmt::Learn { idx, value, line } => {
                let v = self.eval(value, env)?;
                let t = match v.as_tensor() {
                    Some(t) => t,
                    None => {
                        return fail(
                            format!("`learn` needs a number or tensor to start from, not a {}", v.kind()),
                            *line,
                        )
                    }
                };
                let p = tensor::param(t.shape.clone(), t.data.borrow().clone());
                env.vars.borrow_mut()[*idx as usize] = Value::Tensor(p);
            }
            Stmt::SetPath { depth, idx, name, path, value, line } => {
                let v = self.eval(value, env)?;
                let mut keys = Vec::with_capacity(path.len());
                for a in path {
                    keys.push(match a {
                        Access::Field(f) => Key::Field(f.clone()),
                        Access::Index(es) => {
                            let mut vals = Vec::with_capacity(es.len());
                            for e in es {
                                vals.push(self.eval(e, env)?);
                            }
                            Key::Idx(vals)
                        }
                    });
                }
                let mut home: &Env = env;
                for _ in 0..*depth {
                    home = home.parent.as_ref().unwrap();
                }
                let mut target =
                    std::mem::replace(&mut home.vars.borrow_mut()[*idx as usize], Value::Nothing);
                if let Value::Unset = target {
                    home.vars.borrow_mut()[*idx as usize] = Value::Unset;
                    return fail(format!("'{}' doesn't have a value yet", name), *line);
                }
                let r = set_in(&mut target, &keys, v);
                home.vars.borrow_mut()[*idx as usize] = target;
                if let Err(m) = r {
                    return fail(m, *line);
                }
            }
            Stmt::Descend { loss, rate, adam, line } => {
                let l = self.eval(loss, env)?;
                let r = self.eval(rate, env)?;
                let rate = match r.scalar() {
                    Some(x) => x,
                    None => return fail("the rate after `by` must be a number", *line),
                };
                let t = match l.as_tensor() {
                    Some(t) => t,
                    None => {
                        return fail(format!("`descend` needs a number to make smaller, not a {}", l.kind()), *line)
                    }
                };
                tensor::descend(&t, rate, *adam).map_err(|m| Ctrl::Err(m, *line))?;
            }
            _ => {}
        }
        Ok(Value::Nothing)
    }

    pub(crate) fn take_frame(&mut self) -> Rc<Env> {
        self.pool
            .pop()
            .unwrap_or_else(|| Rc::new(Env { vars: RefCell::new(Vec::new()), parent: None }))
    }

    pub(crate) fn release_frame(&mut self, mut frame: Rc<Env>) {
        // A frame captured by a closure is still in use; only recycle free ones.
        if let Some(e) = Rc::get_mut(&mut frame) {
            e.vars.get_mut().clear();
            e.parent = None;
            if self.pool.len() < 256 {
                self.pool.push(frame);
            }
        }
    }

    #[cold]
    pub(crate) fn wrong_count(def: &FnDef, given: usize, line: u32) -> Ctrl {
        Ctrl::Err(
            format!(
                "'{}' takes {} input{} but was given {}",
                def.name,
                def.nparams,
                if def.nparams == 1 { "" } else { "s" },
                given
            ),
            line,
        )
    }

    /// Runs a function whose inputs are already in `frame`.
    pub(crate) fn run_fn(&mut self, c: &Closure, mut frame: Rc<Env>, line: u32) -> R<Value> {
        let def = &c.def;
        if self.depth >= MAX_DEPTH {
            self.release_frame(frame);
            return fail(
                format!("'{}' keeps calling itself without ending ({} levels deep)", def.name, MAX_DEPTH),
                line,
            );
        }
        {
            let e = Rc::get_mut(&mut frame).unwrap();
            e.vars.get_mut().resize(def.nslots, Value::Unset);
            e.parent = Some(c.env.clone());
        }
        self.depth += 1;
        let code = def.code.get_or_init(|| crate::vm::compile(&def.body));
        let r = self.exec_code(code, &frame);
        self.depth -= 1;
        self.release_frame(frame);
        match r {
            Ok(v) | Err(Ctrl::Give(v)) => Ok(v),
            Err(Ctrl::Stop(l)) => fail("`stop` only works inside a loop", l),
            Err(Ctrl::Skip(l)) => fail("`skip` only works inside a loop", l),
            Err(e) => Err(e),
        }
    }

    pub fn call(&mut self, f: &Value, args: Vec<Value>, line: u32) -> R<Value> {
        match f {
            Value::Fn(c) => {
                if args.len() != c.def.nparams {
                    return Err(Self::wrong_count(&c.def, args.len(), line));
                }
                let mut frame = self.take_frame();
                Rc::get_mut(&mut frame).unwrap().vars.get_mut().extend(args);
                self.run_fn(c, frame, line)
            }
            Value::Native(_, nf) => nf(self, args).map_err(|m| Ctrl::Err(m, line)),
            Value::Rust(r) => {
                if args.len() != r.nparams {
                    return fail(
                        format!(
                            "'{}' takes {} input{} but was given {}",
                            r.name,
                            r.nparams,
                            if r.nparams == 1 { "" } else { "s" },
                            args.len()
                        ),
                        line,
                    );
                }
                (r.f)(args).map_err(|e| Ctrl::Err(e.0, if e.1 == 0 { line } else { e.1 }))
            }
            other => fail(format!("a {} can't be called like a function", other.kind()), line),
        }
    }

    /// Call a function value from a built-in (errors become plain messages).
    pub fn call_native(&mut self, f: &Value, args: Vec<Value>) -> Result<Value, String> {
        match self.call(f, args, 0) {
            Ok(v) => Ok(v),
            Err(Ctrl::Err(m, _)) => Err(m),
            Err(_) => Err("unexpected stop/skip/give".into()),
        }
    }

    /// The hot path: the handful of expression kinds that make up almost all
    /// of a running program. Everything else goes to `eval_other`, which keeps
    /// this function small and quick to enter.
    pub(crate) fn eval(&mut self, e: &Expr, env: &Rc<Env>) -> R<Value> {
        match e {
            Expr::Num(n) => Ok(Value::Num(*n)),
            Expr::Var { depth, idx, .. } => {
                let mut home: &Env = env;
                for _ in 0..*depth {
                    home = home.parent.as_ref().unwrap();
                }
                let v = home.vars.borrow()[*idx as usize].clone();
                if let Value::Unset = v {
                    return self.eval_other(e, env);
                }
                Ok(v)
            }
            Expr::Binary(op, a, b, line) => {
                let x = self.eval(a, env)?;
                let y = self.eval(b, env)?;
                if let (Value::Num(p), Value::Num(q)) = (&x, &y) {
                    let (p, q) = (*p, *q);
                    match op {
                        BinOp::Add => return Ok(Value::Num(p + q)),
                        BinOp::Sub => return Ok(Value::Num(p - q)),
                        BinOp::Mul => return Ok(Value::Num(p * q)),
                        BinOp::Div => return Ok(Value::Num(p / q)),
                        BinOp::Lt => return Ok(Value::Bool(p < q)),
                        BinOp::Gt => return Ok(Value::Bool(p > q)),
                        BinOp::Le => return Ok(Value::Bool(p <= q)),
                        BinOp::Ge => return Ok(Value::Bool(p >= q)),
                        BinOp::Eq => return Ok(Value::Bool(p == q)),
                        BinOp::Ne => return Ok(Value::Bool(p != q)),
                        _ => {}
                    }
                }
                binop(*op, &x, &y).map_err(|m| Ctrl::Err(m, *line))
            }
            Expr::Call(f, args, line) => {
                let fv = self.eval(f, env)?;
                if let Value::Fn(c) = &fv {
                    if args.len() != c.def.nparams {
                        return Err(Self::wrong_count(&c.def, args.len(), *line));
                    }
                    // Evaluate the inputs straight into a recycled frame.
                    let mut frame = self.take_frame();
                    for a in args {
                        match self.eval(a, env) {
                            Ok(v) => Rc::get_mut(&mut frame).unwrap().vars.get_mut().push(v),
                            Err(er) => {
                                self.release_frame(frame);
                                return Err(er);
                            }
                        }
                    }
                    return self.run_fn(c, frame, *line);
                }
                let mut av = Vec::with_capacity(args.len());
                for a in args {
                    av.push(self.eval(a, env)?);
                }
                self.call(&fv, av, *line)
            }
            _ => self.eval_other(e, env),
        }
    }

    #[inline(never)]
    fn eval_other(&mut self, e: &Expr, env: &Rc<Env>) -> R<Value> {
        Ok(match e {
            Expr::Num(n) => Value::Num(*n),
            Expr::Str(s) => Value::Str(s.clone()),
            Expr::Bool(b) => Value::Bool(*b),
            Expr::Nothing => Value::Nothing,
            Expr::Interp(parts) => {
                let mut s = String::new();
                for p in parts {
                    s.push_str(&self.eval(p, env)?.show());
                }
                Value::Str(s.into())
            }
            Expr::Var { depth, idx, name, line } => {
                let mut home: &Env = env;
                for _ in 0..*depth {
                    home = home.parent.as_ref().unwrap();
                }
                let v = home.vars.borrow()[*idx as usize].clone();
                if let Value::Unset = v {
                    return fail(format!("'{}' doesn't have a value yet", name), *line);
                }
                v
            }
            Expr::List(items) => {
                let mut vals = Vec::with_capacity(items.len());
                for it in items {
                    vals.push(self.eval(it, env)?);
                }
                pack_list(vals)
            }
            Expr::Map(entries) => {
                let mut m: Vec<(Rc<str>, Value)> = Vec::with_capacity(entries.len());
                for (k, ve) in entries {
                    let v = self.eval(ve, env)?;
                    match m.iter().position(|(k2, _)| k2 == k) {
                        Some(p) => m[p].1 = v,
                        None => m.push((k.clone(), v)),
                    }
                }
                Value::Map(Rc::new(m))
            }
            Expr::Neg(inner, line) => match self.eval(inner, env)? {
                Value::Num(n) => Value::Num(-n),
                Value::Tensor(t) => wrap(tensor::neg(&t)),
                other => return fail(format!("can't put a minus in front of a {}", other.kind()), *line),
            },
            Expr::Not(inner, line) => Value::Bool(!self.truth(inner, env, *line)?),
            Expr::And(a, b, line) => Value::Bool(self.truth(a, env, *line)? && self.truth(b, env, *line)?),
            Expr::Or(a, b, line) => Value::Bool(self.truth(a, env, *line)? || self.truth(b, env, *line)?),
            Expr::Binary(op, a, b, line) => {
                let x = self.eval(a, env)?;
                let y = self.eval(b, env)?;
                if let (Value::Num(p), Value::Num(q)) = (&x, &y) {
                    let (p, q) = (*p, *q);
                    match op {
                        BinOp::Add => return Ok(Value::Num(p + q)),
                        BinOp::Sub => return Ok(Value::Num(p - q)),
                        BinOp::Mul => return Ok(Value::Num(p * q)),
                        BinOp::Div => return Ok(Value::Num(p / q)),
                        BinOp::Lt => return Ok(Value::Bool(p < q)),
                        BinOp::Gt => return Ok(Value::Bool(p > q)),
                        BinOp::Le => return Ok(Value::Bool(p <= q)),
                        BinOp::Ge => return Ok(Value::Bool(p >= q)),
                        BinOp::Eq => return Ok(Value::Bool(p == q)),
                        BinOp::Ne => return Ok(Value::Bool(p != q)),
                        _ => {}
                    }
                }
                binop(*op, &x, &y).map_err(|m| Ctrl::Err(m, *line))?
            }
            Expr::Call(f, args, line) => {
                let fv = self.eval(f, env)?;
                if let Value::Fn(c) = &fv {
                    if args.len() != c.def.nparams {
                        return Err(Self::wrong_count(&c.def, args.len(), *line));
                    }
                    // Evaluate the inputs straight into a recycled frame.
                    let mut frame = self.take_frame();
                    for a in args {
                        match self.eval(a, env) {
                            Ok(v) => Rc::get_mut(&mut frame).unwrap().vars.get_mut().push(v),
                            Err(er) => {
                                self.release_frame(frame);
                                return Err(er);
                            }
                        }
                    }
                    return self.run_fn(c, frame, *line);
                }
                let mut av = Vec::with_capacity(args.len());
                for a in args {
                    av.push(self.eval(a, env)?);
                }
                self.call(&fv, av, *line)?
            }
            Expr::Index(base, idx, line) => {
                let b = self.eval(base, env)?;
                if idx.iter().any(|i| matches!(i, Expr::Range(..))) {
                    let mut ix = Vec::with_capacity(idx.len());
                    for i in idx {
                        ix.push(match i {
                            Expr::Range(a, z, _) => {
                                let a = match a {
                                    Some(x) => Some(self.eval(x, env)?),
                                    None => None,
                                };
                                let z = match z {
                                    Some(x) => Some(self.eval(x, env)?),
                                    None => None,
                                };
                                Ix::Span(a, z)
                            }
                            other => Ix::At(self.eval(other, env)?),
                        });
                    }
                    return slice_value(&b, &ix).map_err(|m| Ctrl::Err(m, *line));
                }
                let mut iv = Vec::with_capacity(idx.len());
                for i in idx {
                    iv.push(self.eval(i, env)?);
                }
                index_value(&b, &iv).map_err(|m| Ctrl::Err(m, *line))?
            }
            Expr::Field(base, name, line) => match self.eval(base, env)? {
                Value::Map(m) => match m.iter().find(|(k, _)| k == name) {
                    Some((_, v)) => v.clone(),
                    None => return fail(format!("this map has no '{}'", name), *line),
                },
                other => {
                    return fail(
                        format!("only maps have named parts, and this is a {} (to call a function on a value, write value -> {}())", other.kind(), name),
                        *line,
                    )
                }
            },
            Expr::Range(_, _, line) => {
                return fail("a range like 1:3 can only be used to read part of a value, as in xs[1:3]", *line)
            }
            Expr::Int(n) => Value::Int(*n),
            Expr::BitNot(inner, line) => {
                let v = self.eval(inner, env)?;
                match v.as_int() {
                    Some(x) => Value::Int(!x),
                    None => return fail(format!("~ works on whole numbers, not on a {}", v.kind()), *line),
                }
            }
            Expr::Lambda(def) => Value::Fn(Rc::new(Closure { def: def.clone(), env: env.clone() })),
            Expr::ModelNew { .. } | Expr::Ask { .. } => self.eval_model(e, env)?,
        })
    }

    #[inline(never)]
    fn eval_model(&mut self, e: &Expr, env: &Rc<Env>) -> R<Value> {
        Ok(match e {
            Expr::ModelNew { spec, opts, line } => {
                let s = self.eval(spec, env)?;
                let o = match opts {
                    Some(o) => Some(self.eval(o, env)?),
                    None => None,
                };
                let m = llm::make_spec(&s, o.as_ref()).map_err(|m| Ctrl::Err(m, *line))?;
                Value::Model(Rc::new(m))
            }
            Expr::Ask { model, prompt, kind, line } => {
                let m = match self.eval(model, env)? {
                    Value::Model(m) => m,
                    other => {
                        return fail(
                            format!("`ask` needs a model first (made with `model name = \"...\"`), not a {}", other.kind()),
                            *line,
                        )
                    }
                };
                let p = self.eval(prompt, env)?.show();
                let suffix = match kind {
                    AskKind::Text => "",
                    AskKind::Number => "\n\nReply with only a number and nothing else.",
                    AskKind::YesNo => "\n\nReply with only the word yes or the word no.",
                    AskKind::List => "\n\nReply with only the items, one per line, with no numbering, bullets or extra text.",
                };
                let reply = llm::ask(&m, &p, suffix).map_err(|m| Ctrl::Err(m, *line))?;
                match kind {
                    AskKind::Text => Value::Str(reply.trim().into()),
                    AskKind::Number => match first_number(&reply) {
                        Some(n) => Value::Num(n),
                        None => {
                            return fail(format!("I asked for a number but the model said: {}", reply.trim()), *line)
                        }
                    },
                    AskKind::YesNo => {
                        let r = reply.trim().to_lowercase();
                        if r.starts_with("yes") {
                            Value::Bool(true)
                        } else if r.starts_with("no") {
                            Value::Bool(false)
                        } else {
                            return fail(format!("I asked for yes or no but the model said: {}", reply.trim()), *line);
                        }
                    }
                    AskKind::List => {
                        let items: Vec<Value> = reply
                            .lines()
                            .map(|l| l.trim().trim_start_matches(|c: char| c == '-' || c == '*' || c == '•').trim())
                            .filter(|l| !l.is_empty())
                            .map(|l| Value::Str(l.into()))
                            .collect();
                        Value::List(Rc::new(items))
                    }
                }
            }
            _ => Value::Nothing,
        })
    }
}
