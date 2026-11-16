use crate::ast::FnDef;
use crate::interp::{Env, Interp};
use crate::llm::ModelSpec;
use crate::tensor::{self, T};
use std::rc::Rc;

pub type NativeFn = fn(&mut Interp, Vec<Value>) -> Result<Value, String>;

#[derive(Clone)]
pub enum Value {
    Unset,
    Nothing,
    Bool(bool),
    Num(f64),
    Tensor(T),
    Str(Rc<str>),
    List(Rc<Vec<Value>>),
    Map(Rc<Vec<(Rc<str>, Value)>>),
    Fn(Rc<Closure>),
    Native(&'static str, NativeFn),
    Model(Rc<ModelSpec>),
    /// A function in a compiled program (made by `antares build`).
    Rust(Rc<RustFn>),
    /// Whole numbers stored in 16 bits each (made by `compact`).
    Packed(Rc<Packed>),
    /// A 64-bit whole number (made by `int`, a 0x literal, or a bit operation).
    Int(i64),
    /// A row of 64-bit whole numbers (made by `ints`).
    Ints(Rc<Vec<i64>>),
}

/// Whole numbers from -32768 to 32767, 2 bytes each instead of 8: one row, or
/// a matrix of `rows` rows. Only the row built-ins work on these.
pub struct Packed {
    pub data: std::cell::RefCell<Vec<i16>>,
    pub rows: usize,
    pub cols: usize,
    pub matrix: bool,
    /// The largest size of any number in it, worked out when first needed
    /// (-1 = not known). Reset whenever the numbers are changed in place.
    pub max_abs: std::cell::Cell<i32>,
}

impl Packed {
    /// The numbers (named like a tensor's, so code reads the same).
    pub fn vals(&self) -> std::cell::Ref<'_, Vec<i16>> {
        self.data.borrow()
    }

    pub fn vals_mut(&self) -> std::cell::RefMut<'_, Vec<i16>> {
        self.data.borrow_mut()
    }
}

/// A compact row made from its numbers.
pub fn packed_row(data: Vec<i16>) -> Value {
    let cols = data.len();
    Value::Packed(Rc::new(Packed { data: std::cell::RefCell::new(data), rows: 1, cols, matrix: false, max_abs: std::cell::Cell::new(-1) }))
}

/// A problem raised in a compiled program: the message and the line.
pub type AErr = Box<(String, u32)>;

pub struct RustFn {
    pub name: String,
    /// The inputs' names, when known.
    pub params: Vec<String>,
    pub nparams: usize,
    pub f: Box<dyn Fn(Vec<Value>) -> Result<Value, AErr>>,
}

pub struct Closure {
    pub def: Rc<FnDef>,
    pub env: Rc<Env>,
}

pub fn fmt_num(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "infinity".into() } else { "-infinity".into() };
    }
    if x == x.trunc() && x.abs() < 1e15 {
        return format!("{}", x as i64);
    }
    let a = x.abs();
    if (1e-4..1e15).contains(&a) {
        let s = format!("{:.6}", x);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        format!("{:.4e}", x)
    }
}

fn fmt_tensor(shape: &[usize], data: &[f64], indent: usize) -> String {
    if shape.is_empty() {
        return fmt_num(data[0]);
    }
    if shape[0] == 0 {
        return "[]".into();
    }
    if shape.len() == 1 {
        let parts: Vec<String> = data.iter().map(|x| fmt_num(*x)).collect();
        return format!("[{}]", parts.join(", "));
    }
    let sub = data.len() / shape[0];
    let rows: Vec<String> = data
        .chunks(sub.max(1))
        .map(|c| fmt_tensor(&shape[1..], c, indent + 1))
        .collect();
    format!("[{}]", rows.join(&format!(",\n{}", " ".repeat(indent + 1))))
}

/// A list of plain numbers (or equally-shaped tensors) is a tensor.
pub fn pack_list(items: Vec<Value>) -> Value {
    if !items.is_empty() {
        if items.iter().all(|v| matches!(v, Value::Num(_))) {
            let data: Vec<f64> = items
                .iter()
                .map(|v| if let Value::Num(n) = v { *n } else { 0.0 })
                .collect();
            return Value::Tensor(tensor::new(vec![data.len()], data));
        }
        if let Value::Tensor(first) = &items[0] {
            let same = items.iter().all(|v| match v {
                Value::Tensor(t) => !t.requires && t.shape == first.shape,
                _ => false,
            });
            if same {
                let mut shape = vec![items.len()];
                shape.extend(first.shape.iter());
                let mut data = Vec::new();
                for v in &items {
                    if let Value::Tensor(t) = v {
                        data.extend(t.vals().iter());
                    }
                }
                return Value::Tensor(tensor::new(shape, data));
            }
        }
    }
    Value::List(Rc::new(items))
}

/// Collapse a tensor holding one plain number back to a number.
pub fn wrap(t: T) -> Value {
    if t.shape.is_empty() && !t.requires {
        let x = t.vals()[0];
        Value::Num(x)
    } else {
        Value::Tensor(t)
    }
}

impl Value {
    pub fn kind(&self) -> &'static str {
        match self {
            Value::Unset | Value::Nothing => "nothing",
            Value::Bool(_) => "yes/no",
            Value::Num(_) => "number",
            Value::Tensor(t) => {
                if t.shape.is_empty() {
                    "number"
                } else {
                    "tensor"
                }
            }
            Value::Str(_) => "text",
            Value::List(_) => "list",
            Value::Map(_) => "map",
            Value::Fn(_) | Value::Native(..) | Value::Rust(_) => "function",
            Value::Model(_) => "model",
            Value::Packed(_) => "compact numbers",
            Value::Int(_) => "integer",
            Value::Ints(_) => "integer row",
        }
    }

    /// The value as a 64-bit whole number, if it is one.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(n) => Some(*n),
            other => match other.scalar() {
                Some(x) if x == x.trunc() && x.abs() < 9.3e18 => Some(x as i64),
                _ => None,
            },
        }
    }

    pub fn scalar(&self) -> Option<f64> {
        match self {
            Value::Int(n) => Some(*n as f64),
            Value::Num(n) => Some(*n),
            Value::Tensor(t) if t.len() == 1 => Some(t.vals()[0]),
            _ => None,
        }
    }

    pub fn as_tensor(&self) -> Option<T> {
        match self {
            Value::Int(n) => Some(tensor::scalar(*n as f64)),
            Value::Num(n) => Some(tensor::scalar(*n)),
            Value::Tensor(t) => Some(t.clone()),
            _ => None,
        }
    }

    /// How a value looks when shown on its own.
    pub fn show(&self) -> String {
        match self {
            Value::Str(s) => s.to_string(),
            other => other.repr(0),
        }
    }

    /// How a value looks inside a list or map (text gets quotes).
    pub fn repr(&self, indent: usize) -> String {
        match self {
            Value::Unset | Value::Nothing => "nothing".into(),
            Value::Bool(b) => if *b { "yes" } else { "no" }.into(),
            Value::Num(n) => fmt_num(*n),
            Value::Tensor(t) => fmt_tensor(&t.shape, &t.vals(), indent),
            Value::Str(s) => format!("\"{}\"", s),
            Value::List(l) => {
                let parts: Vec<String> = l.iter().map(|v| v.repr(indent + 1)).collect();
                format!("[{}]", parts.join(", "))
            }
            Value::Map(m) => {
                let parts: Vec<String> =
                    m.iter().map(|(k, v)| format!("{}: {}", k, v.repr(indent + 1))).collect();
                format!("{{{}}}", parts.join(", "))
            }
            Value::Fn(c) => format!("<function {}>", c.def.name),
            Value::Native(name, _) => format!("<built-in {}>", name),
            Value::Model(m) => format!("<model {}:{}>", m.provider, m.name),
            Value::Rust(r) => format!("<function {}>", r.name),
            Value::Int(n) => n.to_string(),
            Value::Ints(l) => {
                let parts: Vec<String> = l.iter().map(|x| x.to_string()).collect();
                format!("[{}]", parts.join(", "))
            }
            Value::Packed(p) => {
                if p.matrix {
                    format!("<compact numbers, {} rows of {}>", p.rows, p.cols)
                } else {
                    format!("<compact numbers, {}>", p.cols)
                }
            }
        }
    }

    pub fn equals(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Nothing, Value::Nothing) => true,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Ints(a), Value::Ints(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::List(a), Value::List(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.equals(y))
            }
            (Value::Map(a), Value::Map(b)) => {
                a.len() == b.len()
                    && a.iter().all(|(k, v)| {
                        b.iter().any(|(k2, v2)| k == k2 && v.equals(v2))
                    })
            }
            (a, b) => match (a.as_tensor(), b.as_tensor()) {
                (Some(x), Some(y)) => x.shape == y.shape && *x.vals() == *y.vals(),
                _ => false,
            },
        }
    }
}
