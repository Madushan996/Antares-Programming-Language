//! Support code for programs compiled with `antares build`.
//!
//! A compiled program keeps plain numbers in machine registers and number
//! arrays as direct memory, and calls the functions here for everything else
//! (text, lists, maps, tensors, built-ins), so compiled and interpreted
//! programs share one implementation of the language's behaviour.

pub use crate::ast::{AskKind, BinOp};
pub use crate::interp::{Ix, Key};
pub use crate::tensor::T;
pub use crate::value::{RustFn, Value};

use crate::interp::{self, Ctrl, Interp};
use crate::tensor;
use crate::value::pack_list;
use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;

/// A problem: the message and the line it happened on.
pub type E = Box<(String, u32)>;
pub type R<X> = Result<X, E>;

/// How a `try` block ended.
pub enum Flow<X> {
    Next,
    Ret(X),
    Brk,
    Cont,
}

static mut IT: *mut Interp = std::ptr::null_mut();
static mut BUILTINS: *const Vec<Value> = std::ptr::null();
static mut EMPTY_ARR: *const T = std::ptr::null();
static mut DEPTH: usize = 0;

const MAX_DEPTH: usize = 10000;

#[cold]
pub fn fail<X>(msg: impl Into<String>, line: u32) -> R<X> {
    Err(Box::new((msg.into(), line)))
}

#[inline]
fn at<X>(r: Result<X, String>, line: u32) -> R<X> {
    r.map_err(|m| Box::new((m, line)))
}

#[inline]
fn it() -> &'static mut Interp {
    unsafe { &mut *IT }
}

// ---------- starting and ending the program ----------

fn report(files: &[(&str, &str)], main_src: &str, msg: &str, line: u32) {
    let _ = std::io::stdout().flush();
    eprintln!();
    let (file, line) = (line >> 20, line & 0xFFFFF);
    let (src, place) = match if file > 0 { files.get(file as usize - 1) } else { None } {
        Some((name, text)) => (*text, format!(" of {}", name)),
        None => (main_src, String::new()),
    };
    if line > 0 {
        eprintln!("Problem on line {}{}: {}", line, place, msg);
        if let Some(text) = src.lines().nth(line as usize - 1) {
            eprintln!("  {:>4} | {}", line, text.trim_end());
        }
    } else {
        eprintln!("Problem: {}", msg);
    }
}

/// Runs a compiled program's body on a thread with a large stack.
pub fn start(files: &'static [(&'static str, &'static str)], main_src: &'static str, body: fn() -> R<()>) -> ! {
    crate::builtins::set_program_args(std::env::args().skip(1).collect());
    let child = std::thread::Builder::new()
        .stack_size(1 << 30)
        .spawn(move || {
            if let Ok(exe) = std::env::current_exe() {
                crate::builtins::set_program_folder(&exe);
            }
            let b = crate::builtins::all();
            let values: Vec<Value> = b.iter().map(|(n, f)| Value::Native(n, *f)).collect();
            let mut interp = Interp::new(values.clone());
            let empty: T = tensor::new(vec![0], vec![]);
            unsafe {
                IT = &mut interp;
                BUILTINS = &values;
                EMPTY_ARR = &empty;
            }
            match body() {
                Ok(()) => 0,
                Err(e) => {
                    report(files, main_src, &e.0, e.1);
                    1
                }
            }
        })
        .expect("couldn't start");
    let code = child.join().unwrap_or(1);
    std::process::exit(code);
}

/// Registers the files packed into this program (called first by generated code).
pub fn embed(files: &[(&'static str, &'static [u8])]) {
    crate::builtins::set_embedded(files);
}

/// The current call depth (compiled functions called by name receive their
/// depth as an input instead; this is for closures and the program body).
#[inline(always)]
pub fn depth() -> usize {
    unsafe { DEPTH }
}

/// Puts a function's depth into the shared counter while it calls something
/// that may call back into closures (and restores it afterwards).
pub struct AtDepth(usize);

impl Drop for AtDepth {
    #[inline(always)]
    fn drop(&mut self) {
        unsafe { DEPTH = self.0 }
    }
}

#[inline(always)]
pub fn at_depth(d: usize) -> AtDepth {
    unsafe {
        let saved = DEPTH;
        DEPTH = d;
        AtDepth(saved)
    }
}

/// The deepest a function called by name may be (see `too_deep_at`).
pub const MAX_CALL_DEPTH: usize = MAX_DEPTH;

/// Endless recursion in a function that otherwise returns a `Result`.
#[cold]
pub fn too_deep_at<X>(name: &str) -> R<X> {
    fail(format!("'{}' keeps calling itself without ending ({} levels deep)", name, MAX_DEPTH), 0)
}

// Functions that can't fail apart from endless recursion return plain values.
// When one of them goes too deep it records the problem here and returns a
// placeholder; while a problem is recorded every such function returns at
// once, and the first caller that can report problems (`checked`) does so.
// Only loop-free functions without side effects work this way, so the
// placeholders are never seen.
static mut LIMIT: usize = MAX_DEPTH;
static mut FAULT_NAME: &str = "";

#[inline(always)]
pub fn limit() -> usize {
    unsafe { LIMIT }
}

#[cold]
#[inline(never)]
pub fn fault<X>(name: &'static str, placeholder: X) -> X {
    unsafe {
        if LIMIT != 0 {
            LIMIT = 0;
            FAULT_NAME = name;
        }
    }
    placeholder
}

#[cold]
#[inline(never)]
fn faulted<X>() -> R<X> {
    unsafe {
        LIMIT = MAX_DEPTH;
        too_deep_at(FAULT_NAME)
    }
}

/// The result of a function that may have gone too deep (see `fault`).
#[inline(always)]
pub fn checked<X>(x: X) -> R<X> {
    if unsafe { LIMIT } == 0 {
        return faulted();
    }
    Ok(x)
}

pub struct Depth;

impl Drop for Depth {
    #[inline(always)]
    fn drop(&mut self) {
        unsafe { DEPTH -= 1 }
    }
}

#[cold]
fn too_deep(name: &str) -> R<Depth> {
    fail(format!("'{}' keeps calling itself without ending ({} levels deep)", name, MAX_DEPTH), 0)
}

/// Counts nested calls so endless recursion is reported instead of crashing.
#[inline(always)]
pub fn enter(name: &str) -> R<Depth> {
    unsafe {
        if DEPTH >= MAX_DEPTH {
            return too_deep(name);
        }
        DEPTH += 1;
    }
    Ok(Depth)
}

// ---------- values ----------

pub fn cell(v: Value) -> Rc<RefCell<Value>> {
    Rc::new(RefCell::new(v))
}

#[cold]
fn unset(name: &str, line: u32) -> R<Value> {
    fail(format!("'{}' doesn't have a value yet", name), line)
}

/// Reads a variable, checking it has been given a value.
#[inline]
pub fn get(v: &Value, name: &str, line: u32) -> R<Value> {
    if let Value::Unset = v {
        return unset(name, line);
    }
    Ok(v.clone())
}

pub fn builtin(i: usize) -> Value {
    unsafe { (&*BUILTINS)[i].clone() }
}

/// Calls built-in number `i`.
pub fn bcall(i: usize, args: Vec<Value>, line: u32) -> R<Value> {
    match unsafe { &(&*BUILTINS)[i] } {
        Value::Native(_, f) => at(f(it(), args), line),
        _ => fail("not a built-in", line),
    }
}

pub use crate::builtins::{screlu_dot_slice, update_rows_slice};

/// within(x, lo, hi, name) for an integer: x, or the problem.
#[inline(always)]
pub fn within_i(x: i64, lo: f64, hi: f64, name: &str, line: u32) -> R<i64> {
    let f = x as f64;
    if f >= lo && f <= hi {
        Ok(x)
    } else {
        within_fail(x.to_string(), lo, hi, name, line)
    }
}

/// within(x, lo, hi, name) for a number.
#[inline(always)]
pub fn within_n(x: f64, lo: f64, hi: f64, name: &str, line: u32) -> R<f64> {
    if x >= lo && x <= hi {
        Ok(x)
    } else {
        within_fail(crate::value::fmt_num(x), lo, hi, name, line)
    }
}

/// within(x, lo, hi, name) for any value.
pub fn within_v(x: Value, lo: f64, hi: f64, name: &str, line: u32) -> R<Value> {
    at(crate::builtins::within_value(x, lo, hi, name), line)
}

#[cold]
#[inline(never)]
fn within_fail<X>(x: String, lo: f64, hi: f64, name: &str, line: u32) -> R<X> {
    fail(crate::builtins::within_message(&x, lo, hi, name), line)
}

/// Calls a built-in that takes its inputs as a slice (no list is allocated).
#[inline]
pub fn scall(f: fn(&[Value]) -> Result<Value, String>, args: &[Value], line: u32) -> R<Value> {
    at(f(args), line)
}

/// name[key] = v for one index. Lists are changed in place without building a path.
pub fn set1(target: &mut Value, key: Value, v: Value, name: &str, line: u32) -> R<()> {
    if let Value::List(rc) = &mut *target {
        let k = match key {
            Value::Num(i) => {
                let k = whole(i);
                if k as f64 == i { k } else { -1 }
            }
            Value::Int(k) => k,
            _ => -1,
        };
        if (k as u64) < rc.len() as u64 {
            Rc::make_mut(rc)[k as usize] = v;
            return Ok(());
        }
    }
    set_path(target, vec![Key::Idx(vec![key])], v, name, line)
}

/// Fails if a variable hasn't been given a value yet.
#[inline(always)]
pub fn has_value(v: &Value, name: &str, line: u32) -> R<()> {
    if let Value::Unset = v {
        return unset(name, line).map(|_| ());
    }
    Ok(())
}

/// b[key] for one position, reading a variable in place (no copy of a list is
/// made). `name` is the variable's name, for the "doesn't have a value" problem.
#[inline]
pub fn index1(b: &Value, key: Value, name: &str, line: u32) -> R<Value> {
    if let Value::List(l) = b {
        let k = match key {
            Value::Num(i) => {
                let k = whole(i);
                if k as f64 == i { k } else { -1 }
            }
            Value::Int(k) => k,
            _ => -1,
        };
        if (k as u64) < l.len() as u64 {
            return Ok(l[k as usize].clone());
        }
    }
    if let Value::Unset = b {
        return unset(name, line);
    }
    index(b, &[key], line)
}

/// Calls any function value.
pub fn call(f: &Value, args: Vec<Value>, line: u32) -> R<Value> {
    match f {
        Value::Rust(r) if args.len() == r.nparams => (r.f)(args).map_err(|mut e| {
            if e.1 == 0 {
                e.1 = line;
            }
            e
        }),
        _ => match it().call(f, args, line) {
            Ok(v) => Ok(v),
            Err(Ctrl::Err(m, l)) => fail(m, if l == 0 { line } else { l }),
            Err(_) => fail("unexpected stop, skip or give", line),
        },
    }
}

pub fn rust_fn(name: &str, nparams: usize, f: impl Fn(Vec<Value>) -> R<Value> + 'static) -> Value {
    Value::Rust(Rc::new(RustFn { name: name.to_string(), nparams, f: Box::new(f) }))
}

pub fn binop(op: BinOp, a: &Value, b: &Value, line: u32) -> R<Value> {
    at(interp::binop(op, a, b), line)
}

pub fn neg(v: &Value, line: u32) -> R<Value> {
    match v {
        Value::Num(n) => Ok(Value::Num(-n)),
        Value::Tensor(t) => Ok(crate::value::wrap(tensor::neg(t))),
        other => fail(format!("can't put a minus in front of a {}", other.kind()), line),
    }
}

#[cold]
fn not_yes_no(v: &Value, line: u32) -> R<bool> {
    match v {
        Value::Nothing => Ok(false),
        other => fail(
            format!("I need a yes or no here, but got a {}. Compare it with something, like x > 0", other.kind()),
            line,
        ),
    }
}

#[inline]
pub fn truth(v: &Value, line: u32) -> R<bool> {
    match v {
        Value::Bool(b) => Ok(*b),
        other => not_yes_no(other, line),
    }
}

pub fn index(b: &Value, idx: &[Value], line: u32) -> R<Value> {
    at(interp::index_value(b, idx), line)
}

pub fn slice(b: &Value, ix: Vec<Ix>, line: u32) -> R<Value> {
    at(interp::slice_value(b, &ix), line)
}

pub fn field(b: &Value, name: &str, line: u32) -> R<Value> {
    match b {
        Value::Map(m) => match m.iter().find(|(k, _)| &**k == name) {
            Some((_, v)) => Ok(v.clone()),
            None => fail(format!("this map has no '{}'", name), line),
        },
        other => fail(
            format!(
                "only maps have named parts, and this is a {} (to call a function on a value, write value -> {}())",
                other.kind(),
                name
            ),
            line,
        ),
    }
}

/// name[...]... = v
pub fn set_path(target: &mut Value, keys: Vec<Key>, v: Value, name: &str, line: u32) -> R<()> {
    if let Value::Unset = target {
        return fail(format!("'{}' doesn't have a value yet", name), line);
    }
    at(interp::set_in(target, &keys, v), line)
}

pub fn list(items: Vec<Value>) -> Value {
    pack_list(items)
}

pub fn map(entries: Vec<(&str, Value)>) -> Value {
    let mut m: Vec<(Rc<str>, Value)> = Vec::with_capacity(entries.len());
    for (k, v) in entries {
        match m.iter().position(|(k2, _)| &**k2 == k) {
            Some(p) => m[p].1 = v,
            None => m.push((k.into(), v)),
        }
    }
    Value::Map(Rc::new(m))
}

pub fn text(s: &str) -> Value {
    Value::Str(s.into())
}

pub fn interp_str(parts: &[Value]) -> Value {
    let mut s = String::new();
    for p in parts {
        s.push_str(&p.show());
    }
    Value::Str(s.into())
}

pub fn show(parts: &[Value]) {
    let texts: Vec<String> = parts.iter().map(|v| v.show()).collect();
    println!("{}", texts.join(" "));
}

pub fn items(v: &Value, line: u32) -> R<Vec<Value>> {
    at(interp::iter_items(v), line)
}

#[cold]
fn bad_count(line: u32) -> R<f64> {
    fail("`repeat` needs a count that is zero or more", line)
}

#[inline]
pub fn repeat_n(n: f64, line: u32) -> R<f64> {
    if n >= 0.0 {
        Ok((n as u64) as f64)
    } else {
        bad_count(line)
    }
}

pub fn repeat_dyn(v: &Value, line: u32) -> R<f64> {
    match v.scalar() {
        Some(n) => repeat_n(n, line),
        None => bad_count(line),
    }
}

pub fn learn(v: &Value, line: u32) -> R<Value> {
    match v.as_tensor() {
        Some(t) => Ok(Value::Tensor(tensor::param(t.shape.clone(), t.data.borrow().clone()))),
        None => fail(format!("`learn` needs a number or tensor to start from, not a {}", v.kind()), line),
    }
}

pub fn descend(loss: &Value, rate: &Value, adam: bool, line: u32) -> R<()> {
    let rate = match rate.scalar() {
        Some(x) => x,
        None => return fail("the rate after `by` must be a number", line),
    };
    let t = match loss.as_tensor() {
        Some(t) => t,
        None => return fail(format!("`descend` needs a number to make smaller, not a {}", loss.kind()), line),
    };
    at(tensor::descend(&t, rate, adam), line)
}

pub fn model(spec: &Value, opts: Option<&Value>, line: u32) -> R<Value> {
    Ok(Value::Model(Rc::new(at(crate::llm::make_spec(spec, opts), line)?)))
}

pub fn ask(m: &Value, prompt: &Value, kind: AskKind, line: u32) -> R<Value> {
    at(interp::ask_value(m, &prompt.show(), kind), line)
}

// ---------- unboxing: values the compiler has proved are numbers ----------

#[cold]
fn wrong_kind<X>(v: &Value, want: &str, line: u32) -> R<X> {
    fail(format!("expected a {} here but got a {}", want, v.kind()), line)
}

#[inline]
pub fn as_num(v: Value, line: u32) -> R<f64> {
    match v {
        Value::Num(n) => Ok(n),
        other => match other.scalar() {
            Some(n) => Ok(n),
            None => wrong_kind(&other, "number", line),
        },
    }
}

#[inline]
pub fn as_bool(v: Value, line: u32) -> R<bool> {
    match v {
        Value::Bool(b) => Ok(b),
        other => wrong_kind(&other, "yes/no", line),
    }
}

// ---------- 64-bit whole numbers ----------

pub use crate::interp::{shl, shr};

/// A row of 64-bit whole numbers.
pub type IA = Rc<Vec<i64>>;

static mut EMPTY_IARR: *const IA = std::ptr::null();

#[inline]
pub fn iarr0() -> IA {
    unsafe {
        if EMPTY_IARR.is_null() {
            EMPTY_IARR = Box::into_raw(Box::new(Rc::new(Vec::new())));
        }
        (*EMPTY_IARR).clone()
    }
}

#[cold]
fn not_whole<X>(x: f64, line: u32) -> R<X> {
    fail(format!("{} has a fraction, so it can't be used as an integer; use number(x) to work with fractions", crate::value::fmt_num(x)), line)
}

/// A number that must be whole, as an integer.
#[inline(always)]
pub fn n2i(x: f64, line: u32) -> R<i64> {
    let k = whole(x);
    if k as f64 == x {
        Ok(k)
    } else {
        not_whole(x, line)
    }
}

#[inline]
pub fn to_int(v: &Value, line: u32) -> R<i64> {
    match v {
        Value::Int(n) => Ok(*n),
        other => match other.as_int() {
            Some(n) => Ok(n),
            None => fail(format!("expected a whole number here but got {}", other.show()), line),
        },
    }
}

#[inline]
pub fn as_int(v: Value, line: u32) -> R<i64> {
    to_int(&v, line)
}

pub fn as_iarr(v: Value, line: u32) -> R<IA> {
    match v {
        Value::Ints(a) => Ok(a),
        other => wrong_kind(&other, "integer row", line),
    }
}

pub fn bitnot(v: &Value, line: u32) -> R<Value> {
    match v.as_int() {
        Some(x) => Ok(Value::Int(!x)),
        None => fail(format!("~ works on whole numbers, not on a {}", v.kind()), line),
    }
}

#[cold]
fn div_zero<X>(what: &str, line: u32) -> R<X> {
    fail(format!("can't {} 0", what), line)
}

#[inline(always)]
pub fn irem(x: i64, y: i64, line: u32) -> R<i64> {
    if y == 0 {
        return div_zero("take the remainder of dividing by", line);
    }
    Ok(x.wrapping_rem_euclid(y))
}

#[inline(always)]
pub fn idiv(x: i64, y: i64, line: u32) -> R<i64> {
    if y == 0 {
        return div_zero("divide by", line);
    }
    Ok(x.wrapping_div(y))
}

#[cold]
fn iget_slow(a: &IA, i: i64, line: u32) -> R<i64> {
    let v = at(interp::index_value(&Value::Ints(a.clone()), &[Value::Int(i)]), line)?;
    to_int(&v, line)
}

/// a[i] for an integer row.
#[inline(always)]
pub fn iget(a: &IA, i: i64, line: u32) -> R<i64> {
    if (i as u64) < a.len() as u64 {
        Ok(unsafe { *a.get_unchecked(i as usize) })
    } else {
        iget_slow(a, i, line)
    }
}

#[cold]
fn iset_slow(a: &mut IA, i: i64, x: i64, line: u32) -> R<()> {
    let mut v = Value::Ints(std::mem::replace(a, iarr0()));
    let r = interp::set_index(&mut v, Value::Int(i), Value::Int(x));
    *a = as_iarr(v, line)?;
    at(r, line)
}

/// a[i] = x for an integer row. Changed in place unless another variable shares it.
#[inline(always)]
pub fn iset(a: &mut IA, i: i64, x: i64, line: u32) -> R<()> {
    if (i as u64) < a.len() as u64 {
        if let Some(d) = Rc::get_mut(a) {
            unsafe { *d.get_unchecked_mut(i as usize) = x };
            return Ok(());
        }
    }
    iset_slow(a, i, x, line)
}

/// a[i] for a row of ordinary numbers, with an integer position.
#[inline(always)]
pub fn aget_i(a: &T, i: i64, line: u32) -> R<f64> {
    let d = unsafe { &*a.data.as_ptr() };
    if (i as u64) < d.len() as u64 {
        Ok(unsafe { *d.get_unchecked(i as usize) })
    } else {
        aget_slow(a, i as f64, line)
    }
}

#[inline(always)]
pub fn aset_i(a: &mut T, i: i64, x: f64, line: u32) -> R<()> {
    if Rc::strong_count(a) == 1 {
        let d = unsafe { &mut *a.data.as_ptr() };
        if (i as u64) < d.len() as u64 {
            unsafe { *d.get_unchecked_mut(i as usize) = x };
            return Ok(());
        }
    }
    aset_slow(a, i as f64, x, line)
}

// ---------- direct access for loops that checked positions up front ----------

/// The numbers of a row, for a loop that only reads them.
#[inline(always)]
pub fn arr_ptr(a: &T) -> *mut f64 {
    unsafe { (&mut *a.data.as_ptr()).as_mut_ptr() }
}

/// The numbers of a row for a loop that writes them: a row another variable
/// shares is copied first (as the first write would do anyway).
#[inline]
pub fn arr_unique(a: &mut T) -> *mut f64 {
    if Rc::strong_count(a) != 1 {
        let copy = tensor::new(a.shape.clone(), a.data.borrow().clone());
        *a = copy;
    }
    arr_ptr(a)
}

#[inline(always)]
pub fn iarr_ptr(a: &IA) -> *mut i64 {
    a.as_ptr() as *mut i64
}

#[inline]
pub fn iarr_unique(a: &mut IA) -> *mut i64 {
    if Rc::get_mut(a).is_none() {
        *a = Rc::new((**a).clone());
    }
    match Rc::get_mut(a) {
        Some(v) => v.as_mut_ptr(),
        None => std::ptr::null_mut(),
    }
}

// ---------- fixed-size tables ----------
// Top-level rows that are never shared and never change length are kept as
// plain arrays. Positions out of range (and negative ones, which count from the
// end) go the general way, so they behave exactly as for any row.

#[inline(always)]
fn tkey(i: i64, num: bool) -> Value {
    if num { Value::Num(i as f64) } else { Value::Int(i) }
}

#[cold]
#[inline(never)]
fn tget_slow_i(t: &[i64], k: Value, line: u32) -> R<i64> {
    let v = at(interp::index_value(&Value::Ints(Rc::new(t.to_vec())), &[k]), line)?;
    to_int(&v, line)
}

#[cold]
#[inline(never)]
fn tget_slow_a(t: &[f64], k: Value, line: u32) -> R<f64> {
    let v = at(interp::index_value(&Value::Tensor(arr_of(t.to_vec())), &[k]), line)?;
    as_num(v, line)
}

#[cold]
#[inline(never)]
fn tset_slow_i(t: &mut [i64], k: Value, x: i64, line: u32) -> R<()> {
    let mut v = Value::Ints(Rc::new(t.to_vec()));
    at(interp::set_index(&mut v, k, Value::Int(x)), line)?;
    if let Value::Ints(a) = v {
        t.copy_from_slice(&a);
    }
    Ok(())
}

#[cold]
#[inline(never)]
fn tset_slow_a(t: &mut [f64], k: Value, x: f64, line: u32) -> R<()> {
    let mut v = Value::Tensor(arr_of(t.to_vec()));
    at(interp::set_index(&mut v, k, Value::Num(x)), line)?;
    if let Value::Tensor(a) = v {
        t.copy_from_slice(&a.data.borrow());
    }
    Ok(())
}

/// t[i] for an integer table; `num`: the position is an ordinary number.
#[inline(always)]
pub fn tget_i(t: &[i64], i: i64, num: bool, line: u32) -> R<i64> {
    if (i as u64) < t.len() as u64 { Ok(unsafe { *t.get_unchecked(i as usize) }) } else { tget_slow_i(t, tkey(i, num), line) }
}

#[inline(always)]
pub fn tget_if(t: &[i64], x: f64, line: u32) -> R<i64> {
    let k = whole(x);
    if k as f64 == x && (k as u64) < t.len() as u64 { Ok(unsafe { *t.get_unchecked(k as usize) }) } else { tget_slow_i(t, Value::Num(x), line) }
}

#[inline(always)]
pub fn tget_a(t: &[f64], i: i64, num: bool, line: u32) -> R<f64> {
    if (i as u64) < t.len() as u64 { Ok(unsafe { *t.get_unchecked(i as usize) }) } else { tget_slow_a(t, tkey(i, num), line) }
}

#[inline(always)]
pub fn tget_af(t: &[f64], x: f64, line: u32) -> R<f64> {
    let k = whole(x);
    if k as f64 == x && (k as u64) < t.len() as u64 { Ok(unsafe { *t.get_unchecked(k as usize) }) } else { tget_slow_a(t, Value::Num(x), line) }
}

#[inline(always)]
pub fn tset_i(t: &mut [i64], i: i64, num: bool, x: i64, line: u32) -> R<()> {
    if (i as u64) < t.len() as u64 {
        unsafe { *t.get_unchecked_mut(i as usize) = x };
        Ok(())
    } else {
        tset_slow_i(t, tkey(i, num), x, line)
    }
}

#[inline(always)]
pub fn tset_if(t: &mut [i64], i: f64, x: i64, line: u32) -> R<()> {
    let k = whole(i);
    if k as f64 == i && (k as u64) < t.len() as u64 {
        unsafe { *t.get_unchecked_mut(k as usize) = x };
        Ok(())
    } else {
        tset_slow_i(t, Value::Num(i), x, line)
    }
}

#[inline(always)]
pub fn tset_a(t: &mut [f64], i: i64, num: bool, x: f64, line: u32) -> R<()> {
    if (i as u64) < t.len() as u64 {
        unsafe { *t.get_unchecked_mut(i as usize) = x };
        Ok(())
    } else {
        tset_slow_a(t, tkey(i, num), x, line)
    }
}

#[inline(always)]
pub fn tset_af(t: &mut [f64], i: f64, x: f64, line: u32) -> R<()> {
    let k = whole(i);
    if k as f64 == i && (k as u64) < t.len() as u64 {
        unsafe { *t.get_unchecked_mut(k as usize) = x };
        Ok(())
    } else {
        tset_slow_a(t, Value::Num(i), x, line)
    }
}

/// t[key] for a table with a position of any kind (the result as a value).
#[inline]
pub fn tget_iv(t: &[i64], key: Value, line: u32) -> R<Value> {
    let k = match key {
        Value::Int(k) => k,
        Value::Num(x) if whole(x) as f64 == x => whole(x),
        _ => -1,
    };
    if (k as u64) < t.len() as u64 {
        return Ok(Value::Int(t[k as usize]));
    }
    at(interp::index_value(&Value::Ints(Rc::new(t.to_vec())), &[key]), line)
}

#[inline]
pub fn tget_av(t: &[f64], key: Value, line: u32) -> R<Value> {
    let k = match key {
        Value::Int(k) => k,
        Value::Num(x) if whole(x) as f64 == x => whole(x),
        _ => -1,
    };
    if (k as u64) < t.len() as u64 {
        return Ok(Value::Num(t[k as usize]));
    }
    at(interp::index_value(&Value::Tensor(arr_of(t.to_vec())), &[key]), line)
}

// ---------- compact rows and matrices ----------
// A compact matrix's rows are read and written with plain row assignment
// (m[d] = update_rows(m[s], ...), m[d] = m[s], screlu_dot(m[r], w[k], cap)).
// Compiled programs do these in place on the matrix's memory; the results
// and problems are the same as doing them step by step.

pub use crate::value::Packed;
/// Compact numbers known to be compact numbers.
pub type PK = Rc<Packed>;

pub fn pk0() -> PK {
    Rc::new(Packed { data: RefCell::new(vec![]), rows: 1, cols: 0, matrix: false, max_abs: std::cell::Cell::new(-1) })
}

pub fn as_pk(v: Value, line: u32) -> R<PK> {
    match v {
        Value::Packed(p) => Ok(p),
        other => wrong_kind(&other, "compact numbers", line),
    }
}

/// A position into something with `n` parts: the fast case, otherwise the
/// general rules (negative positions count from the end) and messages.
#[inline(always)]
fn pos_of(k: &Value, n: usize, line: u32) -> R<usize> {
    let i = match k {
        Value::Int(i) => *i,
        Value::Num(x) => {
            let w = whole(*x);
            if w as f64 == *x { w } else { -1 }
        }
        _ => -1,
    };
    if (i as u64) < n as u64 {
        return Ok(i as usize);
    }
    at(interp::to_index(k, n), line)
}

/// A row position given to the compact-matrix operations: an integer or a
/// number when the compiler knows it, any value otherwise.
pub trait PkKey {
    fn pos(&self, n: usize, line: u32) -> R<usize>;
    fn value(self) -> Value;
}

impl PkKey for i64 {
    #[inline(always)]
    fn pos(&self, n: usize, line: u32) -> R<usize> {
        if (*self as u64) < n as u64 {
            return Ok(*self as usize);
        }
        at(interp::to_index(&Value::Int(*self), n), line)
    }
    fn value(self) -> Value {
        Value::Int(self)
    }
}

impl PkKey for f64 {
    #[inline(always)]
    fn pos(&self, n: usize, line: u32) -> R<usize> {
        let w = whole(*self);
        if w as f64 == *self && (w as u64) < n as u64 {
            return Ok(w as usize);
        }
        at(interp::to_index(&Value::Num(*self), n), line)
    }
    fn value(self) -> Value {
        Value::Num(self)
    }
}

impl PkKey for Value {
    #[inline(always)]
    fn pos(&self, n: usize, line: u32) -> R<usize> {
        pos_of(self, n, line)
    }
    fn value(self) -> Value {
        self
    }
}

/// Makes a compact value unshared before it is changed in place.
#[inline]
fn pk_unique(m: &mut PK) {
    if Rc::strong_count(m) != 1 {
        let copy = Packed {
            data: RefCell::new(m.data.borrow().clone()),
            rows: m.rows,
            cols: m.cols,
            matrix: m.matrix,
            max_abs: std::cell::Cell::new(-1),
        };
        *m = Rc::new(copy);
    }
    m.max_abs.set(-1);
}

/// m[k] read the general way.
pub fn pk_index(m: &PK, k: Value, line: u32) -> R<Value> {
    at(interp::index_value(&Value::Packed(m.clone()), &[k]), line)
}

/// m[k] = v the general way.
pub fn pk_set(m: &mut PK, k: Value, v: Value, line: u32) -> R<()> {
    let mut val = Value::Packed(std::mem::replace(m, pk0()));
    let r = interp::set_index(&mut val, k, v);
    *m = as_pk(val, line)?;
    at(r, line)
}

fn bcall_update_rows(base: Value, w: &PK, adds: &T, n_add: f64, subs: &T, n_sub: f64, line: u32) -> R<Value> {
    at(
        crate::builtins::update_rows_slice(&[
            base,
            Value::Packed(w.clone()),
            Value::Tensor(adds.clone()),
            Value::Num(n_add),
            Value::Tensor(subs.clone()),
            Value::Num(n_sub),
        ]),
        line,
    )
}

/// m[d] = update_rows(base, w, ...) done step by step.
fn pk_update_general(m: &mut PK, d: Value, base: Value, w: &PK, adds: &T, n_add: f64, subs: &T, n_sub: f64, line: u32) -> R<()> {
    let row = bcall_update_rows(base, w, adds, n_add, subs, n_sub, line)?;
    pk_set(m, d, row, line)
}

/// m[d] = update_rows(m[s], w, adds, n_add, subs, n_sub)
pub fn pk_update_from<D: PkKey, S: PkKey>(m: &mut PK, d: D, s: S, w: &PK, adds: &T, n_add: f64, subs: &T, n_sub: f64, line: u32) -> R<()> {
    if !m.matrix || Rc::ptr_eq(m, w) || !w.matrix || w.cols != m.cols {
        let base = pk_index(m, s.value(), line)?;
        return pk_update_general(m, d.value(), base, w, adds, n_add, subs, n_sub, line);
    }
    let si = s.pos(m.rows, line)?;
    let h = m.cols;
    // Usual case: a different row of the same matrix, written in one pass with
    // no row in between. Row lists are checked before anything is written.
    if let Ok(di) = d.pos(m.rows, line) {
        if di != si {
            pk_unique(m);
            let wd = w.data.borrow();
            let mut md = m.data.borrow_mut();
            let (base, out): (&[i16], &mut [i16]) = if si < di {
                let (lo, hi) = md.split_at_mut(di * h);
                (&lo[si * h..(si + 1) * h], &mut hi[..h])
            } else {
                let (lo, hi) = md.split_at_mut(si * h);
                (&hi[..h], &mut lo[di * h..(di + 1) * h])
            };
            return at(crate::builtins::update_rows_into(out, base, w.rows, &wd, adds, n_add, subs, n_sub), line);
        }
    }
    let mut out = vec![0i16; h];
    {
        let (md, wd) = (m.data.borrow(), w.data.borrow());
        at(crate::builtins::update_rows_into(&mut out, &md[si * h..(si + 1) * h], w.rows, &wd, adds, n_add, subs, n_sub), line)?;
    }
    let di = d.pos(m.rows, line)?;
    pk_unique(m);
    m.data.borrow_mut()[di * h..(di + 1) * h].copy_from_slice(&out);
    Ok(())
}

/// m[d] = update_rows(base, w, ...) with `base` a compact row.
pub fn pk_update_with<D: PkKey>(m: &mut PK, d: D, base: &PK, w: &PK, adds: &T, n_add: f64, subs: &T, n_sub: f64, line: u32) -> R<()> {
    let plain = m.matrix && !base.matrix && !Rc::ptr_eq(m, w) && !Rc::ptr_eq(m, base) && w.matrix && w.cols == base.cols && base.cols == m.cols;
    // Usual case: written straight into row d (the row lists are checked
    // before anything is written). A bad row d is left to the general way, so
    // which problem is reported stays the same.
    if let (true, Ok(di)) = (plain, d.pos(m.rows, line)) {
        let h = m.cols;
        pk_unique(m);
        let (bd, wd) = (base.data.borrow(), w.data.borrow());
        let mut md = m.data.borrow_mut();
        return at(crate::builtins::update_rows_into(&mut md[di * h..(di + 1) * h], &bd, w.rows, &wd, adds, n_add, subs, n_sub), line);
    }
    pk_update_general(m, d.value(), Value::Packed(base.clone()), w, adds, n_add, subs, n_sub, line)
}

/// m[d] = m[s]
pub fn pk_copy_row<D: PkKey, S: PkKey>(m: &mut PK, d: D, s: S, line: u32) -> R<()> {
    if !m.matrix {
        let v = pk_index(m, s.value(), line)?;
        return pk_set(m, d.value(), v, line);
    }
    let s = s.pos(m.rows, line)?;
    let d = d.pos(m.rows, line)?;
    pk_unique(m);
    let h = m.cols;
    m.data.borrow_mut().copy_within(s * h..(s + 1) * h, d * h);
    Ok(())
}

/// screlu_dot(x, w, cap) where x and w are compact rows, or rows `xr` / `wr`
/// of compact matrices.
pub fn pk_screlu<X: PkKey, W: PkKey>(x: &PK, xr: Option<X>, w: &PK, wr: Option<W>, cap: f64, line: u32) -> R<f64> {
    if xr.is_some() != x.matrix || wr.is_some() != w.matrix {
        // not the plain case: do it step by step
        let xv = match xr {
            Some(k) => pk_index(x, k.value(), line)?,
            None => Value::Packed(x.clone()),
        };
        let wv = match wr {
            Some(k) => pk_index(w, k.value(), line)?,
            None => Value::Packed(w.clone()),
        };
        return as_num(at(crate::builtins::screlu_dot_slice(&[xv, wv, Value::Num(cap)]), line)?, line);
    }
    let xi = match &xr {
        Some(k) => k.pos(x.rows, line)?,
        None => 0,
    };
    let wi = match &wr {
        Some(k) => k.pos(w.rows, line)?,
        None => 0,
    };
    let (xd, wd) = (x.data.borrow(), w.data.borrow());
    let (xs, ws) = (&xd[xi * x.cols..(xi + 1) * x.cols], &wd[wi * w.cols..(wi + 1) * w.cols]);
    at(crate::builtins::screlu_dot_rows(xs, ws, || crate::builtins::max_abs(w), cap), line)
}

/// argmax(table, first, last) for a fixed table of numbers.
#[inline]
pub fn t_argmax(t: &[f64], first: f64, last: f64, line: u32) -> R<f64> {
    at(crate::builtins::argmax_range(t, first, last), line)
}

/// argmax(row, first, last) for a row variable.
#[inline]
pub fn a_argmax(a: &T, first: f64, last: f64, line: u32) -> R<f64> {
    if a.shape.len() != 1 {
        return fail("argmax with a range needs a row of numbers", line);
    }
    let d = unsafe { &*a.data.as_ptr() };
    at(crate::builtins::argmax_range(d, first, last), line)
}

/// number(x) without the cost of a general built-in call.
#[inline]
pub fn to_number(v: &Value, line: u32) -> R<f64> {
    match v {
        Value::Num(n) => Ok(*n),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Str(st) => match st.trim().replace('_', "").parse::<f64>() {
            Ok(n) => Ok(n),
            Err(_) => fail(format!("\"{}\" isn't a number", st), line),
        },
        other => match other.scalar() {
            Some(n) => Ok(n),
            None => fail(format!("number needs a number, not a {}", other.kind()), line),
        },
    }
}

/// A plain row of numbers.
pub fn as_arr(v: Value, line: u32) -> R<T> {
    match v {
        Value::Tensor(t) if t.shape.len() == 1 && !t.requires => Ok(t),
        other => wrong_kind(&other, "row of numbers", line),
    }
}

/// A row of numbers known when the program was compiled.
pub fn arr_of(data: Vec<f64>) -> T {
    tensor::new(vec![data.len()], data)
}

/// The starting value of an array variable.
#[inline]
pub fn arr0() -> T {
    unsafe { (*EMPTY_ARR).clone() }
}

#[inline(always)]
pub fn alen(a: &T) -> usize {
    unsafe { (&*a.data.as_ptr()).len() }
}

/// Element k of an array, for loops that already know k is in range.
#[inline(always)]
pub fn aidx(a: &T, k: usize) -> f64 {
    unsafe { *(&*a.data.as_ptr()).get_unchecked(k) }
}

/// A number as a whole number, in one machine instruction. Anything that is
/// not a whole number in range comes back as a value that fails the
/// `k as f64 == x` test the callers make.
#[inline(always)]
pub fn whole(x: f64) -> i64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_cvttsd_si64(core::arch::x86_64::_mm_set_sd(x))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        x as i64
    }
}

/// x % c for a whole-number constant c (the compiler turns the division into
/// multiplication and shifts).
#[inline(always)]
pub fn rem_i(x: f64, c: i64) -> f64 {
    let k = whole(x);
    if k as f64 == x {
        k.rem_euclid(c) as f64
    } else {
        x.rem_euclid(c as f64)
    }
}

/// x % y in general.
#[inline(always)]
pub fn rem_f(x: f64, y: f64) -> f64 {
    let (a, b) = (whole(x), whole(y));
    if a as f64 == x && b as f64 == y && b != 0 {
        if a as i32 as i64 == a && b as i32 as i64 == b && b != -1 {
            (a as i32).rem_euclid(b as i32) as f64
        } else if b != -1 {
            a.rem_euclid(b) as f64
        } else {
            0.0
        }
    } else {
        x.rem_euclid(y)
    }
}

#[inline(always)]
pub fn floor_f(x: f64) -> f64 {
    let k = whole(x);
    let f = k as f64;
    if k != i64::MIN {
        if f > x {
            f - 1.0
        } else {
            f
        }
    } else {
        x.floor()
    }
}

#[inline(always)]
pub fn trunc_f(x: f64) -> f64 {
    let k = whole(x);
    if k != i64::MIN {
        k as f64
    } else {
        x.trunc()
    }
}

#[cold]
fn aget_slow(a: &T, i: f64, line: u32) -> R<f64> {
    let v = at(interp::index_value(&Value::Tensor(a.clone()), &[Value::Num(i)]), line)?;
    as_num(v, line)
}

/// a[i] where a is a plain row of numbers.
#[inline(always)]
pub fn aget(a: &T, i: f64, line: u32) -> R<f64> {
    let d = unsafe { &*a.data.as_ptr() };
    let k = whole(i);
    if (k as u64) < d.len() as u64 && k as f64 == i {
        Ok(unsafe { *d.get_unchecked(k as usize) })
    } else {
        aget_slow(a, i, line)
    }
}

#[cold]
fn aset_slow(a: &mut T, i: f64, x: f64, line: u32) -> R<()> {
    let mut v = Value::Tensor(a.clone());
    // drop our own handle first so the row is changed in place when possible
    *a = arr0();
    let r = interp::set_index(&mut v, Value::Num(i), Value::Num(x));
    *a = as_arr(v, line)?;
    at(r, line)
}

/// a[i] = x. Changes the row in place unless another variable shares it.
#[inline(always)]
pub fn aset(a: &mut T, i: f64, x: f64, line: u32) -> R<()> {
    let k = whole(i);
    if k as f64 == i && Rc::strong_count(a) == 1 {
        let d = unsafe { &mut *a.data.as_ptr() };
        if (k as u64) < d.len() as u64 {
            unsafe { *d.get_unchecked_mut(k as usize) = x };
            return Ok(());
        }
    }
    aset_slow(a, i, x, line)
}
