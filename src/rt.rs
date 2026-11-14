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

/// Calls a built-in that takes its inputs as a slice (no list is allocated).
#[inline]
pub fn scall(f: fn(&[Value]) -> Result<Value, String>, args: &[Value], line: u32) -> R<Value> {
    at(f(args), line)
}

/// name[key] = v for one index. Lists are changed in place without building a path.
pub fn set1(target: &mut Value, key: Value, v: Value, name: &str, line: u32) -> R<()> {
    if let (Value::List(rc), Value::Num(i)) = (&mut *target, &key) {
        let k = whole(*i);
        if k as f64 == *i && (k as u64) < rc.len() as u64 {
            Rc::make_mut(rc)[k as usize] = v;
            return Ok(());
        }
    }
    set_path(target, vec![Key::Idx(vec![key])], v, name, line)
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
