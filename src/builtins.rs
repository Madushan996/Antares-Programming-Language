//! The functions every Antares program starts with.

use crate::interp::{iter_items, Interp};
use crate::json::{self, J};
use crate::tensor::{self, UFn, T};
use crate::value::{fmt_num, pack_list, wrap, NativeFn, Packed, Value};
use std::cell::Cell;
use std::io::Write;
use std::rc::Rc;

type Res = Result<Value, String>;

// ---------- random numbers ----------

thread_local! {
    static RNG: Cell<u64> = Cell::new(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    );
}

fn next_u64() -> u64 {
    RNG.with(|r| {
        let mut z = r.get().wrapping_add(0x9E3779B97F4A7C15);
        r.set(z);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    })
}

pub(crate) fn uniform() -> f64 {
    (next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

pub(crate) fn normal() -> f64 {
    let u1 = uniform().max(1e-300);
    let u2 = uniform();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ---------- argument helpers ----------

pub(crate) fn want(a: &[Value], n: usize, name: &str) -> Result<(), String> {
    if a.len() != n {
        Err(format!("{} takes {} input{} but was given {}", name, n, if n == 1 { "" } else { "s" }, a.len()))
    } else {
        Ok(())
    }
}

pub(crate) fn num(v: &Value, name: &str) -> Result<f64, String> {
    v.scalar().ok_or_else(|| format!("{} needs a number, not a {}", name, v.kind()))
}

pub(crate) fn ten(v: &Value, name: &str) -> Result<T, String> {
    v.as_tensor().ok_or_else(|| format!("{} needs numbers, not a {}", name, v.kind()))
}

pub(crate) fn text(v: &Value, name: &str) -> Result<Rc<str>, String> {
    match v {
        Value::Str(s) => Ok(s.clone()),
        other => Err(format!("{} needs text, not a {}", name, other.kind())),
    }
}

pub(crate) fn s(x: impl AsRef<str>) -> Value {
    Value::Str(x.as_ref().into())
}

fn dims(a: &[Value], name: &str) -> Result<Vec<usize>, String> {
    let vals: Vec<f64> = match a {
        [Value::Tensor(t)] if t.shape.len() == 1 => t.vals().clone(),
        _ => {
            let mut v = vec![];
            for x in a {
                v.push(num(x, name)?);
            }
            v
        }
    };
    let mut out = vec![];
    for x in vals {
        if x < 0.0 || x != x.trunc() {
            return Err(format!("{} needs whole, non-negative sizes", name));
        }
        out.push(x as usize);
    }
    Ok(out)
}

fn filled(a: Vec<Value>, name: &str, mut f: impl FnMut() -> f64) -> Res {
    let shape = dims(&a, name)?;
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n).map(|_| f()).collect();
    Ok(wrap(tensor::new(shape, data)))
}

fn unary(a: Vec<Value>, name: &str, f: UFn) -> Res {
    want(&a, 1, name)?;
    match &a[0] {
        Value::Num(x) => Ok(Value::Num(tensor::apply(f, *x))),
        Value::Tensor(t) => Ok(wrap(tensor::unary(f, t))),
        other => Err(format!("{} needs a number or tensor, not a {}", name, other.kind())),
    }
}

fn elementwise(a: Vec<Value>, name: &str, f: fn(f64) -> f64) -> Res {
    want(&a, 1, name)?;
    let t = ten(&a[0], name)?;
    let d: Vec<f64> = t.vals().iter().map(|x| f(*x)).collect();
    Ok(wrap(tensor::new(t.shape.clone(), d)))
}

macro_rules! un {
    ($fname:ident, $name:expr, $f:expr) => {
        fn $fname(_: &mut Interp, a: Vec<Value>) -> Res {
            unary(a, $name, $f)
        }
    };
}

un!(b_exp, "exp", UFn::Exp);
un!(b_log, "log", UFn::Log);
un!(b_tanh, "tanh", UFn::Tanh);
un!(b_sigmoid, "sigmoid", UFn::Sigmoid);
un!(b_relu, "relu", UFn::Relu);
un!(b_sqrt, "sqrt", UFn::Sqrt);
un!(b_abs, "abs", UFn::Abs);
un!(b_sin, "sin", UFn::Sin);
un!(b_cos, "cos", UFn::Cos);

// ---------- tensors ----------

fn b_zeros(_: &mut Interp, a: Vec<Value>) -> Res {
    filled(a, "zeros", || 0.0)
}

fn b_ones(_: &mut Interp, a: Vec<Value>) -> Res {
    filled(a, "ones", || 1.0)
}

fn b_rand(_: &mut Interp, a: Vec<Value>) -> Res {
    filled(a, "rand", uniform)
}

fn b_randn(_: &mut Interp, a: Vec<Value>) -> Res {
    filled(a, "randn", normal)
}

fn b_seed(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "seed")?;
    let n = num(&a[0], "seed")?;
    RNG.with(|r| r.set(n.to_bits() ^ 0xA076_1D64_78BD_642F));
    Ok(Value::Nothing)
}

fn b_range(_: &mut Interp, a: Vec<Value>) -> Res {
    let (start, end, step) = match a.len() {
        1 => (0.0, num(&a[0], "range")?, 1.0),
        2 => (num(&a[0], "range")?, num(&a[1], "range")?, 1.0),
        3 => (num(&a[0], "range")?, num(&a[1], "range")?, num(&a[2], "range")?),
        _ => return Err("range takes 1 to 3 inputs: range(end), range(start, end) or range(start, end, step)".into()),
    };
    if step == 0.0 {
        return Err("range can't count in steps of 0".into());
    }
    let mut data = vec![];
    let mut x = start;
    while (step > 0.0 && x < end) || (step < 0.0 && x > end) {
        data.push(x);
        x += step;
    }
    Ok(Value::Tensor(tensor::new(vec![data.len()], data)))
}

fn b_shape(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "shape")?;
    let t = ten(&a[0], "shape")?;
    let d: Vec<f64> = t.shape.iter().map(|x| *x as f64).collect();
    Ok(Value::Tensor(tensor::new(vec![d.len()], d)))
}

fn b_reshape(_: &mut Interp, a: Vec<Value>) -> Res {
    if a.len() < 2 {
        return Err("reshape needs a tensor and the new sizes, like reshape(x, 2, 3)".into());
    }
    let t = ten(&a[0], "reshape")?;
    Ok(wrap(tensor::reshape(&t, dims(&a[1..], "reshape")?)?))
}

fn b_t(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "t")?;
    Ok(wrap(tensor::transpose(&ten(&a[0], "t")?)?))
}

fn reduce_axis(a: Vec<Value>, name: &str, all: fn(&T) -> T, along: fn(&T, i64) -> Result<T, String>) -> Res {
    match a.len() {
        1 => Ok(wrap(all(&ten(&a[0], name)?))),
        2 => {
            let ax = num(&a[1], name)?;
            if ax != ax.trunc() {
                return Err(format!("{}: the dimension must be a whole number", name));
            }
            Ok(wrap(along(&ten(&a[0], name)?, ax as i64)?))
        }
        n => Err(format!("{} takes a tensor and, optionally, a dimension to work along, but was given {} inputs", name, n)),
    }
}

fn b_sum(_: &mut Interp, a: Vec<Value>) -> Res {
    reduce_axis(a, "sum", tensor::sum, tensor::sum_axis)
}

fn b_mean(_: &mut Interp, a: Vec<Value>) -> Res {
    reduce_axis(a, "mean", tensor::mean, tensor::mean_axis)
}

fn b_softmax(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "softmax")?;
    Ok(wrap(tensor::softmax(&ten(&a[0], "softmax")?)))
}

fn extreme(a: Vec<Value>, name: &str, better: fn(f64, f64) -> bool) -> Res {
    let mut best: Option<f64> = None;
    for v in &a {
        let t = ten(v, name)?;
        for &x in t.vals().iter() {
            if best.map_or(true, |b| better(x, b)) {
                best = Some(x);
            }
        }
    }
    best.map(Value::Num).ok_or_else(|| format!("{} needs at least one number", name))
}

fn b_max(_: &mut Interp, a: Vec<Value>) -> Res {
    extreme(a, "max", |x, b| x > b)
}

fn b_min(_: &mut Interp, a: Vec<Value>) -> Res {
    extreme(a, "min", |x, b| x < b)
}

/// The position of the first largest number of `s` (as from a plain scan that
/// keeps the first value unless a later one is larger), plus `base`; -1 if
/// `s` is empty. Four running lanes, so the processor can compare several
/// numbers at once.
pub fn argmax_slice(s: &[f64], base: usize) -> f64 {
    if s.is_empty() {
        return -1.0;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if s.len() >= 8 && !s[0].is_nan() && std::arch::is_x86_feature_detected!("avx2") {
            return (base + unsafe { argmax_avx2(s) }) as f64;
        }
    }
    let mut bi = 0;
    for i in 1..s.len() {
        if s[i] > s[bi] {
            bi = i;
        }
    }
    (base + bi) as f64
}

/// The same for at least 8 numbers, the first not NaN: four numbers per step,
/// each lane keeping its largest value and where it was first seen.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn argmax_avx2(s: &[f64]) -> usize {
    unsafe {
        use std::arch::x86_64::*;
        let n = s.len() / 4 * 4;
        let mut vmax = _mm256_set1_pd(f64::NEG_INFINITY);
        let mut vat = _mm256_set1_pd(-1.0);
        let mut cur = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
        let four = _mm256_set1_pd(4.0);
        let mut i = 0;
        while i < n {
            let x = _mm256_loadu_pd(s.as_ptr().add(i));
            let gt = _mm256_cmp_pd::<_CMP_GT_OQ>(x, vmax);
            vmax = _mm256_blendv_pd(vmax, x, gt);
            vat = _mm256_blendv_pd(vat, cur, gt);
            cur = _mm256_add_pd(cur, four);
            i += 4;
        }
        let mut maxes = [0.0f64; 4];
        let mut ats = [0.0f64; 4];
        _mm256_storeu_pd(maxes.as_mut_ptr(), vmax);
        _mm256_storeu_pd(ats.as_mut_ptr(), vat);
        let (mut best, mut at) = (f64::NEG_INFINITY, usize::MAX);
        for l in 0..4 {
            if ats[l] >= 0.0 && (maxes[l] > best || (maxes[l] == best && (ats[l] as usize) < at)) {
                best = maxes[l];
                at = ats[l] as usize;
            }
        }
        for k in n..s.len() {
            if s[k] > best {
                best = s[k];
                at = k;
            }
        }
        // nothing above minus infinity: the scan keeps the first number
        if at == usize::MAX { 0 } else { at }
    }
}

/// argmax(row, first, last) for a row of numbers `d`.
pub fn argmax_range(d: &[f64], first: f64, last: f64) -> Result<f64, String> {
    if first != first.trunc() || last != last.trunc() || first < 0.0 || last < first || last > d.len() as f64 {
        return Err(format!(
            "argmax: positions {} to {} aren't a range inside this row (0 to {})",
            fmt_num(first),
            fmt_num(last),
            d.len()
        ));
    }
    Ok(argmax_slice(&d[first as usize..last as usize], first as usize))
}

fn b_argmax(_: &mut Interp, a: Vec<Value>) -> Res {
    if a.len() == 3 {
        // argmax(row, first, last): the first largest number from first up to (not including) last
        let t = ten(&a[0], "argmax")?;
        if t.shape.len() != 1 {
            return Err("argmax with a range needs a row of numbers".into());
        }
        let (first, last) = (num(&a[1], "argmax")?, num(&a[2], "argmax")?);
        return Ok(Value::Num(argmax_range(&t.vals(), first, last)?));
    }
    want(&a, 1, "argmax")?;
    let t = ten(&a[0], "argmax")?;
    let d = t.shape.last().copied().unwrap_or(1).max(1);
    let data = t.vals();
    let picks: Vec<f64> = data
        .chunks(d)
        .map(|row| {
            let mut bi = 0;
            for (i, x) in row.iter().enumerate() {
                if *x > row[bi] {
                    bi = i;
                }
            }
            bi as f64
        })
        .collect();
    let shape = if t.shape.is_empty() { vec![] } else { t.shape[..t.shape.len() - 1].to_vec() };
    Ok(wrap(tensor::new(shape, picks)))
}

fn b_onehot(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "onehot")?;
    let labels = ten(&a[0], "onehot")?;
    let n = num(&a[1], "onehot")? as usize;
    let ld = labels.vals();
    let mut data = vec![0.0; ld.len() * n];
    for (i, &l) in ld.iter().enumerate() {
        if l < 0.0 || l as usize >= n {
            return Err(format!("onehot: label {} doesn't fit in {} classes", fmt_num(l), n));
        }
        data[i * n + l as usize] = 1.0;
    }
    let mut shape = labels.shape.clone();
    shape.push(n);
    Ok(Value::Tensor(tensor::new(shape, data)))
}

fn b_grad(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "grad")?;
    let loss = ten(&a[0], "grad")?;
    let wrt = ten(&a[1], "grad")?;
    let g = tensor::grad_of(&loss, &wrt);
    Ok(wrap(tensor::new(wrt.shape.clone(), g)))
}

fn b_floor(_: &mut Interp, a: Vec<Value>) -> Res {
    elementwise(a, "floor", f64::floor)
}

fn b_round(_: &mut Interp, a: Vec<Value>) -> Res {
    elementwise(a, "round", f64::round)
}

fn b_trunc(_: &mut Interp, a: Vec<Value>) -> Res {
    elementwise(a, "trunc", f64::trunc)
}

/// clip(x, low, high): every number forced into the range low..high.
fn b_clip(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 3, "clip")?;
    let t = ten(&a[0], "clip")?;
    if t.requires {
        return Err("clip can't be learned through; use relu or your own formula on `learn` values".into());
    }
    let (lo, hi) = (num(&a[1], "clip")?, num(&a[2], "clip")?);
    let d: Vec<f64> = t.vals().iter().map(|x| x.max(lo).min(hi)).collect();
    Ok(wrap(tensor::new(t.shape.clone(), d)))
}

/// sum_rows(matrix, rows, count?): adds up the chosen rows of a matrix.
fn b_sum_rows(_: &mut Interp, a: Vec<Value>) -> Res {
    if a.len() < 2 || a.len() > 3 {
        return Err("sum_rows takes a matrix, the row numbers, and optionally how many of them to use".into());
    }
    let m = ten(&a[0], "sum_rows")?;
    let rows = ten(&a[1], "sum_rows")?;
    if m.shape.len() != 2 {
        return Err("sum_rows needs a matrix (rows and columns)".into());
    }
    let (n, h) = (m.shape[0], m.shape[1]);
    let rd = rows.vals();
    let count = if a.len() == 3 { num(&a[2], "sum_rows")? as usize } else { rd.len() };
    if count > rd.len() {
        return Err(format!("sum_rows was asked for {} rows but only {} row numbers were given", count, rd.len()));
    }
    let md = m.vals();
    let mut out = vec![0.0; h];
    for &r in &rd[..count] {
        if r < 0.0 || r as usize >= n {
            return Err(format!("sum_rows: row {} doesn't exist (the matrix has {} rows)", fmt_num(r), n));
        }
        let row = &md[r as usize * h..(r as usize + 1) * h];
        for (o, x) in out.iter_mut().zip(row) {
            *o += x;
        }
    }
    Ok(Value::Tensor(tensor::new(vec![h], out)))
}

fn packed(data: Vec<i16>, rows: usize, cols: usize, matrix: bool) -> Value {
    Value::Packed(Rc::new(Packed { data: std::cell::RefCell::new(data), rows, cols, matrix, max_abs: std::cell::Cell::new(-1) }))
}

/// compact(x) or compact(x, rows, cols): whole numbers stored in 2 bytes each
/// instead of 8. Much faster for the row built-ins on large tables.
fn b_compact(_: &mut Interp, a: Vec<Value>) -> Res {
    if a.len() != 1 && a.len() != 3 {
        return Err("compact takes a row or matrix of numbers, and optionally the rows and columns to arrange it in".into());
    }
    let t = plain(&a[0], "compact")?;
    let d = t.vals();
    let (rows, cols, matrix) = if a.len() == 3 {
        (num(&a[1], "compact")? as usize, num(&a[2], "compact")? as usize, true)
    } else {
        match t.shape.len() {
            1 => (1, t.shape[0], false),
            2 => (t.shape[0], t.shape[1], true),
            _ => return Err("compact works on a row or a matrix".into()),
        }
    };
    if rows * cols != d.len() {
        return Err(format!("compact: {} numbers can't be arranged as {} rows of {}", d.len(), rows, cols));
    }
    let mut out = Vec::with_capacity(d.len());
    for &x in d.iter() {
        if x != x.trunc() || !(-32768.0..=32767.0).contains(&x) {
            return Err(format!("compact needs whole numbers from -32768 to 32767, but found {}", fmt_num(x)));
        }
        out.push(x as i16);
    }
    Ok(packed(out, rows, cols, matrix))
}

/// expand(c): compact numbers back as an ordinary row or matrix.
fn b_expand(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "expand")?;
    match &a[0] {
        Value::Packed(p) => {
            let d: Vec<f64> = p.vals().iter().map(|x| *x as f64).collect();
            let shape = if p.matrix { vec![p.rows, p.cols] } else { vec![p.cols] };
            Ok(Value::Tensor(tensor::new(shape, d)))
        }
        other => Err(format!("expand needs compact numbers, not a {}", other.kind())),
    }
}

/// The compact versions of update_rows and screlu_dot.
fn update_rows_packed(a: &[Value], base: &Rc<Packed>, m: &Rc<Packed>) -> Res {
    if !m.matrix || base.matrix || base.cols != m.cols {
        return Err("update_rows needs a compact matrix whose rows are as long as the compact base row".into());
    }
    let h = m.cols;
    let md = m.vals();
    let recycled: Option<Rc<Packed>> = match a.get(6) {
        Some(Value::Packed(t)) if !t.matrix && t.cols == h && Rc::strong_count(t) == 2 && !Rc::ptr_eq(t, base) => {
            Some(t.clone())
        }
        _ => None,
    };
    let (adds, subs) = (ten(&a[2], "update_rows")?, ten(&a[4], "update_rows")?);
    let (n_add, n_sub) = (num(&a[3], "update_rows")?, num(&a[5], "update_rows")?);
    let bd = base.vals();
    let mut fresh_row: Vec<i16> = Vec::new();
    let mut held;
    let out: &mut [i16] = match &recycled {
        Some(t) => {
            t.max_abs.set(-1);
            held = t.vals_mut();
            &mut held[..]
        }
        None => {
            fresh_row = vec![0i16; h];
            &mut fresh_row[..]
        }
    };
    update_rows_into(out, &bd, m.rows, &md, &adds, n_add, &subs, n_sub)?;
    drop(bd);
    match &recycled {
        Some(t) => Ok(Value::Packed(t.clone())),
        None => Ok(packed(fresh_row, 1, h, false)),
    }
}

/// update_rows on compact numbers, into `out`: base plus the rows of the
/// matrix `md` (`rows` rows) listed in `adds` (the first `n_add` of them), minus
/// those listed in `subs`. Checks the row lists as update_rows does.
pub fn update_rows_into(
    out: &mut [i16],
    base: &[i16],
    rows: usize,
    md: &[i16],
    adds: &T,
    n_add: f64,
    subs: &T,
    n_sub: f64,
) -> Result<(), String> {
    let h = base.len();
    // Collect the rows to add and to remove, then do everything in one pass.
    // (kept on the stack, and not filled in advance: allocating or clearing even
    // a small list here costs more than the arithmetic)
    use std::mem::MaybeUninit;
    let mut add_rows: [MaybeUninit<&[i16]>; 32] = [const { MaybeUninit::uninit() }; 32];
    let mut sub_rows: [MaybeUninit<&[i16]>; 32] = [const { MaybeUninit::uninit() }; 32];
    let (mut na, mut ns) = (0usize, 0usize);
    for (list, count, add) in [(adds, n_add, true), (subs, n_sub, false)] {
        let rd = list.vals();
        let count = count as usize;
        if count > rd.len() {
            return Err("update_rows was asked for more rows than row numbers were given".into());
        }
        for &r in &rd[..count] {
            if r < 0.0 || r as usize >= rows {
                return Err(format!("update_rows: row {} doesn't exist (the matrix has {} rows)", fmt_num(r), rows));
            }
            let row = &md[r as usize * h..(r as usize + 1) * h];
            if add {
                if na == 32 {
                    return Err("update_rows adds at most 32 rows at a time".into());
                }
                add_rows[na].write(row);
                na += 1;
            } else {
                if ns == 32 {
                    return Err("update_rows removes at most 32 rows at a time".into());
                }
                sub_rows[ns].write(row);
                ns += 1;
            }
        }
    }
    // only the first na and ns entries were written, and only those are read
    let (adds, subs) = unsafe {
        (
            std::slice::from_raw_parts(add_rows.as_ptr() as *const &[i16], na),
            std::slice::from_raw_parts(sub_rows.as_ptr() as *const &[i16], ns),
        )
    };
    combine_rows(out, base, adds, subs);
    Ok(())
}

/// out = base + every row in `adds` - every row in `subs`, wrapping in 16 bits.
fn combine_rows(out: &mut [i16], base: &[i16], adds: &[&[i16]], subs: &[&[i16]]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // A move adds and removes one or two rows; with the counts fixed the
            // inner loops disappear and only the arithmetic is left.
            if out.len() % 16 == 0 && base.len() == out.len() {
                macro_rules! fixed {
                    ($a:literal, $s:literal) => {{
                        let mut ap = [std::ptr::null::<i16>(); $a];
                        let mut sp = [std::ptr::null::<i16>(); $s];
                        for k in 0..$a {
                            ap[k] = adds[k].as_ptr();
                        }
                        for k in 0..$s {
                            sp[k] = subs[k].as_ptr();
                        }
                        unsafe { combine_fixed_avx2::<$a, $s>(out.as_mut_ptr(), base.as_ptr(), ap, sp, out.len()) };
                        return;
                    }};
                }
                if adds.len() + subs.len() <= 4 && adds.iter().chain(subs.iter()).all(|r| r.len() == out.len()) {
                    match (adds.len(), subs.len()) {
                        (1, 1) => fixed!(1, 1),
                        (1, 2) => fixed!(1, 2),
                        (2, 2) => fixed!(2, 2),
                        (2, 1) => fixed!(2, 1),
                        (1, 0) => fixed!(1, 0),
                        (0, 1) => fixed!(0, 1),
                        _ => {}
                    }
                }
            }
            unsafe { combine_rows_avx2(out, base, adds, subs) };
            return;
        }
    }
    for i in 0..out.len() {
        let mut v = base[i];
        for r in adds {
            v = v.wrapping_add(r[i]);
        }
        for r in subs {
            v = v.wrapping_sub(r[i]);
        }
        out[i] = v;
    }
}

/// combine_rows for a fixed number of rows to add (A) and remove (S), on rows
/// whose length `n` is a multiple of 16.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn combine_fixed_avx2<const A: usize, const S: usize>(
    out: *mut i16,
    base: *const i16,
    adds: [*const i16; A],
    subs: [*const i16; S],
    n: usize,
) {
    unsafe {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i < n {
            let mut v = _mm256_loadu_si256(base.add(i) as *const __m256i);
            for r in adds {
                v = _mm256_add_epi16(v, _mm256_loadu_si256(r.add(i) as *const __m256i));
            }
            for r in subs {
                v = _mm256_sub_epi16(v, _mm256_loadu_si256(r.add(i) as *const __m256i));
            }
            _mm256_storeu_si256(out.add(i) as *mut __m256i, v);
            i += 16;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn combine_rows_avx2(out: &mut [i16], base: &[i16], adds: &[&[i16]], subs: &[&[i16]]) {
    unsafe {
        use std::arch::x86_64::*;
        let n = out.len() / 16 * 16;
        let mut i = 0;
        while i < n {
            let mut v = _mm256_loadu_si256(base.as_ptr().add(i) as *const __m256i);
            for r in adds {
                v = _mm256_add_epi16(v, _mm256_loadu_si256(r.as_ptr().add(i) as *const __m256i));
            }
            for r in subs {
                v = _mm256_sub_epi16(v, _mm256_loadu_si256(r.as_ptr().add(i) as *const __m256i));
            }
            _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, v);
            i += 16;
        }
        for k in n..out.len() {
            let mut v = base[k];
            for r in adds {
                v = v.wrapping_add(r[k]);
            }
            for r in subs {
                v = v.wrapping_sub(r[k]);
            }
            out[k] = v;
        }
    }
}

/// The largest size of any number in a compact row (remembered after the first time).
pub fn max_abs(p: &Packed) -> i32 {
    let known = p.max_abs.get();
    if known >= 0 {
        return known;
    }
    let m = p.vals().iter().map(|x| (*x as i32).abs()).max().unwrap_or(0);
    p.max_abs.set(m);
    m
}

/// screlu_dot in 16-bit steps, as the C++ engine does it: sixteen numbers per
/// step. Exact when cap * (largest weight) fits in 16 bits, which the caller checks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn screlu_dot_avx2_16(x: &[i16], w: &[i16], cap: i32) -> i64 {
    unsafe {
        use std::arch::x86_64::*;
        let n = x.len() / 16 * 16;
        let zero = _mm256_setzero_si256();
        let capv = _mm256_set1_epi16(cap as i16);
        let mut total = 0i64;
        let mut i = 0;
        while i < n {
            // at most 512 numbers per block keeps each 32-bit running total in range
            let end = (i + 512).min(n);
            // two running totals, so consecutive steps don't wait for each other
            let (mut acc, mut acc2) = (zero, zero);
            while i + 32 <= end {
                let xv = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
                let wv = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
                let xv2 = _mm256_loadu_si256(x.as_ptr().add(i + 16) as *const __m256i);
                let wv2 = _mm256_loadu_si256(w.as_ptr().add(i + 16) as *const __m256i);
                let c = _mm256_min_epi16(_mm256_max_epi16(xv, zero), capv);
                let c2 = _mm256_min_epi16(_mm256_max_epi16(xv2, zero), capv);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_mullo_epi16(c, wv), c));
                acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_mullo_epi16(c2, wv2), c2));
                i += 32;
            }
            while i < end {
                let xv = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
                let wv = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
                let c = _mm256_min_epi16(_mm256_max_epi16(xv, zero), capv);
                let t = _mm256_mullo_epi16(c, wv);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(t, c));
                i += 16;
            }
            let acc = _mm256_add_epi32(acc, acc2);
            let mut parts = [0i32; 8];
            _mm256_storeu_si256(parts.as_mut_ptr() as *mut __m256i, acc);
            total += parts.iter().map(|v| *v as i64).sum::<i64>();
        }
        total + screlu_dot_plain(&x[n..], &w[n..], cap)
    }
}

/// The 32-bit form of screlu_dot for one stretch: each product c * c * w fits
/// in 32 bits when cap <= 255; its upper and lower halves are added separately
/// so every running total stays in 32 bits.
fn screlu_dot_plain(x: &[i16], w: &[i16], cap: i32) -> i64 {
    let mut total: i64 = 0;
    for (xc, wc) in x.chunks(1024).zip(w.chunks(1024)) {
        let (mut lo, mut hi) = (0i32, 0i32);
        for (p, q) in xc.iter().zip(wc.iter()) {
            let c = (*p as i32).max(0).min(cap);
            let t = c * c * (*q as i32);
            lo += t & 0xFFFF;
            hi += t >> 16;
        }
        total += (hi as i64) * 65536 + lo as i64;
    }
    total
}

fn screlu_dot_packed(x: &Packed, w: &Packed, cap: f64) -> Res {
    let (xd, wd) = (x.vals(), w.vals());
    Ok(Value::Num(screlu_dot_rows(&xd, &wd, || max_abs(w), cap)?))
}

/// screlu_dot of two rows of compact numbers. `w_max` gives (an upper bound
/// of) the largest size of a weight; it is only asked for when needed.
pub fn screlu_dot_rows(xd: &[i16], wd: &[i16], w_max: impl FnOnce() -> i32, cap: f64) -> Result<f64, String> {
    if xd.len() != wd.len() {
        return Err("screlu_dot needs two rows of the same length".into());
    }
    let cap = cap.max(0.0).min(32767.0) as i32;
    if cap <= 255 {
        #[cfg(target_arch = "x86_64")]
        {
            // cap * weight must fit in 16 bits for the 16-bit steps to be exact
            if std::arch::is_x86_feature_detected!("avx2") && (cap * w_max() <= 32767 || cap * row_max_abs(wd) <= 32767) {
                return Ok(unsafe { screlu_dot_avx2_16(xd, wd, cap) } as f64);
            }
        }
        return Ok(screlu_dot_plain(xd, wd, cap) as f64);
    }
    let mut total: i64 = 0;
    for (p, q) in xd.iter().zip(wd.iter()) {
        let c = (*p as i32).max(0).min(cap);
        total += (c as i64) * (c as i64) * (*q as i64);
    }
    Ok(total as f64)
}

fn row_max_abs(d: &[i16]) -> i32 {
    d.iter().map(|x| (*x as i32).abs()).max().unwrap_or(0)
}

/// rows(matrix): the rows of a matrix as a list, so one can be picked without copying.
fn b_rows(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "rows")?;
    if let Value::Packed(p) = &a[0] {
        let d = p.vals();
        let out: Vec<Value> = d.chunks(p.cols.max(1)).map(|c| packed(c.to_vec(), 1, p.cols, false)).collect();
        return Ok(Value::List(Rc::new(out)));
    }
    let m = ten(&a[0], "rows")?;
    if m.shape.len() < 2 {
        return Err("rows needs a matrix (rows and columns)".into());
    }
    let mut out = Vec::with_capacity(m.shape[0]);
    for i in 0..m.shape[0] {
        out.push(Value::Tensor(tensor::index(&m, &[i as i64])?));
    }
    Ok(Value::List(Rc::new(out)))
}

fn plain(v: &Value, name: &str) -> Result<T, String> {
    let t = ten(v, name)?;
    if t.requires {
        return Err(format!("{} works on plain numbers, not on values being learned", name));
    }
    Ok(t)
}

/// screlu(x, cap): every number forced into 0..cap and then squared (an activation function).
fn b_screlu(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "screlu")?;
    let t = plain(&a[0], "screlu")?;
    let cap = num(&a[1], "screlu")?;
    let d: Vec<f64> = t
        .vals()
        .iter()
        .map(|x| {
            let c = x.max(0.0).min(cap);
            c * c
        })
        .collect();
    Ok(wrap(tensor::new(t.shape.clone(), d)))
}

/// sum_product(a, b, ...): multiplies the rows position by position and adds
/// everything up, without building the rows in between.
fn b_sum_product(_: &mut Interp, a: Vec<Value>) -> Res {
    if a.is_empty() {
        return Err("sum_product needs at least one row of numbers".into());
    }
    let ts: Vec<T> = a.iter().map(|v| plain(v, "sum_product")).collect::<Result<_, _>>()?;
    let n = ts[0].len();
    if ts.iter().any(|t| t.len() != n) {
        return Err("sum_product needs rows of the same length".into());
    }
    let total = match &ts[..] {
        [x] => x.vals().iter().sum(),
        [x, y] => {
            let (x, y) = (x.vals(), y.vals());
            let mut lanes = [0.0f64; 8];
            let mut xc = x.chunks_exact(8);
            let mut yc = y.chunks_exact(8);
            for (px, py) in (&mut xc).zip(&mut yc) {
                for k in 0..8 {
                    lanes[k] += px[k] * py[k];
                }
            }
            let rest: f64 = xc.remainder().iter().zip(yc.remainder()).map(|(p, q)| p * q).sum();
            lanes.iter().sum::<f64>() + rest
        }
        _ => {
            let ds: Vec<_> = ts.iter().map(|t| t.vals()).collect();
            let mut total = 0.0;
            for i in 0..n {
                let mut p = 1.0;
                for d in &ds {
                    p *= d[i];
                }
                total += p;
            }
            total
        }
    };
    Ok(Value::Num(total))
}

/// screlu_dot(x, weights, cap): each number of x forced into 0..cap and squared,
/// times the matching weight, all added up. The same as
/// sum_product(screlu(x, cap), weights) without building the row in between.
fn b_screlu_dot(_: &mut Interp, a: Vec<Value>) -> Res {
    screlu_dot_slice(&a)
}

pub fn screlu_dot_slice(a: &[Value]) -> Res {
    want(&a, 3, "screlu_dot")?;
    if let (Value::Packed(x), Value::Packed(w)) = (&a[0], &a[1]) {
        return screlu_dot_packed(x, w, num(&a[2], "screlu_dot")?);
    }
    let x = plain(&a[0], "screlu_dot")?;
    let w = plain(&a[1], "screlu_dot")?;
    let cap = num(&a[2], "screlu_dot")?;
    if x.len() != w.len() {
        return Err("screlu_dot needs two rows of the same length".into());
    }
    let (xd, wd) = (x.vals(), w.vals());
    // Eight running totals, so the processor can work on eight numbers at once.
    let mut lanes = [0.0f64; 8];
    let mut xc = xd.chunks_exact(8);
    let mut wc = wd.chunks_exact(8);
    for (px, pw) in (&mut xc).zip(&mut wc) {
        for k in 0..8 {
            let c = px[k].max(0.0).min(cap);
            lanes[k] += c * c * pw[k];
        }
    }
    let mut total: f64 = lanes.iter().sum();
    for (p, q) in xc.remainder().iter().zip(wc.remainder()) {
        let c = p.max(0.0).min(cap);
        total += c * c * q;
    }
    Ok(Value::Num(total))
}

/// update_rows(base, matrix, add, n_add, remove, n_remove, old?):
/// base plus the chosen rows of the matrix, minus other chosen rows, in one step.
/// `old` is optional: a row that is about to be replaced by the result. If
/// nothing else is using it, its memory is reused instead of making a new row.
fn b_update_rows(_: &mut Interp, a: Vec<Value>) -> Res {
    update_rows_slice(&a)
}

pub fn update_rows_slice(a: &[Value]) -> Res {
    if a.len() != 6 && a.len() != 7 {
        return Err(format!("update_rows takes 6 or 7 inputs but was given {}", a.len()));
    }
    if let (Value::Packed(base), Value::Packed(m)) = (&a[0], &a[1]) {
        return update_rows_packed(&a, base, m);
    }
    let base = plain(&a[0], "update_rows")?;
    let m = plain(&a[1], "update_rows")?;
    if m.shape.len() != 2 || base.len() != m.shape[1] {
        return Err("update_rows needs a matrix whose rows are as long as the base row".into());
    }
    let (n, h) = (m.shape[0], m.shape[1]);
    let md = m.vals();
    // Reuse the old row when the only holders are the variable it came from and this call.
    let recycled: Option<T> = match a.get(6) {
        Some(Value::Tensor(t))
            if !t.requires
                && t.shape == base.shape
                && Rc::strong_count(t) == 2
                && !Rc::ptr_eq(t, &base)
                && !Rc::ptr_eq(t, &m) =>
        {
            Some(t.clone())
        }
        _ => None,
    };
    let mut fresh_row: Vec<f64> = Vec::new();
    let mut held;
    let out: &mut Vec<f64> = match &recycled {
        Some(t) => {
            held = t.vals_mut();
            held.copy_from_slice(&base.vals());
            &mut held
        }
        None => {
            fresh_row = base.vals().clone();
            &mut fresh_row
        }
    };
    for (rows_arg, count_arg, sign) in [(&a[2], &a[3], 1.0), (&a[4], &a[5], -1.0)] {
        let rows = ten(rows_arg, "update_rows")?;
        let rd = rows.vals();
        let count = num(count_arg, "update_rows")? as usize;
        if count > rd.len() {
            return Err("update_rows was asked for more rows than row numbers were given".into());
        }
        for &r in &rd[..count] {
            if r < 0.0 || r as usize >= n {
                return Err(format!("update_rows: row {} doesn't exist (the matrix has {} rows)", fmt_num(r), n));
            }
            let row = &md[r as usize * h..(r as usize + 1) * h];
            if sign > 0.0 {
                for (o, x) in out.iter_mut().zip(row) {
                    *o += x;
                }
            } else {
                for (o, x) in out.iter_mut().zip(row) {
                    *o -= x;
                }
            }
        }
    }
    match &recycled {
        Some(t) => Ok(Value::Tensor(t.clone())),
        None => Ok(Value::Tensor(tensor::new(base.shape.clone(), fresh_row))),
    }
}

/// read_binary(file, kind, start, count): whole numbers stored in a binary file.
fn b_read_binary(_: &mut Interp, a: Vec<Value>) -> Res {
    use std::io::{Read, Seek, SeekFrom};
    want(&a, 4, "read_binary")?;
    let path = text(&a[0], "read_binary")?;
    let kind = text(&a[1], "read_binary")?;
    let start = num(&a[2], "read_binary")? as u64;
    let count = num(&a[3], "read_binary")? as usize;
    let size = match &*kind {
        "u8" | "i8" => 1,
        "u16" | "i16" => 2,
        "u32" | "i32" => 4,
        other => return Err(format!("read_binary knows the kinds u8, i8, u16, i16, u32 and i32, not \"{}\"", other)),
    };
    let too_short = || format!("\"{}\" is too short: it doesn't hold {} numbers from position {}", path, count, start);
    let mut owned = Vec::new();
    let buf: &[u8] = match embedded_file(&path) {
        Some(Some(bytes)) => {
            let from = start as usize;
            bytes.get(from..from + count * size).ok_or_else(too_short)?
        }
        Some(None) => return Err(format!("no file was embedded as \"{}\"", path)),
        None => {
            let mut f = std::fs::File::open(&*path).map_err(|e| format!("couldn't open \"{}\": {}", path, e))?;
            f.seek(SeekFrom::Start(start)).map_err(|e| format!("couldn't read \"{}\": {}", path, e))?;
            owned.resize(count * size, 0u8);
            f.read_exact(&mut owned).map_err(|_| too_short())?;
            &owned
        }
    };
    let data: Vec<f64> = buf
        .chunks_exact(size)
        .map(|c| match &*kind {
            "u8" => c[0] as f64,
            "i8" => c[0] as i8 as f64,
            "u16" => u16::from_le_bytes([c[0], c[1]]) as f64,
            "i16" => i16::from_le_bytes([c[0], c[1]]) as f64,
            "u32" => u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64,
            _ => i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64,
        })
        .collect();
    Ok(Value::Tensor(tensor::new(vec![data.len()], data)))
}

// ---------- collections ----------

// ---------- 64-bit whole numbers ----------

fn whole_arg(v: &Value, name: &str) -> Result<i64, String> {
    v.as_int().ok_or_else(|| format!("{} needs a whole number, not {}", name, v.show()))
}

/// int(x): x as a 64-bit whole number.
fn b_int(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "int")?;
    Ok(Value::Int(whole_arg(&a[0], "int")?))
}

/// ints(n): a row of n integers, all 0. ints(list): the same numbers as integers.
fn b_ints(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "ints")?;
    if let Some(n) = a[0].as_int() {
        if n < 0 {
            return Err("ints needs a size that is zero or more".into());
        }
        return Ok(Value::Ints(Rc::new(vec![0; n as usize])));
    }
    let mut out = vec![];
    for v in iter_items(&a[0])? {
        out.push(whole_arg(&v, "ints")?);
    }
    Ok(Value::Ints(Rc::new(out)))
}

/// The first number in `v` (a number, or every number of a row, matrix or
/// list) that `bad` rejects, as text.
fn first_outside(v: &Value, bad: &dyn Fn(f64) -> bool) -> Result<Option<String>, String> {
    Ok(match v {
        Value::Int(i) => bad(*i as f64).then(|| i.to_string()),
        Value::Num(x) => bad(*x).then(|| fmt_num(*x)),
        Value::Ints(r) => r.iter().find(|&&i| bad(i as f64)).map(|i| i.to_string()),
        Value::Tensor(t) => t.vals().iter().find(|&&x| bad(x)).map(|x| fmt_num(*x)),
        Value::Packed(p) => p.vals().iter().find(|&&x| bad(x as f64)).map(|x| x.to_string()),
        Value::List(items) => {
            for x in items.iter() {
                if let Some(t) = first_outside(x, bad)? {
                    return Ok(Some(t));
                }
            }
            None
        }
        other => return Err(format!("within checks numbers, not a {}", other.kind())),
    })
}

/// within(x, lo, hi [, name]): x itself, if it (or every number in it) is from
/// lo to hi; a problem otherwise. `name = ... within lo to hi` makes every
/// value given to a variable go through this.
fn b_within(_: &mut Interp, a: Vec<Value>) -> Res {
    within_slice(a)
}

fn within_slice(a: Vec<Value>) -> Res {
    if a.len() != 3 && a.len() != 4 {
        return Err(format!("within takes 3 or 4 inputs but was given {}", a.len()));
    }
    let lo = num(&a[1], "within")?;
    let hi = num(&a[2], "within")?;
    if let Some(x) = first_outside(&a[0], &|x: f64| !(x >= lo && x <= hi))? {
        let name = match a.get(3) {
            Some(Value::Str(name)) => name.to_string(),
            _ => String::new(),
        };
        return Err(within_message(&x, lo, hi, &name));
    }
    Ok(a.into_iter().next().unwrap())
}

/// The problem when `x` (as text) is outside lo to hi (`name`: what holds it, if known).
pub fn within_message(x: &str, lo: f64, hi: f64, name: &str) -> String {
    if name.is_empty() {
        format!("{} is not from {} to {}", x, fmt_num(lo), fmt_num(hi))
    } else {
        format!("{} only holds numbers from {} to {}, not {}", name, fmt_num(lo), fmt_num(hi), x)
    }
}

/// within for any value, in a compiled program.
pub fn within_value(x: Value, lo: f64, hi: f64, name: &str) -> Result<Value, String> {
    let mut args = vec![x, Value::Num(lo), Value::Num(hi)];
    if !name.is_empty() {
        args.push(Value::Str(name.into()));
    }
    within_slice(args)
}

/// popcount(x): how many of the 64 bits are set.
fn b_popcount(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "popcount")?;
    Ok(Value::Int(whole_arg(&a[0], "popcount")?.count_ones() as i64))
}

/// lowest_bit(x): the position of the lowest set bit (64 if there is none).
fn b_lowest_bit(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "lowest_bit")?;
    Ok(Value::Int(whole_arg(&a[0], "lowest_bit")?.trailing_zeros() as i64))
}

/// highest_bit(x): the position of the highest set bit (-1 if there is none).
fn b_highest_bit(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "highest_bit")?;
    Ok(Value::Int(63 - whole_arg(&a[0], "highest_bit")?.leading_zeros() as i64))
}

/// div(a, b): a divided by b, dropping any remainder.
fn b_div(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "div")?;
    let (x, y) = (whole_arg(&a[0], "div")?, whole_arg(&a[1], "div")?);
    if y == 0 {
        return Err("can't divide by 0".into());
    }
    Ok(Value::Int(x.wrapping_div(y)))
}

/// rand_bits(): 64 random bits as an integer.
fn b_rand_bits(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 0, "rand_bits")?;
    Ok(Value::Int(next_u64() as i64))
}

fn b_len(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "len")?;
    if let Value::Ints(l) = &a[0] {
        return Ok(Value::Num(l.len() as f64));
    }
    Ok(Value::Num(match &a[0] {
        Value::List(l) => l.len(),
        Value::Str(st) => st.chars().count(),
        Value::Map(m) => m.len(),
        Value::Tensor(t) if !t.shape.is_empty() => t.shape[0],
        Value::Packed(p) => {
            if p.matrix {
                p.rows
            } else {
                p.cols
            }
        }
        other => return Err(format!("a {} has no length", other.kind())),
    } as f64))
}

fn b_push(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "push")?;
    let mut items = iter_items(&a[0])?;
    items.push(a[1].clone());
    Ok(pack_list(items))
}

fn b_map(it: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "map")?;
    let mut out = vec![];
    for item in iter_items(&a[0])? {
        out.push(it.call_native(&a[1], vec![item])?);
    }
    Ok(pack_list(out))
}

fn b_filter(it: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "filter")?;
    let mut out = vec![];
    for item in iter_items(&a[0])? {
        match it.call_native(&a[1], vec![item.clone()])? {
            Value::Bool(true) => out.push(item),
            Value::Bool(false) | Value::Nothing => {}
            other => return Err(format!("the function given to filter must answer yes or no, not a {}", other.kind())),
        }
    }
    Ok(pack_list(out))
}

fn b_reduce(it: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 3, "reduce")?;
    let mut total = a[2].clone();
    for item in iter_items(&a[0])? {
        total = it.call_native(&a[1], vec![total, item])?;
    }
    Ok(total)
}

fn b_keys(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "keys")?;
    match &a[0] {
        Value::Map(m) => Ok(Value::List(Rc::new(m.iter().map(|(k, _)| Value::Str(k.clone())).collect()))),
        other => Err(format!("keys needs a map, not a {}", other.kind())),
    }
}

fn b_values(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "values")?;
    match &a[0] {
        Value::Map(m) => Ok(pack_list(m.iter().map(|(_, v)| v.clone()).collect())),
        other => Err(format!("values needs a map, not a {}", other.kind())),
    }
}

fn b_has(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "has")?;
    Ok(Value::Bool(match (&a[0], &a[1]) {
        (Value::Str(hay), Value::Str(needle)) => hay.contains(&**needle),
        (Value::Map(m), Value::Str(k)) => m.iter().any(|(k2, _)| k2 == k),
        (coll, x) => iter_items(coll)?.iter().any(|v| v.equals(x)),
    }))
}

fn b_join(_: &mut Interp, a: Vec<Value>) -> Res {
    if a.is_empty() || a.len() > 2 {
        return Err("join takes a list and, optionally, the text to put between items".into());
    }
    let sep = if a.len() == 2 { text(&a[1], "join")? } else { "".into() };
    let parts: Vec<String> = iter_items(&a[0])?.iter().map(|v| v.show()).collect();
    Ok(s(parts.join(&sep)))
}

fn b_split(_: &mut Interp, a: Vec<Value>) -> Res {
    if a.is_empty() || a.len() > 2 {
        return Err("split takes text and, optionally, the separator".into());
    }
    let src = text(&a[0], "split")?;
    let parts: Vec<Value> = if a.len() == 2 {
        let sep = text(&a[1], "split")?;
        src.split(&*sep).map(s).collect()
    } else {
        src.split_whitespace().map(s).collect()
    };
    Ok(Value::List(Rc::new(parts)))
}

fn b_sort(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "sort")?;
    let mut items = iter_items(&a[0])?;
    if items.iter().all(|v| v.scalar().is_some()) {
        items.sort_by(|x, y| x.scalar().partial_cmp(&y.scalar()).unwrap_or(std::cmp::Ordering::Equal));
    } else if items.iter().all(|v| matches!(v, Value::Str(_))) {
        items.sort_by(|x, y| x.show().cmp(&y.show()));
    } else {
        return Err("sort needs all numbers or all text".into());
    }
    Ok(pack_list(items))
}

fn b_reverse(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "reverse")?;
    if let Value::Str(st) = &a[0] {
        return Ok(s(st.chars().rev().collect::<String>()));
    }
    let mut items = iter_items(&a[0])?;
    items.reverse();
    Ok(pack_list(items))
}

fn b_slice(_: &mut Interp, a: Vec<Value>) -> Res {
    if a.len() < 2 || a.len() > 3 {
        return Err("slice takes a list or text, a start, and optionally an end".into());
    }
    let items = iter_items(&a[0])?;
    let n = items.len() as i64;
    let fix = |x: f64| -> usize {
        let i = x as i64;
        (if i < 0 { i + n } else { i }).clamp(0, n) as usize
    };
    let start = fix(num(&a[1], "slice")?);
    let end = if a.len() == 3 { fix(num(&a[2], "slice")?) } else { n as usize };
    let part: Vec<Value> = if start < end { items[start..end].to_vec() } else { vec![] };
    if let Value::Str(_) = &a[0] {
        return Ok(s(part.iter().map(|v| v.show()).collect::<String>()));
    }
    Ok(pack_list(part))
}

// ---------- text ----------

fn b_upper(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "upper")?;
    Ok(s(text(&a[0], "upper")?.to_uppercase()))
}

fn b_lower(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "lower")?;
    Ok(s(text(&a[0], "lower")?.to_lowercase()))
}

fn b_trim(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "trim")?;
    Ok(s(text(&a[0], "trim")?.trim()))
}

fn b_replace(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 3, "replace")?;
    let (src, from, to) = (text(&a[0], "replace")?, text(&a[1], "replace")?, text(&a[2], "replace")?);
    Ok(s(src.replace(&*from, &to)))
}

fn b_lines(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "lines")?;
    Ok(Value::List(Rc::new(text(&a[0], "lines")?.lines().map(s).collect())))
}

fn b_text(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "text")?;
    Ok(s(a[0].show()))
}

fn b_number(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "number")?;
    match &a[0] {
        Value::Str(st) => st
            .trim()
            .replace('_', "")
            .parse::<f64>()
            .map(Value::Num)
            .map_err(|_| format!("\"{}\" isn't a number", st)),
        Value::Bool(b) => Ok(Value::Num(if *b { 1.0 } else { 0.0 })),
        v => num(v, "number").map(Value::Num),
    }
}

fn b_kind(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "kind")?;
    Ok(s(a[0].kind()))
}

// ---------- the outside world ----------

thread_local! {
    /// Files packed into a compiled program with `antares build --embed`.
    static EMBEDDED: std::cell::RefCell<Vec<(&'static str, &'static [u8])>> = std::cell::RefCell::new(Vec::new());
}

/// Makes embedded files available under the names "embedded:<file name>".
pub fn set_embedded(files: &[(&'static str, &'static [u8])]) {
    EMBEDDED.with(|e| *e.borrow_mut() = files.to_vec());
}

/// The bytes of "embedded:<name>", if the path has that form and the file was embedded.
pub(crate) fn embedded_file(path: &str) -> Option<Option<&'static [u8]>> {
    let name = path.strip_prefix("embedded:")?;
    Some(EMBEDDED.with(|e| e.borrow().iter().find(|(n, _)| *n == name).map(|(_, b)| *b)))
}

thread_local! {
    static PROGRAM_FOLDER: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
}

static PROGRAM_ARGS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();

/// Records what was typed after the program's name on the command line.
pub fn set_program_args(args: Vec<String>) {
    let _ = PROGRAM_ARGS.set(args);
}

/// args(): what was typed after the program's name, as a list of text.
fn b_args(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 0, "args")?;
    let args = PROGRAM_ARGS.get().cloned().unwrap_or_default();
    Ok(Value::List(Rc::new(args.into_iter().map(s).collect())))
}

/// env(name): a setting of the computer (an environment variable), or nothing.
fn b_env(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "env")?;
    let name = text(&a[0], "env")?;
    Ok(match std::env::var(&*name) {
        Ok(v) => s(v),
        Err(_) => Value::Nothing,
    })
}

/// append(file, value): adds the value (as text) to the end of a file.
fn b_append(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "append")?;
    let p = text(&a[0], "append")?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&*p)
        .map_err(|e| format!("couldn't open the file \"{}\": {}", p, e))?;
    f.write_all(a[1].show().as_bytes()).map_err(|e| format!("couldn't write to the file \"{}\": {}", p, e))?;
    Ok(Value::Nothing)
}

/// delete(file): removes a file.
fn b_delete(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "delete")?;
    let p = text(&a[0], "delete")?;
    std::fs::remove_file(&*p).map_err(|e| format!("couldn't delete the file \"{}\": {}", p, e))?;
    Ok(Value::Nothing)
}

/// files(folder): the names of the files and folders in a folder, sorted.
fn b_files(_: &mut Interp, a: Vec<Value>) -> Res {
    if a.len() > 1 {
        return Err(format!("files takes a folder (or nothing, for the current folder) but was given {} inputs", a.len()));
    }
    let dir = match a.first() {
        Some(v) => text(v, "files")?.to_string(),
        None => ".".to_string(),
    };
    let rd = std::fs::read_dir(&dir).map_err(|e| format!("couldn't read the folder \"{}\": {}", dir, e))?;
    let mut names: Vec<String> = rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().to_string()).collect();
    names.sort();
    Ok(Value::List(Rc::new(names.into_iter().map(s).collect())))
}

/// The local date and time: year, month, day, hour, minute, second, weekday (0 = Sunday).
#[cfg(windows)]
fn local_now() -> [i64; 7] {
    #[repr(C)]
    struct SystemTime {
        year: u16,
        month: u16,
        weekday: u16,
        day: u16,
        hour: u16,
        minute: u16,
        second: u16,
        ms: u16,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetLocalTime(t: *mut SystemTime);
    }
    let mut t = SystemTime { year: 0, month: 0, weekday: 0, day: 0, hour: 0, minute: 0, second: 0, ms: 0 };
    unsafe { GetLocalTime(&mut t) };
    [t.year, t.month, t.day, t.hour, t.minute, t.second, t.weekday].map(|x| x as i64)
}

#[cfg(unix)]
fn local_now() -> [i64; 7] {
    #[repr(C)]
    struct Tm {
        sec: i32,
        min: i32,
        hour: i32,
        mday: i32,
        mon: i32,
        year: i32,
        wday: i32,
        yday: i32,
        isdst: i32,
        gmtoff: i64,
        zone: *const u8,
    }
    unsafe extern "C" {
        fn time(t: *mut i64) -> i64;
        fn localtime_r(t: *const i64, tm: *mut Tm) -> *mut Tm;
    }
    let mut tm = Tm { sec: 0, min: 0, hour: 0, mday: 0, mon: 0, year: 0, wday: 0, yday: 0, isdst: 0, gmtoff: 0, zone: std::ptr::null() };
    unsafe {
        let now = time(std::ptr::null_mut());
        localtime_r(&now, &mut tm);
    }
    [tm.year as i64 + 1900, tm.mon as i64 + 1, tm.mday as i64, tm.hour as i64, tm.min as i64, tm.sec as i64, tm.wday as i64]
}

#[cfg(not(any(windows, unix)))]
fn local_now() -> [i64; 7] {
    // no local time: the time in UTC, from the system clock
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    // civil date from days since 1970-01-01 (Howard Hinnant's algorithm)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    [y, m, d, rem / 3600, rem % 3600 / 60, rem % 60, (days + 4).rem_euclid(7)]
}

/// now(): the local date and time as a map.
fn b_now(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 0, "now")?;
    let [y, mo, d, h, mi, se, wd] = local_now();
    const DAYS: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    let n = |x: i64| Value::Num(x as f64);
    Ok(Value::Map(Rc::new(vec![
        ("year".into(), n(y)),
        ("month".into(), n(mo)),
        ("day".into(), n(d)),
        ("hour".into(), n(h)),
        ("minute".into(), n(mi)),
        ("second".into(), n(se)),
        ("weekday".into(), s(DAYS[wd.rem_euclid(7) as usize].to_string())),
    ])))
}

/// fixed(x, places): x as text with exactly that many digits after the point.
fn b_fixed(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "fixed")?;
    let x = num(&a[0], "fixed")?;
    let places = num(&a[1], "fixed")?;
    if places != places.trunc() || !(0.0..=20.0).contains(&places) {
        return Err(format!("fixed needs a whole number of places from 0 to 20, not {}", fmt_num(places)));
    }
    let mut t = format!("{:.*}", places as usize, x);
    // no "-0.00"
    if t.starts_with('-') && t[1..].chars().all(|c| c == '0' || c == '.') {
        t.remove(0);
    }
    Ok(s(t))
}

/// pad(value, width [, side [, fill]]): the value as text, with spaces (or the
/// fill character) added to make it `width` characters wide. Numbers are lined
/// up on the right and everything else on the left, as in a table; side
/// "left" or "right" chooses. pad(5, 2, "right", "0") is "05"; a minus sign
/// stays in front of zeros ("-05").
fn b_pad(_: &mut Interp, a: Vec<Value>) -> Res {
    if !(2..=4).contains(&a.len()) {
        return Err(format!("pad takes a value, a width, and optionally a side and a fill character, but was given {} inputs", a.len()));
    }
    let width = num(&a[1], "pad")?;
    if width != width.trunc() || width < 0.0 {
        return Err(format!("pad needs a width that is a whole number, 0 or more, not {}", fmt_num(width)));
    }
    let right = match a.get(2) {
        None => matches!(a[0], Value::Num(_) | Value::Int(_)),
        Some(v) => match &*text(v, "pad")? {
            "left" => false,
            "right" => true,
            other => return Err(format!("pad lines things up on the \"left\" or the \"right\", not \"{}\"", other)),
        },
    };
    let fill_char = match a.get(3) {
        None => ' ',
        Some(v) => {
            let f = text(v, "pad")?;
            let mut cs = f.chars();
            match (cs.next(), cs.next()) {
                (Some(c), None) => c,
                _ => return Err(format!("pad fills with one character, not \"{}\"", f)),
            }
        }
    };
    let t = a[0].show();
    let fill: String = std::iter::repeat_n(fill_char, (width as usize).saturating_sub(t.chars().count())).collect();
    // zeros go after the sign of a negative number
    if right && fill_char == '0' && !fill.is_empty() && matches!(a[0], Value::Num(_) | Value::Int(_)) {
        if let Some(digits) = t.strip_prefix('-') {
            return Ok(s(format!("-{}{}", fill, digits)));
        }
    }
    Ok(s(if right { fill + &t } else { t + &fill }))
}

/// code(text): the number of the first character (its Unicode code point).
fn b_code(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "code")?;
    let t = text(&a[0], "code")?;
    match t.chars().next() {
        Some(c) => Ok(Value::Num(c as u32 as f64)),
        None => Err("code needs some text, not empty text".into()),
    }
}

/// char(n): the character with that number.
fn b_char(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "char")?;
    let n = num(&a[0], "char")?;
    match (n == n.trunc() && n >= 0.0 && n <= u32::MAX as f64).then(|| char::from_u32(n as u32)).flatten() {
        Some(c) => Ok(s(c.to_string())),
        None => Err(format!("{} isn't the number of a character", fmt_num(n))),
    }
}

/// A position for insert and remove: negative counts from the end.
fn list_pos(v: &Value, len: usize, extra: usize, name: &str) -> Result<usize, String> {
    let k = num(v, name)?;
    if k != k.trunc() {
        return Err(format!("{} needs a whole-number position, not {}", name, fmt_num(k)));
    }
    let n = (len + extra) as f64;
    let k2 = if k < 0.0 { k + n } else { k };
    if k2 < 0.0 || k2 >= n {
        let last = len + extra - 1;
        return Err(if len + extra == 0 {
            format!("{} was given position {}, but the list is empty", name, fmt_num(k))
        } else {
            format!("{}: position {} is out of range (valid: 0 to {})", name, fmt_num(k), last)
        });
    }
    Ok(k2 as usize)
}

/// insert(list, position, item): a new list with the item put in at that position.
fn b_insert(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 3, "insert")?;
    let mut items = iter_items(&a[0])?;
    let k = list_pos(&a[1], items.len(), 1, "insert")?;
    items.insert(k, a[2].clone());
    Ok(pack_list(items))
}

/// remove(list, position): a new list without the item at that position.
fn b_remove(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "remove")?;
    let mut items = iter_items(&a[0])?;
    let k = list_pos(&a[1], items.len(), 0, "remove")?;
    items.remove(k);
    Ok(pack_list(items))
}

/// find(x, item): the position of the first item equal to `item` in a list,
/// or where a text first appears in a text; -1 if it isn't there.
fn b_find(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "find")?;
    if let Value::Str(t) = &a[0] {
        let part = text(&a[1], "find")?;
        return Ok(Value::Num(match t.find(&*part) {
            Some(byte) => t[..byte].chars().count() as f64,
            None => -1.0,
        }));
    }
    let items = iter_items(&a[0])?;
    Ok(Value::Num(items.iter().position(|v| v.equals(&a[1])).map_or(-1.0, |k| k as f64)))
}

/// unique(list): the items without repeats, in the order they first appear.
fn b_unique(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "unique")?;
    let mut out: Vec<Value> = vec![];
    for v in iter_items(&a[0])? {
        if !out.iter().any(|o| o.equals(&v)) {
            out.push(v);
        }
    }
    Ok(pack_list(out))
}

/// sort_by(list, f): a copy sorted by f(item), which gives a number or text for
/// each item. Items with the same key keep their order.
fn b_sort_by(it: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "sort_by")?;
    let items = iter_items(&a[0])?;
    let mut keyed = vec![];
    for v in items {
        let k = it.call_native(&a[1], vec![v.clone()])?;
        keyed.push((k, v));
    }
    if keyed.iter().all(|(k, _)| k.scalar().is_some()) {
        keyed.sort_by(|x, y| x.0.scalar().partial_cmp(&y.0.scalar()).unwrap_or(std::cmp::Ordering::Equal));
    } else if keyed.iter().all(|(k, _)| matches!(k, Value::Str(_))) {
        keyed.sort_by(|x, y| x.0.show().cmp(&y.0.show()));
    } else {
        return Err("sort_by needs its function to give all numbers or all text".into());
    }
    Ok(pack_list(keyed.into_iter().map(|(_, v)| v).collect()))
}

/// Records where the running program lives (the script, or the compiled executable).
pub fn set_program_folder(file: &std::path::Path) {
    let dir = std::fs::canonicalize(file)
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let mut text = dir.to_string_lossy().replace('\\', "/");
    if let Some(rest) = text.strip_prefix("//?/") {
        text = rest.to_string();
    }
    PROGRAM_FOLDER.with(|f| *f.borrow_mut() = Some(text));
}

/// program_folder(): the folder the running program is in, so it can find
/// files that sit next to it wherever it is started from.
fn b_program_folder(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 0, "program_folder")?;
    Ok(s(PROGRAM_FOLDER.with(|f| f.borrow().clone()).unwrap_or_else(|| ".".into())))
}

fn b_read(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "read")?;
    let p = text(&a[0], "read")?;
    std::fs::read_to_string(&*p).map(s).map_err(|e| format!("couldn't read the file \"{}\": {}", p, e))
}

fn b_write(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "write")?;
    let p = text(&a[0], "write")?;
    std::fs::write(&*p, a[1].show()).map_err(|e| format!("couldn't write the file \"{}\": {}", p, e))?;
    Ok(Value::Nothing)
}

fn b_input(_: &mut Interp, a: Vec<Value>) -> Res {
    if let Some(p) = a.first() {
        print!("{}", p.show());
        let _ = std::io::stdout().flush();
    }
    // A whole line needs the terminal's normal mode, even if key() switched it.
    let was_raw = console::is_raw();
    console::leave_raw();
    let mut line = String::new();
    let got = std::io::stdin().read_line(&mut line);
    console::resume_raw(was_raw);
    let n = got.map_err(|e| format!("couldn't read input: {}", e))?;
    if n == 0 {
        return Ok(Value::Nothing);
    }
    Ok(s(line.trim_end_matches(&['\n', '\r'][..])))
}

fn b_clock(_: &mut Interp, _: Vec<Value>) -> Res {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    Ok(Value::Num(t))
}

fn b_fail(_: &mut Interp, a: Vec<Value>) -> Res {
    Err(a.first().map(|v| v.show()).unwrap_or_else(|| "the program asked to stop".into()))
}

fn b_check(_: &mut Interp, a: Vec<Value>) -> Res {
    match a.first() {
        Some(Value::Bool(true)) => Ok(Value::Nothing),
        Some(Value::Bool(false)) => Err(a.get(1).map(|v| v.show()).unwrap_or_else(|| "a check failed".into())),
        _ => Err("check needs a yes/no value, like check(x > 0, \"x must be positive\")".into()),
    }
}

// ---------- games and live screens ----------

#[cfg(windows)]
mod console {
    use std::ffi::c_void;
    unsafe extern "C" {
        fn _kbhit() -> i32;
        fn _getch() -> i32;
    }
    unsafe extern "system" {
        fn GetStdHandle(which: u32) -> *mut c_void;
        fn GetConsoleMode(handle: *mut c_void, mode: *mut u32) -> i32;
        fn SetConsoleMode(handle: *mut c_void, mode: u32) -> i32;
        fn PeekNamedPipe(
            handle: *mut c_void,
            buf: *mut c_void,
            size: u32,
            read: *mut u32,
            avail: *mut u32,
            left: *mut u32,
        ) -> i32;
    }

    pub fn leave_raw() {}
    pub fn resume_raw(_: bool) {}
    pub fn is_raw() -> bool {
        false
    }

    /// Whether input is waiting to be read (a pipe from another program, or keys typed in a console).
    pub fn input_waiting() -> bool {
        unsafe {
            let h = GetStdHandle(-10i32 as u32);
            let mut avail = 0u32;
            if PeekNamedPipe(h, std::ptr::null_mut(), 0, std::ptr::null_mut(), &mut avail, std::ptr::null_mut()) != 0 {
                return avail > 0;
            }
            _kbhit() != 0
        }
    }

    /// Lets the Windows console understand cursor-movement codes.
    pub fn enable_ansi() {
        unsafe {
            let h = GetStdHandle(-11i32 as u32);
            let mut mode = 0u32;
            if GetConsoleMode(h, &mut mode) != 0 {
                SetConsoleMode(h, mode | 0x0004);
            }
        }
    }

    /// The key waiting to be read, if any, without pausing the program.
    pub fn poll_key() -> Option<String> {
        unsafe {
            if _kbhit() == 0 {
                return None;
            }
            let c = _getch();
            if c == 0 || c == 224 {
                return Some(
                    match _getch() {
                        72 => "up",
                        80 => "down",
                        75 => "left",
                        77 => "right",
                        _ => "other",
                    }
                    .to_string(),
                );
            }
            Some(match c {
                13 => "enter".to_string(),
                27 => "escape".to_string(),
                32 => "space".to_string(),
                8 => "backspace".to_string(),
                c if (33..127).contains(&c) => (c as u8 as char).to_ascii_lowercase().to_string(),
                _ => "other".to_string(),
            })
        }
    }
}

/// macOS and Linux: the terminal is switched to key-at-a-time mode (no
/// waiting for Enter, no echo) the first time `key()` is called, and put back
/// the way it was when the program ends, fails, or is stopped with Ctrl+C.
#[cfg(any(
    target_os = "macos",
    all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "x86", target_arch = "arm", target_arch = "riscv64"))
))]
mod console {
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[cfg(target_os = "linux")]
    mod sys {
        pub type Flag = u32;
        pub type Nfds = u64;
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct Termios {
            pub c_iflag: Flag,
            pub c_oflag: Flag,
            pub c_cflag: Flag,
            pub c_lflag: Flag,
            pub c_line: u8,
            pub c_cc: [u8; 32],
            pub c_ispeed: u32,
            pub c_ospeed: u32,
        }
        pub const ICANON: Flag = 0o2;
        pub const ECHO: Flag = 0o10;
        pub const VTIME: usize = 5;
        pub const VMIN: usize = 6;
    }

    #[cfg(target_os = "macos")]
    mod sys {
        pub type Flag = u64;
        pub type Nfds = u32;
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct Termios {
            pub c_iflag: Flag,
            pub c_oflag: Flag,
            pub c_cflag: Flag,
            pub c_lflag: Flag,
            pub c_cc: [u8; 20],
            pub c_ispeed: u64,
            pub c_ospeed: u64,
        }
        pub const ICANON: Flag = 0x100;
        pub const ECHO: Flag = 0x8;
        pub const VMIN: usize = 16;
        pub const VTIME: usize = 17;
    }

    use sys::*;

    #[repr(C)]
    struct PollFd {
        fd: i32,
        events: i16,
        revents: i16,
    }

    unsafe extern "C" {
        fn tcgetattr(fd: i32, t: *mut Termios) -> i32;
        fn tcsetattr(fd: i32, when: i32, t: *const Termios) -> i32;
        fn isatty(fd: i32) -> i32;
        fn poll(fds: *mut PollFd, n: Nfds, timeout: i32) -> i32;
        fn atexit(f: extern "C" fn()) -> i32;
        fn signal(sig: i32, handler: usize) -> usize;
        fn raise(sig: i32) -> i32;
    }

    const TCSANOW: i32 = 0;
    const SIGINT: i32 = 2;
    const SIGTERM: i32 = 15;

    /// The terminal settings from before the program changed them.
    static mut SAVED: Option<Termios> = None;
    /// Whether key-at-a-time mode is on right now.
    static RAW: AtomicBool = AtomicBool::new(false);

    thread_local! {
        /// Bytes read from the terminal but not yet turned into keys.
        static PENDING: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    }

    extern "C" fn restore_at_exit() {
        leave_raw();
    }

    extern "C" fn restore_on_signal(sig: i32) {
        // tcsetattr is safe to call from a signal handler.
        leave_raw();
        unsafe {
            signal(sig, 0);
            raise(sig);
        }
    }

    fn saved() -> Option<Termios> {
        unsafe { *std::ptr::addr_of!(SAVED) }
    }

    /// Switches to key-at-a-time mode. Returns false if input is not a terminal.
    fn enter_raw() -> bool {
        if RAW.load(Ordering::SeqCst) {
            return true;
        }
        unsafe {
            if isatty(0) == 0 {
                return false;
            }
            let first = saved().is_none();
            if first {
                let mut t: Termios = std::mem::zeroed();
                if tcgetattr(0, &mut t) != 0 {
                    return false;
                }
                *std::ptr::addr_of_mut!(SAVED) = Some(t);
            }
            let mut raw = saved().unwrap();
            raw.c_lflag &= !(ICANON | ECHO);
            raw.c_cc[VMIN] = 0;
            raw.c_cc[VTIME] = 0;
            if tcsetattr(0, TCSANOW, &raw) != 0 {
                return false;
            }
            RAW.store(true, Ordering::SeqCst);
            if first {
                atexit(restore_at_exit);
                let handler = restore_on_signal as extern "C" fn(i32) as usize;
                signal(SIGINT, handler);
                signal(SIGTERM, handler);
            }
        }
        true
    }

    /// Puts the terminal back the way it was (does nothing if it was never changed).
    pub fn leave_raw() {
        if RAW.swap(false, Ordering::SeqCst) {
            if let Some(t) = saved() {
                unsafe {
                    tcsetattr(0, TCSANOW, &t);
                }
            }
        }
    }

    /// Turns key-at-a-time mode back on after `leave_raw`, if it was on before.
    pub fn resume_raw(was_raw: bool) {
        if was_raw {
            enter_raw();
        }
    }

    pub fn is_raw() -> bool {
        RAW.load(Ordering::SeqCst)
    }

    pub fn enable_ansi() {}

    /// Whether input is waiting to be read (a pipe from another program, or keys typed in a terminal).
    pub fn input_waiting() -> bool {
        if PENDING.with(|p| !p.borrow().is_empty()) {
            return true;
        }
        let mut fd = PollFd { fd: 0, events: 1, revents: 0 };
        unsafe { poll(&mut fd, 1, 0) > 0 && (fd.revents & 1) != 0 }
    }

    /// Reads whatever bytes the terminal has ready, without waiting.
    fn fill() {
        use std::io::Read;
        use std::os::unix::io::FromRawFd;
        // Read straight from the terminal (not through stdin's buffer), and
        // never close it: ManuallyDrop keeps the File from closing fd 0.
        let mut tty = std::mem::ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(0) });
        let mut buf = [0u8; 64];
        if let Ok(n) = tty.read(&mut buf) {
            PENDING.with(|p| p.borrow_mut().extend_from_slice(&buf[..n]));
        }
    }

    /// The key waiting to be read, if any, without pausing the program.
    pub fn poll_key() -> Option<String> {
        if !enter_raw() {
            return None;
        }
        fill();
        PENDING.with(|p| {
            let mut p = p.borrow_mut();
            if p.is_empty() {
                return None;
            }
            // Arrow keys arrive as ESC [ A..D (or ESC O A..D).
            if p[0] == 27 {
                if p.len() >= 3 && (p[1] == b'[' || p[1] == b'O') {
                    let name = match p[2] {
                        b'A' => "up",
                        b'B' => "down",
                        b'C' => "right",
                        b'D' => "left",
                        _ => "other",
                    };
                    // Skip the rest of a longer sequence such as ESC [ 3 ~.
                    let end = p[2..]
                        .iter()
                        .position(|b| (0x40..=0x7e).contains(b))
                        .map_or(p.len(), |i| i + 3);
                    p.drain(..end);
                    return Some(name.to_string());
                }
                p.drain(..1);
                return Some("escape".to_string());
            }
            let c = p.remove(0);
            Some(match c {
                b'\r' | b'\n' => "enter".to_string(),
                b' ' => "space".to_string(),
                8 | 127 => "backspace".to_string(),
                33..=126 => (c as char).to_ascii_lowercase().to_string(),
                _ => "other".to_string(),
            })
        })
    }
}

#[cfg(not(any(
    windows,
    target_os = "macos",
    all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "x86", target_arch = "arm", target_arch = "riscv64"))
)))]
mod console {
    pub fn enable_ansi() {}
    pub fn poll_key() -> Option<String> {
        None
    }
    pub fn input_waiting() -> bool {
        false
    }
    pub fn leave_raw() {}
    pub fn resume_raw(_: bool) {}
    pub fn is_raw() -> bool {
        false
    }
}

fn ansi_ready() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(console::enable_ansi);
}

fn b_key(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 0, "key")?;
    if let Some(k) = crate::gfx::next_key() {
        return Ok(k.map(s).unwrap_or(Value::Nothing));
    }
    Ok(match console::poll_key() {
        Some(k) => s(k),
        None => Value::Nothing,
    })
}

fn b_input_waiting(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 0, "input_waiting")?;
    Ok(Value::Bool(console::input_waiting()))
}

fn b_wait(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "wait")?;
    let secs = num(&a[0], "wait")?;
    if secs > 0.0 && secs.is_finite() && crate::gfx::window_is_open() {
        crate::gfx::wait(secs);
    } else if secs > 0.0 && secs.is_finite() {
        std::thread::sleep(std::time::Duration::from_secs_f64(secs));
    }
    Ok(Value::Nothing)
}

fn b_clear(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 0, "clear")?;
    ansi_ready();
    print!("\x1b[2J\x1b[H");
    let _ = std::io::stdout().flush();
    Ok(Value::Nothing)
}

/// Redraws the screen in place: writes the text from the top-left corner
/// in one go, so animation doesn't flicker.
fn b_draw(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "draw")?;
    ansi_ready();
    let mut out = String::from("\x1b[H");
    for line in a[0].show().lines() {
        out.push_str(line);
        out.push_str("\x1b[K\n");
    }
    out.push_str("\x1b[J");
    let mut so = std::io::stdout().lock();
    let _ = so.write_all(out.as_bytes());
    let _ = so.flush();
    Ok(Value::Nothing)
}

// ---------- JSON ----------

pub(crate) fn from_json(j: &J) -> Value {
    match j {
        J::Null => Value::Nothing,
        J::Bool(b) => Value::Bool(*b),
        J::Num(n) => Value::Num(*n),
        J::Str(st) => s(st),
        J::Arr(a) => pack_list(a.iter().map(from_json).collect()),
        J::Obj(m) => Value::Map(Rc::new(m.iter().map(|(k, v)| (k.as_str().into(), from_json(v))).collect())),
    }
}

fn json_num(x: f64) -> String {
    if x.is_finite() {
        format!("{}", x)
    } else {
        "null".into()
    }
}

fn json_tensor(shape: &[usize], data: &[f64]) -> String {
    if shape.is_empty() {
        return json_num(data[0]);
    }
    if shape[0] == 0 {
        return "[]".into();
    }
    let sub = (data.len() / shape[0]).max(1);
    let parts: Vec<String> = data.chunks(sub).map(|c| json_tensor(&shape[1..], c)).collect();
    format!("[{}]", parts.join(","))
}

pub(crate) fn to_json(v: &Value) -> Result<String, String> {
    Ok(match v {
        Value::Unset | Value::Nothing => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Num(n) => json_num(*n),
        Value::Tensor(t) => json_tensor(&t.shape, &t.vals()),
        Value::Str(st) => json::escape(st),
        Value::List(l) => {
            let mut parts = vec![];
            for x in l.iter() {
                parts.push(to_json(x)?);
            }
            format!("[{}]", parts.join(","))
        }
        Value::Map(m) => {
            let mut parts = vec![];
            for (k, x) in m.iter() {
                parts.push(format!("{}:{}", json::escape(k), to_json(x)?));
            }
            format!("{{{}}}", parts.join(","))
        }
        other => return Err(format!("a {} can't be written as JSON", other.kind())),
    })
}

fn b_save(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "save")?;
    let p = text(&a[1], "save").map_err(|_| "save takes the value first and the file name second, like save(weights, \"weights.json\")".to_string())?;
    std::fs::write(&*p, to_json(&a[0])?).map_err(|e| format!("couldn't save to \"{}\": {}", p, e))?;
    Ok(Value::Nothing)
}

fn b_load(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "load")?;
    let p = text(&a[0], "load")?;
    let src = std::fs::read_to_string(&*p).map_err(|e| format!("couldn't load \"{}\": {}", p, e))?;
    let j = json::parse(&src).map_err(|e| format!("\"{}\" isn't a file made by save: {}", p, e))?;
    Ok(from_json(&j))
}

fn b_exists(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "exists")?;
    let p = text(&a[0], "exists")?;
    if let Some(found) = embedded_file(&p) {
        return Ok(Value::Bool(found.is_some()));
    }
    Ok(Value::Bool(std::path::Path::new(&*p).exists()))
}

fn b_json(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "json")?;
    Ok(from_json(&json::parse(&text(&a[0], "json")?)?))
}

fn b_to_json(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "to_json")?;
    Ok(s(to_json(&a[0])?))
}

pub fn all() -> Vec<(&'static str, NativeFn)> {
    vec![
        ("zeros", b_zeros as NativeFn),
        ("ones", b_ones),
        ("rand", b_rand),
        ("randn", b_randn),
        ("seed", b_seed),
        ("range", b_range),
        ("shape", b_shape),
        ("reshape", b_reshape),
        ("t", b_t),
        ("sum", b_sum),
        ("mean", b_mean),
        ("max", b_max),
        ("min", b_min),
        ("argmax", b_argmax),
        ("softmax", b_softmax),
        ("onehot", b_onehot),
        ("grad", b_grad),
        ("exp", b_exp),
        ("log", b_log),
        ("tanh", b_tanh),
        ("sigmoid", b_sigmoid),
        ("relu", b_relu),
        ("sqrt", b_sqrt),
        ("abs", b_abs),
        ("sin", b_sin),
        ("cos", b_cos),
        ("floor", b_floor),
        ("round", b_round),
        ("trunc", b_trunc),
        ("clip", b_clip),
        ("sum_rows", b_sum_rows),
        ("rows", b_rows),
        ("compact", b_compact),
        ("expand", b_expand),
        ("screlu", b_screlu),
        ("sum_product", b_sum_product),
        ("screlu_dot", b_screlu_dot),
        ("update_rows", b_update_rows),
        ("read_binary", b_read_binary),
        ("input_waiting", b_input_waiting),
        ("len", b_len),
        ("int", b_int),
        ("ints", b_ints),
        ("popcount", b_popcount),
        ("within", b_within),
        ("args", b_args),
        ("env", b_env),
        ("append", b_append),
        ("delete", b_delete),
        ("files", b_files),
        ("now", b_now),
        ("fixed", b_fixed),
        ("pad", b_pad),
        ("code", b_code),
        ("char", b_char),
        ("insert", b_insert),
        ("remove", b_remove),
        ("find", b_find),
        ("unique", b_unique),
        ("sort_by", b_sort_by),
        ("lowest_bit", b_lowest_bit),
        ("highest_bit", b_highest_bit),
        ("div", b_div),
        ("rand_bits", b_rand_bits),
        ("push", b_push),
        ("map", b_map),
        ("filter", b_filter),
        ("reduce", b_reduce),
        ("keys", b_keys),
        ("values", b_values),
        ("has", b_has),
        ("join", b_join),
        ("split", b_split),
        ("sort", b_sort),
        ("reverse", b_reverse),
        ("slice", b_slice),
        ("upper", b_upper),
        ("lower", b_lower),
        ("trim", b_trim),
        ("replace", b_replace),
        ("lines", b_lines),
        ("text", b_text),
        ("number", b_number),
        ("kind", b_kind),
        ("read", b_read),
        ("write", b_write),
        ("input", b_input),
        ("clock", b_clock),
        ("fail", b_fail),
        ("check", b_check),
        ("json", b_json),
        ("to_json", b_to_json),
        ("save", b_save),
        ("load", b_load),
        ("exists", b_exists),
        ("program_folder", b_program_folder),
        ("key", b_key),
        ("wait", b_wait),
        ("clear", b_clear),
        ("draw", b_draw),
        ("window", crate::gfx::b_window),
        ("window_open", crate::gfx::b_window_open),
        ("frame", crate::gfx::b_frame),
        ("fill", crate::gfx::b_fill),
        ("rect", crate::gfx::b_rect),
        ("circle", crate::gfx::b_circle),
        ("line", crate::gfx::b_line),
        ("pixel", crate::gfx::b_pixel),
        ("label", crate::gfx::b_label),
        ("sprite", crate::gfx::b_sprite),
        ("held", crate::gfx::b_held),
        ("pressed", crate::gfx::b_pressed),
        ("mouse", crate::gfx::b_mouse),
        ("screenshot", crate::gfx::b_screenshot),
        ("sound", crate::sound::b_sound),
        ("music", crate::sound::b_music),
        ("beep", crate::sound::b_beep),
        ("tune", crate::sound::b_tune),
        ("effect", crate::sound::b_effect),
        ("volume", crate::sound::b_volume),
        ("silence", crate::sound::b_silence),
        ("playing", crate::sound::b_playing),
        ("gelu", crate::nn::b_gelu),
        ("silu", crate::nn::b_silu),
        ("softplus", crate::nn::b_softplus),
        ("leaky_relu", crate::nn::b_leaky_relu),
        ("log_softmax", crate::nn::b_log_softmax),
        ("cross_entropy", crate::nn::b_cross_entropy),
        ("mse", crate::nn::b_mse),
        ("bce", crate::nn::b_bce),
        ("take", crate::nn::b_take),
        ("concat", crate::nn::b_concat),
        ("layer_norm", crate::nn::b_layer_norm),
        ("dropout", crate::nn::b_dropout),
        ("causal_mask", crate::nn::b_causal_mask),
        ("xavier", crate::nn::b_xavier),
        ("he", crate::nn::b_he),
        ("accuracy", crate::nn::b_accuracy),
        ("top", crate::nn::b_top),
        ("sample", crate::nn::b_sample),
        ("read_csv", crate::nn::b_read_csv),
        ("columns", crate::nn::b_columns),
        ("shuffle", crate::nn::b_shuffle),
        ("batches", crate::nn::b_batches),
        ("conversation", crate::llm::b_conversation),
        ("history", crate::llm::b_history),
        ("image", crate::llm::b_image),
        ("embed", crate::llm::b_embed),
        ("similarity", crate::llm::b_similarity),
        ("gpu", crate::gpu::b_gpu),
        ("use_gpu", crate::gpu::b_use_gpu),
    ]
}
