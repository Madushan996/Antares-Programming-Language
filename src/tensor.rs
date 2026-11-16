//! Tensors with built-in reverse-mode automatic differentiation.
//!
//! Every numeric array in Antares is a `Node`. Operations on values that
//! depend on a `learn` variable remember their inputs, so `descend` can walk
//! the graph backwards and compute gradients. Values that don't depend on a
//! learnable variable keep no history, so plain maths costs nothing extra.

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

pub type T = Rc<Node>;

#[derive(Clone, Copy, PartialEq)]
pub enum BinK {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
}

#[derive(Clone, Copy, PartialEq)]
pub enum UFn {
    Exp,
    Log,
    Tanh,
    Sigmoid,
    Relu,
    Sqrt,
    Abs,
    Sin,
    Cos,
    /// GELU (the tanh approximation used by GPT models).
    Gelu,
    /// SiLU, also called swish: x * sigmoid(x).
    Silu,
    /// log(1 + exp(x)), a smooth relu.
    Softplus,
    /// relu that lets 1% through below zero.
    LeakyRelu,
}

/// How `descend` changes `learn` variables.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Optim {
    Sgd,
    Adam,
    /// Adam with decoupled weight decay (0.01).
    AdamW,
    /// Gradient descent with momentum (0.9).
    Momentum,
}

pub enum Op {
    Leaf,
    Bin(BinK, T, T),
    MatMul(T, T),
    Neg(T),
    Unary(UFn, T),
    Sum(T),
    Mean(T),
    Softmax(T),
    Transpose(T),
    Reshape(T),
    Index(T, usize),
    /// Picks elements by flat position (slices).
    Gather(T, Vec<usize>),
    /// Sum along one dimension.
    SumAxis(T, usize),
    /// Log of softmax along the last dimension.
    LogSoftmax(T),
    /// Mean cross-entropy of softmax(logits) against target probabilities
    /// (one row of targets per row of logits).
    CrossEntropy(T, Rc<Vec<f64>>),
    /// Tensors joined along one dimension.
    Concat(Vec<T>, usize),
}

pub struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: i32,
}

pub struct Node {
    pub shape: Vec<usize>,
    pub data: RefCell<Vec<f64>>,
    pub grad: RefCell<Vec<f64>>,
    pub requires: bool,
    pub op: Op,
    pub opt: RefCell<Option<Adam>>,
}

impl Node {
    pub fn len(&self) -> usize {
        self.data.borrow().len()
    }

    fn parents(&self) -> Vec<&T> {
        match &self.op {
            Op::Leaf => vec![],
            Op::Bin(_, a, b) | Op::MatMul(a, b) => vec![a, b],
            Op::Neg(a)
            | Op::Unary(_, a)
            | Op::Sum(a)
            | Op::Mean(a)
            | Op::Softmax(a)
            | Op::Transpose(a)
            | Op::Reshape(a)
            | Op::Index(a, _)
            | Op::Gather(a, _)
            | Op::SumAxis(a, _)
            | Op::LogSoftmax(a)
            | Op::CrossEntropy(a, _) => vec![a],
            Op::Concat(parts, _) => parts.iter().collect(),
        }
    }
}

fn make(shape: Vec<usize>, data: Vec<f64>, op: Op, requires: bool) -> T {
    Rc::new(Node {
        shape,
        data: RefCell::new(data),
        grad: RefCell::new(Vec::new()),
        requires,
        op: if requires { op } else { Op::Leaf },
        opt: RefCell::new(None),
    })
}

/// A plain tensor with no gradient tracking.
pub fn new(shape: Vec<usize>, data: Vec<f64>) -> T {
    make(shape, data, Op::Leaf, false)
}

pub fn scalar(x: f64) -> T {
    new(vec![], vec![x])
}

/// A learnable tensor: `descend` will adjust it.
pub fn param(shape: Vec<usize>, data: Vec<f64>) -> T {
    make(shape, data, Op::Leaf, true)
}

pub fn fmt_shape(s: &[usize]) -> String {
    if s.is_empty() {
        return "a single number".to_string();
    }
    let parts: Vec<String> = s.iter().map(|d| d.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

// ---------- broadcasting ----------

enum BMap {
    Same,
    Scalar,
    Idx(Vec<usize>),
}

impl BMap {
    #[inline]
    fn at(&self, i: usize) -> usize {
        match self {
            BMap::Same => i,
            BMap::Scalar => 0,
            BMap::Idx(v) => v[i],
        }
    }
}

fn bshape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let n = a.len().max(b.len());
    let mut out = vec![0; n];
    for i in 0..n {
        let da = if i < n - a.len() { 1 } else { a[i - (n - a.len())] };
        let db = if i < n - b.len() { 1 } else { b[i - (n - b.len())] };
        out[i] = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return Err(format!(
                "shapes {} and {} don't fit together",
                fmt_shape(a),
                fmt_shape(b)
            ));
        };
    }
    Ok(out)
}

fn bmap(src: &[usize], out: &[usize]) -> BMap {
    if src == out {
        return BMap::Same;
    }
    let n_src: usize = src.iter().product();
    if n_src == 1 {
        return BMap::Scalar;
    }
    let total: usize = out.iter().product();
    let off = out.len() - src.len();
    let mut strides = vec![0usize; out.len()];
    let mut s = 1;
    for i in (0..src.len()).rev() {
        strides[i + off] = if src[i] == 1 { 0 } else { s };
        s *= src[i];
    }
    let mut idx = Vec::with_capacity(total);
    let mut counter = vec![0usize; out.len()];
    for _ in 0..total {
        let mut k = 0;
        for d in 0..out.len() {
            k += counter[d] * strides[d];
        }
        idx.push(k);
        for d in (0..out.len()).rev() {
            counter[d] += 1;
            if counter[d] < out[d] {
                break;
            }
            counter[d] = 0;
        }
    }
    BMap::Idx(idx)
}

// ---------- forward operations ----------

fn bin_fn(k: BinK, x: f64, y: f64) -> f64 {
    match k {
        BinK::Add => x + y,
        BinK::Sub => x - y,
        BinK::Mul => x * y,
        BinK::Div => x / y,
        BinK::Pow => x.powf(y),
        BinK::Mod => x.rem_euclid(y),
    }
}

pub fn binary(k: BinK, a: &T, b: &T) -> Result<T, String> {
    let shape = bshape(&a.shape, &b.shape)?;
    let ma = bmap(&a.shape, &shape);
    let mb = bmap(&b.shape, &shape);
    let n: usize = shape.iter().product();
    let ad = a.data.borrow();
    let bd = b.data.borrow();
    let mut out = Vec::with_capacity(n);
    match (&ma, &mb) {
        (BMap::Same, BMap::Same) => {
            for i in 0..n {
                out.push(bin_fn(k, ad[i], bd[i]));
            }
        }
        _ => {
            for i in 0..n {
                out.push(bin_fn(k, ad[ma.at(i)], bd[mb.at(i)]));
            }
        }
    }
    drop(ad);
    drop(bd);
    let req = a.requires || b.requires;
    Ok(make(shape, out, Op::Bin(k, a.clone(), b.clone()), req))
}

fn mat_dims(a: &T, b: &T) -> Result<(usize, usize, usize), String> {
    let (m, n) = match a.shape.len() {
        1 => (1, a.shape[0]),
        2 => (a.shape[0], a.shape[1]),
        _ => return Err("@ works on vectors and matrices (use * to multiply by a number)".into()),
    };
    let (n2, p) = match b.shape.len() {
        1 => (b.shape[0], 1),
        2 => (b.shape[0], b.shape[1]),
        _ => return Err("@ works on vectors and matrices (use * to multiply by a number)".into()),
    };
    if n != n2 {
        return Err(format!(
            "@ can't combine shapes {} and {}: the inner sizes {} and {} differ",
            fmt_shape(&a.shape),
            fmt_shape(&b.shape),
            n,
            n2
        ));
    }
    Ok((m, n, p))
}

/// Whether `use gpu` is on.
static GPU_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Below about 256 x 256 x 256 multiply-adds, copying to the GPU and back
/// takes longer than the CPU takes to do the work.
const GPU_MIN_WORK: usize = 1 << 24;

/// Turns the GPU on or off for big matrix multiplications (`use gpu`).
pub fn set_gpu(on: bool) {
    GPU_ON.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The m x n matrix `a` times the n x p matrix `b` on the GPU, when the GPU
/// is on and the work is big enough to be worth it. None: do it on the CPU
/// (also when the GPU reports a problem).
fn gpu_mm(a: &[f64], b: &[f64], m: usize, n: usize, p: usize) -> Option<Vec<f64>> {
    if !GPU_ON.load(std::sync::atomic::Ordering::Relaxed) || m.saturating_mul(n).saturating_mul(p) < GPU_MIN_WORK {
        return None;
    }
    crate::gpu::matmul(a, b, m, n, p).ok()
}

/// A rows x cols matrix turned on its side.
fn transposed(x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut t = vec![0.0; x.len()];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = x[i * cols + j];
        }
    }
    t
}

/// Splits `out` (rows of `width` numbers) among the CPU's cores and runs
/// `f(first_row, rows)` on each part, when there is enough work (`work`
/// multiply-adds) to be worth it. Each row is still worked out by one core
/// in the usual order, so the results are exactly the same either way.
fn par_rows(out: &mut [f64], width: usize, work: usize, f: impl Fn(usize, &mut [f64]) + Sync) {
    let rows = if width == 0 { 0 } else { out.len() / width };
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(16);
    if cores < 2 || rows < 2 || work < 1_000_000 {
        f(0, out);
        return;
    }
    let per = rows.div_ceil(cores.min(rows));
    std::thread::scope(|sc| {
        for (k, part) in out.chunks_mut(per * width).enumerate() {
            let f = &f;
            sc.spawn(move || f(k * per, part));
        }
    });
}

fn mm(a: &[f64], b: &[f64], m: usize, n: usize, p: usize) -> Vec<f64> {
    if let Some(o) = gpu_mm(a, b, m, n, p) {
        return o;
    }
    let mut o = vec![0.0; m * p];
    par_rows(&mut o, p, m * n * p, |first, part| {
        for (r, orow) in part.chunks_mut(p.max(1)).enumerate() {
            let i = first + r;
            for k in 0..n {
                let x = a[i * n + k];
                if x == 0.0 {
                    continue;
                }
                let row = &b[k * p..(k + 1) * p];
                for j in 0..p {
                    orow[j] += x * row[j];
                }
            }
        }
    });
    o
}

pub fn matmul(a: &T, b: &T) -> Result<T, String> {
    let (m, n, p) = mat_dims(a, b)?;
    let out = mm(&a.data.borrow(), &b.data.borrow(), m, n, p);
    let mut shape = vec![];
    if a.shape.len() == 2 {
        shape.push(m);
    }
    if b.shape.len() == 2 {
        shape.push(p);
    }
    let req = a.requires || b.requires;
    Ok(make(shape, out, Op::MatMul(a.clone(), b.clone()), req))
}

pub fn neg(a: &T) -> T {
    let d: Vec<f64> = a.data.borrow().iter().map(|x| -x).collect();
    make(a.shape.clone(), d, Op::Neg(a.clone()), a.requires)
}

pub fn apply(f: UFn, x: f64) -> f64 {
    match f {
        UFn::Exp => x.exp(),
        UFn::Log => x.ln(),
        UFn::Tanh => x.tanh(),
        UFn::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        UFn::Relu => x.max(0.0),
        UFn::Sqrt => x.sqrt(),
        UFn::Abs => x.abs(),
        UFn::Sin => x.sin(),
        UFn::Cos => x.cos(),
        UFn::Gelu => 0.5 * x * (1.0 + (GELU_K * (x + 0.044715 * x * x * x)).tanh()),
        UFn::Silu => x / (1.0 + (-x).exp()),
        UFn::Softplus => {
            // log(1 + e^x) without overflow for large x
            if x > 30.0 {
                x
            } else {
                x.exp().ln_1p()
            }
        }
        UFn::LeakyRelu => {
            if x > 0.0 {
                x
            } else {
                0.01 * x
            }
        }
    }
}

/// sqrt(2 / pi), for GELU.
const GELU_K: f64 = 0.797_884_560_802_865_4;

pub fn unary(f: UFn, a: &T) -> T {
    let d: Vec<f64> = a.data.borrow().iter().map(|&x| apply(f, x)).collect();
    make(a.shape.clone(), d, Op::Unary(f, a.clone()), a.requires)
}

pub fn sum(a: &T) -> T {
    let s: f64 = a.data.borrow().iter().sum();
    make(vec![], vec![s], Op::Sum(a.clone()), a.requires)
}

pub fn mean(a: &T) -> T {
    let n = a.len().max(1) as f64;
    let s: f64 = a.data.borrow().iter().sum();
    make(vec![], vec![s / n], Op::Mean(a.clone()), a.requires)
}

/// Softmax along the last dimension.
pub fn softmax(a: &T) -> T {
    let d = a.shape.last().copied().unwrap_or(1).max(1);
    let src = a.data.borrow();
    let mut out = vec![0.0; src.len()];
    for (row, orow) in src.chunks(d).zip(out.chunks_mut(d)) {
        let mx = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0;
        for (o, &x) in orow.iter_mut().zip(row) {
            *o = (x - mx).exp();
            total += *o;
        }
        for o in orow.iter_mut() {
            *o /= total;
        }
    }
    drop(src);
    make(a.shape.clone(), out, Op::Softmax(a.clone()), a.requires)
}

pub fn transpose(a: &T) -> Result<T, String> {
    match a.shape.len() {
        0 | 1 => Ok(a.clone()),
        2 => {
            let (m, n) = (a.shape[0], a.shape[1]);
            let src = a.data.borrow();
            let mut out = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    out[j * m + i] = src[i * n + j];
                }
            }
            drop(src);
            Ok(make(vec![n, m], out, Op::Transpose(a.clone()), a.requires))
        }
        _ => Err("t() flips rows and columns, so it needs a matrix".into()),
    }
}

pub fn reshape(a: &T, shape: Vec<usize>) -> Result<T, String> {
    let n: usize = shape.iter().product();
    if n != a.len() {
        return Err(format!(
            "can't reshape {} values into shape {}",
            a.len(),
            fmt_shape(&shape)
        ));
    }
    let d = a.data.borrow().clone();
    Ok(make(shape, d, Op::Reshape(a.clone()), a.requires))
}

pub fn index(t: &T, idx: &[i64]) -> Result<T, String> {
    if idx.len() > t.shape.len() {
        return Err(format!(
            "too many indexes: this value has {} dimension(s)",
            t.shape.len()
        ));
    }
    let mut off = 0;
    let mut size = t.len();
    for (d, &i) in idx.iter().enumerate() {
        let n = t.shape[d] as i64;
        let j = if i < 0 { i + n } else { i };
        if j < 0 || j >= n {
            return Err(format!(
                "index {} is out of range (valid: 0 to {})",
                i,
                n - 1
            ));
        }
        size /= n as usize;
        off += j as usize * size;
    }
    let shape = t.shape[idx.len()..].to_vec();
    let data = t.data.borrow()[off..off + size].to_vec();
    Ok(make(shape, data, Op::Index(t.clone(), off), t.requires))
}

/// Builds a new tensor from chosen positions along each dimension.
/// `sel[d]` lists which indexes of dimension `d` to keep.
pub fn gather(t: &T, shape: Vec<usize>, sel: &[Vec<usize>]) -> T {
    let mut strides = vec![1usize; t.shape.len()];
    for d in (0..t.shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * t.shape[d + 1];
    }
    let total: usize = sel.iter().map(|s| s.len()).product();
    let mut offsets = Vec::with_capacity(total);
    let mut counter = vec![0usize; sel.len()];
    for _ in 0..total {
        let mut off = 0;
        for d in 0..sel.len() {
            off += sel[d][counter[d]] * strides[d];
        }
        offsets.push(off);
        for d in (0..sel.len()).rev() {
            counter[d] += 1;
            if counter[d] < sel[d].len() {
                break;
            }
            counter[d] = 0;
        }
    }
    let src = t.data.borrow();
    let data: Vec<f64> = offsets.iter().map(|&o| src[o]).collect();
    drop(src);
    make(shape, data, Op::Gather(t.clone(), offsets), t.requires)
}

/// (outer, n, inner) element counts around one dimension.
fn axis_parts(shape: &[usize], axis: usize) -> (usize, usize, usize) {
    (shape[..axis].iter().product(), shape[axis], shape[axis + 1..].iter().product())
}

pub fn sum_axis(t: &T, axis: i64) -> Result<T, String> {
    let rank = t.shape.len() as i64;
    let ax = if axis < 0 { axis + rank } else { axis };
    if ax < 0 || ax >= rank {
        return Err(format!(
            "dimension {} doesn't exist: this value has {} dimension(s), numbered from 0",
            axis, rank
        ));
    }
    let ax = ax as usize;
    let (outer, n, inner) = axis_parts(&t.shape, ax);
    let src = t.data.borrow();
    let mut out = vec![0.0; outer * inner];
    for o in 0..outer {
        for j in 0..n {
            let row = &src[(o * n + j) * inner..(o * n + j + 1) * inner];
            for (k, x) in row.iter().enumerate() {
                out[o * inner + k] += x;
            }
        }
    }
    drop(src);
    let mut shape = t.shape.clone();
    shape.remove(ax);
    Ok(make(shape, out, Op::SumAxis(t.clone(), ax), t.requires))
}

pub fn mean_axis(t: &T, axis: i64) -> Result<T, String> {
    let s = sum_axis(t, axis)?;
    let n = t.len() / s.len().max(1);
    binary(BinK::Div, &s, &scalar(n.max(1) as f64))
}

/// log(softmax(x)) along the last dimension, computed without overflow.
pub fn log_softmax(a: &T) -> T {
    let d = a.shape.last().copied().unwrap_or(1).max(1);
    let src = a.data.borrow();
    let mut out = vec![0.0; src.len()];
    for (row, orow) in src.chunks(d).zip(out.chunks_mut(d)) {
        let mx = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse = mx + row.iter().map(|x| (x - mx).exp()).sum::<f64>().ln();
        for (o, &x) in orow.iter_mut().zip(row) {
            *o = x - lse;
        }
    }
    drop(src);
    make(a.shape.clone(), out, Op::LogSoftmax(a.clone()), a.requires)
}

/// The mean cross-entropy loss of `logits` (one row per example, one
/// column per class) against `target`, the right probabilities for each
/// class (usually one 1 per row). Softmax is part of it, so `logits` are
/// the raw scores.
pub fn cross_entropy(logits: &T, target: Vec<f64>) -> Result<T, String> {
    if target.len() != logits.len() {
        return Err(format!("cross_entropy has {} scores but {} target values", logits.len(), target.len()));
    }
    let lsm = log_softmax(logits);
    let d = logits.shape.last().copied().unwrap_or(1).max(1);
    let rows = (logits.len() / d).max(1);
    let total: f64 = lsm.data.borrow().iter().zip(&target).map(|(l, t)| if *t == 0.0 { 0.0 } else { l * t }).sum();
    let loss = -total / rows as f64;
    Ok(make(vec![], vec![loss], Op::CrossEntropy(logits.clone(), Rc::new(target)), logits.requires))
}

/// Joins tensors along dimension `axis`; the other dimensions must match.
pub fn concat(parts: &[T], axis: i64) -> Result<T, String> {
    let first = parts.first().ok_or("concat needs at least one tensor")?;
    let rank = first.shape.len() as i64;
    let ax = if axis < 0 { axis + rank } else { axis };
    if rank == 0 || ax < 0 || ax >= rank {
        return Err(format!(
            "concat can't join along dimension {}: the tensors have {} dimension(s), numbered from 0",
            axis, rank
        ));
    }
    let ax = ax as usize;
    for p in parts {
        let same = p.shape.len() == first.shape.len()
            && p.shape.iter().zip(&first.shape).enumerate().all(|(i, (x, y))| i == ax || x == y);
        if !same {
            return Err(format!(
                "concat can't join shapes {} and {} along dimension {}",
                fmt_shape(&first.shape),
                fmt_shape(&p.shape),
                ax
            ));
        }
    }
    let outer: usize = first.shape[..ax].iter().product();
    let inner: usize = first.shape[ax + 1..].iter().product();
    let total: usize = parts.iter().map(|p| p.shape[ax]).sum();
    let mut out = Vec::with_capacity(outer * total * inner);
    let datas: Vec<_> = parts.iter().map(|p| p.data.borrow()).collect();
    for o in 0..outer {
        for (p, d) in parts.iter().zip(&datas) {
            let n = p.shape[ax] * inner;
            out.extend_from_slice(&d[o * n..(o + 1) * n]);
        }
    }
    drop(datas);
    let mut shape = first.shape.clone();
    shape[ax] = total;
    let req = parts.iter().any(|p| p.requires);
    Ok(make(shape, out, Op::Concat(parts.to_vec(), ax), req))
}

// ---------- backward pass ----------

fn acc(t: &T, d: &[f64]) {
    if !t.requires {
        return;
    }
    let mut g = t.grad.borrow_mut();
    if g.is_empty() {
        *g = d.to_vec();
    } else {
        for (x, y) in g.iter_mut().zip(d) {
            *x += y;
        }
    }
}

fn topo(root: &T) -> Vec<T> {
    let mut order = vec![];
    let mut seen: HashSet<*const Node> = HashSet::new();
    let mut stack: Vec<(T, bool)> = vec![(root.clone(), false)];
    while let Some((n, done)) = stack.pop() {
        if done {
            order.push(n);
            continue;
        }
        if !seen.insert(Rc::as_ptr(&n)) {
            continue;
        }
        stack.push((n.clone(), true));
        for p in n.parents() {
            if p.requires && !seen.contains(&Rc::as_ptr(p)) {
                stack.push((p.clone(), false));
            }
        }
    }
    order
}

fn back(node: &Node, g: &[f64]) {
    match &node.op {
        Op::Leaf => {}
        Op::Bin(k, a, b) => {
            let ma = bmap(&a.shape, &node.shape);
            let mb = bmap(&b.shape, &node.shape);
            let ad = a.data.borrow();
            let bd = b.data.borrow();
            let od = node.data.borrow();
            let mut da = if a.requires { vec![0.0; ad.len()] } else { vec![] };
            let mut db = if b.requires { vec![0.0; bd.len()] } else { vec![] };
            for i in 0..g.len() {
                let (ia, ib) = (ma.at(i), mb.at(i));
                let (x, y, gi) = (ad[ia], bd[ib], g[i]);
                let (ga, gb) = match k {
                    BinK::Add => (gi, gi),
                    BinK::Sub => (gi, -gi),
                    BinK::Mul => (gi * y, gi * x),
                    BinK::Div => (gi / y, -gi * x / (y * y)),
                    BinK::Pow => (
                        gi * y * x.powf(y - 1.0),
                        if b.requires && x > 0.0 { gi * od[i] * x.ln() } else { 0.0 },
                    ),
                    BinK::Mod => (gi, 0.0),
                };
                if a.requires {
                    da[ia] += ga;
                }
                if b.requires {
                    db[ib] += gb;
                }
            }
            drop(ad);
            drop(bd);
            drop(od);
            acc(a, &da);
            acc(b, &db);
        }
        Op::MatMul(a, b) => {
            let (m, n, p) = mat_dims(a, b).unwrap();
            let ad = a.data.borrow();
            let bd = b.data.borrow();
            let mut da = if a.requires { vec![0.0; m * n] } else { vec![] };
            let mut db = if b.requires { vec![0.0; n * p] } else { vec![] };
            // plain slices, so the cores can share them
            let (ad, bd) = (&ad[..], &bd[..]);
            // with `use gpu`: da = g @ t(b) and db = t(a) @ g on the GPU
            let mut on_gpu_a = false;
            let mut on_gpu_b = false;
            if a.requires {
                if let Some(d) = gpu_mm(g, &transposed(bd, n, p), m, p, n) {
                    da = d;
                    on_gpu_a = true;
                }
            }
            if b.requires {
                if let Some(d) = gpu_mm(&transposed(ad, m, n), g, n, m, p) {
                    db = d;
                    on_gpu_b = true;
                }
            }
            if a.requires && !on_gpu_a {
                // da[i, k] = sum over j of g[i, j] * b[k, j]
                par_rows(&mut da, n, m * n * p, |first, part| {
                    for (r, drow) in part.chunks_mut(n.max(1)).enumerate() {
                        let i = first + r;
                        let grow = &g[i * p..(i + 1) * p];
                        for (k, d) in drow.iter_mut().enumerate() {
                            let brow = &bd[k * p..(k + 1) * p];
                            let mut s = 0.0;
                            for j in 0..p {
                                s += grow[j] * brow[j];
                            }
                            *d = s;
                        }
                    }
                });
            }
            if b.requires && !on_gpu_b {
                // db[k, j] = sum over i of a[i, k] * g[i, j], adding i in order
                par_rows(&mut db, p, m * n * p, |first, part| {
                    let rows = part.len() / p.max(1);
                    for i in 0..m {
                        let grow = &g[i * p..(i + 1) * p];
                        for r in 0..rows {
                            let x = ad[i * n + first + r];
                            let drow = &mut part[r * p..(r + 1) * p];
                            for j in 0..p {
                                drow[j] += x * grow[j];
                            }
                        }
                    }
                });
            }
            acc(a, &da);
            acc(b, &db);
        }
        Op::Neg(a) => {
            let d: Vec<f64> = g.iter().map(|x| -x).collect();
            acc(a, &d);
        }
        Op::Unary(f, a) => {
            let d: Vec<f64> = {
                let xd = a.data.borrow();
                let yd = node.data.borrow();
                (0..g.len())
                    .map(|i| {
                        let (x, y) = (xd[i], yd[i]);
                        g[i] * match f {
                            UFn::Exp => y,
                            UFn::Log => 1.0 / x,
                            UFn::Tanh => 1.0 - y * y,
                            UFn::Sigmoid => y * (1.0 - y),
                            UFn::Relu => {
                                if x > 0.0 {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            UFn::Sqrt => 0.5 / y,
                            UFn::Abs => {
                                if x >= 0.0 {
                                    1.0
                                } else {
                                    -1.0
                                }
                            }
                            UFn::Sin => x.cos(),
                            UFn::Cos => -x.sin(),
                            UFn::Gelu => {
                                let u = GELU_K * (x + 0.044715 * x * x * x);
                                let th = u.tanh();
                                0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * GELU_K * (1.0 + 3.0 * 0.044715 * x * x)
                            }
                            UFn::Silu => {
                                let sg = 1.0 / (1.0 + (-x).exp());
                                sg * (1.0 + x * (1.0 - sg))
                            }
                            UFn::Softplus => 1.0 / (1.0 + (-x).exp()),
                            UFn::LeakyRelu => {
                                if x > 0.0 {
                                    1.0
                                } else {
                                    0.01
                                }
                            }
                        }
                    })
                    .collect()
            };
            acc(a, &d);
        }
        Op::Sum(a) => acc(a, &vec![g[0]; a.len()]),
        Op::Mean(a) => {
            let n = a.len();
            acc(a, &vec![g[0] / n.max(1) as f64; n]);
        }
        Op::Softmax(a) => {
            let d = node.shape.last().copied().unwrap_or(1).max(1);
            let mut out = vec![0.0; g.len()];
            {
                let s = node.data.borrow();
                for ((srow, grow), orow) in s.chunks(d).zip(g.chunks(d)).zip(out.chunks_mut(d)) {
                    let dot: f64 = srow.iter().zip(grow).map(|(a, b)| a * b).sum();
                    for j in 0..srow.len() {
                        orow[j] = srow[j] * (grow[j] - dot);
                    }
                }
            }
            acc(a, &out);
        }
        Op::Transpose(a) => {
            let (m, n) = (a.shape[0], a.shape[1]);
            let mut d = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    d[i * n + j] = g[j * m + i];
                }
            }
            acc(a, &d);
        }
        Op::Reshape(a) => acc(a, g),
        Op::Gather(a, offsets) => {
            let mut d = vec![0.0; a.len()];
            for (i, &o) in offsets.iter().enumerate() {
                d[o] += g[i];
            }
            acc(a, &d);
        }
        Op::SumAxis(a, ax) => {
            let (outer, n, inner) = axis_parts(&a.shape, *ax);
            let mut d = vec![0.0; a.len()];
            for o in 0..outer {
                for j in 0..n {
                    for k in 0..inner {
                        d[(o * n + j) * inner + k] = g[o * inner + k];
                    }
                }
            }
            acc(a, &d);
        }
        Op::Index(a, off) => {
            let mut d = vec![0.0; a.len()];
            d[*off..*off + g.len()].copy_from_slice(g);
            acc(a, &d);
        }
        Op::LogSoftmax(a) => {
            // d/dx: g - softmax(x) * sum(g), row by row
            let d = node.shape.last().copied().unwrap_or(1).max(1);
            let mut out = vec![0.0; g.len()];
            {
                let y = node.data.borrow();
                for ((yrow, grow), orow) in y.chunks(d).zip(g.chunks(d)).zip(out.chunks_mut(d)) {
                    let total: f64 = grow.iter().sum();
                    for j in 0..yrow.len() {
                        orow[j] = grow[j] - yrow[j].exp() * total;
                    }
                }
            }
            acc(a, &out);
        }
        Op::CrossEntropy(a, target) => {
            // d/dlogits: (softmax(logits) * sum(target row) - target) / rows
            let d = a.shape.last().copied().unwrap_or(1).max(1);
            let rows = (a.len() / d).max(1) as f64;
            let mut out = vec![0.0; a.len()];
            {
                let x = a.data.borrow();
                for ((xrow, trow), orow) in x.chunks(d).zip(target.chunks(d)).zip(out.chunks_mut(d)) {
                    let mx = xrow.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let total: f64 = xrow.iter().map(|v| (v - mx).exp()).sum();
                    let tsum: f64 = trow.iter().sum();
                    for j in 0..xrow.len() {
                        orow[j] = g[0] * ((xrow[j] - mx).exp() / total * tsum - trow[j]) / rows;
                    }
                }
            }
            acc(a, &out);
        }
        Op::Concat(parts, ax) => {
            let outer: usize = parts[0].shape[..*ax].iter().product();
            let inner: usize = parts[0].shape[*ax + 1..].iter().product();
            let total: usize = parts.iter().map(|p| p.shape[*ax]).sum();
            let mut start = 0;
            for p in parts {
                let n = p.shape[*ax] * inner;
                if p.requires {
                    let mut d = Vec::with_capacity(p.len());
                    for o in 0..outer {
                        let row = o * total * inner + start;
                        d.extend_from_slice(&g[row..row + n]);
                    }
                    acc(p, &d);
                }
                start += n;
            }
        }
    }
}

fn backward(root: &T) -> Vec<T> {
    let order = topo(root);
    *root.grad.borrow_mut() = vec![1.0; root.len()];
    for n in order.iter().rev() {
        if matches!(n.op, Op::Leaf) {
            continue;
        }
        let g = std::mem::take(&mut *n.grad.borrow_mut());
        if g.is_empty() {
            continue;
        }
        back(n, &g);
    }
    order
}

/// One optimisation step: nudge every `learn` variable that `loss` depends on.
/// With `clip`, the gradients are first scaled down if their overall size
/// (the square root of the sum of their squares) is larger.
pub fn descend(loss: &T, rate: f64, optim: Optim, clip: Option<f64>) -> Result<(), String> {
    if !loss.requires {
        return Err(
            "there is nothing to learn here: this value doesn't depend on any `learn` variable"
                .into(),
        );
    }
    if loss.data.borrow().iter().any(|x| !x.is_finite()) {
        return Err(
            "the value being minimised is no longer a real number (it blew up). Try a smaller rate after `by`"
                .into(),
        );
    }
    let order = backward(loss);
    let mut scale = 1.0;
    if let Some(limit) = clip {
        let total: f64 = order
            .iter()
            .filter(|n| matches!(n.op, Op::Leaf) && n.requires)
            .map(|n| n.grad.borrow().iter().map(|x| x * x).sum::<f64>())
            .sum();
        let norm = total.sqrt();
        if norm > limit && norm > 0.0 {
            scale = limit / norm;
        }
    }
    for n in &order {
        if matches!(n.op, Op::Leaf) && n.requires {
            if scale != 1.0 {
                n.grad.borrow_mut().iter_mut().for_each(|x| *x *= scale);
            }
            let g = n.grad.borrow();
            if !g.is_empty() {
                let mut d = n.data.borrow_mut();
                if optim == Optim::Adam || optim == Optim::AdamW {
                    let mut o = n.opt.borrow_mut();
                    let len = d.len();
                    let st = o.get_or_insert_with(|| Adam {
                        m: vec![0.0; len],
                        v: vec![0.0; len],
                        t: 0,
                    });
                    st.t += 1;
                    let (b1, b2) = (0.9f64, 0.999f64);
                    let c1 = 1.0 - b1.powi(st.t);
                    let c2 = 1.0 - b2.powi(st.t);
                    let decay = if optim == Optim::AdamW { 0.01 } else { 0.0 };
                    for i in 0..len {
                        st.m[i] = b1 * st.m[i] + (1.0 - b1) * g[i];
                        st.v[i] = b2 * st.v[i] + (1.0 - b2) * g[i] * g[i];
                        // (written exactly as before AdamW existed, so Adam's results don't change)
                        let step = rate * (st.m[i] / c1) / ((st.v[i] / c2).sqrt() + 1e-8);
                        d[i] -= if decay == 0.0 { step } else { step + rate * decay * d[i] };
                    }
                } else if optim == Optim::Momentum {
                    let mut o = n.opt.borrow_mut();
                    let len = d.len();
                    let st = o.get_or_insert_with(|| Adam { m: vec![0.0; len], v: vec![], t: 0 });
                    for i in 0..len {
                        st.m[i] = 0.9 * st.m[i] + g[i];
                        d[i] -= rate * st.m[i];
                    }
                } else {
                    for i in 0..d.len() {
                        d[i] -= rate * g[i];
                    }
                }
            }
        }
        n.grad.borrow_mut().clear();
    }
    Ok(())
}

/// The gradient of `loss` with respect to `wrt`, without changing anything.
pub fn grad_of(loss: &T, wrt: &T) -> Vec<f64> {
    if !loss.requires {
        return vec![0.0; wrt.len()];
    }
    let order = backward(loss);
    let mut g = wrt.grad.borrow().clone();
    if g.is_empty() {
        g = vec![0.0; wrt.len()];
    }
    for n in &order {
        n.grad.borrow_mut().clear();
    }
    g
}
