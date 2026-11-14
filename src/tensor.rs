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
            | Op::SumAxis(a, _) => vec![a],
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

fn mm(a: &[f64], b: &[f64], m: usize, n: usize, p: usize) -> Vec<f64> {
    let mut o = vec![0.0; m * p];
    for i in 0..m {
        let orow = &mut o[i * p..(i + 1) * p];
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
    }
}

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
            for i in 0..m {
                for k in 0..n {
                    let x = ad[i * n + k];
                    let mut s = 0.0;
                    for j in 0..p {
                        let gij = g[i * p + j];
                        s += gij * bd[k * p + j];
                        if b.requires {
                            db[k * p + j] += x * gij;
                        }
                    }
                    if a.requires {
                        da[i * n + k] = s;
                    }
                }
            }
            drop(ad);
            drop(bd);
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
pub fn descend(loss: &T, rate: f64, adam: bool) -> Result<(), String> {
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
    for n in &order {
        if matches!(n.op, Op::Leaf) && n.requires {
            let g = n.grad.borrow();
            if !g.is_empty() {
                let mut d = n.data.borrow_mut();
                if adam {
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
                    for i in 0..len {
                        st.m[i] = b1 * st.m[i] + (1.0 - b1) * g[i];
                        st.v[i] = b2 * st.v[i] + (1.0 - b2) * g[i] * g[i];
                        d[i] -= rate * (st.m[i] / c1) / ((st.v[i] / c2).sqrt() + 1e-8);
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
