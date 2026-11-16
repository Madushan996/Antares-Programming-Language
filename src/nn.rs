//! Deep learning and data: activations, losses, layers' building blocks,
//! weight initialisation, sampling, and tools for loading and batching data.
//!
//! Everything that takes a tensor keeps working with `learn` variables:
//! gradients flow through all of it except where a function says otherwise.

use crate::builtins::{normal, num, ten, text, uniform, want};
use crate::interp::Interp;
use crate::tensor::{self, BinK, UFn, T};
use crate::value::{wrap, Value};
use std::rc::Rc;

type Res = Result<Value, String>;

fn inputs(a: &[Value], lo: usize, hi: usize, name: &str, what: &str) -> Result<(), String> {
    if a.len() < lo || a.len() > hi {
        return Err(format!("{} takes {}, but was given {} input{}", name, what, a.len(), if a.len() == 1 { "" } else { "s" }));
    }
    Ok(())
}

fn whole(v: &Value, name: &str, what: &str) -> Result<usize, String> {
    let n = num(v, name)?;
    if n != n.trunc() || n < 0.0 {
        return Err(format!("{} needs {} that is a whole number, 0 or more, not {}", name, what, v.show()));
    }
    Ok(n as usize)
}

fn bin(k: BinK, a: &T, b: &T) -> Result<T, String> {
    tensor::binary(k, a, b)
}

fn scalar(x: f64) -> T {
    tensor::scalar(x)
}

// ---------- activations ----------

macro_rules! activation {
    ($f:ident, $name:expr, $u:expr) => {
        pub fn $f(_: &mut Interp, a: Vec<Value>) -> Res {
            want(&a, 1, $name)?;
            Ok(wrap(tensor::unary($u, &ten(&a[0], $name)?)))
        }
    };
}

activation!(b_gelu, "gelu", UFn::Gelu);
activation!(b_silu, "silu", UFn::Silu);
activation!(b_softplus, "softplus", UFn::Softplus);
activation!(b_leaky_relu, "leaky_relu", UFn::LeakyRelu);

/// log_softmax(x): the log of softmax along the last dimension, without overflow.
pub fn b_log_softmax(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "log_softmax")?;
    Ok(wrap(tensor::log_softmax(&ten(&a[0], "log_softmax")?)))
}

// ---------- losses ----------

/// cross_entropy(scores, answers): the usual loss for choosing between
/// classes. `scores` has one row per example and one column per class (raw
/// scores: softmax is built in). `answers` are the right class numbers, one
/// per row, or the right probabilities, the same shape as `scores`.
pub fn b_cross_entropy(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "cross_entropy")?;
    let logits = ten(&a[0], "cross_entropy")?;
    let classes = logits.shape.last().copied().unwrap_or(1);
    let rows = logits.len() / classes.max(1);
    let answers = ten(&a[1], "cross_entropy")?;
    let target = if answers.shape == logits.shape {
        answers.vals().clone()
    } else if answers.len() == rows {
        let mut t = vec![0.0; logits.len()];
        for (r, &c) in answers.vals().iter().enumerate() {
            if c != c.trunc() || c < 0.0 || c as usize >= classes {
                return Err(format!("cross_entropy: answer {} isn't a class number from 0 to {}", crate::value::fmt_num(c), classes - 1));
            }
            t[r * classes + c as usize] = 1.0;
        }
        t
    } else {
        return Err(format!(
            "cross_entropy needs one answer per row ({} here), or answers shaped like the scores {}, not {}",
            rows,
            tensor::fmt_shape(&logits.shape),
            tensor::fmt_shape(&answers.shape)
        ));
    };
    Ok(wrap(tensor::cross_entropy(&logits, target)?))
}

/// mse(guess, target): the mean squared error.
pub fn b_mse(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "mse")?;
    let d = bin(BinK::Sub, &ten(&a[0], "mse")?, &ten(&a[1], "mse")?)?;
    Ok(wrap(tensor::mean(&bin(BinK::Mul, &d, &d)?)))
}

/// bce(probability, target): binary cross-entropy, for yes/no predictions
/// that come out of sigmoid.
pub fn b_bce(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "bce")?;
    let p = ten(&a[0], "bce")?;
    let y = ten(&a[1], "bce")?;
    let eps = scalar(1e-7);
    let one = scalar(1.0);
    let lp = tensor::unary(UFn::Log, &bin(BinK::Add, &p, &eps)?);
    let lq = tensor::unary(UFn::Log, &bin(BinK::Add, &bin(BinK::Sub, &one, &p)?, &eps)?);
    let terms = bin(BinK::Add, &bin(BinK::Mul, &y, &lp)?, &bin(BinK::Mul, &bin(BinK::Sub, &one, &y)?, &lq)?)?;
    Ok(wrap(tensor::neg(&tensor::mean(&terms))))
}

// ---------- building blocks ----------

fn index_list(v: &Value, name: &str, n: usize) -> Result<Vec<usize>, String> {
    let raw: Vec<f64> = match v {
        Value::List(l) => l.iter().map(|x| x.scalar()).collect::<Option<Vec<_>>>().ok_or_else(|| format!("{} needs whole numbers as positions", name))?,
        other => match other.as_tensor() {
            Some(t) => t.vals().clone(),
            None => return Err(format!("{} needs positions: a number or a list of numbers, not a {}", name, other.kind())),
        },
    };
    raw.iter()
        .map(|&x| {
            let i = if x < 0.0 { x + n as f64 } else { x };
            if i != i.trunc() || i < 0.0 || i as usize >= n {
                Err(format!("{}: position {} is out of range (valid: 0 to {})", name, crate::value::fmt_num(x), n as i64 - 1))
            } else {
                Ok(i as usize)
            }
        })
        .collect()
}

/// take(table, positions): the rows of `table` at those positions, in that
/// order (also called an embedding lookup). Gradients flow back to the rows.
pub fn b_take(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "take")?;
    let t = ten(&a[0], "take")?;
    if t.shape.is_empty() {
        return Err("take needs a row or a table, not a single number".into());
    }
    let single = a[1].scalar().is_some() && !matches!(a[1], Value::List(_)) && a[1].as_tensor().is_some_and(|x| x.shape.is_empty());
    let idx = index_list(&a[1], "take", t.shape[0])?;
    let mut sel = vec![idx.clone()];
    for &d in &t.shape[1..] {
        sel.push((0..d).collect());
    }
    let mut shape = t.shape.clone();
    shape[0] = idx.len();
    if single {
        shape.remove(0);
    }
    Ok(wrap(tensor::gather(&t, shape, &sel)))
}

/// concat(tensors [, dimension]): joins tensors along a dimension (0 unless given).
pub fn b_concat(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 1, 2, "concat", "a list of tensors, and optionally a dimension")?;
    let Value::List(items) = &a[0] else {
        return Err(format!("concat needs a list of tensors, like concat([a, b]), not a {}", a[0].kind()));
    };
    let parts: Vec<T> = items.iter().map(|v| ten(v, "concat")).collect::<Result<_, _>>()?;
    let axis = match a.get(1) {
        Some(v) => num(v, "concat")? as i64,
        None => 0,
    };
    Ok(wrap(tensor::concat(&parts, axis)?))
}

/// layer_norm(x): each row scaled to mean 0 and spread 1 (along the last
/// dimension). Multiply by a learned gain and add a learned bias after it.
pub fn b_layer_norm(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 1, 2, "layer_norm", "a tensor, and optionally a small number to keep it stable")?;
    let x = ten(&a[0], "layer_norm")?;
    let eps = match a.get(1) {
        Some(v) => num(v, "layer_norm")?,
        None => 1e-5,
    };
    if x.shape.is_empty() {
        return Err("layer_norm needs a row or a table, not a single number".into());
    }
    let mut keep = x.shape.clone();
    *keep.last_mut().unwrap() = 1;
    let mu = tensor::reshape(&tensor::mean_axis(&x, -1)?, keep.clone())?;
    let c = bin(BinK::Sub, &x, &mu)?;
    let var = tensor::reshape(&tensor::mean_axis(&bin(BinK::Mul, &c, &c)?, -1)?, keep)?;
    let sd = tensor::unary(UFn::Sqrt, &bin(BinK::Add, &var, &scalar(eps))?);
    Ok(wrap(bin(BinK::Div, &c, &sd)?))
}

/// dropout(x, p): while training, sets each value to 0 with probability p
/// (and scales up the rest), so a network can't rely on any one value.
pub fn b_dropout(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "dropout")?;
    let x = ten(&a[0], "dropout")?;
    let p = num(&a[1], "dropout")?;
    if !(0.0..1.0).contains(&p) {
        return Err(format!("dropout needs a probability from 0 up to (not including) 1, not {}", a[1].show()));
    }
    if p == 0.0 {
        return Ok(wrap(x));
    }
    let keep = 1.0 / (1.0 - p);
    let mask: Vec<f64> = (0..x.len()).map(|_| if uniform() < p { 0.0 } else { keep }).collect();
    Ok(wrap(bin(BinK::Mul, &x, &tensor::new(x.shape.clone(), mask))?))
}

/// causal_mask(n): an n by n table that is 0 on and below the diagonal and
/// a large negative number above it. Added to attention scores before
/// softmax, it stops each position from looking at later ones.
pub fn b_causal_mask(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "causal_mask")?;
    let n = whole(&a[0], "causal_mask", "a size")?;
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in i + 1..n {
            d[i * n + j] = -1e9;
        }
    }
    Ok(wrap(tensor::new(vec![n, n], d)))
}

fn init(a: &[Value], name: &str, spread: fn(usize, usize) -> f64) -> Res {
    want(a, 2, name)?;
    let rows = whole(&a[0], name, "a number of inputs")?;
    let cols = whole(&a[1], name, "a number of outputs")?;
    let s = spread(rows, cols);
    Ok(wrap(tensor::new(vec![rows, cols], (0..rows * cols).map(|_| normal() * s).collect())))
}

/// xavier(inputs, outputs): random starting weights suited to tanh and sigmoid.
pub fn b_xavier(_: &mut Interp, a: Vec<Value>) -> Res {
    init(&a, "xavier", |i, o| (2.0 / (i + o).max(1) as f64).sqrt())
}

/// he(inputs, outputs): random starting weights suited to relu and gelu.
pub fn b_he(_: &mut Interp, a: Vec<Value>) -> Res {
    init(&a, "he", |i, _| (2.0 / i.max(1) as f64).sqrt())
}

// ---------- reading results ----------

/// accuracy(scores, answers): the fraction of rows whose highest score is
/// the right class. `answers` are class numbers, or rows like the scores.
pub fn b_accuracy(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "accuracy")?;
    let s = ten(&a[0], "accuracy")?;
    let classes = s.shape.last().copied().unwrap_or(1).max(1);
    let rows = s.len() / classes;
    let ans = ten(&a[1], "accuracy")?;
    let best = |row: &[f64]| row.iter().enumerate().fold(0, |b, (i, x)| if *x > row[b] { i } else { b });
    let sd = s.vals();
    let ad = ans.vals();
    let right: Vec<usize> = if ans.len() == rows {
        ad.iter().map(|x| *x as usize).collect()
    } else if ans.len() == s.len() {
        ad.chunks(classes).map(best).collect()
    } else {
        return Err(format!("accuracy needs one answer per row ({} here), not {}", rows, ans.len()));
    };
    let hits = sd.chunks(classes).zip(&right).filter(|(row, r)| best(row) == **r).count();
    Ok(Value::Num(hits as f64 / rows.max(1) as f64))
}

/// top(values, k): the positions of the k largest values, largest first.
pub fn b_top(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "top")?;
    let t = ten(&a[0], "top")?;
    let k = whole(&a[1], "top", "a count")?;
    let d = t.vals();
    let mut order: Vec<usize> = (0..d.len()).collect();
    order.sort_by(|&i, &j| d[j].partial_cmp(&d[i]).unwrap_or(std::cmp::Ordering::Equal).then(i.cmp(&j)));
    order.truncate(k);
    Ok(wrap(tensor::new(vec![order.len()], order.iter().map(|&i| i as f64).collect())))
}

/// sample(probabilities [, temperature]): a random position, chosen with
/// those probabilities. A temperature below 1 makes likely choices more
/// likely; above 1 makes the choice more adventurous.
pub fn b_sample(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 1, 2, "sample", "probabilities, and optionally a temperature")?;
    let t = ten(&a[0], "sample")?;
    let temp = match a.get(1) {
        Some(v) => num(v, "sample")?,
        None => 1.0,
    };
    if !(temp > 0.0) {
        return Err(format!("sample needs a temperature above 0, not {}", a.get(1).map(|v| v.show()).unwrap_or_default()));
    }
    let d = t.vals();
    if d.is_empty() || d.iter().any(|p| *p < 0.0 || !p.is_finite()) {
        return Err("sample needs probabilities: numbers of 0 or more, like the output of softmax".into());
    }
    let w: Vec<f64> = d.iter().map(|p| if temp == 1.0 { *p } else { p.powf(1.0 / temp) }).collect();
    let total: f64 = w.iter().sum();
    if total <= 0.0 {
        return Err("sample needs probabilities that aren't all 0".into());
    }
    let mut r = uniform() * total;
    for (i, x) in w.iter().enumerate() {
        r -= x;
        if r < 0.0 {
            return Ok(Value::Num(i as f64));
        }
    }
    Ok(Value::Num((w.len() - 1) as f64))
}

// ---------- data ----------

/// Splits one CSV line into fields, honouring quotes ("a, b" and "" inside).
fn csv_fields(line: &str) -> Vec<String> {
    let mut out = vec![];
    let mut cur = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, quoted) {
            ('"', true) if chars.peek() == Some(&'"') => {
                cur.push('"');
                chars.next();
            }
            ('"', _) => quoted = !quoted,
            (',', false) => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out
}

fn csv_value(s: &str) -> Value {
    let t = s.trim();
    match t.parse::<f64>() {
        Ok(n) if !t.is_empty() => Value::Num(n),
        _ if t.is_empty() => Value::Nothing,
        _ => Value::Str(t.into()),
    }
}

/// read_csv(file): a CSV file as a list of maps, one per row, named by the
/// first line. Numbers become numbers and empty fields become nothing.
pub fn b_read_csv(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "read_csv")?;
    let file = text(&a[0], "read_csv")?;
    let bytes = match crate::builtins::embedded_file(&file) {
        Some(Some(b)) => b.to_vec(),
        Some(None) => return Err(format!("read_csv: no file was embedded as \"{}\"", file)),
        None => std::fs::read(&*file).map_err(|e| format!("read_csv couldn't read \"{}\": {}", file, e))?,
    };
    let content = String::from_utf8_lossy(&bytes);
    let content = content.trim_start_matches('\u{feff}');
    let mut lines = content.lines().filter(|l| !l.trim().is_empty());
    let Some(head) = lines.next() else { return Ok(Value::List(Rc::new(vec![]))) };
    let names: Vec<Rc<str>> = csv_fields(head).iter().map(|n| Rc::from(n.trim())).collect();
    let mut rows = vec![];
    for (k, line) in lines.enumerate() {
        let fields = csv_fields(line);
        if fields.len() != names.len() {
            return Err(format!("read_csv: line {} of \"{}\" has {} fields, but the first line names {}", k + 2, file, fields.len(), names.len()));
        }
        rows.push(Value::Map(Rc::new(names.iter().cloned().zip(fields.iter().map(|f| csv_value(f))).collect())));
    }
    Ok(Value::List(Rc::new(rows)))
}

/// columns(rows, names): numbers from a list of maps as a table, one row per
/// map and one column per name. With one name (not a list), a single row of numbers.
pub fn b_columns(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "columns")?;
    let Value::List(rows) = &a[0] else {
        return Err(format!("columns needs a list of maps, like the result of read_csv, not a {}", a[0].kind()));
    };
    let (names, single): (Vec<Rc<str>>, bool) = match &a[1] {
        Value::Str(s) => (vec![s.clone()], true),
        Value::List(l) => (l.iter().map(|v| text(v, "columns")).collect::<Result<_, _>>()?, false),
        other => return Err(format!("columns needs a name or a list of names, not a {}", other.kind())),
    };
    let mut data = Vec::with_capacity(rows.len() * names.len());
    for (i, r) in rows.iter().enumerate() {
        let Value::Map(m) = r else {
            return Err(format!("columns: item {} is a {}, not a map", i, r.kind()));
        };
        for n in &names {
            let v = m.iter().find(|(k, _)| k == n).map(|(_, v)| v).ok_or_else(|| format!("columns: row {} has no \"{}\"", i, n))?;
            data.push(v.scalar().ok_or_else(|| format!("columns: \"{}\" in row {} is {}, not a number", n, i, v.show()))?);
        }
    }
    let shape = if single { vec![rows.len()] } else { vec![rows.len(), names.len()] };
    Ok(wrap(tensor::new(shape, data)))
}

/// shuffle(x): the items of a list, or the rows of a tensor, in random order.
pub fn b_shuffle(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "shuffle")?;
    let n = match &a[0] {
        Value::List(l) => l.len(),
        other => match other.as_tensor() {
            Some(t) if !t.shape.is_empty() => t.shape[0],
            _ => return Err(format!("shuffle needs a list or a tensor, not a {}", other.kind())),
        },
    };
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (uniform() * (i + 1) as f64) as usize;
        order.swap(i, j.min(i));
    }
    match &a[0] {
        Value::List(l) => Ok(Value::List(Rc::new(order.iter().map(|&i| l[i].clone()).collect()))),
        other => {
            let t = other.as_tensor().unwrap();
            let mut sel = vec![order];
            for &d in &t.shape[1..] {
                sel.push((0..d).collect());
            }
            Ok(wrap(tensor::gather(&t, t.shape.clone(), &sel)))
        }
    }
}

/// batches(x, size): a list of pieces of x, each `size` items or rows long
/// (the last may be shorter).
pub fn b_batches(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "batches")?;
    let size = whole(&a[1], "batches", "a batch size")?;
    if size == 0 {
        return Err("batches needs a batch size of 1 or more".into());
    }
    let out: Vec<Value> = match &a[0] {
        Value::List(l) => l.chunks(size).map(|c| Value::List(Rc::new(c.to_vec()))).collect(),
        other => {
            let t = other.as_tensor().filter(|t| !t.shape.is_empty()).ok_or_else(|| format!("batches needs a list or a tensor, not a {}", other.kind()))?;
            let n = t.shape[0];
            let mut out = vec![];
            let mut start = 0;
            while start < n {
                let end = (start + size).min(n);
                let mut sel = vec![(start..end).collect::<Vec<_>>()];
                for &d in &t.shape[1..] {
                    sel.push((0..d).collect());
                }
                let mut shape = t.shape.clone();
                shape[0] = end - start;
                out.push(wrap(tensor::gather(&t, shape, &sel)));
                start = end;
            }
            out
        }
    };
    Ok(Value::List(Rc::new(out)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    /// The gradient of `f` at `x`, worked out by nudging each input a little.
    fn numeric_grad(x: &[f64], shape: &[usize], f: &dyn Fn(&T) -> T) -> Vec<f64> {
        (0..x.len())
            .map(|i| {
                let h = 1e-6;
                let mut up = x.to_vec();
                up[i] += h;
                let mut dn = x.to_vec();
                dn[i] -= h;
                let fu = f(&tensor::new(shape.to_vec(), up)).vals()[0];
                let fd = f(&tensor::new(shape.to_vec(), dn)).vals()[0];
                (fu - fd) / (2.0 * h)
            })
            .collect()
    }

    fn check_grad(x: Vec<f64>, shape: Vec<usize>, f: &dyn Fn(&T) -> T) {
        let p = tensor::param(shape.clone(), x.clone());
        let g = tensor::grad_of(&f(&p), &p);
        let n = numeric_grad(&x, &shape, f);
        for (a, b) in g.iter().zip(&n) {
            assert!((a - b).abs() < 1e-4, "autodiff {:?} vs numeric {:?}", g, n);
        }
    }

    #[test]
    fn activation_gradients() {
        for u in [UFn::Gelu, UFn::Silu, UFn::Softplus, UFn::LeakyRelu] {
            check_grad(vec![-2.0, -0.3, 0.4, 1.7], vec![4], &|t| tensor::sum(&tensor::unary(u, t)));
        }
        assert!(close(tensor::apply(UFn::Gelu, 0.0), 0.0));
        assert!(close(tensor::apply(UFn::Softplus, 0.0), 2f64.ln()));
        assert!(close(tensor::apply(UFn::Softplus, 100.0), 100.0));
    }

    #[test]
    fn log_softmax_and_cross_entropy() {
        let x = tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 1000.0, 0.0, -1000.0]);
        let l = tensor::log_softmax(&x);
        let ld = l.vals();
        assert!(ld.iter().all(|v| v.is_finite()));
        let row0: f64 = ld[0..3].iter().map(|v| v.exp()).sum();
        assert!(close(row0, 1.0));
        drop(ld);
        check_grad(vec![0.2, -1.0, 0.5, 2.0, 0.1, -0.4], vec![2, 3], &|t| {
            let w = tensor::new(vec![2, 3], vec![1.0, -2.0, 0.5, 0.3, 0.7, -1.1]);
            tensor::sum(&tensor::binary(BinK::Mul, &tensor::log_softmax(t), &w).unwrap())
        });
        // uniform scores over 4 classes: loss is ln 4
        let ce = tensor::cross_entropy(&tensor::new(vec![1, 4], vec![0.0; 4]), vec![0.0, 0.0, 1.0, 0.0]).unwrap();
        assert!(close(ce.vals()[0], 4f64.ln()));
        check_grad(vec![0.2, -1.0, 0.5, 2.0, 0.1, -0.4], vec![2, 3], &|t| {
            tensor::cross_entropy(t, vec![0.0, 1.0, 0.0, 0.25, 0.25, 0.5]).unwrap()
        });
    }

    #[test]
    fn concat_forward_and_back() {
        let a = tensor::new(vec![2, 1], vec![1.0, 2.0]);
        let b = tensor::new(vec![2, 2], vec![3.0, 4.0, 5.0, 6.0]);
        let c = tensor::concat(&[a.clone(), b.clone()], 1).unwrap();
        assert_eq!(c.shape, vec![2, 3]);
        assert_eq!(*c.vals(), vec![1.0, 3.0, 4.0, 2.0, 5.0, 6.0]);
        let r = tensor::concat(&[b.clone(), b.clone()], 0).unwrap();
        assert_eq!(r.shape, vec![4, 2]);
        assert!(tensor::concat(&[a, tensor::new(vec![3, 1], vec![0.0; 3])], 1).is_err());
        check_grad(vec![0.5, -1.0, 2.0, 0.3], vec![2, 2], &|t| {
            let other = tensor::new(vec![2, 1], vec![7.0, 8.0]);
            let j = tensor::concat(&[t.clone(), other], 1).unwrap();
            let w = tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, -1.0, 0.5, 2.0]);
            tensor::sum(&tensor::binary(BinK::Mul, &j, &w).unwrap())
        });
    }

    #[test]
    fn optimisers_and_clipping_lower_the_loss() {
        for optim in [tensor::Optim::Sgd, tensor::Optim::Adam, tensor::Optim::AdamW, tensor::Optim::Momentum] {
            let w = tensor::param(vec![2], vec![3.0, -2.0]);
            let loss = |w: &T| tensor::sum(&tensor::binary(BinK::Mul, w, w).unwrap());
            let start = loss(&w).vals()[0];
            for _ in 0..50 {
                tensor::descend(&loss(&w), 0.05, optim, Some(1.0)).unwrap();
            }
            let end = loss(&w).vals()[0];
            assert!(end < start * 0.5, "{:?}: {} -> {}", optim, start, end);
        }
        // clipping limits the step: a gradient of size 10 is scaled to 1
        let w = tensor::param(vec![1], vec![5.0]);
        tensor::descend(&tensor::binary(BinK::Mul, &w, &tensor::scalar(10.0)).unwrap(), 1.0, tensor::Optim::Sgd, Some(1.0)).unwrap();
        assert!(close(w.vals()[0], 4.0));
    }

    #[test]
    fn csv_parsing() {
        assert_eq!(csv_fields(r#"a,"b, c","say ""hi""",,"#), vec!["a", "b, c", "say \"hi\"", "", ""]);
        assert!(matches!(csv_value(" 3.5 "), Value::Num(x) if x == 3.5));
        assert!(matches!(csv_value(""), Value::Nothing));
        assert!(matches!(csv_value("Paris"), Value::Str(_)));
    }
}
