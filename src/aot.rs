//! `antares build`: compiles a program to native code.
//!
//! The compiler works out, for the whole program at once, which variables
//! only ever hold a plain number, a yes/no, or a plain row of numbers. Those
//! become machine numbers and direct memory. Everything else stays a general
//! value and uses the same runtime as the interpreter (`rt.rs`), so both ways
//! of running a program behave the same.
//!
//! The output is Rust source, which `cargo` turns into the final program.

use crate::ast::*;
use crate::builtins;
use crate::lexer;
use crate::parser::Parser;
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::rc::Rc;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    /// Nothing known yet (only during inference).
    Bot,
    Num,
    Bool,
    /// A plain one-dimensional tensor that is not being learned.
    Arr,
    /// Any value.
    Dyn,
    /// No value at all (only as what a function gives back).
    Unit,
    /// A whole-number constant: becomes an integer next to integers, and an
    /// ordinary number otherwise (only during inference).
    ILit,
    /// A 64-bit integer.
    Int,
    /// A row of 64-bit integers.
    IArr,
    /// A compact row (16-bit numbers, from `compact` or `update_rows`).
    PkR,
    /// A compact matrix (rows of 16-bit numbers, from `compact(x, rows, cols)`).
    PkM,
    /// An ordinary number that only ever holds whole numbers counted in small
    /// steps (loop counters and the like), kept as a machine integer. It
    /// behaves exactly like `Num`: it is turned into one wherever it is used,
    /// except as a position in a row, in comparisons and with `%`.
    Whole,
}

fn join(a: Ty, b: Ty) -> Ty {
    use Ty::*;
    match (a, b) {
        (Bot, x) | (x, Bot) => x,
        (x, y) if x == y => x,
        (ILit, Num) | (Num, ILit) => Num,
        (ILit, Int) | (Int, ILit) => Int,
        _ => Dyn,
    }
}

fn rust_ty(t: Ty) -> &'static str {
    match t {
        Ty::Unit => "()",
        Ty::Int | Ty::Whole => "i64",
        Ty::IArr => "IA",
        Ty::Num | Ty::ILit => "f64",
        Ty::Bool => "bool",
        Ty::Arr => "T",
        Ty::PkR | Ty::PkM => "PK",
        _ => "Value",
    }
}

fn zero_of(t: Ty) -> &'static str {
    match t {
        Ty::Num | Ty::ILit => "0.0f64",
        Ty::Int | Ty::Whole => "0i64",
        Ty::IArr => "iarr0()",
        Ty::Bool => "false",
        Ty::Arr => "arr0()",
        Ty::PkR | Ty::PkM => "pk0()",
        _ => "Value::Unset",
    }
}

/// Converts generated code of one type into another.
fn conv(code: String, from: Ty, to: Ty) -> String {
    if from == to {
        return code;
    }
    match (from, to) {
        (Ty::Int, Ty::Num) => return format!("(({}) as f64)", code),
        (Ty::Num, Ty::Int) => return format!("n2i({}, 0)?", code),
        (Ty::Whole, Ty::Num) => return format!("(({}) as f64)", code),
        (Ty::Whole, Ty::Int) => return code,
        (Ty::Num, Ty::Whole) => return format!("(({}) as i64)", code),
        _ => {}
    }
    let as_dyn = match from {
        Ty::Whole => format!("Value::Num(({}) as f64)", code),
        Ty::Int => format!("Value::Int({})", code),
        Ty::IArr => format!("Value::Ints({})", code),
        Ty::Unit => format!("{{ let _ = {}; Value::Nothing }}", code),
        Ty::Num => format!("Value::Num({})", code),
        Ty::Bool => format!("Value::Bool({})", code),
        Ty::Arr => format!("Value::Tensor({})", code),
        Ty::PkR | Ty::PkM => format!("Value::Packed({})", code),
        _ => code,
    };
    match to {
        Ty::Num => format!("as_num({}, 0)?", as_dyn),
        Ty::Bool => format!("as_bool({}, 0)?", as_dyn),
        Ty::Arr => format!("as_arr({}, 0)?", as_dyn),
        Ty::Int => format!("as_int({}, 0)?", as_dyn),
        Ty::IArr => format!("as_iarr({}, 0)?", as_dyn),
        Ty::Whole => format!("(as_num({}, 0)? as i64)", as_dyn),
        Ty::PkR | Ty::PkM => format!("as_pk({}, 0)?", as_dyn),
        _ => as_dyn,
    }
}

fn binop_name(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "Add",
        BinOp::Sub => "Sub",
        BinOp::Mul => "Mul",
        BinOp::Div => "Div",
        BinOp::Mod => "Mod",
        BinOp::Pow => "Pow",
        BinOp::MatMul => "MatMul",
        BinOp::Eq => "Eq",
        BinOp::Ne => "Ne",
        BinOp::Lt => "Lt",
        BinOp::Gt => "Gt",
        BinOp::Le => "Le",
        BinOp::Ge => "Ge",
        BinOp::BitAnd => "BitAnd",
        BinOp::BitOr => "BitOr",
        BinOp::BitXor => "BitXor",
        BinOp::Shl => "Shl",
        BinOp::Shr => "Shr",
    }
}

/// Whether working out `e` may call a function (and so change variables).
fn calls_anything(e: &Expr) -> bool {
    match e {
        Expr::Call(..) | Expr::Ask { .. } | Expr::ModelNew { .. } | Expr::Lambda(_) => true,
        Expr::Interp(p) | Expr::List(p) => p.iter().any(calls_anything),
        Expr::Map(m) => m.iter().any(|(_, x)| calls_anything(x)),
        Expr::Neg(a, _) | Expr::Not(a, _) | Expr::Field(a, _, _) | Expr::BitNot(a, _) => calls_anything(a),
        Expr::Binary(_, a, b, _) | Expr::And(a, b, _) | Expr::Or(a, b, _) => calls_anything(a) || calls_anything(b),
        Expr::Index(b, idx, _) => calls_anything(b) || idx.iter().any(calls_anything),
        Expr::Range(a, z, _) => a.as_deref().is_some_and(calls_anything) || z.as_deref().is_some_and(calls_anything),
        _ => false,
    }
}

/// Built-ins that call functions given to them.
fn calls_back(name: &str) -> bool {
    matches!(name, "map" | "filter" | "reduce" | "sort_by")
}

fn is_bit_op(op: BinOp) -> bool {
    matches!(op, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr)
}

fn bin_ty(op: BinOp, a: Ty, b: Ty) -> Ty {
    use Ty::*;
    if a == Dyn || b == Dyn || a == Unit || b == Unit {
        return Dyn;
    }
    if a == Bot || b == Bot {
        return Bot;
    }
    let number = |t: Ty| matches!(t, Num | ILit | Int);
    let plain = |t: Ty| matches!(t, Num | ILit);
    let both_numbers = number(a) && number(b);
    let any_int = a == Int || b == Int;
    let arr_mix = (a == Arr && (plain(b) || b == Arr)) || (b == Arr && plain(a));
    match op {
        BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
            if both_numbers {
                Int
            } else {
                Dyn
            }
        }
        BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
            if both_numbers {
                Bool
            } else {
                Dyn
            }
        }
        BinOp::Eq | BinOp::Ne => {
            if both_numbers || (a == Bool && b == Bool) {
                Bool
            } else {
                Dyn
            }
        }
        BinOp::MatMul => Dyn,
        BinOp::Div => {
            if both_numbers {
                Num
            } else if arr_mix {
                Arr
            } else {
                Dyn
            }
        }
        BinOp::Pow => {
            if plain(a) && plain(b) {
                Num
            } else if arr_mix {
                Arr
            } else {
                Dyn
            }
        }
        _ => {
            if both_numbers {
                if any_int {
                    Int
                } else if a == ILit && b == ILit {
                    ILit
                } else {
                    Num
                }
            } else if arr_mix {
                Arr
            } else {
                Dyn
            }
        }
    }
}

/// What a built-in gives back, when that is certain from its inputs.
fn builtin_ret(name: &str, tys: &[Ty]) -> Ty {
    use Ty::*;
    let tys: Vec<Ty> = tys.iter().map(|t| if *t == ILit { Num } else { *t }).collect();
    let tys = &tys[..];
    match name {
        // within gives back what it was given
        "within" => match tys.first() {
            Some(t @ (Int | Num | IArr | Arr | PkR | PkM | Bot)) => *t,
            _ => Dyn,
        },
        "int" | "popcount" | "lowest_bit" | "highest_bit" | "div" | "rand_bits" => Int,
        // always an ordinary number
        "code" | "find" => Num,
        "ints" => IArr,
        "abs" | "floor" | "trunc" | "round" | "sqrt" | "exp" | "log" | "sin" | "cos" | "tanh" | "sigmoid" | "relu" => {
            match tys {
                [Num] => Num,
                [Arr] => Arr,
                [Bot] => Bot,
                _ => Dyn,
            }
        }
        "min" | "max" | "len" | "clock" | "number" | "sum_product" | "screlu_dot" => Num,
        "zeros" | "ones" | "rand" | "randn" => match tys {
            [] => Num,
            [Num] => Arr,
            [Bot] => Bot,
            _ => Dyn,
        },
        "range" | "read_binary" | "shape" => Arr,
        "argmax" if tys.len() == 3 => match tys[0] {
            Arr => Num,
            Bot => Bot,
            _ => Dyn,
        },
        // compact(x, rows, cols) is a matrix; compact(row) a row
        "compact" => match tys {
            [_, _, _] => PkM,
            [Arr] => PkR,
            [Bot] => Bot,
            _ => Dyn,
        },
        "update_rows" => match tys {
            [PkR, PkM, ..] if tys.len() == 6 || tys.len() == 7 => PkR,
            [Bot, ..] | [_, Bot, ..] => Bot,
            _ => Dyn,
        },
        "has" | "exists" | "input_waiting" => Bool,
        _ => Dyn,
    }
}

struct Scope {
    def: *const FnDef,
    parent: usize,
    vars: Vec<Ty>,
    captured: Vec<bool>,
    nparams: usize,
    ret: Ty,
    /// Used as a value (passed around, stored), not only called by name.
    escapes: bool,
    /// Variables of enclosing functions this one (or one inside it) uses.
    free: BTreeSet<(usize, usize)>,
    name: String,
}

enum Callee {
    Builtin(usize),
    Static(usize),
    Dynamic,
}

struct Cx<'a> {
    scopes: Vec<Scope>,
    by_def: HashMap<*const FnDef, usize>,
    /// Global slots that hold one function for the whole program.
    static_fn: HashMap<usize, usize>,
    gcount: Vec<u32>,
    gdef: Vec<Option<*const FnDef>>,
    /// The literal number a global was given at the top level, if any.
    glit: Vec<Option<f64>>,
    /// Globals that are that one number for the whole program.
    consts: HashMap<usize, f64>,
    /// The same for integers: a global given one 0x literal or int(literal).
    glit_i: Vec<Option<i64>>,
    iconsts: HashMap<usize, i64>,
    nb: usize,
    bnames: Vec<&'static str>,
    gnames: &'a [String],
    /// Set once inference is done: constants are then plain numbers.
    codegen: bool,
    pass0: bool,
    changed: bool,
    // code generation
    strs: Vec<String>,
    /// Lists made only of constant numbers; each is built once and shared.
    arrs: Vec<Vec<f64>>,
    items: String,
    tmp: usize,
    loops: Vec<(usize, usize)>,
    try_depth: usize,
    ret: Option<Ty>,
    /// Functions called by name that can call themselves again (directly, through
    /// other functions, or through functions passed around as values): only
    /// these check for endless recursion.
    rec: Vec<bool>,
    /// Functions called by name that can't fail: they return a plain value.
    inf: Vec<bool>,
    /// Plain-value functions that may return with endless recursion recorded
    /// (see `rt::fault`); callers that can report problems check for it.
    prop: Vec<bool>,
    /// The function called by name whose body is being generated, if any.
    cur: Option<usize>,
    /// Whether the body being generated returns a plain value.
    plain: bool,
    /// Rows read and written through a direct pointer in the loop body being
    /// generated: (scope, variable) and the pointer's name.
    fast: Vec<((usize, usize), String)>,
    /// Top-level rows kept as fixed-size tables: variable -> (length, integers?).
    tables: HashMap<usize, (usize, bool)>,
    /// The numbers a table made from a constant list starts with.
    table_init: HashMap<usize, Vec<i64>>,
    /// What integer variables, results and tables can hold (see `find_ranges`).
    ranges: HashMap<(usize, usize), Rg>,
    /// `lowest_bit(x)` / `highest_bit(x)` calls where `x` is certainly not 0.
    nonzero_bits: std::collections::HashSet<*const Expr>,
}

/// What an innermost counting loop does with rows (see `Cx::fast_rows`).
struct FastScan {
    ok: bool,
    /// Variables used in some other way than `row[counter + constant]`.
    bad: BTreeSet<usize>,
    /// Row variable: lowest and highest constant added to the counter, and
    /// whether the loop writes to it.
    rows: std::collections::BTreeMap<usize, (i64, i64, bool)>,
    /// Fixed tables used as `table[counter + constant]`: lowest and highest constant.
    tables: std::collections::BTreeMap<usize, (i64, i64)>,
    /// Variables of the loop's own scope that the body gives a value.
    assigned: BTreeSet<usize>,
}

/// A row or fixed table a counting loop reaches only at counter + constant.
#[derive(Clone, Copy)]
struct FastRow {
    var: usize,
    lo: i64,
    hi: i64,
    write: bool,
    table: bool,
}

impl FastScan {
    fn note_table(&mut self, r: usize, off: i64) {
        let e = self.tables.entry(r).or_insert((off, off));
        e.0 = e.0.min(off);
        e.1 = e.1.max(off);
    }

    fn note(&mut self, r: usize, off: i64, write: bool) {
        let e = self.rows.entry(r).or_insert((off, off, false));
        e.0 = e.0.min(off);
        e.1 = e.1.max(off);
        e.2 |= write;
    }
}

/// Where a variable gets a value from (for finding whole-number variables).
enum Src {
    /// An assignment.
    Expr(*const Expr, usize),
    /// An input given to a function called by name.
    Arg(*const Expr, usize),
    /// The variable of an `each` loop over this.
    Each(*const Expr, usize),
    /// The counter of `repeat count as i`.
    Repeat(*const Expr, usize),
    /// Anything else.
    Other,
}

/// Source keys that aren't variables: (function scope, RET) is what the
/// function gives back, and (TABLE, global) the numbers stored in a table.
const RET: usize = usize::MAX;
const TABLE: usize = usize::MAX;

/// The whole numbers an integer expression can have: none known yet (only
/// while working them out), all from `lo` to `hi`, or any.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Rg {
    Bot,
    In(i64, i64),
    Any,
}

impl Rg {
    fn join(self, o: Rg) -> Rg {
        match (self, o) {
            (Rg::Bot, x) | (x, Rg::Bot) => x,
            (Rg::In(a, b), Rg::In(c, d)) => Rg::In(a.min(c), b.max(d)),
            _ => Rg::Any,
        }
    }

    /// Both known: the range from `f` on their bounds, or any if that overflows.
    fn with(self, o: Rg, f: impl Fn(i64, i64, i64, i64) -> Option<(i64, i64)>) -> Rg {
        match (self, o) {
            (Rg::Any, _) | (_, Rg::Any) => Rg::Any,
            (Rg::Bot, _) | (_, Rg::Bot) => Rg::Bot,
            (Rg::In(a, b), Rg::In(c, d)) => f(a, b, c, d).map_or(Rg::Any, |(lo, hi)| Rg::In(lo, hi)),
        }
    }

    /// The highest number, if none are negative.
    fn nonneg(self) -> Option<i64> {
        match self {
            Rg::In(lo, hi) if lo >= 0 => Some(hi),
            _ => None,
        }
    }
}

/// What a function body does, for deciding how to call it.
#[derive(Default)]
struct Scan {
    /// Functions called by name.
    calls: Vec<usize>,
    /// Calls something that may call functions passed around as values.
    unknown: bool,
    /// Has a loop, shows something or changes a variable outside itself.
    effects: bool,
}

impl<'a> Cx<'a> {
    fn resolve(&self, mut sc: usize, depth: u32) -> usize {
        for _ in 0..depth {
            sc = self.scopes[sc].parent;
        }
        sc
    }

    fn scope_for(&mut self, def: &Rc<FnDef>, parent: usize) -> usize {
        let p = Rc::as_ptr(def);
        if let Some(&s) = self.by_def.get(&p) {
            return s;
        }
        self.scopes.push(Scope {
            def: p,
            parent,
            vars: vec![Ty::Bot; def.nslots],
            captured: vec![false; def.nslots],
            nparams: def.nparams,
            ret: Ty::Bot,
            escapes: false,
            free: BTreeSet::new(),
            name: def.name.to_string(),
        });
        self.by_def.insert(p, self.scopes.len() - 1);
        self.scopes.len() - 1
    }

    fn is_builtin(&self, idx: usize) -> bool {
        idx < self.nb && self.gcount[idx] == 0
    }

    fn callee(&self, f: &Expr, sc: usize) -> Callee {
        if let Expr::Var { depth, idx, .. } = f {
            if self.resolve(sc, *depth) == 0 {
                let i = *idx as usize;
                if self.is_builtin(i) {
                    return Callee::Builtin(i);
                }
                if let Some(&s) = self.static_fn.get(&i) {
                    return Callee::Static(s);
                }
            }
        }
        Callee::Dynamic
    }

    /// Arithmetic on numbers and top-level constants, worked out when compiling
    /// exactly as the program would (ordinary decimal arithmetic).
    fn const_expr(&self, e: &Expr) -> Option<f64> {
        match e {
            Expr::Num(n) => Some(*n),
            Expr::Neg(a, _) => Some(-self.const_expr(a)?),
            Expr::Var { depth: 0, idx, .. } => self.consts.get(&(*idx as usize)).copied(),
            Expr::Binary(op @ (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div), a, b, _) => {
                let (x, y) = (self.const_expr(a)?, self.const_expr(b)?);
                Some(match op {
                    BinOp::Add => x + y,
                    BinOp::Sub => x - y,
                    BinOp::Mul => x * y,
                    _ => x / y,
                })
            }
            _ => None,
        }
    }

    /// The value of an expression that is a known number when compiling.
    fn const_of(&self, e: &Expr, sc: usize) -> Option<f64> {
        match e {
            Expr::Num(n) => Some(*n),
            Expr::Neg(a, _) => self.const_of(a, sc).map(|x| -x),
            Expr::Var { depth, idx, .. } if self.resolve(sc, *depth) == 0 => self.consts.get(&(*idx as usize)).copied(),
            _ => None,
        }
    }

    /// Like `const_of`, also working out arithmetic on constants
    /// (`SIZE * SIZE - 1`), as the program would.
    fn fold_const(&self, e: &Expr, sc: usize) -> Option<f64> {
        if let Some(c) = self.const_of(e, sc) {
            return Some(c);
        }
        match e {
            Expr::Int(n) => Some(*n as f64),
            Expr::Var { depth, idx, .. } if self.resolve(sc, *depth) == 0 => self.iconsts.get(&(*idx as usize)).map(|&c| c as f64),
            Expr::Call(f, args, _) if args.len() == 1 => match self.callee(f, sc) {
                // int(constant)
                Callee::Builtin(b) if self.bnames[b] == "int" => self.fold_const(&args[0], sc).filter(|c| *c == c.trunc()),
                _ => None,
            },
            Expr::Neg(a, _) => self.fold_const(a, sc).map(|x| -x),
            Expr::Binary(op @ (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div), a, b, _) => {
                let (x, y) = (self.fold_const(a, sc)?, self.fold_const(b, sc)?);
                Some(match op {
                    BinOp::Add => x + y,
                    BinOp::Sub => x - y,
                    BinOp::Mul => x * y,
                    _ => x / y,
                })
            }
            _ => None,
        }
    }

    fn var_ty(&self, t: usize, idx: usize) -> Ty {
        if self.scopes[t].captured[idx] {
            Ty::Dyn
        } else {
            self.scopes[t].vars[idx]
        }
    }

    // ---------- types of expressions ----------

    fn ty(&self, e: &Expr, sc: usize) -> Ty {
        use Ty::*;
        match e {
            Expr::Num(n) => {
                if !self.codegen && *n == n.trunc() && n.abs() < 9.0e15 {
                    ILit
                } else {
                    Num
                }
            }
            Expr::Int(_) => Int,
            Expr::BitNot(a, _) => match self.ty(a, sc) {
                Num | ILit | Int => Int,
                Bot => Bot,
                _ => Dyn,
            },
            Expr::Bool(_) | Expr::Not(..) | Expr::And(..) | Expr::Or(..) => Bool,
            Expr::Var { depth, idx, .. } => {
                let t = self.resolve(sc, *depth);
                let i = *idx as usize;
                if t == 0 {
                    if let Some(c) = self.consts.get(&i) {
                        return if !self.codegen && *c == c.trunc() && c.abs() < 9.0e15 { ILit } else { Num };
                    }
                    if self.iconsts.contains_key(&i) {
                        return Int;
                    }
                }
                if t == 0 && (self.is_builtin(i) || self.static_fn.contains_key(&i)) {
                    return Dyn;
                }
                match self.var_ty(t, i) {
                    Whole => Num,
                    other => other,
                }
            }
            Expr::List(items) => {
                if items.is_empty() {
                    return Dyn;
                }
                let mut bot = false;
                for it in items {
                    match self.ty(it, sc) {
                        Num | ILit => {}
                        Bot => bot = true,
                        _ => return Dyn,
                    }
                }
                if bot {
                    Bot
                } else {
                    Arr
                }
            }
            Expr::Neg(a, _) => match self.ty(a, sc) {
                Num => Num,
                ILit => ILit,
                Int => Int,
                Arr => Arr,
                Bot => Bot,
                _ => Dyn,
            },
            Expr::Binary(op, a, b, _) => bin_ty(*op, self.ty(a, sc), self.ty(b, sc)),
            Expr::Call(f, args, _) => match self.callee(f, sc) {
                Callee::Builtin(i) => {
                    let tys: Vec<Ty> = args.iter().map(|a| self.ty(a, sc)).collect();
                    builtin_ret(self.bnames[i], &tys)
                }
                Callee::Static(s) => {
                    if args.len() == self.scopes[s].nparams && self.scopes[s].ret != Unit {
                        self.scopes[s].ret
                    } else {
                        Dyn
                    }
                }
                Callee::Dynamic => Dyn,
            },
            Expr::Index(base, idx, _) => {
                if idx.len() == 1 && !matches!(idx[0], Expr::Range(..)) {
                    match (self.ty(base, sc), self.ty(&idx[0], sc)) {
                        (Arr, Num | ILit | Int) => Num,
                        (IArr, Num | ILit | Int) => Int,
                        (PkM, Num | ILit | Int) => PkR,
                        (PkR, Num | ILit | Int) => Num,
                        (PkM | PkR, Bot) => Bot,
                        (Arr | IArr | Bot, Num | ILit | Int | Bot) => Bot,
                        _ => Dyn,
                    }
                } else {
                    Dyn
                }
            }
            _ => Dyn,
        }
    }

    /// The type of `e` as what a function gives back: like `ty`, except that
    /// `nothing` and a call to a function that gives back nothing are no value
    /// at all, so a function ending that way doesn't give back a general value.
    fn result_ty(&self, e: &Expr, sc: usize) -> Ty {
        match e {
            Expr::Nothing => Ty::Unit,
            Expr::Call(f, args, _) => match self.callee(f, sc) {
                Callee::Static(s) if args.len() == self.scopes[s].nparams && self.scopes[s].ret == Ty::Unit => Ty::Unit,
                _ => self.ty(e, sc),
            },
            _ => self.ty(e, sc),
        }
    }

    fn tail_ty(&self, stmts: &[Stmt], sc: usize) -> Ty {
        match stmts.last() {
            None => Ty::Unit,
            Some(Stmt::Expr(e)) => self.result_ty(e, sc),
            Some(Stmt::Give(_)) | Some(Stmt::Stop(_)) | Some(Stmt::Skip(_)) => Ty::Bot,
            Some(Stmt::When { arms, otherwise }) => {
                let mut t = match otherwise {
                    Some(b) => self.tail_ty(b, sc),
                    None => Ty::Unit,
                };
                for (_, body, _) in arms {
                    t = join(t, self.tail_ty(body, sc));
                }
                t
            }
            Some(Stmt::Try { body, handler, .. }) => join(self.tail_ty(body, sc), self.tail_ty(handler, sc)),
            Some(_) => Ty::Unit,
        }
    }

    // ---------- inference: walking the program and recording facts ----------

    fn join_var(&mut self, t: usize, idx: usize, ty: Ty) {
        let ty = if self.scopes[t].captured[idx] { Ty::Dyn } else { ty };
        let new = join(self.scopes[t].vars[idx], ty);
        if new == Ty::Dyn && !matches!(self.scopes[t].vars[idx], Ty::Dyn | Ty::Bot) && !self.pass0 && std::env::var("ANTARES_EXPLAIN").is_ok() {
            let name = if t == 0 { self.gnames.get(idx).cloned().unwrap_or_default() } else { format!("variable {}", idx) };
            eprintln!("explain: in {}, {} was {:?} and is now also given {:?}, so it becomes a general value", self.scopes[t].name, name, self.scopes[t].vars[idx], ty);
        }
        if new != self.scopes[t].vars[idx] {
            self.scopes[t].vars[idx] = new;
            self.changed = true;
        }
    }

    fn join_ret(&mut self, sc: usize, ty: Ty) {
        let new = join(self.scopes[sc].ret, ty);
        if new == Ty::Dyn && !matches!(self.scopes[sc].ret, Ty::Dyn | Ty::Bot) && !self.pass0 && std::env::var("ANTARES_EXPLAIN").is_ok() {
            eprintln!("explain: {} gives back {:?} in one place and {:?} in another, so its result becomes a general value", self.scopes[sc].name, self.scopes[sc].ret, ty);
        }
        if new != self.scopes[sc].ret {
            self.scopes[sc].ret = new;
            self.changed = true;
        }
    }

    fn set_escapes(&mut self, s: usize) {
        if !self.scopes[s].escapes {
            self.scopes[s].escapes = true;
            self.changed = true;
        }
        for i in 0..self.scopes[s].nparams {
            self.join_var(s, i, Ty::Dyn);
        }
    }

    fn count(&mut self, sc: usize, idx: usize) {
        if self.pass0 && sc == 0 {
            self.gcount[idx] += 1;
        }
    }

    /// A use of a variable from scope `sc`: notes captures and escaping functions.
    fn touch_var(&mut self, sc: usize, depth: u32, idx: usize, read: bool) {
        let t = self.resolve(sc, depth);
        if t == 0 {
            if read {
                if let Some(&s) = self.static_fn.get(&idx) {
                    self.set_escapes(s);
                }
            }
        } else if depth > 0 {
            if !self.scopes[t].captured[idx] {
                self.scopes[t].captured[idx] = true;
                self.scopes[t].vars[idx] = Ty::Dyn;
                self.changed = true;
            }
            let mut s = sc;
            while s != t {
                if self.scopes[s].free.insert((t, idx)) {
                    self.changed = true;
                }
                s = self.scopes[s].parent;
            }
        }
    }

    fn walk_fn(&mut self, s: usize) {
        let def: &FnDef = unsafe { &*self.scopes[s].def };
        if self.scopes[s].escapes {
            for i in 0..def.nparams {
                self.join_var(s, i, Ty::Dyn);
            }
        }
        self.walk_block(&def.body, s);
        let t = self.tail_ty(&def.body, s);
        self.join_ret(s, t);
    }

    fn walk_block(&mut self, stmts: &[Stmt], sc: usize) {
        for s in stmts {
            self.walk_stmt(s, sc);
        }
    }

    fn walk_stmt(&mut self, s: &Stmt, sc: usize) {
        match s {
            Stmt::Expr(e) => self.walk_expr(e, sc),
            Stmt::Assign { idx, value } => {
                let i = *idx as usize;
                if let (Expr::Lambda(def), 0) = (value, sc) {
                    let fs = self.scope_for(def, 0);
                    if self.pass0 {
                        self.gdef[i] = Some(Rc::as_ptr(def));
                    }
                    if self.static_fn.get(&i) == Some(&fs) {
                        self.walk_fn(fs);
                        return;
                    }
                }
                self.count(sc, i);
                if self.pass0 && sc == 0 {
                    self.glit_i[i] = match value {
                        Expr::Int(n) => Some(*n),
                        // int(8) and int(-8)
                        Expr::Call(f, args, _) if args.len() == 1 => match (&**f, &args[0]) {
                            (Expr::Var { name, .. }, Expr::Num(n)) if &**name == "int" && *n == n.trunc() => Some(*n as i64),
                            (Expr::Var { name, .. }, Expr::Neg(inner, _)) if &**name == "int" => match &**inner {
                                Expr::Num(n) if *n == n.trunc() => Some(-(*n as i64)),
                                _ => None,
                            },
                            _ => None,
                        },
                        _ => None,
                    };
                    self.glit[i] = match value {
                        Expr::Num(n) => Some(*n),
                        Expr::Neg(inner, _) => match &**inner {
                            Expr::Num(n) => Some(-*n),
                            _ => None,
                        },
                        _ => None,
                    };
                }
                self.walk_expr(value, sc);
                let t = self.ty(value, sc);
                self.join_var(sc, i, t);
            }
            Stmt::AssignOuter { depth, idx, value } => {
                self.walk_expr(value, sc);
                let t = self.resolve(sc, *depth);
                self.count(t, *idx as usize);
                self.touch_var(sc, *depth, *idx as usize, false);
                let ty = self.ty(value, sc);
                self.join_var(t, *idx as usize, ty);
            }
            Stmt::Learn { idx, value, .. } => {
                self.walk_expr(value, sc);
                self.count(sc, *idx as usize);
                self.join_var(sc, *idx as usize, Ty::Dyn);
            }
            Stmt::SetPath { depth, idx, path, value, .. } => {
                self.walk_expr(value, sc);
                for a in path {
                    if let Access::Index(es) = a {
                        for e in es {
                            self.walk_expr(e, sc);
                        }
                    }
                }
                let t = self.resolve(sc, *depth);
                self.touch_var(sc, *depth, *idx as usize, false);
                let ok = match self.fast_set(path) {
                    Some(ie) => {
                        use Ty::*;
                        let vt = self.var_ty(t, *idx as usize);
                        let it = self.ty(ie, sc);
                        let val = self.ty(value, sc);
                        matches!(it, Num | ILit | Int | Bot)
                            && ((matches!(vt, Arr | IArr | Bot) && matches!(val, Num | ILit | Int | Bot))
                                || (vt == PkM && matches!(val, PkR | Bot))
                                || (vt == PkR && matches!(val, Num | ILit | Int | Bot)))
                    }
                    None => false,
                };
                if !ok {
                    self.join_var(t, *idx as usize, Ty::Dyn);
                }
            }
            Stmt::Show(items) => {
                for e in items {
                    self.walk_expr(e, sc);
                }
            }
            Stmt::Give(e) => {
                let t = match e {
                    Some(e) => {
                        self.walk_expr(e, sc);
                        self.result_ty(e, sc)
                    }
                    None => Ty::Unit,
                };
                if sc != 0 {
                    self.join_ret(sc, t);
                }
            }
            Stmt::Repeat { count, var, body, .. } => {
                self.walk_expr(count, sc);
                if let Some(v) = var {
                    self.count(sc, *v as usize);
                    self.join_var(sc, *v as usize, Ty::Num);
                }
                self.walk_block(body, sc);
            }
            Stmt::Each { var, iter, body, .. } => {
                self.walk_expr(iter, sc);
                self.count(sc, *var as usize);
                match self.ty(iter, sc) {
                    Ty::Arr => self.join_var(sc, *var as usize, Ty::Num),
                    Ty::IArr => self.join_var(sc, *var as usize, Ty::Int),
                    Ty::Bot => {}
                    _ => self.join_var(sc, *var as usize, Ty::Dyn),
                }
                self.walk_block(body, sc);
            }
            Stmt::While { cond, body, .. } => {
                self.walk_expr(cond, sc);
                self.walk_block(body, sc);
            }
            Stmt::When { arms, otherwise } => {
                for (c, b, _) in arms {
                    self.walk_expr(c, sc);
                    self.walk_block(b, sc);
                }
                if let Some(b) = otherwise {
                    self.walk_block(b, sc);
                }
            }
            Stmt::Descend { loss, rate, clip, .. } => {
                self.walk_expr(loss, sc);
                self.walk_expr(rate, sc);
                if let Some(c) = clip {
                    self.walk_expr(c, sc);
                }
            }
            Stmt::Try { body, var, handler } => {
                self.walk_block(body, sc);
                if let Some(v) = var {
                    self.count(sc, *v as usize);
                    self.join_var(sc, *v as usize, Ty::Dyn);
                }
                self.walk_block(handler, sc);
            }
            Stmt::Stop(_) | Stmt::Skip(_) => {}
        }
    }

    /// `name[i] = v` with one plain index: the index expression.
    fn fast_set<'e>(&self, path: &'e [Access]) -> Option<&'e Expr> {
        match path {
            [Access::Index(es)] if es.len() == 1 && !matches!(es[0], Expr::Range(..)) => Some(&es[0]),
            _ => None,
        }
    }

    fn walk_expr(&mut self, e: &Expr, sc: usize) {
        match e {
            Expr::Var { depth, idx, .. } => self.touch_var(sc, *depth, *idx as usize, true),
            Expr::Interp(p) | Expr::List(p) => {
                for x in p {
                    self.walk_expr(x, sc);
                }
            }
            Expr::Map(m) => {
                for (_, x) in m {
                    self.walk_expr(x, sc);
                }
            }
            Expr::Neg(a, _) | Expr::Not(a, _) | Expr::Field(a, _, _) | Expr::BitNot(a, _) => self.walk_expr(a, sc),
            Expr::Binary(_, a, b, _) | Expr::And(a, b, _) | Expr::Or(a, b, _) => {
                self.walk_expr(a, sc);
                self.walk_expr(b, sc);
            }
            Expr::Call(f, args, _) => {
                for a in args {
                    self.walk_expr(a, sc);
                }
                match self.callee(f, sc) {
                    Callee::Static(s) => {
                        if args.len() == self.scopes[s].nparams && !self.scopes[s].escapes {
                            for (i, a) in args.iter().enumerate() {
                                let t = self.ty(a, sc);
                                self.join_var(s, i, t);
                            }
                        }
                    }
                    Callee::Builtin(_) => {}
                    Callee::Dynamic => self.walk_expr(f, sc),
                }
            }
            Expr::Index(b, idx, _) => {
                self.walk_expr(b, sc);
                for x in idx {
                    self.walk_expr(x, sc);
                }
            }
            Expr::Range(a, z, _) => {
                if let Some(a) = a {
                    self.walk_expr(a, sc);
                }
                if let Some(z) = z {
                    self.walk_expr(z, sc);
                }
            }
            Expr::Lambda(def) => {
                let s = self.scope_for(def, sc);
                self.set_escapes(s);
                self.walk_fn(s);
            }
            Expr::Ask { model, prompt, .. } => {
                self.walk_expr(model, sc);
                self.walk_expr(prompt, sc);
            }
            Expr::ModelNew { spec, opts, .. } => {
                self.walk_expr(spec, sc);
                if let Some(o) = opts {
                    self.walk_expr(o, sc);
                }
            }
            _ => {}
        }
    }

    fn infer(&mut self, top: &[Stmt]) {
        // Pass 0: find the functions that are defined once at the top level.
        self.pass0 = true;
        self.walk_block(top, 0);
        self.pass0 = false;
        for i in 0..self.gcount.len() {
            if let (1, Some(p)) = (self.gcount[i], self.gdef[i]) {
                self.static_fn.insert(i, self.by_def[&p]);
            }
            if let (1, Some(n)) = (self.gcount[i], self.glit[i]) {
                self.consts.insert(i, n);
            }
            if let (1, Some(n)) = (self.gcount[i], self.glit_i[i]) {
                self.iconsts.insert(i, n);
            }
        }
        // A top-level variable given once, directly at the top level, arithmetic
        // on constants given before it (AH = HIST + 1) is a constant too.
        for st in top {
            if let Stmt::Assign { idx, value } = st {
                let i = *idx as usize;
                if self.gcount[i] == 1 && !self.consts.contains_key(&i) && !self.iconsts.contains_key(&i) && !self.static_fn.contains_key(&i) {
                    if let Some(v) = self.const_expr(value) {
                        if v.is_finite() {
                            self.consts.insert(i, v);
                        }
                    }
                }
            }
        }
        for s in &mut self.scopes {
            for v in &mut s.vars {
                *v = Ty::Bot;
            }
            for c in &mut s.captured {
                *c = false;
            }
            s.free.clear();
            s.ret = Ty::Bot;
            s.escapes = false;
        }
        for _ in 0..200 {
            self.changed = false;
            self.walk_block(top, 0);
            if !self.changed {
                break;
            }
        }
        for s in &mut self.scopes {
            for v in &mut s.vars {
                if *v == Ty::Bot {
                    *v = Ty::Dyn;
                }
                if *v == Ty::ILit {
                    *v = Ty::Num;
                }
            }
            if s.ret == Ty::Bot {
                s.ret = Ty::Dyn;
            }
            if s.ret == Ty::ILit {
                s.ret = Ty::Num;
            }
        }
        self.codegen = true;
    }

    // ---------- code generation ----------

    fn fresh(&mut self) -> usize {
        self.tmp += 1;
        self.tmp
    }

    fn sconst(&mut self, s: &str) -> String {
        let i = match self.strs.iter().position(|x| x == s) {
            Some(i) => i,
            None => {
                self.strs.push(s.to_string());
                self.strs.len() - 1
            }
        };
        format!("sc({})", i)
    }

    /// The Rust place holding a variable (not valid for captured ones).
    fn place(&self, t: usize, idx: usize) -> String {
        if t == 0 {
            format!("(*GV.as_mut_ptr()).v{}", idx)
        } else {
            format!("v{}", idx)
        }
    }

    fn assign(&self, t: usize, idx: usize, code: String) -> String {
        if t != 0 && self.scopes[t].captured[idx] {
            format!("{{ let x_ = {}; *c{}_{}.borrow_mut() = x_; }}\n", code, t, idx)
        } else {
            format!("{} = {};\n", self.place(t, idx), code)
        }
    }

    fn read_var(&self, sc: usize, depth: u32, idx: usize, name: &str, line: u32) -> (String, Ty) {
        let t = self.resolve(sc, depth);
        if t == 0 {
            if let Some(n) = self.consts.get(&idx) {
                return (format!("{:?}f64", n), Ty::Num);
            }
            if let Some(n) = self.iconsts.get(&idx) {
                return (format!("({}i64)", n), Ty::Int);
            }
            if self.is_builtin(idx) {
                return (format!("builtin({})", idx), Ty::Dyn);
            }
            if self.static_fn.contains_key(&idx) {
                return (format!("(*GV.as_mut_ptr()).v{}.clone()", idx), Ty::Dyn);
            }
        } else if self.scopes[t].captured[idx] {
            return (format!("{{ let b_ = c{}_{}.borrow(); get(&b_, {:?}, {})? }}", t, idx, name, line), Ty::Dyn);
        }
        let p = self.place(t, idx);
        match self.var_ty(t, idx) {
            Ty::Num => (p, Ty::Num),
            Ty::Bool => (p, Ty::Bool),
            Ty::Arr => (format!("{}.clone()", p), Ty::Arr),
            Ty::Int => (p, Ty::Int),
            Ty::Whole => (format!("({} as f64)", p), Ty::Num),
            Ty::PkR => (format!("{}.clone()", p), Ty::PkR),
            Ty::PkM => (format!("{}.clone()", p), Ty::PkM),
            Ty::IArr => (format!("{}.clone()", p), Ty::IArr),
            _ => (format!("get(&{}, {:?}, {})?", p, name, line), Ty::Dyn),
        }
    }

    /// Code for an expression as a machine integer.
    fn int_code(&mut self, e: &Expr, sc: usize, line: u32) -> String {
        if let Some(c) = self.const_of(e, sc) {
            if c == c.trunc() && c.abs() < 9.0e15 {
                return format!("({}i64)", c as i64);
            }
        }
        if let Some(c) = self.icode(e, sc, false) {
            return c;
        }
        let (c, t) = self.expr(e, sc);
        match t {
            Ty::Int => format!("({})", c),
            Ty::Num => format!("n2i({}, {})?", c, line),
            other => format!("{{ let v_ = {}; to_int(&v_, {})? }}", conv(c, other, Ty::Dyn), line),
        }
    }

    /// Code for an expression as a machine decimal.
    fn num_code(&mut self, e: &Expr, sc: usize) -> String {
        let (c, t) = self.expr(e, sc);
        conv(c, t, Ty::Num)
    }

    fn dyn_code(&mut self, e: &Expr, sc: usize) -> String {
        let (c, t) = self.expr(e, sc);
        conv(c, t, Ty::Dyn)
    }

    fn cond(&mut self, e: &Expr, sc: usize, line: u32) -> String {
        let (c, t) = self.expr(e, sc);
        match t {
            Ty::Bool => c,
            _ => format!("truth(&{}, {})?", conv(c, t, Ty::Dyn), line),
        }
    }

    fn dyn_list(&mut self, es: &[Expr], sc: usize) -> String {
        let parts: Vec<String> = es.iter().map(|x| self.dyn_code(x, sc)).collect();
        parts.join(", ")
    }

    fn expr(&mut self, e: &Expr, sc: usize) -> (String, Ty) {
        let want = self.ty(e, sc);
        match e {
            Expr::Num(n) => (format!("{:?}f64", n), Ty::Num),
            Expr::Int(n) => (format!("({}i64)", n), Ty::Int),
            Expr::BitNot(a, line) => {
                if want == Ty::Int {
                    (format!("(!{})", self.int_code(a, sc, *line)), Ty::Int)
                } else {
                    let d = self.dyn_code(a, sc);
                    (format!("{{ let v_ = {}; bitnot(&v_, {})? }}", d, line), Ty::Dyn)
                }
            }
            Expr::Bool(b) => (b.to_string(), Ty::Bool),
            Expr::Nothing => ("Value::Nothing".into(), Ty::Dyn),
            Expr::Str(s) => (self.sconst(s), Ty::Dyn),
            Expr::Interp(parts) => (format!("interp_str(&[{}])", self.dyn_list(parts, sc)), Ty::Dyn),
            Expr::Var { depth, idx, name, line } => self.read_var(sc, *depth, *idx as usize, name, *line),
            Expr::List(items) => {
                let known: Option<Vec<f64>> = items.iter().map(|x| self.const_of(x, sc)).collect();
                if let (Some(vals), false) = (known, items.is_empty()) {
                    let bits: Vec<u64> = vals.iter().map(|x| x.to_bits()).collect();
                    let i = match self.arrs.iter().position(|a| a.iter().map(|x| x.to_bits()).collect::<Vec<u64>>() == bits) {
                        Some(i) => i,
                        None => {
                            self.arrs.push(vals);
                            self.arrs.len() - 1
                        }
                    };
                    return (format!("ac({})", i), Ty::Arr);
                }
                let code = format!("list(vec![{}])", self.dyn_list(items, sc));
                (conv(code, Ty::Dyn, want), want)
            }
            Expr::Map(entries) => {
                let parts: Vec<String> =
                    entries.iter().map(|(k, v)| format!("({:?}, {})", &**k, self.dyn_code(v, sc))).collect();
                (format!("map(vec![{}])", parts.join(", ")), Ty::Dyn)
            }
            Expr::Neg(a, line) => {
                let (c, t) = self.expr(a, sc);
                if t == Ty::Num {
                    (format!("(-({}))", c), Ty::Num)
                } else if t == Ty::Int {
                    (format!("({}).wrapping_neg()", c), Ty::Int)
                } else {
                    let code = format!("{{ let a_ = {}; neg(&a_, {})? }}", conv(c, t, Ty::Dyn), line);
                    (conv(code, Ty::Dyn, want), want)
                }
            }
            Expr::Not(a, line) => (format!("(!({}))", self.cond(a, sc, *line)), Ty::Bool),
            Expr::And(a, b, line) => {
                (format!("(({}) && ({}))", self.cond(a, sc, *line), self.cond(b, sc, *line)), Ty::Bool)
            }
            Expr::Or(a, b, line) => {
                (format!("(({}) || ({}))", self.cond(a, sc, *line), self.cond(b, sc, *line)), Ty::Bool)
            }
            Expr::Binary(op, a, b, line) => {
                // Anything that involves an integer (or is a bit operation on numbers)
                // is done with machine integers.
                let (ta0, tb0) = (self.ty(a, sc), self.ty(b, sc));
                let number = |t: Ty| t == Ty::Num || t == Ty::Int;
                if number(ta0) && number(tb0) && (ta0 == Ty::Int || tb0 == Ty::Int || is_bit_op(*op)) {
                    match op {
                        BinOp::Div => {
                            let (x, y) = (self.num_code(a, sc), self.num_code(b, sc));
                            return (format!("({} / {})", x, y), Ty::Num);
                        }
                        BinOp::Pow | BinOp::MatMul => {}
                        BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge | BinOp::Eq | BinOp::Ne => {
                            let sym = match op {
                                BinOp::Lt => "<",
                                BinOp::Gt => ">",
                                BinOp::Le => "<=",
                                BinOp::Ge => ">=",
                                BinOp::Eq => "==",
                                _ => "!=",
                            };
                            let whole_const = |s: &Self, e: &Expr| {
                                s.const_of(e, sc).map_or(false, |c| c == c.trunc() && c.abs() < 9.0e15)
                            };
                            let ints = (ta0 == Ty::Int || whole_const(self, a)) && (tb0 == Ty::Int || whole_const(self, b));
                            let (x, y) = if ints {
                                (self.int_code(a, sc, *line), self.int_code(b, sc, *line))
                            } else {
                                (self.num_code(a, sc), self.num_code(b, sc))
                            };
                            return (format!("({} {} {})", x, sym, y), Ty::Bool);
                        }
                        _ => {
                            let x = self.int_code(a, sc, *line);
                            let y = self.int_code(b, sc, *line);
                            let code = match op {
                                BinOp::Add => format!("{}.wrapping_add({})", x, y),
                                BinOp::Sub => format!("{}.wrapping_sub({})", x, y),
                                BinOp::Mul => format!("{}.wrapping_mul({})", x, y),
                                BinOp::Mod => match self.const_of(b, sc) {
                                    Some(c) if c != 0.0 && c == c.trunc() && c.abs() < 9.0e15 => {
                                        format!("{}.wrapping_rem_euclid({}i64)", x, c as i64)
                                    }
                                    _ => format!("irem({}, {}, {})?", x, y, line),
                                },
                                BinOp::BitAnd => format!("({} & {})", x, y),
                                BinOp::BitOr => format!("({} | {})", x, y),
                                BinOp::BitXor => format!("({} ^ {})", x, y),
                                BinOp::Shl => format!("shl({}, {})", x, y),
                                _ => format!("shr({}, {})", x, y),
                            };
                            return (code, Ty::Int);
                        }
                    }
                }
                if ta0 == Ty::Num && tb0 == Ty::Num {
                    let sym = match op {
                        BinOp::Lt => "<",
                        BinOp::Gt => ">",
                        BinOp::Le => "<=",
                        BinOp::Ge => ">=",
                        BinOp::Eq => "==",
                        BinOp::Ne => "!=",
                        _ => "",
                    };
                    if !sym.is_empty() {
                        if let (Some(x), Some(y)) = (self.icode(a, sc, false), self.icode(b, sc, false)) {
                            return (format!("({} {} {})", x, sym, y), Ty::Bool);
                        }
                    }
                    if *op == BinOp::Mod {
                        if let Some(x) = self.icode(e, sc, false) {
                            return (format!("(({}) as f64)", x), Ty::Num);
                        }
                    }
                }
                let (ca, ta) = self.expr(a, sc);
                let (cb, tb) = self.expr(b, sc);
                if ta == Ty::Num && tb == Ty::Num {
                    let code = match op {
                        BinOp::Add => format!("({} + {})", ca, cb),
                        BinOp::Sub => format!("({} - {})", ca, cb),
                        BinOp::Mul => format!("({} * {})", ca, cb),
                        BinOp::Div => format!("({} / {})", ca, cb),
                        BinOp::Mod => match self.const_of(b, sc) {
                            Some(c) if c != 0.0 && c == c.trunc() && c.abs() < 4.0e18 => {
                                format!("rem_i({}, {}i64)", ca, c as i64)
                            }
                            _ => format!("rem_f({}, {})", ca, cb),
                        },
                        BinOp::Pow => format!("({}).powf({})", ca, cb),
                        BinOp::Lt => format!("({} < {})", ca, cb),
                        BinOp::Gt => format!("({} > {})", ca, cb),
                        BinOp::Le => format!("({} <= {})", ca, cb),
                        BinOp::Ge => format!("({} >= {})", ca, cb),
                        BinOp::Eq => format!("({} == {})", ca, cb),
                        BinOp::Ne => format!("({} != {})", ca, cb),
                        _ => String::new(),
                    };
                    if !code.is_empty() {
                        return (code, want);
                    }
                }
                if ta == Ty::Bool && tb == Ty::Bool && matches!(op, BinOp::Eq | BinOp::Ne) {
                    let sym = if *op == BinOp::Eq { "==" } else { "!=" };
                    return (format!("({} {} {})", ca, sym, cb), Ty::Bool);
                }
                let code = format!(
                    "{{ let a_ = {}; let b_ = {}; binop(BinOp::{}, &a_, &b_, {})? }}",
                    conv(ca, ta, Ty::Dyn),
                    conv(cb, tb, Ty::Dyn),
                    binop_name(*op),
                    line
                );
                (conv(code, Ty::Dyn, want), want)
            }
            Expr::Call(f, args, line) => match self.callee(f, sc) {
                Callee::Builtin(i) => {
                    let name = self.bnames[i];
                    match (name, args.len()) {
                        ("within", 3 | 4) if matches!(self.ty(&args[0], sc), Ty::Int | Ty::Num | Ty::ILit) => {
                            return self.within_code(args, sc, *line);
                        }
                        ("int", 1) => return (self.int_code(&args[0], sc, *line), Ty::Int),
                        ("popcount", 1) => {
                            return (format!("({}.count_ones() as i64)", self.int_code(&args[0], sc, *line)), Ty::Int)
                        }
                        ("lowest_bit", 1) => {
                            return (format!("({}.trailing_zeros() as i64)", self.int_code(&args[0], sc, *line)), Ty::Int)
                        }
                        ("highest_bit", 1) => {
                            return (format!("(63 - {}.leading_zeros() as i64)", self.int_code(&args[0], sc, *line)), Ty::Int)
                        }
                        ("div", 2) => {
                            let x = self.int_code(&args[0], sc, *line);
                            let y = self.int_code(&args[1], sc, *line);
                            return (format!("idiv({}, {}, {})?", x, y, line), Ty::Int);
                        }
                        _ => {}
                    }
                    if name == "argmax" && args.len() == 3 && matches!(self.ty(&args[1], sc), Ty::Num | Ty::Int) && matches!(self.ty(&args[2], sc), Ty::Num | Ty::Int) {
                        if let Expr::Var { depth, idx: vi, .. } = &args[0] {
                            let target = if let Some((tb, false)) = self.table_of(sc, *depth, *vi) {
                                Some(format!("t_argmax(&*std::ptr::addr_of!({})", tb))
                            } else if self.ty(&args[0], sc) == Ty::Arr {
                                self.plain_place(&args[0], sc).map(|p| format!("a_argmax(&{}", p))
                            } else {
                                None
                            };
                            if let Some(t) = target {
                                let first = self.num_code(&args[1], sc);
                                let last = self.num_code(&args[2], sc);
                                return (format!("{{ let a_ = {}; let b_ = {}; {}, a_, b_, {})? }}", first, last, t, line), Ty::Num);
                            }
                        }
                    }
                    if name == "screlu_dot" && args.len() == 3 {
                        if let Some(code) = self.pk_screlu(args, sc, *line) {
                            return (code, Ty::Num);
                        }
                    }
                    let typed: Vec<(String, Ty)> = args
                        .iter()
                        .map(|a| {
                            let (c, t) = self.expr(a, sc);
                            // number helpers take integers as ordinary numbers
                            if t == Ty::Int && matches!(name, "min" | "max" | "number" | "abs" | "floor" | "trunc" | "round" | "sqrt") {
                                (conv(c, Ty::Int, Ty::Num), Ty::Num)
                            } else {
                                (c, t)
                            }
                        })
                        .collect();
                    if let [(c, Ty::Num)] = &typed[..] {
                        if name == "floor" || name == "trunc" {
                            return (format!("{}_f({})", name, c), Ty::Num);
                        }
                        let m = match name {
                            "abs" => "abs",
                            "round" => "round",
                            "sqrt" => "sqrt",
                            "exp" => "exp",
                            "log" => "ln",
                            "sin" => "sin",
                            "cos" => "cos",
                            "tanh" => "tanh",
                            _ => "",
                        };
                        if !m.is_empty() {
                            return (format!("({}).{}()", c, m), Ty::Num);
                        }
                    }
                    if let ("number", [(c, t)]) = (name, &typed[..]) {
                        if *t == Ty::Num {
                            return (c.clone(), Ty::Num);
                        }
                        return (format!("{{ let v_ = {}; to_number(&v_, {})? }}", conv(c.clone(), *t, Ty::Dyn), line), Ty::Num);
                    }
                    if let [(a, Ty::Num), (b, Ty::Num)] = &typed[..] {
                        if name == "min" || name == "max" {
                            return (format!("f64::{}({}, {})", name, a, b), Ty::Num);
                        }
                    }
                    let parts: Vec<String> = typed.into_iter().map(|(c, t)| conv(c, t, Ty::Dyn)).collect();
                    if name == "screlu_dot" || name == "update_rows" {
                        let code = format!("scall({}_slice, &[{}], {})?", name, parts.join(", "), line);
                        return (conv(code, Ty::Dyn, want), want);
                    }
                    let mut code = format!("bcall({}, vec![{}], {})?", i, parts.join(", "), line);
                    if calls_back(name) {
                        code = self.callback(code);
                    }
                    (conv(code, Ty::Dyn, want), want)
                }
                Callee::Static(s) => {
                    let n = self.scopes[s].nparams;
                    if args.len() != n {
                        let msg = format!(
                            "'{}' takes {} input{} but was given {}",
                            self.scopes[s].name,
                            n,
                            if n == 1 { "" } else { "s" },
                            args.len()
                        );
                        return (format!("fail::<Value>({:?}, {})?", msg, line), Ty::Dyn);
                    }
                    let mut parts = vec![];
                    for (i, a) in args.iter().enumerate() {
                        if self.var_ty_param(s, i) == Ty::Whole {
                            parts.push(self.whole_code(a, sc));
                            continue;
                        }
                        let (c, t) = self.expr(a, sc);
                        parts.push(conv(c, t, self.var_ty_param(s, i)));
                    }
                    // inputs given a range: checked here, unless certainly in it
                    let mut checks = vec![];
                    for (k, lo, hi, pname, pline) in self.input_checks(s).0 {
                        let (lo, hi) = unsafe { (&*lo, &*hi) };
                        let int = matches!(self.var_ty_param(s, k), Ty::Int | Ty::Whole);
                        let proven = int
                            && matches!(self.ty(&args[k], sc), Ty::Int | Ty::Whole)
                            && matches!((self.erange(&args[k], sc), self.within_bounds(lo, hi, s)), (Rg::In(x, y), Some((l, h))) if x >= l && y <= h);
                        let lc = self.num_code(lo, s);
                        let hc = self.num_code(hi, s);
                        if !proven {
                            let f = match self.var_ty_param(s, k) {
                                Ty::Int | Ty::Whole => "within_i",
                                Ty::Num => "within_n",
                                _ => "within_v",
                            };
                            let clone = if f == "within_v" { ".clone()" } else { "" };
                            checks.push(format!("{}(a{}_{}, {}, {}, {:?}, {})?;", f, k, clone, lc, hc, pname, pline));
                        }
                    }
                    let call = if checks.is_empty() {
                        parts.push(self.depth_arg().to_string());
                        self.static_call(s, format!("f{}({})", s, parts.join(", ")))
                    } else {
                        // the inputs are worked out first, in order, then checked
                        let mut lets = String::new();
                        let mut names = vec![];
                        for (k, p) in parts.iter().enumerate() {
                            lets.push_str(&format!("let a{}_ = {}; ", k, p));
                            names.push(format!("a{}_", k));
                        }
                        names.push(self.depth_arg().to_string());
                        let call = self.static_call(s, format!("f{}({})", s, names.join(", ")));
                        format!("{{ {}{} {} }}", lets, checks.join(" "), call)
                    };
                    if self.scopes[s].ret == Ty::Unit {
                        (format!("{{ {}; Value::Nothing }}", call), Ty::Dyn)
                    } else {
                        (call, self.scopes[s].ret)
                    }
                }
                Callee::Dynamic => {
                    let fc = self.dyn_code(f, sc);
                    let code = format!("{{ let f_ = {}; call(&f_, vec![{}], {})? }}", fc, self.dyn_list(args, sc), line);
                    (self.callback(code), Ty::Dyn)
                }
            },
            Expr::Index(base, idx, line) => {
                if idx.iter().any(|i| matches!(i, Expr::Range(..))) {
                    let mut parts = vec![];
                    for i in idx {
                        parts.push(match i {
                            Expr::Range(a, z, _) => {
                                let a = match a {
                                    Some(x) => format!("Some({})", self.dyn_code(x, sc)),
                                    None => "None".into(),
                                };
                                let z = match z {
                                    Some(x) => format!("Some({})", self.dyn_code(x, sc)),
                                    None => "None".into(),
                                };
                                format!("Ix::Span({}, {})", a, z)
                            }
                            other => format!("Ix::At({})", self.dyn_code(other, sc)),
                        });
                    }
                    let b = self.dyn_code(base, sc);
                    return (format!("{{ let b_ = {}; slice(&b_, vec![{}], {})? }}", b, parts.join(", "), line), Ty::Dyn);
                }
                if matches!(self.ty(base, sc), Ty::PkM | Ty::PkR) && matches!(want, Ty::PkR | Ty::Num) {
                    let k = self.dyn_code(&idx[0], sc);
                    let code = match self.plain_place(base, sc) {
                        Some(p) if !calls_anything(&idx[0]) => format!("{{ let k_ = {}; pk_index(&{}, k_, {})? }}", k, p, line),
                        _ => {
                            let (b, _) = self.expr(base, sc);
                            format!("{{ let b_ = {}; let k_ = {}; pk_index(&b_, k_, {})? }}", b, k, line)
                        }
                    };
                    return (conv(code, Ty::Dyn, want), want);
                }
                if want == Ty::Num || want == Ty::Int {
                    if let Expr::Var { depth, idx: vi, .. } = &**base {
                        let t = self.resolve(sc, *depth);
                        if let Some(p) = self.fast_ptr(t, *vi as usize) {
                            if let Some(ic) = self.fast_pos(&idx[0], sc) {
                                return (format!("(*{}.add(({}) as usize))", p, ic), want);
                            }
                        }
                    }
                    if let Expr::Var { depth, idx: vi, .. } = &**base {
                        if let Some((tb, int)) = self.table_of(sc, *depth, *vi) {
                            let k = if int { "i" } else { "a" };
                            let safe = self.in_table(&idx[0], sc, *vi as usize);
                            let code = match self.table_pos(&idx[0], sc, *line) {
                                Ok((ic, _)) if safe => {
                                    format!("(*(std::ptr::addr_of!({}) as *const {}).add(({}) as usize))", tb, if int { "i64" } else { "f64" }, ic)
                                }
                                Ok((ic, num)) => format!("tget_{}(&*std::ptr::addr_of!({}), {}, {}, {})?", k, tb, ic, num, line),
                                Err(fc) => format!("tget_{}f(&*std::ptr::addr_of!({}), {}, {})?", k, tb, fc, line),
                            };
                            return (code, want);
                        }
                    }
                    let it = self.ty(&idx[0], sc);
                    // which reader: (row kind, position kind)
                    let at = self.icode(&idx[0], sc, true);
                    let (ic, reader) = if want == Ty::Int {
                        (at.unwrap_or_else(|| self.int_code(&idx[0], sc, *line)), "iget")
                    } else if it == Ty::Int {
                        (self.int_code(&idx[0], sc, *line), "aget_i")
                    } else if let Some(c) = at {
                        (c, "aget_i")
                    } else {
                        (self.expr(&idx[0], sc).0, "aget")
                    };
                    // A variable is read in place; no copy of the handle is needed
                    // (unless working out the position could change the variable).
                    if let Expr::Var { depth, idx: vi, .. } = &**base {
                        let t = self.resolve(sc, *depth);
                        if !(t != 0 && self.scopes[t].captured[*vi as usize])
                            && !(t == 0 && self.consts.contains_key(&(*vi as usize)))
                            && !calls_anything(&idx[0])
                        {
                            let p = self.place(t, *vi as usize);
                            return (format!("{{ let i_ = {}; {}(&{}, i_, {})? }}", ic, reader, p, line), want);
                        }
                    }
                    let (bc, _) = self.expr(base, sc);
                    return (format!("{{ let a_ = {}; let i_ = {}; {}(&a_, i_, {})? }}", bc, ic, reader, line), want);
                }
                // a fixed table read with a position of any kind
                if let ([ie], Expr::Var { depth, idx: vi, .. }) = (&idx[..], &**base) {
                    if let Some((tb, int)) = self.table_of(sc, *depth, *vi) {
                        let k = self.dyn_code(ie, sc);
                        let f = if int { "tget_iv" } else { "tget_av" };
                        return (format!("{{ let k_ = {}; {}(&*std::ptr::addr_of!({}), k_, {})? }}", k, f, tb, line), Ty::Dyn);
                    }
                }
                // One position into a variable: read it in place. (Only when
                // working out the position can't change the variable first.)
                if let ([ie], Expr::Var { depth, idx: vi, name, .. }) = (&idx[..], &**base) {
                    let t = self.resolve(sc, *depth);
                    let i = *vi as usize;
                    let plain_var = !(t != 0 && self.scopes[t].captured[i])
                        && !(t == 0 && (self.consts.contains_key(&i) || self.iconsts.contains_key(&i) || self.is_builtin(i) || self.static_fn.contains_key(&i)))
                        && self.var_ty(t, i) == Ty::Dyn;
                    if plain_var && !calls_anything(ie) {
                        let k = self.dyn_code(ie, sc);
                        let p = self.place(t, i);
                        return (
                            format!("{{ has_value(&{p}, {n:?}, {l})?; let k_ = {k}; index1(&{p}, k_, {n:?}, {l})? }}", p = p, n = &**name, l = line, k = k),
                            Ty::Dyn,
                        );
                    }
                }
                let b = self.dyn_code(base, sc);
                (format!("{{ let b_ = {}; index(&b_, &[{}], {})? }}", b, self.dyn_list(idx, sc), line), Ty::Dyn)
            }
            Expr::Field(base, name, line) => {
                let b = self.dyn_code(base, sc);
                (format!("{{ let b_ = {}; field(&b_, {:?}, {})? }}", b, &**name, line), Ty::Dyn)
            }
            Expr::Range(_, _, line) => (
                format!(
                    "fail::<Value>(\"a range like 1:3 can only be used to read part of a value, as in xs[1:3]\", {})?",
                    line
                ),
                Ty::Dyn,
            ),
            Expr::Lambda(def) => {
                let s = self.scope_for(def, sc);
                (self.closure(s), Ty::Dyn)
            }
            Expr::Ask { model, prompt, kind, line } => {
                let k = match kind {
                    AskKind::Text => "AskKind::Text".to_string(),
                    AskKind::Number => "AskKind::Number".to_string(),
                    AskKind::YesNo => "AskKind::YesNo".to_string(),
                    AskKind::List => "AskKind::List".to_string(),
                    AskKind::Shape(s) => format!("AskKind::Shape(std::rc::Rc::new(antares::llm::Shape::from_spec({:?}).unwrap()))", s.spec()),
                };
                let m = self.dyn_code(model, sc);
                let p = self.dyn_code(prompt, sc);
                (format!("{{ let m_ = {}; let p_ = {}; ask(&m_, &p_, {}, {})? }}", m, p, k, line), Ty::Dyn)
            }
            Expr::ModelNew { spec, opts, line } => {
                let s = self.dyn_code(spec, sc);
                let o = match opts {
                    Some(o) => format!("Some(&{})", self.dyn_code(o, sc)),
                    None => "None".into(),
                };
                (format!("{{ let s_ = {}; model(&s_, {}, {})? }}", s, o, line), Ty::Dyn)
            }
        }
    }

    fn var_ty_param(&self, s: usize, i: usize) -> Ty {
        self.var_ty(s, i)
    }

    /// `counter`, `counter + c`, `c + counter` or `counter - c` for the loop
    /// counter `v` of scope `sc` and a small whole constant c: the c.
    fn fast_offset(&self, e: &Expr, sc: usize, v: usize) -> Option<i64> {
        let is_v = |x: &Expr| matches!(x, Expr::Var { depth, idx, .. } if self.resolve(sc, *depth) == sc && *idx as usize == v);
        let c = |x: &Expr| self.const_of(x, sc).filter(|c| *c == c.trunc() && c.abs() <= 1048576.0).map(|c| c as i64);
        match e {
            _ if is_v(e) => Some(0),
            Expr::Binary(BinOp::Add, a, b, _) => {
                if is_v(a) {
                    c(b)
                } else if is_v(b) {
                    c(a)
                } else {
                    None
                }
            }
            Expr::Binary(BinOp::Sub, a, b, _) if is_v(a) => c(b).map(|x| -x),
            _ => None,
        }
    }

    /// A row variable (plain or integer) of scope `sc` itself.
    fn row_var(&self, sc: usize, depth: u32, idx: u32) -> Option<usize> {
        let i = idx as usize;
        let table = sc == 0 && self.tables.contains_key(&i);
        (self.resolve(sc, depth) == sc && matches!(self.var_ty(sc, i), Ty::Arr | Ty::IArr) && !table).then_some(i)
    }

    fn fast_note_set(&self, sc: usize, depth: u32, idx: u32, v: usize, f: &mut FastScan) {
        if self.resolve(sc, depth) == sc {
            if idx as usize == v {
                f.ok = false;
            }
            f.bad.insert(idx as usize);
            f.assigned.insert(idx as usize);
        }
    }

    /// A fixed table, if `depth`/`idx` from scope `sc` is one.
    fn is_table(&self, sc: usize, depth: u32, idx: u32) -> bool {
        self.resolve(sc, depth) == 0 && self.tables.contains_key(&(idx as usize))
    }

    fn fast_scan_block(&self, stmts: &[Stmt], sc: usize, v: usize, f: &mut FastScan) {
        for st in stmts {
            match st {
                Stmt::Expr(e) => self.fast_scan_expr(e, sc, v, f),
                Stmt::Assign { idx, value } => {
                    self.fast_note_set(sc, 0, *idx, v, f);
                    self.fast_scan_expr(value, sc, v, f);
                }
                Stmt::AssignOuter { depth, idx, value } => {
                    self.fast_note_set(sc, *depth, *idx, v, f);
                    self.fast_scan_expr(value, sc, v, f);
                }
                Stmt::Learn { idx, value, .. } => {
                    self.fast_note_set(sc, 0, *idx, v, f);
                    self.fast_scan_expr(value, sc, v, f);
                }
                Stmt::SetPath { depth, idx, path, value, .. } => {
                    self.fast_scan_expr(value, sc, v, f);
                    for a in path {
                        if let Access::Index(es) = a {
                            for e in es {
                                self.fast_scan_expr(e, sc, v, f);
                            }
                        }
                    }
                    if self.resolve(sc, *depth) == sc && *idx as usize == v {
                        f.ok = false;
                    }
                    let off = self.fast_set(path).and_then(|ie| self.fast_offset(ie, sc, v));
                    if self.is_table(sc, *depth, *idx) {
                        if let Some(o) = off {
                            f.note_table(*idx as usize, o);
                        }
                        continue;
                    }
                    match (self.row_var(sc, *depth, *idx), off) {
                        (Some(r), Some(o)) => f.note(r, o, true),
                        (Some(r), None) => {
                            f.bad.insert(r);
                        }
                        _ => self.fast_note_set(sc, *depth, *idx, v, f),
                    }
                }
                Stmt::Show(es) => {
                    for e in es {
                        self.fast_scan_expr(e, sc, v, f);
                    }
                }
                Stmt::Give(e) => {
                    if let Some(e) = e {
                        self.fast_scan_expr(e, sc, v, f);
                    }
                }
                // only innermost loops get a direct version
                Stmt::Repeat { .. } | Stmt::Each { .. } | Stmt::While { .. } => f.ok = false,
                Stmt::When { arms, otherwise } => {
                    for (c, b, _) in arms {
                        self.fast_scan_expr(c, sc, v, f);
                        self.fast_scan_block(b, sc, v, f);
                    }
                    if let Some(b) = otherwise {
                        self.fast_scan_block(b, sc, v, f);
                    }
                }
                Stmt::Descend { loss, rate, clip, .. } => {
                    self.fast_scan_expr(loss, sc, v, f);
                    self.fast_scan_expr(rate, sc, v, f);
                    if let Some(c) = clip {
                        self.fast_scan_expr(c, sc, v, f);
                    }
                }
                Stmt::Try { body, var, handler } => {
                    if let Some(x) = var {
                        self.fast_note_set(sc, 0, *x, v, f);
                    }
                    self.fast_scan_block(body, sc, v, f);
                    self.fast_scan_block(handler, sc, v, f);
                }
                Stmt::Stop(_) | Stmt::Skip(_) => {}
            }
        }
    }

    fn fast_scan_expr(&self, e: &Expr, sc: usize, v: usize, f: &mut FastScan) {
        match e {
            Expr::Var { depth, idx, .. } => {
                if self.resolve(sc, *depth) == sc {
                    f.bad.insert(*idx as usize);
                }
            }
            Expr::Index(base, idx, _) => {
                if let (Expr::Var { depth, idx: bi, .. }, [ie]) = (&**base, &idx[..]) {
                    if self.is_table(sc, *depth, *bi) {
                        if let Some(o) = self.fast_offset(ie, sc, v) {
                            f.note_table(*bi as usize, o);
                        }
                        self.fast_scan_expr(ie, sc, v, f);
                        return;
                    }
                    if let (Some(r), Some(o)) = (self.row_var(sc, *depth, *bi), self.fast_offset(ie, sc, v)) {
                        f.note(r, o, false);
                        self.fast_scan_expr(ie, sc, v, f);
                        return;
                    }
                }
                self.fast_scan_expr(base, sc, v, f);
                for x in idx {
                    self.fast_scan_expr(x, sc, v, f);
                }
            }
            Expr::Interp(p) | Expr::List(p) => {
                for x in p {
                    self.fast_scan_expr(x, sc, v, f);
                }
            }
            Expr::Map(m) => {
                for (_, x) in m {
                    self.fast_scan_expr(x, sc, v, f);
                }
            }
            Expr::Neg(a, _) | Expr::Not(a, _) | Expr::Field(a, _, _) | Expr::BitNot(a, _) => self.fast_scan_expr(a, sc, v, f),
            Expr::Binary(_, a, b, _) | Expr::And(a, b, _) | Expr::Or(a, b, _) => {
                self.fast_scan_expr(a, sc, v, f);
                self.fast_scan_expr(b, sc, v, f);
            }
            Expr::Call(fe, args, _) => {
                for a in args {
                    self.fast_scan_expr(a, sc, v, f);
                }
                match self.callee(fe, sc) {
                    Callee::Builtin(_) => {}
                    // a function could change a top-level variable
                    Callee::Static(_) => {
                        if sc == 0 {
                            f.ok = false;
                        }
                    }
                    Callee::Dynamic => {
                        if sc == 0 {
                            f.ok = false;
                        }
                        self.fast_scan_expr(fe, sc, v, f);
                    }
                }
            }
            Expr::Range(a, z, _) => {
                for x in [a, z].into_iter().flatten() {
                    self.fast_scan_expr(x, sc, v, f);
                }
            }
            Expr::Lambda(_) => f.ok = false,
            Expr::Ask { model, prompt, .. } => {
                self.fast_scan_expr(model, sc, v, f);
                self.fast_scan_expr(prompt, sc, v, f);
            }
            Expr::ModelNew { spec, opts, .. } => {
                self.fast_scan_expr(spec, sc, v, f);
                if let Some(o) = opts {
                    self.fast_scan_expr(o, sc, v, f);
                }
            }
            _ => {}
        }
    }

    /// For an innermost counting loop over the whole-number counter `v`: the
    /// row variables used only as `row[v + constant]` in its body, with the
    /// lowest and highest constant and whether the loop writes to the row.
    /// Such a loop also gets a version that checks positions and sharing once
    /// before it starts and then reads and writes the rows directly.
    fn fast_rows(&self, body: &[Stmt], sc: usize, v: usize) -> Vec<FastRow> {
        self.fast_scan(body, sc, v).1
    }

    fn fast_scan(&self, body: &[Stmt], sc: usize, v: usize) -> (FastScan, Vec<FastRow>) {
        let mut f = FastScan { ok: true, bad: BTreeSet::new(), rows: Default::default(), tables: Default::default(), assigned: BTreeSet::new() };
        self.fast_scan_block(body, sc, v, &mut f);
        if !f.ok {
            return (f, vec![]);
        }
        let mut out: Vec<FastRow> = f
            .rows
            .iter()
            .filter(|(r, _)| !f.bad.contains(r))
            .map(|(r, (lo, hi, w))| FastRow { var: *r, lo: *lo, hi: *hi, write: *w, table: false })
            .collect();
        out.extend(f.tables.iter().map(|(r, (lo, hi))| FastRow { var: *r, lo: *lo, hi: *hi, write: true, table: true }));
        (f, out)
    }

    /// `while i < end { ...; i = i + 1 }` (or `<=`) where i is a whole-number or
    /// integer variable changed only by that last statement and `end` doesn't
    /// change in the loop: the counter, `end`, whether it is `<=`, and the rows
    /// and tables reached at counter + constant.
    fn while_plan<'e>(&self, cond: &'e Expr, body: &[Stmt], sc: usize) -> Option<(usize, &'e Expr, bool, Vec<FastRow>)> {
        let Expr::Binary(op @ (BinOp::Lt | BinOp::Le), a, end, _) = cond else { return None };
        let Expr::Var { depth, idx, .. } = &**a else { return None };
        if self.resolve(sc, *depth) != sc || !matches!(self.var_ty(sc, *idx as usize), Ty::Whole | Ty::Int) {
            return None;
        }
        let v = *idx as usize;
        let (last, rest) = body.split_last()?;
        let Stmt::Assign { idx: li, value } = last else { return None };
        let step_ok = *li as usize == v
            && matches!(value, Expr::Binary(BinOp::Add, x, one, _)
                if matches!(&**x, Expr::Var { depth: d, idx: xi, .. } if self.resolve(sc, *d) == sc && *xi as usize == v)
                && (self.const_of(one, sc) == Some(1.0) || matches!(&**one, Expr::Int(1))));
        if !step_ok || calls_anything(end) || !matches!(self.ty(end, sc), Ty::Num | Ty::Int) {
            return None;
        }
        let (f, rows) = self.fast_scan(rest, sc, v);
        if !f.ok || rows.is_empty() {
            return None;
        }
        // `end` may only use constants and variables of this scope the loop leaves alone
        if !self.invariant(end, sc, &f.assigned) {
            return None;
        }
        Some((v, &**end, *op == BinOp::Le, rows))
    }

    /// Whether `e` reads only constants and variables of scope `sc` not in `assigned`.
    fn invariant(&self, e: &Expr, sc: usize, assigned: &BTreeSet<usize>) -> bool {
        if self.const_of(e, sc).is_some() {
            return true;
        }
        match e {
            Expr::Num(_) | Expr::Int(_) => true,
            Expr::Var { depth, idx, .. } => self.resolve(sc, *depth) == sc && sc != 0 && !assigned.contains(&(*idx as usize)),
            Expr::Neg(x, _) => self.invariant(x, sc, assigned),
            Expr::Binary(_, x, y, _) => self.invariant(x, sc, assigned) && self.invariant(y, sc, assigned),
            _ => false,
        }
    }

    /// The loop body over `rows` with direct pointers, and the code that gets
    /// those pointers (to run after the loop's checks have passed).
    fn fast_body(&mut self, body: &[Stmt], sc: usize, k: usize, rows: &[FastRow]) -> (String, String) {
        let mut setup = String::new();
        let before = self.fast.len();
        for row in rows {
            let (r, write) = (row.var, row.write);
            if row.table {
                // a fixed table never moves: its own address
                let (_, int) = self.tables[&r];
                let name = format!("(std::ptr::addr_of_mut!(TB{}) as *mut {})", r, if int { "i64" } else { "f64" });
                self.fast.push(((0, r), name));
                continue;
            }
            let name = format!("p{}_{}", k, r);
            let int = self.var_ty(sc, r) == Ty::IArr;
            let get = match (int, write) {
                (false, true) => format!("arr_unique(&mut {})", self.place(sc, r)),
                (false, false) => format!("arr_ptr(&{})", self.place(sc, r)),
                (true, true) => format!("iarr_unique(&mut {})", self.place(sc, r)),
                (true, false) => format!("iarr_ptr(&{})", self.place(sc, r)),
            };
            setup.push_str(&format!("let {} = {};\n", name, get));
            self.fast.push(((sc, r), name));
        }
        self.loops.push((k, self.try_depth));
        let b = self.block(body, sc, false);
        self.loops.pop();
        self.fast.truncate(before);
        (setup, b)
    }

    /// The length of row variable (or fixed table) `r` of scope `sc`.
    fn row_len(&self, sc: usize, row: &FastRow) -> String {
        let r = row.var;
        if row.table {
            return format!("{}usize", self.tables[&r].0);
        }
        if self.var_ty(sc, r) == Ty::IArr {
            format!("({}.len())", self.place(sc, r))
        } else {
            format!("alen(&{})", self.place(sc, r))
        }
    }

    fn fast_ptr(&self, t: usize, i: usize) -> Option<String> {
        self.fast.iter().find(|(r, _)| *r == (t, i)).map(|(_, p)| p.clone())
    }

    /// Whether `fast_pos` has code for this position.
    fn fast_pos_ok(&self, ie: &Expr, sc: usize) -> bool {
        self.icode(ie, sc, true).is_some() || (self.ty(ie, sc) == Ty::Int && !calls_anything(ie))
    }

    /// Integer code for a position at counter + constant inside a fast loop.
    fn fast_pos(&mut self, ie: &Expr, sc: usize) -> Option<String> {
        if let Some(c) = self.icode(ie, sc, true) {
            return Some(c);
        }
        if self.ty(ie, sc) == Ty::Int && !calls_anything(ie) {
            return Some(self.expr(ie, sc).0);
        }
        None
    }

    /// `range(...)` with number inputs and a constant step: its start (if
    /// given), end and step.
    fn range_parts<'e>(&self, iter: &'e Expr, sc: usize) -> Option<(Option<&'e Expr>, &'e Expr, f64)> {
        let Expr::Call(f, args, _) = iter else { return None };
        let Callee::Builtin(i) = self.callee(f, sc) else { return None };
        if self.bnames[i] != "range" || !args.iter().all(|a| matches!(self.ty(a, sc), Ty::Num | Ty::Int)) {
            return None;
        }
        match &args[..] {
            [z] => Some((None, z, 1.0)),
            [a, z] => Some((Some(a), z, 1.0)),
            [a, z, st] => {
                let c = self.const_of(st, sc)?;
                if c == 0.0 || !c.is_finite() {
                    None
                } else {
                    Some((Some(a), z, c))
                }
            }
            _ => None,
        }
    }

    /// The place of a variable kept as a machine integer (`Ty::Whole`).
    fn whole_place(&self, sc: usize, depth: u32, idx: usize) -> Option<String> {
        let t = self.resolve(sc, depth);
        if t == 0 && (self.consts.contains_key(&idx) || self.is_builtin(idx)) {
            return None;
        }
        (self.scopes[t].vars.get(idx) == Some(&Ty::Whole) && !self.scopes[t].captured[idx]).then(|| self.place(t, idx))
    }

    /// Integer code for an ordinary-number expression whose value is certainly
    /// whole, giving exactly the number the ordinary calculation would.
    /// Without `linear` that is only constants, whole-number variables, such a
    /// variable plus or minus a constant, and `% constant` of these: whole-number
    /// variables count in small steps, so these stay far below 2^53, where
    /// ordinary numbers stop being exact. With `linear` (for positions in a
    /// row) it also allows sums and constant multiples, whose results are
    /// exact unless far too large to be a position anyway.
    fn icode(&self, e: &Expr, sc: usize, linear: bool) -> Option<String> {
        let lit = |c: f64, lim: f64| c == c.trunc() && c.abs() <= lim;
        if let Some(c) = self.const_of(e, sc) {
            return lit(c, 9.0e15).then(|| format!("({}i64)", c as i64));
        }
        match e {
            Expr::Var { depth, idx, .. } => self.whole_place(sc, *depth, *idx as usize),
            Expr::Neg(a, _) if linear => Some(format!("({}).wrapping_neg()", self.icode(a, sc, true)?)),
            Expr::Binary(op @ (BinOp::Add | BinOp::Sub), a, b, _) => {
                if !linear {
                    let small = |x: &Expr| self.const_of(x, sc).is_some_and(|c| lit(c, 4294967296.0));
                    let var = |x: &Expr| matches!(x, Expr::Var { .. });
                    if !((var(a) && small(b)) || (*op == BinOp::Add && small(a) && var(b))) {
                        return None;
                    }
                }
                let f = if *op == BinOp::Add { "wrapping_add" } else { "wrapping_sub" };
                Some(format!("({}).{}({})", self.icode(a, sc, linear)?, f, self.icode(b, sc, linear)?))
            }
            Expr::Binary(BinOp::Mul, a, b, _) if linear => {
                let small = |x: &Expr| self.const_of(x, sc).is_some_and(|c| lit(c, 1048576.0));
                if !(small(a) || small(b)) {
                    return None;
                }
                Some(format!("({}).wrapping_mul({})", self.icode(a, sc, true)?, self.icode(b, sc, true)?))
            }
            Expr::Binary(BinOp::Mod, a, b, _) => {
                let c = self.const_of(b, sc)?;
                if c == 0.0 || !lit(c, 4294967296.0) {
                    return None;
                }
                Some(format!("({}).wrapping_rem_euclid({}i64)", self.icode(a, sc, linear)?, c as i64))
            }
            _ => None,
        }
    }

    /// Code for a value given to a whole-number variable (one of the forms
    /// `counts` accepts).
    fn whole_code(&self, e: &Expr, sc: usize) -> String {
        self.icode(e, sc, false).expect("a whole-number variable is only given whole-number counts")
    }

    /// Whether `e` is a whole number counted in small steps, given the
    /// variables `w` that are: a constant, one of those variables, or one of
    /// them plus or minus a small constant.
    fn counts(&self, e: &Expr, sc: usize, w: &std::collections::HashSet<(usize, usize)>) -> bool {
        let lit = |c: f64, lim: f64| c == c.trunc() && c.abs() <= lim;
        if let Some(c) = self.const_of(e, sc) {
            return lit(c, 4294967296.0);
        }
        let var = |x: &Expr| match x {
            Expr::Var { depth, idx, .. } => {
                let t = self.resolve(sc, *depth);
                !(t == 0 && self.consts.contains_key(&(*idx as usize))) && w.contains(&(t, *idx as usize))
            }
            _ => false,
        };
        let step = |x: &Expr| self.const_of(x, sc).is_some_and(|c| lit(c, 64.0));
        match e {
            Expr::Var { .. } => var(e),
            Expr::Binary(op @ (BinOp::Add | BinOp::Sub), a, b, _) => {
                (var(a) && step(b)) || (*op == BinOp::Add && step(a) && var(b))
            }
            _ => false,
        }
    }

    fn whole_sources(&self, stmts: &[Stmt], sc: usize, out: &mut Vec<((usize, usize), Src)>) {
        for st in stmts {
            match st {
                Stmt::Assign { idx, value } => {
                    let i = *idx as usize;
                    if let (Expr::Lambda(def), 0) = (value, sc) {
                        let fs = self.by_def.get(&Rc::as_ptr(def)).copied();
                        if fs.is_some() && self.static_fn.get(&i).copied() == fs {
                            self.whole_sources(&def.body, fs.unwrap(), out);
                            continue;
                        }
                    }
                    out.push(((sc, i), Src::Expr(value, sc)));
                    self.whole_sources_expr(value, sc, out);
                }
                Stmt::AssignOuter { depth, idx, value } => {
                    out.push(((self.resolve(sc, *depth), *idx as usize), Src::Expr(value, sc)));
                    self.whole_sources_expr(value, sc, out);
                }
                Stmt::Learn { idx, value, .. } => {
                    out.push(((sc, *idx as usize), Src::Other));
                    self.whole_sources_expr(value, sc, out);
                }
                Stmt::SetPath { depth, idx, path, value, .. } => {
                    let t = self.resolve(sc, *depth);
                    out.push(((t, *idx as usize), Src::Other));
                    // a number stored at one position of a table (anything else
                    // means it isn't a table)
                    if t == 0 {
                        let src = match &path[..] {
                            [Access::Index(es)] if es.len() == 1 && !matches!(es[0], Expr::Range(..)) => Src::Expr(value, sc),
                            _ => Src::Other,
                        };
                        out.push(((TABLE, *idx as usize), src));
                    }
                    self.whole_sources_expr(value, sc, out);
                    for a in path {
                        if let Access::Index(es) = a {
                            for e in es {
                                self.whole_sources_expr(e, sc, out);
                            }
                        }
                    }
                }
                Stmt::Show(es) => {
                    for e in es {
                        self.whole_sources_expr(e, sc, out);
                    }
                }
                Stmt::Give(e) => {
                    if let Some(e) = e {
                        if sc != 0 {
                            out.push(((sc, RET), Src::Expr(e, sc)));
                        }
                        self.whole_sources_expr(e, sc, out);
                    }
                }
                Stmt::Repeat { count, var, body, .. } => {
                    // the counter of `repeat ... as i` counts in steps of 1
                    if let Some(v) = var {
                        out.push(((sc, *v as usize), Src::Repeat(count, sc)));
                    }
                    self.whole_sources_expr(count, sc, out);
                    self.whole_sources(body, sc, out);
                }
                Stmt::Each { var, iter, body, .. } => {
                    out.push(((sc, *var as usize), Src::Each(iter, sc)));
                    self.whole_sources_expr(iter, sc, out);
                    self.whole_sources(body, sc, out);
                }
                Stmt::While { cond, body, .. } => {
                    self.whole_sources_expr(cond, sc, out);
                    self.whole_sources(body, sc, out);
                }
                Stmt::When { arms, otherwise } => {
                    for (c, b, _) in arms {
                        self.whole_sources_expr(c, sc, out);
                        self.whole_sources(b, sc, out);
                    }
                    if let Some(b) = otherwise {
                        self.whole_sources(b, sc, out);
                    }
                }
                Stmt::Descend { loss, rate, clip, .. } => {
                    self.whole_sources_expr(loss, sc, out);
                    self.whole_sources_expr(rate, sc, out);
                    if let Some(c) = clip {
                        self.whole_sources_expr(c, sc, out);
                    }
                }
                Stmt::Try { body, var, handler } => {
                    if let Some(v) = var {
                        out.push(((sc, *v as usize), Src::Other));
                    }
                    self.whole_sources(body, sc, out);
                    self.whole_sources(handler, sc, out);
                }
                Stmt::Expr(e) => self.whole_sources_expr(e, sc, out),
                Stmt::Stop(_) | Stmt::Skip(_) => {}
            }
        }
    }

    fn whole_sources_expr(&self, e: &Expr, sc: usize, out: &mut Vec<((usize, usize), Src)>) {
        match e {
            Expr::Interp(p) | Expr::List(p) => {
                for x in p {
                    self.whole_sources_expr(x, sc, out);
                }
            }
            Expr::Map(m) => {
                for (_, x) in m {
                    self.whole_sources_expr(x, sc, out);
                }
            }
            Expr::Neg(a, _) | Expr::Not(a, _) | Expr::Field(a, _, _) | Expr::BitNot(a, _) => {
                self.whole_sources_expr(a, sc, out)
            }
            Expr::Binary(_, a, b, _) | Expr::And(a, b, _) | Expr::Or(a, b, _) => {
                self.whole_sources_expr(a, sc, out);
                self.whole_sources_expr(b, sc, out);
            }
            Expr::Call(f, args, _) => {
                for a in args {
                    self.whole_sources_expr(a, sc, out);
                }
                match self.callee(f, sc) {
                    Callee::Static(s) => {
                        if args.len() == self.scopes[s].nparams && !self.scopes[s].escapes {
                            for (i, a) in args.iter().enumerate() {
                                out.push(((s, i), Src::Arg(a, sc)));
                            }
                        }
                    }
                    Callee::Builtin(_) => {}
                    Callee::Dynamic => self.whole_sources_expr(f, sc, out),
                }
            }
            Expr::Index(b, idx, _) => {
                self.whole_sources_expr(b, sc, out);
                for x in idx {
                    self.whole_sources_expr(x, sc, out);
                }
            }
            Expr::Range(a, z, _) => {
                for x in [a, z].into_iter().flatten() {
                    self.whole_sources_expr(x, sc, out);
                }
            }
            Expr::Lambda(def) => {
                if let Some(&s) = self.by_def.get(&Rc::as_ptr(def)) {
                    self.whole_sources(&def.body, s, out);
                }
            }
            Expr::Ask { model, prompt, .. } => {
                self.whole_sources_expr(model, sc, out);
                self.whole_sources_expr(prompt, sc, out);
            }
            Expr::ModelNew { spec, opts, .. } => {
                self.whole_sources_expr(spec, sc, out);
                if let Some(o) = opts {
                    self.whole_sources_expr(o, sc, out);
                }
            }
            _ => {}
        }
    }

    /// The expressions a function body's value can come from at its end.
    fn tail_exprs<'e>(stmts: &'e [Stmt], out: &mut Vec<&'e Expr>) {
        match stmts.last() {
            Some(Stmt::Expr(e)) => out.push(e),
            Some(Stmt::When { arms, otherwise }) => {
                for (_, b, _) in arms {
                    Self::tail_exprs(b, out);
                }
                if let Some(b) = otherwise {
                    Self::tail_exprs(b, out);
                }
            }
            Some(Stmt::Try { body, handler, .. }) => {
                Self::tail_exprs(body, out);
                Self::tail_exprs(handler, out);
            }
            _ => {}
        }
    }

    /// Works out the whole numbers each integer variable, each result of a
    /// function called by name and each integer table can hold, so positions
    /// that are certainly inside a table need no check (see `in_table`).
    ///
    /// Every way of getting a value counts: assignments, inputs at every call,
    /// loop counters, the 0 a variable holds before its first assignment and
    /// the 0 a function gives back after a problem (see `rt::fault`). Anything
    /// not understood is "any number".
    fn find_ranges(&mut self, top: &[Stmt]) {
        let statics: Vec<usize> = self.static_fn.values().copied().collect();
        self.find_nonzero(top, 0, &[]);
        let mut srcs = vec![];
        self.whole_sources(top, 0, &mut srcs);
        let mut st: HashMap<(usize, usize), Rg> = HashMap::new();
        for t in 0..self.scopes.len() {
            let sc = &self.scopes[t];
            let named = statics.contains(&t) && !sc.escapes;
            for i in 0..sc.vars.len() {
                if !matches!(sc.vars[i], Ty::Int | Ty::Whole) || sc.captured[i] {
                    continue;
                }
                if t == 0
                    && (self.is_builtin(i) || self.static_fn.contains_key(&i) || self.consts.contains_key(&i) || self.iconsts.contains_key(&i))
                {
                    continue;
                }
                if t != 0 && i < sc.nparams {
                    // inputs are known only for functions called by name
                    if named {
                        st.insert((t, i), Rg::Bot);
                    }
                    continue;
                }
                st.insert((t, i), Rg::In(0, 0));
            }
            if named && matches!(sc.ret, Ty::Int | Ty::Whole) {
                st.insert((t, RET), Rg::In(0, 0));
                let def: &FnDef = unsafe { &*sc.def };
                let mut tails = vec![];
                Self::tail_exprs(&def.body, &mut tails);
                for e in tails {
                    srcs.push(((t, RET), Src::Expr(e, t)));
                }
            }
        }
        for (&i, &(_, int)) in &self.tables {
            if int {
                let init = self
                    .table_init
                    .get(&i)
                    .map_or(Rg::In(0, 0), |v| v.iter().fold(Rg::In(0, 0), |r, &x| r.join(Rg::In(x, x))));
                st.insert((TABLE, i), init);
            }
        }
        // inputs checked with `x within lo to hi`: checked when calling, so the
        // body knows the range (see `input_checks`)
        let mut fixed = std::collections::HashSet::new();
        for &s in &statics {
            for (k, lo, hi, _, _) in self.input_checks(s).0 {
                if st.contains_key(&(s, k)) {
                    if let Some((lo, hi)) = self.within_bounds(unsafe { &*lo }, unsafe { &*hi }, s) {
                        st.insert((s, k), if lo <= hi { Rg::In(lo, hi) } else { Rg::Bot });
                        fixed.insert((s, k));
                    }
                }
            }
        }
        self.ranges = st;
        let mut round = 0;
        loop {
            let mut changed = false;
            for (v, src) in &srcs {
                let Some(&old) = self.ranges.get(v) else { continue };
                if old == Rg::Any || (matches!(src, Src::Arg(..)) && fixed.contains(v)) {
                    continue;
                }
                let r = match *src {
                    Src::Expr(e, sc) | Src::Arg(e, sc) => self.erange(unsafe { &*e }, sc),
                    Src::Each(iter, sc) => self.each_range(unsafe { &*iter }, sc),
                    // `repeat n as i` counts from 1 to n
                    Src::Repeat(count, sc) => match self.erange(unsafe { &*count }, sc) {
                        Rg::In(_, hi) => Rg::In(1, hi.max(1)),
                        r => r,
                    },
                    Src::Other => Rg::Any,
                };
                // a value checked with within(..., lo, hi) never grows past lo to hi
                let cap = match *src {
                    Src::Expr(e, sc) => self.within_cap(unsafe { &*e }, sc),
                    _ => None,
                };
                changed |= self.widen_into(*v, r, round, cap);
            }
            if !changed {
                break;
            }
            round += 1;
        }
        if std::env::var("ANTARES_EXPLAIN").is_ok() {
            let mut known: Vec<_> = self.ranges.iter().filter(|(_, r)| matches!(r, Rg::In(..))).collect();
            known.sort_by_key(|(k, _)| **k);
            for (&(t, i), r) in known {
                let what = if t == TABLE {
                    format!("the numbers in table {}", self.gnames.get(i).cloned().unwrap_or_default())
                } else if i == RET {
                    format!("what {} gives back", self.scopes[t].name)
                } else if t == 0 {
                    self.gnames.get(i).cloned().unwrap_or_default()
                } else {
                    format!("variable {} of {}", i, self.scopes[t].name)
                };
                if let Rg::In(lo, hi) = r {
                    eprintln!("explain: {} are from {} to {}", what, lo, hi);
                }
            }
        }
    }

    /// Adds `r` to what `v` can hold; a range still growing after many rounds
    /// grows without end. Whether anything changed.
    fn widen_into(&mut self, v: (usize, usize), r: Rg, round: usize, cap: Option<Rg>) -> bool {
        let Some(&old) = self.ranges.get(&v) else { return false };
        let mut new = old.join(r);
        if new != old {
            match cap {
                Some(c) if round > 2 => new = new.join(c),
                _ if round > 24 => new = Rg::Any,
                _ => {}
            }
        }
        if new != old {
            self.ranges.insert(v, new);
            true
        } else {
            false
        }
    }

    /// For within(x, lo, hi, ...) with known lo and hi: the range nothing it
    /// gives back can be outside.
    fn within_cap(&self, e: &Expr, sc: usize) -> Option<Rg> {
        let Expr::Call(f, args, _) = e else { return None };
        let Callee::Builtin(b) = self.callee(f, sc) else { return None };
        if self.bnames[b] != "within" || !(3..=4).contains(&args.len()) {
            return None;
        }
        self.within_bounds(&args[1], &args[2], sc).map(|(lo, hi)| Rg::In(lo, hi))
    }

    /// The numbers the counter of `each x in range(...)` takes.
    fn each_range(&self, iter: &Expr, sc: usize) -> Rg {
        let Some((start, end, step)) = self.range_parts(iter, sc) else { return Rg::Any };
        let a = match start {
            Some(x) => self.erange(x, sc),
            None => Rg::In(0, 0),
        };
        let z = self.erange(end, sc);
        if step > 0.0 {
            // from the start up to (not including) the end
            a.with(z, |lo, _, _, d| Some((lo, d.checked_sub(1)?)))
        } else {
            // from the start down to (not including) the end
            a.with(z, |_, hi, c, _| Some((c.checked_add(1)?, hi)))
        }
    }

    /// The whole numbers integer expression `e` can have (see `find_ranges`).
    fn erange(&self, e: &Expr, sc: usize) -> Rg {
        if let Some(c) = self.const_of(e, sc) {
            return if c == c.trunc() && c.abs() <= 9.0e15 { Rg::In(c as i64, c as i64) } else { Rg::Any };
        }
        match e {
            Expr::Int(n) => Rg::In(*n, *n),
            Expr::Var { depth, idx, .. } => {
                let t = self.resolve(sc, *depth);
                let i = *idx as usize;
                if t == 0 {
                    if let Some(&c) = self.iconsts.get(&i) {
                        return Rg::In(c, c);
                    }
                }
                self.ranges.get(&(t, i)).copied().unwrap_or(Rg::Any)
            }
            Expr::Neg(a, _) => self.erange(a, sc).with(Rg::In(0, 0), |lo, hi, _, _| Some((hi.checked_neg()?, lo.checked_neg()?))),
            Expr::Binary(op, a, b, _) => {
                let (x, y) = (self.erange(a, sc), self.erange(b, sc));
                let bot = x == Rg::Bot || y == Rg::Bot;
                match op {
                    BinOp::Add => x.with(y, |a, b, c, d| Some((a.checked_add(c)?, b.checked_add(d)?))),
                    BinOp::Sub => x.with(y, |a, b, c, d| Some((a.checked_sub(d)?, b.checked_sub(c)?))),
                    BinOp::Mul => x.with(y, |a, b, c, d| {
                        let p = [a.checked_mul(c)?, a.checked_mul(d)?, b.checked_mul(c)?, b.checked_mul(d)?];
                        Some((*p.iter().min()?, *p.iter().max()?))
                    }),
                    // remainders are never negative (as in rem_euclid)
                    BinOp::Mod => match (x, self.const_of(b, sc)) {
                        (Rg::In(..), Some(c)) if c == c.trunc() && (1.0..=9.0e15).contains(&c) => Rg::In(0, c as i64 - 1),
                        (Rg::Bot, _) => Rg::Bot,
                        _ => Rg::Any,
                    },
                    BinOp::BitAnd => match (x.nonneg(), y.nonneg()) {
                        (Some(p), Some(q)) => Rg::In(0, p.min(q)),
                        (Some(p), None) | (None, Some(p)) => Rg::In(0, p),
                        _ if bot => Rg::Bot,
                        _ => Rg::Any,
                    },
                    BinOp::BitOr | BinOp::BitXor => match (x.nonneg(), y.nonneg()) {
                        (Some(p), Some(q)) => {
                            let m = p.max(q) as u64;
                            Rg::In(0, ((m + 1).next_power_of_two() - 1).min(i64::MAX as u64) as i64)
                        }
                        _ if bot => Rg::Bot,
                        _ => Rg::Any,
                    },
                    // shifts move the bits as an unsigned number; 64 or more gives 0
                    BinOp::Shr => match (x, y) {
                        _ if bot => Rg::Bot,
                        (Rg::In(lo, hi), Rg::In(c, d)) if c == d && (0..64).contains(&c) && lo >= 0 => Rg::In(lo >> c, hi >> c),
                        (_, Rg::In(c, d)) if c == d && (1..64).contains(&c) => Rg::In(0, (u64::MAX >> c) as i64),
                        _ => Rg::Any,
                    },
                    BinOp::Shl => match (x, y) {
                        _ if bot => Rg::Bot,
                        (Rg::In(lo, hi), Rg::In(c, d)) if c == d && (0..63).contains(&c) && lo >= 0 && hi <= (i64::MAX >> c) => {
                            Rg::In(lo << c, hi << c)
                        }
                        _ => Rg::Any,
                    },
                    _ => Rg::Any,
                }
            }
            Expr::Call(f, args, _) => match self.callee(f, sc) {
                Callee::Builtin(i) => match (self.bnames[i], args.len()) {
                    _ if self.nonzero_bits.contains(&(e as *const Expr)) => Rg::In(0, 63),
                    ("within", 3 | 4) if matches!(self.ty(&args[0], sc), Ty::Int | Ty::Whole) => match self.within_bounds(&args[1], &args[2], sc) {
                        Some((lo, hi)) => match self.erange(&args[0], sc) {
                            Rg::Bot => Rg::Bot,
                            Rg::Any => Rg::In(lo, hi),
                            // what gets past the check
                            Rg::In(a, b) if a.max(lo) <= b.min(hi) => Rg::In(a.max(lo), b.min(hi)),
                            Rg::In(..) => Rg::Bot,
                        },
                        None => Rg::Any,
                    },
                    ("lowest_bit", 1) | ("popcount", 1) => Rg::In(0, 64),
                    ("highest_bit", 1) => Rg::In(-1, 63),
                    // a whole number stays the same number
                    ("int", 1) => match self.erange(&args[0], sc) {
                        Rg::In(lo, hi) => Rg::In(lo, hi),
                        r => r,
                    },
                    ("abs", 1) => self.erange(&args[0], sc).with(Rg::In(0, 0), |lo, hi, _, _| {
                        let (a, b) = (lo.checked_abs()?, hi.checked_abs()?);
                        Some((if lo <= 0 && hi >= 0 { 0 } else { a.min(b) }, a.max(b)))
                    }),
                    ("min", 2) => self.erange(&args[0], sc).with(self.erange(&args[1], sc), |a, b, c, d| Some((a.min(c), b.min(d)))),
                    ("max", 2) => self.erange(&args[0], sc).with(self.erange(&args[1], sc), |a, b, c, d| Some((a.max(c), b.max(d)))),
                    _ => Rg::Any,
                },
                Callee::Static(s) if args.len() == self.scopes[s].nparams => self.ranges.get(&(s, RET)).copied().unwrap_or(Rg::Any),
                _ => Rg::Any,
            },
            Expr::Index(b, idx, _) if idx.len() == 1 && !matches!(idx[0], Expr::Range(..)) => match &**b {
                Expr::Var { depth, idx: vi, .. } if self.resolve(sc, *depth) == 0 => {
                    self.ranges.get(&(TABLE, *vi as usize)).copied().unwrap_or(Rg::Any)
                }
                _ => Rg::Any,
            },
            _ => Rg::Any,
        }
    }

    /// A local variable of scope `sc`, if `e` is one.
    fn local_var(&self, e: &Expr, sc: usize) -> Option<usize> {
        match e {
            Expr::Var { depth, idx, .. } if *depth == 0 && sc != 0 && !self.scopes[sc].captured[*idx as usize] => Some(*idx as usize),
            _ => None,
        }
    }

    /// For `a == 0` / `a != 0` (either way round): the local variable compared with 0.
    fn zero_test(&self, a: &Expr, b: &Expr, sc: usize) -> Option<usize> {
        if self.const_of(b, sc) == Some(0.0) {
            self.local_var(a, sc)
        } else if self.const_of(a, sc) == Some(0.0) {
            self.local_var(b, sc)
        } else {
            None
        }
    }

    /// The local variables of scope `sc` that `cond` being true shows aren't 0.
    fn nonzero_vars(&self, cond: &Expr, sc: usize, out: &mut Vec<usize>) {
        let local = |e: &Expr| self.local_var(e, sc);
        let c = |e: &Expr| self.const_of(e, sc);
        match cond {
            Expr::And(a, b, _) => {
                self.nonzero_vars(a, sc, out);
                self.nonzero_vars(b, sc, out);
            }
            Expr::Binary(op, a, b, _) => {
                let v = match (op, local(a), local(b)) {
                    (BinOp::Ne, _, _) => self.zero_test(a, b, sc),
                    (BinOp::Gt, Some(v), _) if c(b).is_some_and(|x| x >= 0.0) => Some(v),
                    (BinOp::Lt, Some(v), _) if c(b).is_some_and(|x| x <= 0.0) => Some(v),
                    (BinOp::Ge, Some(v), _) if c(b).is_some_and(|x| x > 0.0) => Some(v),
                    _ => None,
                };
                out.extend(v);
            }
            _ => {}
        }
    }

    /// The variables of scope `sc` that `stmts` can give a value.
    fn assigned_in(&self, stmts: &[Stmt], sc: usize, out: &mut Vec<usize>) {
        for st in stmts {
            match st {
                Stmt::Assign { idx, .. } | Stmt::Learn { idx, .. } => out.push(*idx as usize),
                Stmt::AssignOuter { depth, idx, .. } | Stmt::SetPath { depth, idx, .. } => {
                    if self.resolve(sc, *depth) == sc {
                        out.push(*idx as usize);
                    }
                }
                Stmt::Repeat { var, body, .. } => {
                    out.extend(var.map(|v| v as usize));
                    self.assigned_in(body, sc, out);
                }
                Stmt::Each { var, body, .. } => {
                    out.push(*var as usize);
                    self.assigned_in(body, sc, out);
                }
                Stmt::While { body, .. } => self.assigned_in(body, sc, out),
                Stmt::When { arms, otherwise } => {
                    for (_, b, _) in arms {
                        self.assigned_in(b, sc, out);
                    }
                    if let Some(b) = otherwise {
                        self.assigned_in(b, sc, out);
                    }
                }
                Stmt::Try { body, var, handler } => {
                    out.extend(var.map(|v| v as usize));
                    self.assigned_in(body, sc, out);
                    self.assigned_in(handler, sc, out);
                }
                _ => {}
            }
        }
    }

    /// Finds the `lowest_bit(x)` / `highest_bit(x)` calls whose `x` is a
    /// local variable certainly not 0 there: inside `while x != 0 { ... }` or
    /// `when x != 0 { ... }` (and the like) before `x` is changed, or after
    /// `when x == 0 { give ... }`. Those give a position from 0 to 63.
    fn find_nonzero(&mut self, stmts: &[Stmt], sc: usize, facts: &[usize]) {
        let mut facts: Vec<usize> = facts.to_vec();
        let drop = |facts: &mut Vec<usize>, gone: &[usize]| facts.retain(|v| !gone.contains(v));
        for st in stmts {
            match st {
                Stmt::While { cond, body, .. } => {
                    // the condition runs again after each round of the body
                    let mut gone = vec![];
                    self.assigned_in(body, sc, &mut gone);
                    drop(&mut facts, &gone);
                    self.nonzero_expr(cond, sc, &facts);
                    let mut inner = facts.clone();
                    self.nonzero_vars(cond, sc, &mut inner);
                    self.find_nonzero(body, sc, &inner);
                }
                Stmt::When { arms, otherwise } => {
                    let mut gone = vec![];
                    for (c, b, _) in arms {
                        self.nonzero_expr(c, sc, &facts);
                        let mut inner = facts.clone();
                        self.nonzero_vars(c, sc, &mut inner);
                        self.find_nonzero(b, sc, &inner);
                        self.assigned_in(b, sc, &mut gone);
                    }
                    if let Some(b) = otherwise {
                        self.find_nonzero(b, sc, &facts);
                        self.assigned_in(b, sc, &mut gone);
                    }
                    drop(&mut facts, &gone);
                    // `when x == 0 { give ... }` with nothing else: x isn't 0 after it
                    if let ([(Expr::Binary(BinOp::Eq, a, b, _), body, _)], None) = (&arms[..], otherwise) {
                        if matches!(body.last(), Some(Stmt::Give(_) | Stmt::Stop(_) | Stmt::Skip(_))) {
                            facts.extend(self.zero_test(a, b, sc));
                        }
                    }
                }
                Stmt::Repeat { count, var, body, .. } => {
                    self.nonzero_expr(count, sc, &facts);
                    let mut gone = vec![];
                    self.assigned_in(body, sc, &mut gone);
                    gone.extend(var.map(|v| v as usize));
                    drop(&mut facts, &gone);
                    self.find_nonzero(body, sc, &facts);
                }
                Stmt::Each { var, iter, body, .. } => {
                    self.nonzero_expr(iter, sc, &facts);
                    let mut gone = vec![*var as usize];
                    self.assigned_in(body, sc, &mut gone);
                    drop(&mut facts, &gone);
                    self.find_nonzero(body, sc, &facts);
                }
                Stmt::Try { body, var, handler } => {
                    let mut gone = vec![];
                    self.assigned_in(std::slice::from_ref(st), sc, &mut gone);
                    gone.extend(var.map(|v| v as usize));
                    drop(&mut facts, &gone);
                    self.find_nonzero(body, sc, &facts);
                    self.find_nonzero(handler, sc, &facts);
                }
                Stmt::Assign { idx, value } => {
                    if let (Expr::Lambda(def), 0) = (value, sc) {
                        if let Some(&fs) = self.by_def.get(&Rc::as_ptr(def)) {
                            if self.static_fn.get(&(*idx as usize)).copied() == Some(fs) {
                                self.find_nonzero(&def.body, fs, &[]);
                                continue;
                            }
                        }
                    }
                    self.nonzero_expr(value, sc, &facts);
                    drop(&mut facts, &[*idx as usize]);
                }
                Stmt::AssignOuter { value, .. } | Stmt::Learn { value, .. } => {
                    self.nonzero_expr(value, sc, &facts);
                    let mut gone = vec![];
                    self.assigned_in(std::slice::from_ref(st), sc, &mut gone);
                    drop(&mut facts, &gone);
                }
                Stmt::SetPath { path, value, .. } => {
                    self.nonzero_expr(value, sc, &facts);
                    for a in path {
                        if let Access::Index(es) = a {
                            for e in es {
                                self.nonzero_expr(e, sc, &facts);
                            }
                        }
                    }
                    let mut gone = vec![];
                    self.assigned_in(std::slice::from_ref(st), sc, &mut gone);
                    drop(&mut facts, &gone);
                }
                Stmt::Show(es) => {
                    for e in es {
                        self.nonzero_expr(e, sc, &facts);
                    }
                }
                Stmt::Give(Some(e)) | Stmt::Expr(e) => self.nonzero_expr(e, sc, &facts),
                Stmt::Descend { loss, rate, .. } => {
                    self.nonzero_expr(loss, sc, &facts);
                    self.nonzero_expr(rate, sc, &facts);
                }
                Stmt::Give(None) | Stmt::Stop(_) | Stmt::Skip(_) => {}
            }
        }
    }

    fn nonzero_expr(&mut self, e: &Expr, sc: usize, facts: &[usize]) {
        match e {
            Expr::Call(f, args, _) => {
                if let (Callee::Builtin(i), [Expr::Var { depth: 0, idx, .. }]) = (self.callee(f, sc), &args[..]) {
                    if matches!(self.bnames[i], "lowest_bit" | "highest_bit") && facts.contains(&(*idx as usize)) {
                        self.nonzero_bits.insert(e as *const Expr);
                    }
                }
                self.nonzero_expr(f, sc, facts);
                for a in args {
                    self.nonzero_expr(a, sc, facts);
                }
            }
            Expr::Interp(p) | Expr::List(p) => {
                for x in p {
                    self.nonzero_expr(x, sc, facts);
                }
            }
            Expr::Map(m) => {
                for (_, x) in m {
                    self.nonzero_expr(x, sc, facts);
                }
            }
            Expr::Neg(a, _) | Expr::Not(a, _) | Expr::Field(a, _, _) | Expr::BitNot(a, _) => self.nonzero_expr(a, sc, facts),
            Expr::Binary(_, a, b, _) => {
                self.nonzero_expr(a, sc, facts);
                self.nonzero_expr(b, sc, facts);
            }
            // the right side runs only when the left one allows it
            Expr::And(a, b, _) => {
                self.nonzero_expr(a, sc, facts);
                let mut inner = facts.to_vec();
                self.nonzero_vars(a, sc, &mut inner);
                self.nonzero_expr(b, sc, &inner);
            }
            Expr::Or(a, b, _) => {
                self.nonzero_expr(a, sc, facts);
                self.nonzero_expr(b, sc, facts);
            }
            Expr::Index(b, idx, _) => {
                self.nonzero_expr(b, sc, facts);
                for x in idx {
                    self.nonzero_expr(x, sc, facts);
                }
            }
            Expr::Range(a, z, _) => {
                for x in [a, z].into_iter().flatten() {
                    self.nonzero_expr(x, sc, facts);
                }
            }
            Expr::Lambda(def) => {
                if let Some(&s) = self.by_def.get(&Rc::as_ptr(def)) {
                    self.find_nonzero(&def.body, s, &[]);
                }
            }
            Expr::Ask { model, prompt, .. } => {
                self.nonzero_expr(model, sc, facts);
                self.nonzero_expr(prompt, sc, facts);
            }
            Expr::ModelNew { spec, opts, .. } => {
                self.nonzero_expr(spec, sc, facts);
                if let Some(o) = opts {
                    self.nonzero_expr(o, sc, facts);
                }
            }
            _ => {}
        }
    }

    /// The range of within(x, lo, hi, ...) when lo and hi are known: the whole
    /// numbers from lo to hi.
    fn within_bounds(&self, lo: &Expr, hi: &Expr, sc: usize) -> Option<(i64, i64)> {
        let (lo, hi) = (self.fold_const(lo, sc)?, self.fold_const(hi, sc)?);
        let (lo, hi) = (lo.ceil(), hi.floor());
        (lo.abs() < 9.0e15 && hi.abs() < 9.0e15).then_some((lo as i64, hi as i64))
    }

    /// within(x, lo, hi [, name]) for a number x: x, after a check that is left
    /// out when x is certainly in range.
    fn within_code(&mut self, args: &[Expr], sc: usize, line: u32) -> (String, Ty) {
        let int = self.ty(&args[0], sc) == Ty::Int;
        let proven = int
            && matches!((self.erange(&args[0], sc), self.within_bounds(&args[1], &args[2], sc)), (Rg::In(a, b), Some((lo, hi))) if a >= lo && b <= hi);
        let (x, t) = if int { (self.int_code(&args[0], sc, line), Ty::Int) } else { (self.num_code(&args[0], sc), Ty::Num) };
        let lo = self.num_code(&args[1], sc);
        let hi = self.num_code(&args[2], sc);
        let name = match args.get(3) {
            Some(Expr::Str(n)) => format!("{:?}", &**n),
            Some(e) => {
                let c = self.dyn_code(e, sc);
                format!("&{{ let n_ = {}; n_.show() }}", c)
            }
            None => "\"\"".to_string(),
        };
        if proven {
            // the bounds and name are still worked out, as when checking
            return (x, t);
        }
        let f = if int { "within_i" } else { "within_n" };
        (format!("{}({}, {}, {}, {}, {})?", f, x, lo, hi, name, line), t)
    }

    /// The inputs of function `s` (called by name) checked first with
    /// `x within lo to hi`: (input, lowest, highest, name, line), and how many
    /// statements at the start of the body do that.
    fn input_checks(&self, s: usize) -> (Vec<(usize, *const Expr, *const Expr, String, u32)>, usize) {
        let mut out = vec![];
        if self.scopes[s].escapes || !self.static_fn.values().any(|&f| f == s) {
            return (out, 0);
        }
        let def: &FnDef = unsafe { &*self.scopes[s].def };
        let mut n = 0;
        for st in &def.body {
            let Stmt::Assign { idx, value: Expr::Call(f, args, line) } = st else { break };
            let k = *idx as usize;
            let Callee::Builtin(b) = self.callee(f, s) else { break };
            match (&args[..], self.bnames[b]) {
                ([Expr::Var { depth: 0, idx: v, .. }, lo, hi, Expr::Str(name)], "within")
                    if *v as usize == k
                        && k < self.scopes[s].nparams
                        && !self.scopes[s].captured[k]
                        && self.fold_const(lo, s).is_some()
                        && self.fold_const(hi, s).is_some() =>
                {
                    out.push((k, lo as *const Expr, hi as *const Expr, name.to_string(), *line));
                    n += 1;
                }
                _ => break,
            }
        }
        (out, n)
    }

    /// Whether position `ie` is certainly inside fixed table `t`.
    fn in_table(&self, ie: &Expr, sc: usize, t: usize) -> bool {
        let Some(&(len, _)) = self.tables.get(&t) else { return false };
        matches!(self.erange(ie, sc), Rg::In(lo, hi) if lo >= 0 && (hi as u64) < len as u64)
    }

    /// Finds the number variables that only ever hold whole numbers counted
    /// in small steps, and keeps them as machine integers.
    fn infer_whole(&mut self, top: &[Stmt]) {
        let statics: Vec<usize> = self.static_fn.values().copied().collect();
        let mut w = std::collections::HashSet::new();
        for t in 0..self.scopes.len() {
            let sc = &self.scopes[t];
            let named = statics.contains(&t) && !sc.escapes;
            for i in 0..sc.vars.len() {
                if sc.vars[i] != Ty::Num || sc.captured[i] {
                    continue;
                }
                if t == 0 && (self.is_builtin(i) || self.static_fn.contains_key(&i) || self.consts.contains_key(&i)) {
                    continue;
                }
                // inputs are known only for functions called by name
                if t != 0 && i < sc.nparams && !named {
                    continue;
                }
                w.insert((t, i));
            }
        }
        let mut srcs = vec![];
        self.whole_sources(top, 0, &mut srcs);
        loop {
            let mut changed = false;
            for (v, src) in &srcs {
                if !w.contains(v) {
                    continue;
                }
                let ok = match *src {
                    Src::Expr(e, sc) | Src::Arg(e, sc) => self.counts(unsafe { &*e }, sc, &w),
                    Src::Each(iter, sc) => match self.range_parts(unsafe { &*iter }, sc) {
                        Some((start, _, step)) => {
                            step == step.trunc() && step.abs() <= 64.0 && start.is_none_or(|x| self.counts(x, sc, &w))
                        }
                        None => false,
                    },
                    Src::Repeat(..) => true,
                    Src::Other => false,
                };
                if !ok {
                    w.remove(v);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let explain = std::env::var("ANTARES_EXPLAIN").is_ok();
        for &(t, i) in &w {
            self.scopes[t].vars[i] = Ty::Whole;
            if explain {
                let name = if t == 0 { self.gnames.get(i).cloned().unwrap_or_default() } else { format!("variable {}", i) };
                eprintln!("explain: in {}, {} counts in whole steps, so it is kept as a machine integer", self.scopes[t].name, name);
            }
        }
    }

    /// A whole number known when compiling: a number, or arithmetic on
    /// top-level variables given one such number earlier in the program (before
    /// top-level statement `k`).
    fn const_size(&self, e: &Expr, k: usize, first: &HashMap<usize, (usize, *const Expr)>, depth: u32) -> Option<f64> {
        if depth > 32 {
            return None;
        }
        let r = match e {
            Expr::Num(n) => *n,
            Expr::Int(n) => *n as f64,
            Expr::Neg(a, _) => -self.const_size(a, k, first, depth + 1)?,
            Expr::Binary(op @ (BinOp::Add | BinOp::Sub | BinOp::Mul), a, b, _) => {
                let (x, y) = (self.const_size(a, k, first, depth + 1)?, self.const_size(b, k, first, depth + 1)?);
                match op {
                    BinOp::Add => x + y,
                    BinOp::Sub => x - y,
                    _ => x * y,
                }
            }
            Expr::Var { depth: 0, idx, .. } => {
                let i = *idx as usize;
                if self.gcount[i] != 1 {
                    return None;
                }
                let &(k2, e2) = first.get(&i)?;
                if k2 >= k {
                    return None;
                }
                self.const_size(unsafe { &*e2 }, k2, first, depth + 1)?
            }
            _ => return None,
        };
        (r == r.trunc() && r.abs() < 1.0e15).then_some(r)
    }

    /// Notes the top-level variables used in `stmts` (in `refs`), and those used
    /// other than as `name[i]` or `name[i] = v` (in `bad`).
    fn table_uses(&self, stmts: &[Stmt], sc: usize, bad: &mut std::collections::HashSet<usize>, refs: &mut BTreeSet<usize>) {
        for st in stmts {
            match st {
                Stmt::Expr(e) | Stmt::Assign { value: e, .. } => self.table_uses_expr(e, sc, bad, refs),
                Stmt::AssignOuter { depth, idx, value } => {
                    if self.resolve(sc, *depth) == 0 {
                        refs.insert(*idx as usize);
                        if self.reset_kind(value, sc).is_none() {
                            bad.insert(*idx as usize);
                        }
                    }
                    self.table_uses_expr(value, sc, bad, refs);
                }
                Stmt::Learn { idx, value, .. } => {
                    if sc == 0 {
                        bad.insert(*idx as usize);
                    }
                    self.table_uses_expr(value, sc, bad, refs);
                }
                Stmt::SetPath { depth, idx, path, value, .. } => {
                    if self.resolve(sc, *depth) == 0 {
                        refs.insert(*idx as usize);
                        if self.fast_set(path).is_none() {
                            bad.insert(*idx as usize);
                        }
                    }
                    self.table_uses_expr(value, sc, bad, refs);
                    for a in path {
                        if let Access::Index(es) = a {
                            for e in es {
                                self.table_uses_expr(e, sc, bad, refs);
                            }
                        }
                    }
                }
                Stmt::Show(es) => {
                    for e in es {
                        self.table_uses_expr(e, sc, bad, refs);
                    }
                }
                Stmt::Give(e) => {
                    if let Some(e) = e {
                        self.table_uses_expr(e, sc, bad, refs);
                    }
                }
                Stmt::Repeat { count, body, .. } => {
                    self.table_uses_expr(count, sc, bad, refs);
                    self.table_uses(body, sc, bad, refs);
                }
                Stmt::Each { iter, body, .. } => {
                    self.table_uses_expr(iter, sc, bad, refs);
                    self.table_uses(body, sc, bad, refs);
                }
                Stmt::While { cond, body, .. } => {
                    self.table_uses_expr(cond, sc, bad, refs);
                    self.table_uses(body, sc, bad, refs);
                }
                Stmt::When { arms, otherwise } => {
                    for (c, b, _) in arms {
                        self.table_uses_expr(c, sc, bad, refs);
                        self.table_uses(b, sc, bad, refs);
                    }
                    if let Some(b) = otherwise {
                        self.table_uses(b, sc, bad, refs);
                    }
                }
                Stmt::Descend { loss, rate, .. } => {
                    self.table_uses_expr(loss, sc, bad, refs);
                    self.table_uses_expr(rate, sc, bad, refs);
                }
                Stmt::Try { body, handler, .. } => {
                    self.table_uses(body, sc, bad, refs);
                    self.table_uses(handler, sc, bad, refs);
                }
                Stmt::Stop(_) | Stmt::Skip(_) => {}
            }
        }
    }

    fn table_uses_expr(&self, e: &Expr, sc: usize, bad: &mut std::collections::HashSet<usize>, refs: &mut BTreeSet<usize>) {
        match e {
            Expr::Var { depth, idx, .. } => {
                if self.resolve(sc, *depth) == 0 {
                    refs.insert(*idx as usize);
                    bad.insert(*idx as usize);
                }
            }
            Expr::Index(base, idx, _) => {
                if let (Expr::Var { depth, idx: vi, .. }, [ie]) = (&**base, &idx[..]) {
                    if self.resolve(sc, *depth) == 0 && !matches!(ie, Expr::Range(..)) {
                        refs.insert(*vi as usize);
                        self.table_uses_expr(ie, sc, bad, refs);
                        return;
                    }
                }
                self.table_uses_expr(base, sc, bad, refs);
                for x in idx {
                    self.table_uses_expr(x, sc, bad, refs);
                }
            }
            Expr::Interp(p) | Expr::List(p) => {
                for x in p {
                    self.table_uses_expr(x, sc, bad, refs);
                }
            }
            Expr::Map(m) => {
                for (_, x) in m {
                    self.table_uses_expr(x, sc, bad, refs);
                }
            }
            Expr::Neg(a, _) | Expr::Not(a, _) | Expr::Field(a, _, _) | Expr::BitNot(a, _) => self.table_uses_expr(a, sc, bad, refs),
            Expr::Binary(_, a, b, _) | Expr::And(a, b, _) | Expr::Or(a, b, _) => {
                self.table_uses_expr(a, sc, bad, refs);
                self.table_uses_expr(b, sc, bad, refs);
            }
            Expr::Call(f, args, _) => {
                self.table_uses_expr(f, sc, bad, refs);
                let reads_range = matches!(self.callee(f, sc), Callee::Builtin(b) if self.bnames[b] == "argmax") && args.len() == 3;
                for (k, a) in args.iter().enumerate() {
                    if let (true, 0, Expr::Var { depth, idx, .. }) = (reads_range, k, a) {
                        if self.resolve(sc, *depth) == 0 {
                            refs.insert(*idx as usize);
                            continue;
                        }
                    }
                    self.table_uses_expr(a, sc, bad, refs);
                }
            }
            Expr::Range(a, z, _) => {
                for x in [a, z].into_iter().flatten() {
                    self.table_uses_expr(x, sc, bad, refs);
                }
            }
            Expr::Ask { model, prompt, .. } => {
                self.table_uses_expr(model, sc, bad, refs);
                self.table_uses_expr(prompt, sc, bad, refs);
            }
            Expr::ModelNew { spec, opts, .. } => {
                self.table_uses_expr(spec, sc, bad, refs);
                if let Some(o) = opts {
                    self.table_uses_expr(o, sc, bad, refs);
                }
            }
            // a function made here is looked at as a scope of its own
            _ => {}
        }
    }

    /// Finds the top-level rows that can be fixed-size tables: given once, by a
    /// top-level `ints(n)` or `zeros(n)` with a size known when compiling, and
    /// otherwise only ever used as `name[i]` and `name[i] = v`, with nothing
    /// that runs before that line able to use them. Such a row is never shared
    /// and never changes length, so it needs no sharing check and its length is
    /// a constant.
    /// `ints(n)` or `zeros(n)`: whether it is ints, and the expression for n.
    /// `e` without a within(..., lo, hi, ...) around it whose check can't fail
    /// for a new table: its zeros (or its constant numbers) are in range.
    fn table_value<'e>(&self, e: &'e Expr, sc: usize) -> &'e Expr {
        let Expr::Call(f, args, _) = e else { return e };
        let Callee::Builtin(b) = self.callee(f, sc) else { return e };
        if self.bnames[b] != "within" || !(3..=4).contains(&args.len()) {
            return e;
        }
        let (Some(lo), Some(hi)) = (self.fold_const(&args[1], sc), self.fold_const(&args[2], sc)) else { return e };
        let inside = |x: f64| x >= lo && x <= hi;
        let ok = match &args[0] {
            Expr::Call(g, a, _) => match (self.callee(g, sc), &a[..]) {
                (Callee::Builtin(c), [Expr::List(items)]) if self.bnames[c] == "ints" => {
                    items.iter().all(|x| self.const_of(x, sc).is_some_and(inside) || matches!(x, Expr::Int(n) if inside(*n as f64)))
                }
                (Callee::Builtin(c), [_]) if matches!(self.bnames[c], "ints" | "zeros") => inside(0.0),
                _ => false,
            },
            _ => false,
        };
        if ok {
            &args[0]
        } else {
            e
        }
    }

    fn reset_kind<'e>(&self, e: &'e Expr, sc: usize) -> Option<(bool, &'e Expr)> {
        let e = self.table_value(e, sc);
        let Expr::Call(f, args, _) = e else { return None };
        let Callee::Builtin(b) = self.callee(f, sc) else { return None };
        match (self.bnames[b], &args[..]) {
            ("ints", [a]) => Some((true, a)),
            ("zeros", [a]) => Some((false, a)),
            _ => None,
        }
    }

    /// The size `e` (seen from scope `sc`) is sure to have, using top-level
    /// variables given a known number once.
    fn size_in(&self, e: &Expr, sc: usize, first: &HashMap<usize, (usize, *const Expr)>) -> Option<f64> {
        if let Some(c) = self.const_of(e, sc) {
            return Some(c);
        }
        match e {
            Expr::Int(n) => Some(*n as f64),
            Expr::Neg(a, _) => Some(-self.size_in(a, sc, first)?),
            Expr::Binary(op @ (BinOp::Add | BinOp::Sub | BinOp::Mul), a, b, _) => {
                let (x, y) = (self.size_in(a, sc, first)?, self.size_in(b, sc, first)?);
                Some(match op {
                    BinOp::Add => x + y,
                    BinOp::Sub => x - y,
                    _ => x * y,
                })
            }
            Expr::Var { depth, idx, .. } if self.resolve(sc, *depth) == 0 => {
                let i = *idx as usize;
                if self.gcount[i] != 1 {
                    return None;
                }
                let &(k2, e2) = first.get(&i)?;
                self.const_size(unsafe { &*e2 }, k2, first, 1)
            }
            _ => None,
        }
    }

    /// Every assignment to a top-level variable: the variable, and for
    /// `ints(n)` / `zeros(n)` with a known n, (n, whether ints).
    fn table_assigns(&self, stmts: &[Stmt], sc: usize, first: &HashMap<usize, (usize, *const Expr)>, out: &mut Vec<(usize, Option<(f64, bool)>)>) {
        for st in stmts {
            match st {
                Stmt::Assign { idx, value } if sc == 0 => {
                    if !matches!(value, Expr::Lambda(_)) {
                        let k = self.reset_kind(value, sc).and_then(|(int, a)| self.size_in(a, sc, first).map(|n| (n, int)));
                        out.push((*idx as usize, k));
                    }
                }
                Stmt::AssignOuter { depth, idx, value } if self.resolve(sc, *depth) == 0 => {
                    let k = self.reset_kind(value, sc).and_then(|(int, a)| self.size_in(a, sc, first).map(|n| (n, int)));
                    out.push((*idx as usize, k));
                }
                Stmt::Repeat { body, .. } | Stmt::Each { body, .. } | Stmt::While { body, .. } => self.table_assigns(body, sc, first, out),
                Stmt::When { arms, otherwise } => {
                    for (_, b, _) in arms {
                        self.table_assigns(b, sc, first, out);
                    }
                    if let Some(b) = otherwise {
                        self.table_assigns(b, sc, first, out);
                    }
                }
                Stmt::Try { body, handler, .. } => {
                    self.table_assigns(body, sc, first, out);
                    self.table_assigns(handler, sc, first, out);
                }
                _ => {}
            }
        }
    }

    fn find_tables(&mut self, top: &[Stmt]) {
        let n = self.scopes.len();
        let mut first: HashMap<usize, (usize, *const Expr)> = HashMap::new();
        for (k, st) in top.iter().enumerate() {
            if let Stmt::Assign { idx, value } = st {
                first.entry(*idx as usize).or_insert((k, value as *const Expr));
            }
        }
        // every assignment to a top-level variable, to allow resets with the same size
        let mut assigns = vec![];
        self.table_assigns(top, 0, &first, &mut assigns);
        for s in 1..n {
            let def: &FnDef = unsafe { &*self.scopes[s].def };
            self.table_assigns(&def.body, s, &first, &mut assigns);
        }
        // where each scope uses top-level variables, and which are used other ways
        let mut bad = std::collections::HashSet::new();
        let mut refs: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        for s in 1..n {
            let def: &FnDef = unsafe { &*self.scopes[s].def };
            let mut r = BTreeSet::new();
            self.table_uses(&def.body, s, &mut bad, &mut r);
            refs[s] = r;
        }
        let mut top_refs: Vec<BTreeSet<usize>> = vec![];
        for st in top {
            let mut r = BTreeSet::new();
            self.table_uses(std::slice::from_ref(st), 0, &mut bad, &mut r);
            top_refs.push(r);
        }
        // calls between functions called by name
        let mut calls: Vec<Scan> = (0..n).map(|_| Scan::default()).collect();
        for s in 1..n {
            let def: &FnDef = unsafe { &*self.scopes[s].def };
            self.scan_block(&def.body, s, &mut calls[s]);
        }
        // A call through a value can run a closure or a function passed around.
        let statics: Vec<usize> = self.static_fn.values().copied().collect();
        let through_values: Vec<usize> = (1..n).filter(|s| !statics.contains(s) || self.scopes[*s].escapes).collect();
        // Walk the top level in order, keeping what may have run so far.
        let mut used_so_far: BTreeSet<usize> = BTreeSet::new();
        let mut reached = vec![false; n];
        for (k, st) in top.iter().enumerate() {
            let made = match st {
                Stmt::Assign { idx, value } => Some((idx, self.table_value(value, 0))),
                _ => None,
            };
            if let Some((idx, Expr::Call(f, args, _))) = made {
                let i = *idx as usize;
                let mut init: Option<Vec<i64>> = None;
                let kind = match (self.callee(f, 0), &args[..]) {
                    (Callee::Builtin(b), [Expr::List(items)]) if self.bnames[b] == "ints" && !items.is_empty() => {
                        let whole = |e: &Expr| -> Option<i64> {
                            match e {
                                Expr::Int(n) => Some(*n),
                                Expr::Neg(x, _) if matches!(&**x, Expr::Int(_)) => {
                                    if let Expr::Int(n) = &**x { Some(n.wrapping_neg()) } else { None }
                                }
                                _ => self.const_of(e, 0).filter(|c| *c == c.trunc() && c.abs() < 9.0e15).map(|c| c as i64),
                            }
                        };
                        let vals: Option<Vec<i64>> = items.iter().map(whole).collect();
                        // given only this once
                        match vals {
                            Some(v) if self.gcount[i] == 1 => {
                                let len = v.len() as f64;
                                init = Some(v);
                                Some((len, true))
                            }
                            _ => None,
                        }
                    }
                    (Callee::Builtin(b), [a]) if matches!(self.bnames[b], "ints" | "zeros") => {
                        self.const_size(a, k, &first, 0).map(|len| (len, self.bnames[b] == "ints"))
                    }
                    _ => None,
                };
                if let Some((len, int)) = kind {
                    let ty_ok = if int { self.scopes[0].vars[i] == Ty::IArr } else { self.scopes[0].vars[i] == Ty::Arr };
                    let used_before = used_so_far.contains(&i);
                    if ty_ok
                        && len >= 1.0
                        && len <= (1u64 << 24) as f64
                        // given only by ints(n) / zeros(n) of this size: later ones reset it to zeros
                        && (init.is_some() || {
                            let mine: Vec<&Option<(f64, bool)>> = assigns.iter().filter(|a| a.0 == i).map(|a| &a.1).collect();
                            mine.len() == self.gcount[i] as usize && mine.iter().all(|a| **a == Some((len, int)))
                        })
                        && first.get(&i).map(|f| f.0) == Some(k)
                        && !bad.contains(&i)
                        && !self.scopes[0].captured[i]
                        && !used_before
                    {
                        self.tables.insert(i, (len as usize, int));
                        if let Some(v) = init.take() {
                            self.table_init.insert(i, v);
                        }
                    }
                }
            }
            // what this statement may run
            used_so_far.extend(top_refs[k].iter().copied());
            let mut sc = Scan::default();
            self.scan_block(std::slice::from_ref(st), 0, &mut sc);
            let mut stack = sc.calls.clone();
            if sc.unknown {
                stack.extend(through_values.iter().copied());
            }
            while let Some(t) = stack.pop() {
                if reached[t] {
                    continue;
                }
                reached[t] = true;
                used_so_far.extend(refs[t].iter().copied());
                stack.extend(calls[t].calls.iter().copied());
                if calls[t].unknown {
                    stack.extend(through_values.iter().copied());
                }
            }
        }
        if std::env::var("ANTARES_EXPLAIN").is_ok() {
            for (&i, &(len, _)) in &self.tables {
                eprintln!("explain: {} is a fixed table of {} numbers", self.gnames.get(i).cloned().unwrap_or_default(), len);
            }
        }
    }

    /// `t = ints(n)` / `zeros(n)` for a fixed table: all zeros again (nothing
    /// else can hold the old row, so this is the same as a new one).
    fn table_fill(&self, i: usize) -> String {
        let zero = if self.tables[&i].1 { "0i64" } else { "0.0f64" };
        format!("(*std::ptr::addr_of_mut!(TB{})).fill({});\n", i, zero)
    }

    /// The table a variable refers to, if it is one: (Rust expression for the
    /// table as a slice, integers?).
    fn table_of(&self, sc: usize, depth: u32, idx: u32) -> Option<(String, bool)> {
        if self.resolve(sc, depth) != 0 {
            return None;
        }
        let &(_, int) = self.tables.get(&(idx as usize))?;
        Some((format!("TB{}", idx), int))
    }

    /// Code for a position into a table: (i64 code, whether the position is an
    /// ordinary number) or the f64 code of a position that may not be whole.
    fn table_pos(&mut self, ie: &Expr, sc: usize, line: u32) -> Result<(String, bool), String> {
        if self.ty(ie, sc) == Ty::Int {
            return Ok((self.int_code(ie, sc, line), false));
        }
        if let Some(c) = self.icode(ie, sc, true) {
            return Ok((c, true));
        }
        Err(self.num_code(ie, sc))
    }

    /// The Rust place of a variable read in place (not captured, not a constant
    /// or a function), if `e` is one.
    fn plain_place(&self, e: &Expr, sc: usize) -> Option<String> {
        let Expr::Var { depth, idx, .. } = e else { return None };
        let t = self.resolve(sc, *depth);
        let i = *idx as usize;
        if t != 0 && self.scopes[t].captured[i] {
            return None;
        }
        if t == 0 && (self.consts.contains_key(&i) || self.iconsts.contains_key(&i) || self.is_builtin(i) || self.static_fn.contains_key(&i)) {
            return None;
        }
        Some(self.place(t, i))
    }

    /// A row position for the compact-matrix operations, as an integer or a
    /// number when its type is known (no general value is made for it).
    fn pk_key(&mut self, e: &Expr, sc: usize) -> String {
        if let Some(c) = self.icode(e, sc, false) {
            return c;
        }
        let (c, t) = self.expr(e, sc);
        match t {
            Ty::Int => c,
            Ty::Num | Ty::ILit => conv(c, t, Ty::Num),
            _ => conv(c, t, Ty::Dyn),
        }
    }

    /// A compact-row argument: a compact row (code, no row) or a row of a compact
    /// matrix variable (the matrix in place, and the row position).
    fn pk_arg(&mut self, e: &Expr, sc: usize) -> Option<(String, Option<String>)> {
        if let Expr::Index(base, idx, _) = e {
            if idx.len() == 1 && !matches!(idx[0], Expr::Range(..)) && self.ty(base, sc) == Ty::PkM && !calls_anything(&idx[0]) {
                let p = self.plain_place(base, sc)?;
                let k = self.pk_key(&idx[0], sc);
                return Some((format!("&{}", p), Some(k)));
            }
        }
        if self.ty(e, sc) == Ty::PkR {
            if let Some(p) = self.plain_place(e, sc) {
                return Some((format!("&{}", p), None));
            }
            if !calls_anything(e) {
                let (c, _) = self.expr(e, sc);
                return Some((format!("&{}", c), None));
            }
        }
        None
    }

    /// screlu_dot(x, w, cap) straight on compact rows.
    fn pk_screlu(&mut self, args: &[Expr], sc: usize, line: u32) -> Option<String> {
        if calls_anything(&args[2]) {
            return None;
        }
        let (x, xr) = self.pk_arg(&args[0], sc)?;
        let (w, wr) = self.pk_arg(&args[1], sc)?;
        let cap = self.num_code(&args[2], sc);
        let opt = |r: Option<String>| match r {
            Some(k) => format!("Some({})", k),
            None => "None::<i64>".to_string(),
        };
        Some(format!("pk_screlu({}, {}, {}, {}, {}, {})?", x, opt(xr), w, opt(wr), cap, line))
    }

    /// name[i] = v for a compact matrix or row variable.
    fn pk_set_stmt(&mut self, t: usize, i: usize, ie: &Expr, value: &Expr, sc: usize, line: u32) -> String {
        let place = self.place(t, i);
        let same = |e: &Expr, s: &Self| matches!(e, Expr::Var { depth, idx, .. } if s.resolve(sc, *depth) == t && *idx as usize == i);
        if self.var_ty(t, i) == Ty::PkM && !calls_anything(ie) {
            // m[d] = m[s]
            if let Expr::Index(b, ix, _) = value {
                if ix.len() == 1 && !matches!(ix[0], Expr::Range(..)) && same(b, self) && !calls_anything(&ix[0]) {
                    let s_ = self.pk_key(&ix[0], sc);
                    let d_ = self.pk_key(ie, sc);
                    return format!("{{ let s_ = {}; let d_ = {}; pk_copy_row(&mut {}, d_, s_, {})?; }}\n", s_, d_, place, line);
                }
            }
            // m[d] = update_rows(base, w, adds, n_add, subs, n_sub)
            if let Expr::Call(f, a, _) = value {
                if let Callee::Builtin(b) = self.callee(f, sc) {
                    if self.bnames[b] == "update_rows" && a.len() == 6 && !a.iter().any(calls_anything) {
                        let w = self.plain_place(&a[1], sc).filter(|_| self.ty(&a[1], sc) == Ty::PkM && !same(&a[1], self));
                        let adds = self.plain_place(&a[2], sc).filter(|_| self.ty(&a[2], sc) == Ty::Arr);
                        let subs = self.plain_place(&a[4], sc).filter(|_| self.ty(&a[4], sc) == Ty::Arr);
                        let num = |e: &Expr, s: &Self| matches!(s.ty(e, sc), Ty::Num | Ty::Int);
                        if let (Some(w), Some(adds), Some(subs), true, true) = (w, adds, subs, num(&a[3], self), num(&a[5], self)) {
                            let from_row = match &a[0] {
                                Expr::Index(b, ix, _) if ix.len() == 1 && !matches!(ix[0], Expr::Range(..)) && same(b, self) => Some(&ix[0]),
                                _ => None,
                            };
                            if let Some(srow) = from_row {
                                let s_ = self.pk_key(srow, sc);
                                let na = self.num_code(&a[3], sc);
                                let ns = self.num_code(&a[5], sc);
                                let d_ = self.pk_key(ie, sc);
                                return format!(
                                    "{{ let s_ = {}; let na_ = {}; let ns_ = {}; let d_ = {}; pk_update_from(&mut {}, d_, s_, &{}, &{}, na_, &{}, ns_, {})?; }}\n",
                                    s_, na, ns, d_, place, w, adds, subs, line
                                );
                            }
                            if self.ty(&a[0], sc) == Ty::PkR {
                                // a compact-row variable other than the target is
                                // read in place; anything else is worked out first
                                let (bc, bref) = match self.plain_place(&a[0], sc).filter(|_| !same(&a[0], self)) {
                                    Some(p) => (String::from("()"), format!("&{}", p)),
                                    None => (self.expr(&a[0], sc).0, String::from("&b_")),
                                };
                                let na = self.num_code(&a[3], sc);
                                let ns = self.num_code(&a[5], sc);
                                let d_ = self.pk_key(ie, sc);
                                return format!(
                                    "{{ let b_ = {}; let na_ = {}; let ns_ = {}; let d_ = {}; pk_update_with(&mut {}, d_, {}, &{}, &{}, na_, &{}, ns_, {})?; }}\n",
                                    bc, na, ns, d_, place, bref, w, adds, subs, line
                                );
                            }
                        }
                    }
                }
            }
        }
        let v = self.dyn_code(value, sc);
        let k = self.dyn_code(ie, sc);
        format!("{{ let v_ = {}; let k_ = {}; pk_set(&mut {}, k_, v_, {})?; }}\n", v, k, place, line)
    }

    /// The depth to give a function called by name.
    fn depth_arg(&self) -> &'static str {
        if self.cur.is_some() {
            "d_ + 1"
        } else {
            "depth() + 1"
        }
    }

    /// A call of function `s` (already written as `call`), handling its problems.
    fn static_call(&self, s: usize, call: String) -> String {
        if !self.inf.get(s).copied().unwrap_or(false) {
            return format!("{}?", call);
        }
        let caller_prop = self.cur.is_some_and(|c| self.prop.get(c).copied().unwrap_or(false));
        if self.prop[s] && !caller_prop {
            format!("checked({})?", call)
        } else {
            call
        }
    }

    /// Code that may call back into closures: inside a function called by
    /// name, the shared depth counter is set to that function's depth first.
    fn callback(&self, code: String) -> String {
        if self.cur.is_some() {
            format!("{{ let _g = at_depth(d_); {} }}", code)
        } else {
            code
        }
    }

    // ---------- what functions do ----------

    fn scan_block(&self, stmts: &[Stmt], sc: usize, out: &mut Scan) {
        for st in stmts {
            match st {
                Stmt::Expr(e) | Stmt::Assign { value: e, .. } => self.scan_expr(e, sc, out),
                Stmt::AssignOuter { value, .. } | Stmt::Learn { value, .. } => {
                    out.effects = true;
                    self.scan_expr(value, sc, out);
                }
                Stmt::SetPath { path, value, .. } => {
                    out.effects = true;
                    self.scan_expr(value, sc, out);
                    for a in path {
                        if let Access::Index(es) = a {
                            for e in es {
                                self.scan_expr(e, sc, out);
                            }
                        }
                    }
                }
                Stmt::Show(es) => {
                    out.effects = true;
                    for e in es {
                        self.scan_expr(e, sc, out);
                    }
                }
                Stmt::Give(e) => {
                    if let Some(e) = e {
                        self.scan_expr(e, sc, out);
                    }
                }
                Stmt::Repeat { count, body, .. } => {
                    out.effects = true;
                    self.scan_expr(count, sc, out);
                    self.scan_block(body, sc, out);
                }
                Stmt::Each { iter, body, .. } => {
                    out.effects = true;
                    self.scan_expr(iter, sc, out);
                    self.scan_block(body, sc, out);
                }
                Stmt::While { cond, body, .. } => {
                    out.effects = true;
                    self.scan_expr(cond, sc, out);
                    self.scan_block(body, sc, out);
                }
                Stmt::When { arms, otherwise } => {
                    for (c, b, _) in arms {
                        self.scan_expr(c, sc, out);
                        self.scan_block(b, sc, out);
                    }
                    if let Some(b) = otherwise {
                        self.scan_block(b, sc, out);
                    }
                }
                Stmt::Descend { loss, rate, .. } => {
                    out.effects = true;
                    self.scan_expr(loss, sc, out);
                    self.scan_expr(rate, sc, out);
                }
                Stmt::Try { body, handler, .. } => {
                    out.effects = true;
                    self.scan_block(body, sc, out);
                    self.scan_block(handler, sc, out);
                }
                Stmt::Stop(_) | Stmt::Skip(_) => {}
            }
        }
    }

    fn scan_expr(&self, e: &Expr, sc: usize, out: &mut Scan) {
        match e {
            Expr::Interp(p) | Expr::List(p) => {
                for x in p {
                    self.scan_expr(x, sc, out);
                }
            }
            Expr::Map(m) => {
                for (_, x) in m {
                    self.scan_expr(x, sc, out);
                }
            }
            Expr::Neg(a, _) | Expr::Not(a, _) | Expr::Field(a, _, _) | Expr::BitNot(a, _) => self.scan_expr(a, sc, out),
            Expr::Binary(_, a, b, _) | Expr::And(a, b, _) | Expr::Or(a, b, _) => {
                self.scan_expr(a, sc, out);
                self.scan_expr(b, sc, out);
            }
            Expr::Call(f, args, _) => {
                for a in args {
                    self.scan_expr(a, sc, out);
                }
                match self.callee(f, sc) {
                    Callee::Static(s) => out.calls.push(s),
                    Callee::Builtin(i) => {
                        if calls_back(self.bnames[i]) {
                            out.unknown = true;
                        }
                    }
                    Callee::Dynamic => {
                        out.unknown = true;
                        self.scan_expr(f, sc, out);
                    }
                }
            }
            Expr::Index(b, idx, _) => {
                self.scan_expr(b, sc, out);
                for x in idx {
                    self.scan_expr(x, sc, out);
                }
            }
            Expr::Range(a, z, _) => {
                for x in [a, z].into_iter().flatten() {
                    self.scan_expr(x, sc, out);
                }
            }
            Expr::Ask { model, prompt, .. } => {
                self.scan_expr(model, sc, out);
                self.scan_expr(prompt, sc, out);
            }
            Expr::ModelNew { spec, opts, .. } => {
                self.scan_expr(spec, sc, out);
                if let Some(o) = opts {
                    self.scan_expr(o, sc, out);
                }
            }
            // a function made here runs when called, as a closure of its own
            _ => {}
        }
    }

    /// Decides which functions called by name need the recursion check and
    /// which can return plain values.
    fn plan_calls(&mut self, statics: &[usize]) {
        let n = self.scopes.len();
        let mut scans: Vec<Scan> = (0..n).map(|_| Scan::default()).collect();
        for &s in statics {
            let def: &FnDef = unsafe { &*self.scopes[s].def };
            let mut out = Scan::default();
            self.scan_block(&def.body, s, &mut out);
            scans[s] = out;
        }
        // Calls through values can reach any function that is passed around,
        // and any function called by name from inside a closure.
        let mut escaping: Vec<usize> = statics.iter().copied().filter(|&s| self.scopes[s].escapes).collect();
        for c in 1..n {
            if !statics.contains(&c) {
                let def: &FnDef = unsafe { &*self.scopes[c].def };
                let mut out = Scan::default();
                self.scan_block(&def.body, c, &mut out);
                escaping.extend(out.calls);
            }
        }
        let succ = |s: usize| -> Vec<usize> {
            let mut v = scans[s].calls.clone();
            if scans[s].unknown {
                v.extend(escaping.iter().copied());
            }
            v
        };
        self.rec = vec![false; n];
        for &s in statics {
            let mut seen = vec![false; n];
            let mut stack = succ(s);
            while let Some(t) = stack.pop() {
                if t == s {
                    self.rec[s] = true;
                    break;
                }
                if !seen[t] {
                    seen[t] = true;
                    stack.extend(succ(t));
                }
            }
        }
        // Loop-free functions without side effects that only call such functions.
        let mut pure = vec![false; n];
        for &s in statics {
            pure[s] = !scans[s].effects;
        }
        loop {
            let mut changed = false;
            for &s in statics {
                if pure[s] && scans[s].calls.iter().any(|&t| !pure[t]) {
                    pure[s] = false;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // Start from "no function can fail" and drop the ones whose code turns
        // out to have a way to fail, until nothing changes.
        self.inf = vec![false; n];
        for &s in statics {
            self.inf[s] = !(self.rec[s] && !pure[s]);
        }
        loop {
            self.prop = vec![false; n];
            loop {
                let mut changed = false;
                for &s in statics {
                    if self.inf[s]
                        && pure[s]
                        && !self.prop[s]
                        && (self.rec[s] || scans[s].calls.iter().any(|&t| self.prop[t]))
                    {
                        self.prop[s] = true;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
            let mut changed = false;
            for &s in statics {
                if self.inf[s] {
                    let text = self.static_text(s);
                    if text.contains('?') || text.contains("fail") || text.contains("Err") || text.contains("R<") {
                        self.inf[s] = false;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    fn ret_str(&self) -> &'static str {
        match self.ret {
            None => "()",
            Some(t) => rust_ty(t),
        }
    }

    fn emit_return(&self, code: &str) -> String {
        if self.try_depth > 0 {
            format!("return Ok(Flow::Ret({}));\n", code)
        } else if self.plain {
            format!("return {};\n", code)
        } else {
            format!("return Ok({});\n", code)
        }
    }

    fn ret_conv(&self, code: String, t: Ty) -> String {
        match self.ret {
            None => format!("{{ let _ = {}; }}", code),
            Some(rt) => conv(code, t, rt),
        }
    }

    fn ret_nothing(&self) -> String {
        let v = match self.ret {
            None => "()",
            Some(Ty::Num) => "0.0f64",
            Some(Ty::Int) => "0i64",
            Some(Ty::IArr) => "iarr0()",
            Some(Ty::Bool) => "false",
            Some(Ty::Arr) => "arr0()",
            Some(_) => "Value::Nothing",
        };
        self.emit_return(v)
    }

    fn stop_code(&self, line: u32) -> String {
        match self.loops.last() {
            Some((label, td)) if *td == self.try_depth => format!("break 'l{};\n", label),
            Some(_) => "return Ok(Flow::Brk);\n".into(),
            None => format!("return fail(\"`stop` only works inside a loop\", {});\n", line),
        }
    }

    fn skip_code(&self, line: u32) -> String {
        match self.loops.last() {
            Some((label, td)) if *td == self.try_depth => format!("continue 'l{};\n", label),
            Some(_) => "return Ok(Flow::Cont);\n".into(),
            None => format!("return fail(\"`skip` only works inside a loop\", {});\n", line),
        }
    }

    fn block(&mut self, stmts: &[Stmt], sc: usize, tail: bool) -> String {
        let mut out = String::new();
        if !tail {
            for s in stmts {
                out.push_str(&self.stmt(s, sc));
            }
            return out;
        }
        match stmts.split_last() {
            None => out.push_str(&self.ret_nothing()),
            Some((last, rest)) => {
                for s in rest {
                    out.push_str(&self.stmt(s, sc));
                }
                out.push_str(&self.tail(last, sc));
            }
        }
        out
    }

    /// The last statement of a function body: its value is the function's result.
    fn tail(&mut self, s: &Stmt, sc: usize) -> String {
        match s {
            Stmt::Expr(e) => {
                let (c, t) = self.expr(e, sc);
                let v = self.ret_conv(c, t);
                if self.ret.is_none() {
                    format!("{}\n{}", v, self.emit_return("()"))
                } else {
                    self.emit_return(&v)
                }
            }
            Stmt::When { arms, otherwise } => {
                let mut out = String::new();
                for (i, (c, body, line)) in arms.iter().enumerate() {
                    let cc = self.cond(c, sc, *line);
                    let b = self.block(body, sc, true);
                    out.push_str(&format!("{}if {} {{\n{}}}", if i == 0 { "" } else { " else " }, cc, b));
                }
                match otherwise {
                    Some(b) => {
                        let bc = self.block(b, sc, true);
                        out.push_str(&format!(" else {{\n{}}}\n", bc));
                    }
                    None => {
                        out.push('\n');
                        out.push_str(&self.ret_nothing());
                    }
                }
                out
            }
            Stmt::Try { body, var, handler } => {
                let mut out = self.try_stmt(body, *var, handler, sc, true);
                out.push_str(&self.ret_nothing());
                out
            }
            Stmt::Give(_) => self.stmt(s, sc),
            other => {
                let mut out = self.stmt(other, sc);
                out.push_str(&self.ret_nothing());
                out
            }
        }
    }

    fn try_stmt(&mut self, body: &[Stmt], var: Option<u32>, handler: &[Stmt], sc: usize, tail: bool) -> String {
        let rt = self.ret_str();
        self.try_depth += 1;
        let b = self.block(body, sc, tail);
        self.try_depth -= 1;
        let h = self.block(handler, sc, tail);
        let set = match var {
            Some(v) => self.assign(sc, v as usize, "text(&e_.0)".into()),
            None => String::new(),
        };
        format!(
            "match (|| -> R<Flow<{rt}>> {{\n{b}Ok(Flow::Next)\n}})() {{\nOk(Flow::Next) => {{}}\nOk(Flow::Ret(r_)) => {{ {ret} }}\nOk(Flow::Brk) => {{ {brk} }}\nOk(Flow::Cont) => {{ {cont} }}\nErr(e_) => {{\n{set}{h}}}\n}}\n",
            rt = rt,
            b = b,
            ret = self.emit_return("r_"),
            brk = if self.loops.is_empty() { String::new() } else { self.stop_code(0) },
            cont = if self.loops.is_empty() { String::new() } else { self.skip_code(0) },
            set = set,
            h = h
        )
    }

    fn stmt(&mut self, s: &Stmt, sc: usize) -> String {
        match s {
            Stmt::Expr(e) => {
                let (c, _) = self.expr(e, sc);
                format!("let _ = {};\n", c)
            }
            Stmt::Assign { idx, value } => {
                let i = *idx as usize;
                if let (Expr::Lambda(def), 0) = (value, sc) {
                    if self.static_fn.get(&i) == self.by_def.get(&Rc::as_ptr(def)) {
                        return String::new();
                    }
                }
                if sc == 0 && self.tables.contains_key(&i) {
                    if self.table_init.contains_key(&i) {
                        // the table already holds the list's numbers
                        return String::new();
                    }
                    return self.table_fill(i);
                }
                if self.var_ty(sc, i) == Ty::Whole {
                    let c = self.whole_code(value, sc);
                    return self.assign(sc, i, c);
                }
                let (c, t) = self.expr(value, sc);
                self.assign(sc, i, conv(c, t, self.var_ty(sc, i)))
            }
            Stmt::AssignOuter { depth, idx, value } => {
                let t = self.resolve(sc, *depth);
                if t == 0 && self.tables.contains_key(&(*idx as usize)) {
                    return self.table_fill(*idx as usize);
                }
                if self.var_ty(t, *idx as usize) == Ty::Whole {
                    let c = self.whole_code(value, sc);
                    return self.assign(t, *idx as usize, c);
                }
                let (c, ty) = self.expr(value, sc);
                self.assign(t, *idx as usize, conv(c, ty, self.var_ty(t, *idx as usize)))
            }
            Stmt::Learn { idx, value, line } => {
                let v = self.dyn_code(value, sc);
                self.assign(sc, *idx as usize, format!("{{ let v_ = {}; learn(&v_, {})? }}", v, line))
            }
            Stmt::SetPath { depth, idx, name, path, value, line } => {
                let t = self.resolve(sc, *depth);
                let i = *idx as usize;
                let captured = t != 0 && self.scopes[t].captured[i];
                if matches!(self.var_ty(t, i), Ty::PkM | Ty::PkR) && !captured {
                    if let Some(ie) = self.fast_set(path) {
                        return self.pk_set_stmt(t, i, ie, value, sc, *line);
                    }
                }
                if let (Some(p), Some(ie)) = (self.fast_ptr(t, i), self.fast_set(path)) {
                    if self.fast_pos_ok(ie, sc) {
                        // the value is worked out before the position, as everywhere
                        let int = self.var_ty(t, i) == Ty::IArr;
                        let vc = if int { self.int_code(value, sc, *line) } else { self.num_code(value, sc) };
                        let ic = self.fast_pos(ie, sc).unwrap_or_default();
                        return format!("{{ let v_ = {}; let i_ = {}; *{}.add(i_ as usize) = v_; }}\n", vc, ic, p);
                    }
                }
                if let (Some((tb, int)), Some(ie)) = (self.table_of(sc, *depth, *idx), self.fast_set(path)) {
                    let vc = if int { self.int_code(value, sc, *line) } else { self.num_code(value, sc) };
                    let k = if int { "i" } else { "a" };
                    let safe = self.in_table(ie, sc, *idx as usize);
                    return match self.table_pos(ie, sc, *line) {
                        Ok((ic, _)) if safe => format!(
                            "{{ let v_ = {}; let i_ = {}; *(std::ptr::addr_of_mut!({}) as *mut {}).add(i_ as usize) = v_; }}\n",
                            vc,
                            ic,
                            tb,
                            if int { "i64" } else { "f64" }
                        ),
                        Ok((ic, num)) => format!(
                            "{{ let v_ = {}; let i_ = {}; tset_{}(&mut *std::ptr::addr_of_mut!({}), i_, {}, v_, {})?; }}\n",
                            vc, ic, k, tb, num, line
                        ),
                        Err(fc) => format!(
                            "{{ let v_ = {}; let i_ = {}; tset_{}f(&mut *std::ptr::addr_of_mut!({}), i_, v_, {})?; }}\n",
                            vc, fc, k, tb, line
                        ),
                    };
                }
                if self.var_ty(t, i) == Ty::Arr {
                    if let Some(ie) = self.fast_set(path) {
                        let vc = self.num_code(value, sc);
                        let (ic, writer) = if self.ty(ie, sc) == Ty::Int {
                            (self.int_code(ie, sc, *line), "aset_i")
                        } else if let Some(c) = self.icode(ie, sc, true) {
                            (c, "aset_i")
                        } else {
                            (self.num_code(ie, sc), "aset")
                        };
                        return format!(
                            "{{ let v_ = {}; let i_ = {}; {}(&mut {}, i_, v_, {})?; }}\n",
                            vc,
                            ic,
                            writer,
                            self.place(t, i),
                            line
                        );
                    }
                }
                if self.var_ty(t, i) == Ty::IArr {
                    if let Some(ie) = self.fast_set(path) {
                        let vc = self.int_code(value, sc, *line);
                        let ic = match self.icode(ie, sc, true) {
                            Some(c) => c,
                            None => self.int_code(ie, sc, *line),
                        };
                        return format!(
                            "{{ let v_ = {}; let i_ = {}; iset(&mut {}, i_, v_, {})?; }}\n",
                            vc,
                            ic,
                            self.place(t, i),
                            line
                        );
                    }
                }
                let v = self.dyn_code(value, sc);
                let target = if captured {
                    format!("&mut *c{}_{}.borrow_mut()", t, i)
                } else {
                    format!("&mut {}", self.place(t, i))
                };
                if let Some(ie) = self.fast_set(path) {
                    let k = self.dyn_code(ie, sc);
                    return format!(
                        "{{ let v_ = {}; let k_ = {}; set1({}, k_, v_, {:?}, {})?; }}\n",
                        v, k, target, &**name, line
                    );
                }
                let mut keys = vec![];
                for a in path {
                    keys.push(match a {
                        Access::Field(f) => format!("Key::Field({:?}.into())", &**f),
                        Access::Index(es) => format!("Key::Idx(vec![{}])", self.dyn_list(es, sc)),
                    });
                }
                format!(
                    "{{ let v_ = {}; let k_ = vec![{}]; set_path({}, k_, v_, {:?}, {})?; }}\n",
                    v,
                    keys.join(", "),
                    target,
                    &**name,
                    line
                )
            }
            Stmt::Show(items) => format!("show(&[{}]);\n", self.dyn_list(items, sc)),
            Stmt::Give(e) => match e {
                Some(e) => {
                    let (c, t) = self.expr(e, sc);
                    let v = self.ret_conv(c, t);
                    if self.ret.is_none() {
                        format!("{}\n{}", v, self.emit_return("()"))
                    } else {
                        self.emit_return(&v)
                    }
                }
                None => self.ret_nothing(),
            },
            Stmt::Repeat { count, var, body, line } => {
                let k = self.fresh();
                let (c, t) = self.expr(count, sc);
                let n = if t == Ty::Num {
                    format!("repeat_n({}, {})?", c, line)
                } else if t == Ty::Int {
                    format!("repeat_n(({}) as f64, {})?", c, line)
                } else {
                    format!("repeat_dyn(&{}, {})?", conv(c, t, Ty::Dyn), line)
                };
                let set = match var {
                    Some(v) => match self.var_ty(sc, *v as usize) {
                        Ty::Whole => self.assign(sc, *v as usize, format!("(i{} as i64)", k)),
                        vt => self.assign(sc, *v as usize, conv(format!("(i{} as f64)", k), Ty::Num, vt)),
                    },
                    None => String::new(),
                };
                self.loops.push((k, self.try_depth));
                let b = self.block(body, sc, false);
                self.loops.pop();
                // A direct version when the counter only reaches rows at counter
                // + constant: the counter runs from 1 to n.
                let rows = match var {
                    Some(v) if self.var_ty(sc, *v as usize) == Ty::Whole => self.fast_rows(body, sc, *v as usize),
                    _ => vec![],
                };
                if !rows.is_empty() && rows.iter().all(|r| r.lo >= -1) {
                    let conds: Vec<String> =
                        rows.iter().map(|r| format!("(n{} as i128) + {} < ({} as i128)", k, r.hi, self.row_len(sc, r))).collect();
                    let (setup, fb) = self.fast_body(body, sc, k, &rows);
                    return format!(
                        "{{ let n{k} = ({n}) as u64; let mut i{k} = 0u64;\nif {conds} {{\n{setup}'l{k}: while i{k} < n{k} {{\ni{k} += 1;\n{set}{fb}}}\n}} else {{\n'l{k}: while i{k} < n{k} {{\ni{k} += 1;\n{set}{b}}}\n}} }}\n",
                        k = k,
                        n = n,
                        conds = conds.join(" && "),
                        setup = setup,
                        set = set,
                        fb = fb,
                        b = b
                    );
                }
                // (the count is a whole number below 2^64, so this counts the same)
                format!(
                    "{{ let n{k} = ({n}) as u64; let mut i{k} = 0u64;\n'l{k}: while i{k} < n{k} {{\ni{k} += 1;\n{set}{b}}} }}\n",
                    k = k,
                    n = n,
                    set = set,
                    b = b
                )
            }
            Stmt::Each { var, iter, body, line } if self.range_parts(iter, sc).is_some() => {
                let (start, end, step) = self.range_parts(iter, sc).unwrap();
                let k = self.fresh();
                let v = *var as usize;
                let whole = self.var_ty(sc, v) == Ty::Whole;
                let s = match (start, whole) {
                    (Some(x), true) => self.whole_code(x, sc),
                    (Some(x), false) => self.num_code(x, sc),
                    (None, true) => "0i64".into(),
                    (None, false) => "0.0f64".into(),
                };
                let e = self.num_code(end, sc);
                let set = if whole {
                    self.assign(sc, v, format!("c{}", k))
                } else {
                    let vt = self.var_ty(sc, v);
                    self.assign(sc, v, conv(format!("c{}", k), Ty::Num, vt))
                };
                self.loops.push((k, self.try_depth));
                let b = self.block(body, sc, false);
                self.loops.pop();
                let cmp = if step > 0.0 { "<" } else { ">" };
                // the same numbers range() would list, one at a time
                let rows = if whole && step > 0.0 { self.fast_rows(body, sc, v) } else { vec![] };
                if !rows.is_empty() {
                    // Every counter value c satisfies start <= c < end, so checking
                    // the first and last positions once covers every iteration.
                    let conds: Vec<String> = rows
                        .iter()
                        .map(|r| {
                            format!("x{k}.wrapping_add({lo}) >= 0 && e{k} + {hi:?}f64 <= ({len} as f64)", k = k, lo = r.lo, hi = r.hi as f64, len = self.row_len(sc, r))
                        })
                        .collect();
                    let (setup, fb) = self.fast_body(body, sc, k, &rows);
                    return format!(
                        "{{ let mut x{k}: i64 = {s}; let e{k}: f64 = {e};\nif {conds} {{\n{setup}let z{k} = e{k}.ceil() as i64;\n'l{k}: while x{k} < z{k} {{\nlet c{k} = x{k}; x{k} += {st}i64;\n{set}{fb}}}\n}} else {{\n'l{k}: while (x{k} as f64) < e{k} {{\nlet c{k} = x{k}; x{k} += {st}i64;\n{set}{b}}}\n}} }}\n",
                        k = k, s = s, e = e, conds = conds.join(" && "), setup = setup, st = step as i64, set = set, fb = fb, b = b
                    );
                }
                if whole {
                    format!(
                        "{{ let mut x{k}: i64 = {s}; let e{k}: f64 = {e};\n'l{k}: while (x{k} as f64) {cmp} e{k} {{\nlet c{k} = x{k}; x{k} += {st}i64;\n{set}{b}}} }}\n",
                        k = k, s = s, e = e, cmp = cmp, st = step as i64, set = set, b = b
                    )
                } else {
                    format!(
                        "{{ let mut x{k}: f64 = {s}; let e{k}: f64 = {e};\n'l{k}: while x{k} {cmp} e{k} {{\nlet c{k} = x{k}; x{k} += {st:?}f64;\n{set}{b}}} }}\n",
                        k = k, s = s, e = e, cmp = cmp, st = step, set = set, b = b
                    )
                }
            }
            Stmt::Each { var, iter, body, line } => {
                let k = self.fresh();
                let (c, t) = self.expr(iter, sc);
                let vt = self.var_ty(sc, *var as usize);
                if t == Ty::IArr {
                    let set = self.assign(sc, *var as usize, conv(format!("x{}", k), Ty::Int, vt));
                    self.loops.push((k, self.try_depth));
                    let b = self.block(body, sc, false);
                    self.loops.pop();
                    return format!(
                        "{{ let a{k} = {c}; let n{k} = a{k}.len(); let mut i{k} = 0usize;\n'l{k}: while i{k} < n{k} {{\nlet x{k} = a{k}[i{k}]; i{k} += 1;\n{set}{b}}} }}\n",
                        k = k,
                        c = c,
                        set = set,
                        b = b
                    );
                }
                if t == Ty::Arr {
                    let set = self.assign(sc, *var as usize, conv(format!("x{}", k), Ty::Num, vt));
                    self.loops.push((k, self.try_depth));
                    let b = self.block(body, sc, false);
                    self.loops.pop();
                    format!(
                        "{{ let a{k} = {c}; let n{k} = alen(&a{k}); let mut i{k} = 0usize;\n'l{k}: while i{k} < n{k} {{\nlet x{k} = aidx(&a{k}, i{k}); i{k} += 1;\n{set}{b}}} }}\n",
                        k = k,
                        c = c,
                        set = set,
                        b = b
                    )
                } else {
                    let set = self.assign(sc, *var as usize, conv(format!("x{}", k), Ty::Dyn, vt));
                    self.loops.push((k, self.try_depth));
                    let b = self.block(body, sc, false);
                    self.loops.pop();
                    format!(
                        "{{ let s{k} = {c}; 'l{k}: for x{k} in items(&s{k}, {line})? {{\n{set}{b}}} }}\n",
                        k = k,
                        c = conv(c, t, Ty::Dyn),
                        line = line,
                        set = set,
                        b = b
                    )
                }
            }
            Stmt::While { cond, body, line } => {
                let k = self.fresh();
                let c = self.cond(cond, sc, *line);
                self.loops.push((k, self.try_depth));
                let b = self.block(body, sc, false);
                self.loops.pop();
                // A counting while loop that reaches rows and tables only at
                // counter + constant gets a version that checks the first and
                // last positions once, before it starts.
                if let Some((v, end, le, rows)) = self.while_plan(cond, body, sc) {
                    let e = self.num_code(end, sc);
                    let i = self.place(sc, v);
                    let conds: Vec<String> = rows
                        .iter()
                        .map(|r| {
                            format!(
                                "{i}.wrapping_add({lo}) >= 0 && e{k} + {hi:?}f64 {cmp} ({len} as f64)",
                                i = i, k = k, lo = r.lo, hi = r.hi as f64, len = self.row_len(sc, r), cmp = if le { "<" } else { "<=" }
                            )
                        })
                        .collect();
                    let (setup, fb) = self.fast_body(body, sc, k, &rows);
                    return format!(
                        "{{ let e{k}: f64 = {e};\nif {conds} {{\n{setup}'l{k}: while {c} {{\n{fb}}}\n}} else {{\n'l{k}: while {c} {{\n{b}}}\n}} }}\n",
                        k = k, e = e, conds = conds.join(" && "), setup = setup, c = c, fb = fb, b = b
                    );
                }
                format!("'l{}: while {} {{\n{}}}\n", k, c, b)
            }
            Stmt::When { arms, otherwise } => {
                let mut out = String::new();
                for (i, (c, body, line)) in arms.iter().enumerate() {
                    let cc = self.cond(c, sc, *line);
                    let b = self.block(body, sc, false);
                    out.push_str(&format!("{}if {} {{\n{}}}", if i == 0 { "" } else { " else " }, cc, b));
                }
                if let Some(b) = otherwise {
                    let bc = self.block(b, sc, false);
                    out.push_str(&format!(" else {{\n{}}}", bc));
                }
                out.push('\n');
                out
            }
            Stmt::Descend { loss, rate, optim, clip, line } => {
                let l = self.dyn_code(loss, sc);
                let r = self.dyn_code(rate, sc);
                let c = match clip {
                    Some(c) => format!("Some({})", self.dyn_code(c, sc)),
                    None => "None".to_string(),
                };
                format!(
                    "{{ let l_ = {}; let r_ = {}; let c_ = {}; descend(&l_, &r_, antares::tensor::Optim::{:?}, c_, {})?; }}\n",
                    l, r, c, optim, line
                )
            }
            Stmt::Try { body, var, handler } => self.try_stmt(body, *var, handler, sc, false),
            Stmt::Stop(line) => self.stop_code(*line),
            Stmt::Skip(line) => self.skip_code(*line),
        }
    }

    fn locals(&self, s: usize) -> String {
        let sc = &self.scopes[s];
        let mut out = String::new();
        for i in sc.nparams..sc.vars.len() {
            if sc.captured[i] {
                out.push_str(&format!("let c{}_{} = cell(Value::Unset);\n", s, i));
            } else {
                out.push_str(&format!("let mut v{}: {} = {};\n", i, rust_ty(sc.vars[i]), zero_of(sc.vars[i])));
            }
        }
        out
    }

    /// The body of function `s`, with code generation state set for it.
    /// `by_name`: a function called by name, which receives its depth as `d_`.
    fn fn_body(&mut self, s: usize, ret: Ty, by_name: bool) -> String {
        let ret = if ret == Ty::Unit { None } else { Some(ret) };
        let def: &FnDef = unsafe { &*self.scopes[s].def };
        let saved = (std::mem::take(&mut self.loops), self.try_depth, self.ret, self.cur, self.plain);
        self.try_depth = 0;
        self.ret = ret;
        self.cur = if by_name { Some(s) } else { None };
        self.plain = by_name && self.inf.get(s).copied().unwrap_or(false);
        let mut out = String::new();
        if !by_name {
            out.push_str(&format!("let _d = enter({:?})?;\n", self.scopes[s].name));
        } else if self.rec.get(s).copied().unwrap_or(true) {
            if self.plain {
                let placeholder = match ret {
                    None => "()",
                    Some(Ty::Dyn) => "Value::Nothing",
                    Some(t) => zero_of(t),
                };
                out.push_str(&format!(
                    "if d_ > limit() {{ return fault({:?}, {}); }}\n",
                    self.scopes[s].name, placeholder
                ));
            } else {
                out.push_str(&format!("if d_ > MAX_CALL_DEPTH {{ return too_deep_at({:?}); }}\n", self.scopes[s].name));
            }
        }
        out.push_str(&self.locals(s));
        // inputs with a range are checked by the callers (see `input_checks`)
        let skip = if by_name { self.input_checks(s).1 } else { 0 };
        out.push_str(&self.block(&def.body[skip..], s, true));
        self.loops = saved.0;
        self.try_depth = saved.1;
        self.ret = saved.2;
        self.cur = saved.3;
        self.plain = saved.4;
        out
    }

    /// The names of a function's inputs, as Rust code: &["a", "b"].
    fn param_names(&self, s: usize, n: usize) -> String {
        let def = self.scopes[s].def;
        let names: Vec<String> = if def.is_null() {
            (1..=n).map(|i| format!("input{}", i)).collect()
        } else {
            unsafe { &*def }.params.iter().map(|p| p.to_string()).collect()
        };
        format!("&{:?}", names)
    }

    /// A function used as a value: a Rust closure wrapped as an Antares function.
    fn closure(&mut self, s: usize) -> String {
        let n = self.scopes[s].nparams;
        let mut params = String::from("let mut a_ = args.into_iter();\n");
        for i in 0..n {
            if self.scopes[s].captured[i] {
                params.push_str(&format!("let c{}_{} = cell(a_.next().unwrap());\n", s, i));
            } else {
                params.push_str(&format!("let mut v{}: Value = a_.next().unwrap();\n", i));
            }
        }
        let body = self.fn_body(s, Ty::Dyn, false);
        let mut clones = String::new();
        for (t, i) in &self.scopes[s].free {
            clones.push_str(&format!("let c{t}_{i} = c{t}_{i}.clone();\n", t = t, i = i));
        }
        format!(
            "{{ {clones}rust_fn_p({name:?}, {names}, move |args: Vec<Value>| -> R<Value> {{ unsafe {{\n{params}{body}}} }}) }}",
            clones = clones,
            name = self.scopes[s].name,
            names = self.param_names(s, n),
            params = params,
            body = body
        )
    }

    /// A top-level function called by name: a plain Rust function with typed inputs.
    fn static_text(&mut self, s: usize) -> String {
        let n = self.scopes[s].nparams;
        let ret = self.scopes[s].ret;
        let mut sig = vec![];
        let mut cells = String::new();
        for i in 0..n {
            if self.scopes[s].captured[i] {
                sig.push(format!("p{}: Value", i));
                cells.push_str(&format!("let c{}_{} = cell(p{});\n", s, i, i));
            } else {
                sig.push(format!("mut v{}: {}", i, rust_ty(self.scopes[s].vars[i])));
            }
        }
        sig.push("d_: usize".into());
        let body = self.fn_body(s, ret, true);
        let ret_ty = if self.inf[s] { rust_ty(ret).to_string() } else { format!("R<{}>", rust_ty(ret)) };
        format!("unsafe fn f{}({}) -> {} {{\n{}{}}}\n\n", s, sig.join(", "), ret_ty, cells, body)
    }

    fn program(&mut self, top: &[Stmt], files: &[(String, String)], main_src: &str) -> String {
        let mut statics: Vec<(usize, usize)> = self.static_fn.iter().map(|(a, b)| (*a, *b)).collect();
        statics.sort();
        let ids: Vec<usize> = statics.iter().map(|(_, s)| *s).collect();
        self.plan_calls(&ids);
        for (_, s) in &statics {
            let item = self.static_text(*s);
            self.items.push_str(&item);
        }
        self.ret = None;
        let body = self.block(top, 0, false);

        // functions that are also passed around as values get a wrapper in their global slot
        let mut init = String::new();
        for (gi, s) in &statics {
            if self.scopes[*s].escapes {
                let n = self.scopes[*s].nparams;
                let mut args: Vec<String> = (0..n).map(|_| "a_.next().unwrap()".to_string()).collect();
                args.push("depth() + 1".into());
                let call = self.static_call(*s, format!("f{}({})", s, args.join(", ")));
                init.push_str(&format!(
                    "(*GV.as_mut_ptr()).v{gi} = rust_fn_p({name:?}, {names}, |args: Vec<Value>| -> R<Value> {{ let mut a_ = args.into_iter(); let r_ = unsafe {{ {call} }}; Ok({conv}) }});\n",
                    gi = gi,
                    name = self.scopes[*s].name,
                    names = self.param_names(*s, n),
                    call = call,
                    conv = conv("r_".into(), self.scopes[*s].ret, Ty::Dyn)
                ));
            }
        }

        let g = &self.scopes[0];
        let mut fields = String::new();
        let mut inits = String::new();
        let mut tables = String::new();
        for i in 0..g.vars.len() {
            if self.is_builtin(i) {
                continue;
            }
            if let Some(&(len, int)) = self.tables.get(&i) {
                let zero = if int { "0i64" } else { "0.0f64" };
                let ty = if int { "i64" } else { "f64" };
                let name = self.gnames.get(i).map(|s| s.as_str()).unwrap_or("");
                match self.table_init.get(&i) {
                    Some(vals) => {
                        let nums: Vec<String> = vals.iter().map(|v| format!("{}i64", v)).collect();
                        tables.push_str(&format!("static mut TB{}: [i64; {}] = [{}];   // {}\n", i, len, nums.join(", "), name));
                    }
                    None => tables.push_str(&format!("static mut TB{}: [{}; {}] = [{}; {}];   // {}\n", i, ty, len, zero, len, name)),
                }
                continue;
            }
            let t = if self.static_fn.contains_key(&i) { Ty::Dyn } else { g.vars[i] };
            fields.push_str(&format!("    v{}: {},   // {}\n", i, rust_ty(t), self.gnames.get(i).map(|s| s.as_str()).unwrap_or("")));
            inits.push_str(&format!("v{}: {}, ", i, zero_of(t)));
        }
        let strs: Vec<String> = self.strs.iter().map(|s| format!("text({:?})", s)).collect();
        let arrs: Vec<String> = self
            .arrs
            .iter()
            .map(|a| {
                let nums: Vec<String> = a.iter().map(|x| format!("{:?}f64", x)).collect();
                format!("arr_of(vec![{}])", nums.join(", "))
            })
            .collect();
        let file_list: Vec<String> = files.iter().map(|(n, s)| format!("({:?}, {:?})", n, s)).collect();

        format!(
            "// Generated by `antares build`. Do not edit.\n#![allow(warnings)]\nuse antares::rt::*;\n\nstruct Globals {{\n{fields}}}\n\n// at a fixed address, so reading a variable needs no pointer to be loaded first\nstatic mut GV: std::mem::MaybeUninit<Globals> = std::mem::MaybeUninit::uninit();\nstatic mut S: *const Vec<Value> = std::ptr::null();\nstatic mut A: *const Vec<T> = std::ptr::null();\n\n{tables}\n#[inline(always)]\nunsafe fn sc(i: usize) -> Value {{ (&*S)[i].clone() }}\n\n#[inline(always)]\nunsafe fn ac(i: usize) -> T {{ (&*A)[i].clone() }}\n\n{items}unsafe fn main_body() -> R<()> {{\n{init}{body}Ok(())\n}}\n\nfn body() -> R<()> {{\n    let g = Globals {{ {inits} }};\n    let strs: Vec<Value> = vec![{strs}];\n    let arrs: Vec<T> = vec![{arrs}];\n    unsafe {{\n        GV.write(g);\n        S = &strs;\n        A = &arrs;\n        main_body()\n    }}\n}}\n\nstatic FILES: &[(&str, &str)] = &[{files}];\nstatic MAIN_SRC: &str = {main:?};\n\nfn main() {{\n    start(FILES, MAIN_SRC, body)\n}}\n",
            fields = fields,
            tables = tables,
            items = self.items,
            init = init,
            body = body,
            inits = inits,
            strs = strs.join(", "),
            arrs = arrs.join(", "),
            files = file_list.join(", "),
            main = main_src
        )
    }
}

/// Compiles `path` and writes the program next to it (or to `out`).
/// `cpu` is the processor to build for: "native" (this machine, the default),
/// "x86-64" (any 64-bit PC), "x86-64-v2" (SSE4.2), "x86-64-v3" (AVX2).
/// `embed` lists files to pack into the program; it reads them as "embedded:<file name>".
const REPOSITORY: &str = "https://github.com/Madushan996/Antares-Programming-Language";

/// The build script that puts an icon into a Windows program. Antares writes
/// the compiled resource (icon.res) itself; Microsoft's linker takes it as it
/// is, and for the GNU toolchain windres only converts it to an object file.
const ICON_BUILD_RS: &str = r#"fn main() {
    println!("cargo:rerun-if-changed=icon.res");
    let here = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let res = format!("{}/icon.res", here);
    if std::env::var("TARGET").map(|t| t.ends_with("-gnu")).unwrap_or(false) {
        let obj = format!("{}/icon.o", std::env::var("OUT_DIR").unwrap());
        let made = std::process::Command::new("windres").args(["-J", "res", "-O", "coff", "-i", &res, "-o", &obj]).status();
        if made.map(|s| s.success()).unwrap_or(false) {
            println!("cargo:rustc-link-arg-bins={}", obj);
        } else {
            println!("cargo:warning=windres couldn't convert the icon; building without it");
        }
    } else {
        println!("cargo:rustc-link-arg-bins={}", res);
    }
}
"#;

/// The pictures in an .ico file, and the icon group that lists them (as a
/// Windows RT_GROUP_ICON resource, the pictures numbered from 1). None if the
/// file isn't an icon.
pub(crate) fn ico_parts(ico: &[u8]) -> Option<(Vec<&[u8]>, Vec<u8>)> {
    let u16_at = |b: &[u8], i: usize| -> Option<u16> { Some(u16::from_le_bytes(b.get(i..i + 2)?.try_into().ok()?)) };
    let u32_at = |b: &[u8], i: usize| -> Option<u32> { Some(u32::from_le_bytes(b.get(i..i + 4)?.try_into().ok()?)) };
    if u16_at(ico, 0)? != 0 || u16_at(ico, 2)? != 1 {
        return None;
    }
    let count = u16_at(ico, 4)? as usize;
    if count == 0 {
        return None;
    }
    let mut pictures = vec![];
    let mut group = Vec::new();
    group.extend_from_slice(&0u16.to_le_bytes());
    group.extend_from_slice(&1u16.to_le_bytes());
    group.extend_from_slice(&(count as u16).to_le_bytes());
    for k in 0..count {
        let e = 6 + 16 * k;
        let entry = ico.get(e..e + 16)?;
        let size = u32_at(ico, e + 8)? as usize;
        let at = u32_at(ico, e + 12)? as usize;
        pictures.push(ico.get(at..at.checked_add(size)?)?);
        group.extend_from_slice(&entry[..12]); // size, colours, planes, bits, bytes
        group.extend_from_slice(&((k + 1) as u16).to_le_bytes());
    }
    Some((pictures, group))
}

/// An .ico file as a compiled Windows resource (.res): each picture becomes an
/// icon resource and the list of them an icon group, as a resource compiler
/// would make them. None if the file isn't an icon.
fn ico_to_res(ico: &[u8]) -> Option<Vec<u8>> {
    let (pictures, group) = ico_parts(ico)?;
    // one resource: header (type and name as numbers) and data, padded to 4 bytes
    fn resource(out: &mut Vec<u8>, kind: u16, name: u16, flags: u16, data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&32u32.to_le_bytes());
        out.extend_from_slice(&[0xFF, 0xFF]);
        out.extend_from_slice(&kind.to_le_bytes());
        out.extend_from_slice(&[0xFF, 0xFF]);
        out.extend_from_slice(&name.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // data version
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&0x0409u16.to_le_bytes()); // language: English (United States)
        out.extend_from_slice(&0u32.to_le_bytes()); // version
        out.extend_from_slice(&0u32.to_le_bytes()); // characteristics
        out.extend_from_slice(data);
        while out.len() % 4 != 0 {
            out.push(0);
        }
    }
    let mut out = Vec::new();
    resource(&mut out, 0, 0, 0, &[]); // a .res file starts with an empty entry
    for (k, picture) in pictures.iter().enumerate() {
        resource(&mut out, 3, (k + 1) as u16, 0x1010, picture); // RT_ICON
    }
    resource(&mut out, 14, 1, 0x1030, &group); // RT_GROUP_ICON
    Some(out)
}

/// Writes (or removes) the files that give the program an icon. False on a
/// problem worth stopping for.
fn write_icon_files(dir: &std::path::Path, icon: Option<&str>) -> bool {
    let (res, script) = (dir.join("icon.res"), dir.join("build.rs"));
    let _ = std::fs::remove_file(&res);
    let _ = std::fs::remove_file(&script);
    let Some(icon) = icon else { return true };
    if !cfg!(windows) {
        eprintln!("Note: --icon only applies to Windows programs, so this one is built without it.");
        return true;
    }
    let data = match std::fs::read(icon) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Couldn't open the icon \"{}\": {}", icon, e);
            return false;
        }
    };
    let Some(compiled) = ico_to_res(&data) else {
        eprintln!("\"{}\" isn't an .ico file.", icon);
        return false;
    };
    // the toolchain that built Antares is the one that builds the program
    if cfg!(target_env = "gnu") {
        let found = std::process::Command::new("windres")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok();
        if !found {
            eprintln!("Note: adding an icon needs `windres` (it comes with MinGW binutils), which wasn't found; building without the icon.");
            return true;
        }
    }
    std::fs::write(&res, compiled).is_ok() && std::fs::write(&script, ICON_BUILD_RS).is_ok()
}

/// Whether a folder holds the Antares source (compiled programs build against it).
fn is_source(dir: &std::path::Path) -> bool {
    dir.join("src").join("rt.rs").is_file()
        && std::fs::read_to_string(dir.join("Cargo.toml")).is_ok_and(|t| t.contains("name = \"antares\""))
}

/// The Antares source folder to build against: ANTARES_HOME, or the folder
/// this copy of Antares was built from if it is still there. None means the
/// matching release is fetched from GitHub instead.
fn find_source() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("ANTARES_HOME").filter(|h| !h.is_empty()) {
        let home = PathBuf::from(home);
        if is_source(&home) {
            return Some(home);
        }
        eprintln!("ANTARES_HOME is set to {}, but that isn't the Antares source folder; ignoring it.", home.display());
    }
    let built_from = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    is_source(&built_from).then_some(built_from)
}

/// Where to keep build files when there is no local source folder.
fn cache_dir() -> Option<PathBuf> {
    let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let base = if cfg!(windows) {
        var("LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        var("HOME").map(|h| h.join("Library").join("Caches"))
    } else {
        var("XDG_CACHE_HOME").or_else(|| var("HOME").map(|h| h.join(".cache")))
    };
    base.map(|b| b.join("antares").join(env!("CARGO_PKG_VERSION")))
}

/// The name a built program gets: the source file's name in lower case, with
/// anything but letters and digits made `_`.
pub(crate) fn exe_stem(path: &str) -> String {
    PathBuf::from(path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "program".into())
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect()
}

/// Whether `cargo` (and so Rust) is installed.
pub fn have_cargo() -> bool {
    std::process::Command::new("cargo")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// `icon`: an .ico file to show for the program in Windows (ignored elsewhere).
pub fn build(path: &str, out: Option<&str>, cpu: Option<&str>, embed: &[String], icon: Option<&str>) -> i32 {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Couldn't open \"{}\": {}", path, e);
            return 1;
        }
    };
    let b = builtins::all();
    let names: Vec<&'static str> = b.iter().map(|(n, _)| *n).collect();
    let mut parser = Parser::new(&names);
    parser.set_main(path);
    let parsed = lexer::lex(&src, 1).and_then(|toks| parser.parse(toks).map_err(|e| (e.msg, e.line)));
    let stmts = match parsed {
        Ok(s) => s,
        Err((msg, line)) => {
            let (file, line) = (line >> 20, line & 0xFFFFF);
            let place = if file > 0 { parser.file(file).map(|(n, _)| format!(" of {}", n)).unwrap_or_default() } else { String::new() };
            eprintln!("\nProblem on line {}{}: {}", line, place, msg);
            return 1;
        }
    };

    let ng = parser.global_count();
    let mut cx = Cx {
        scopes: vec![Scope {
            def: std::ptr::null(),
            parent: 0,
            vars: vec![Ty::Bot; ng],
            captured: vec![false; ng],
            nparams: 0,
            ret: Ty::Dyn,
            escapes: false,
            free: BTreeSet::new(),
            name: "program".into(),
        }],
        by_def: HashMap::new(),
        static_fn: HashMap::new(),
        gcount: vec![0; ng],
        gdef: vec![None; ng],
        glit: vec![None; ng],
        consts: HashMap::new(),
        glit_i: vec![None; ng],
        iconsts: HashMap::new(),
        nb: names.len(),
        bnames: names.clone(),
        gnames: parser.global_names(),
        codegen: false,
        pass0: false,
        changed: false,
        strs: vec![],
        arrs: vec![],
        items: String::new(),
        tmp: 0,
        loops: vec![],
        try_depth: 0,
        ret: None,
        rec: vec![],
        inf: vec![],
        prop: vec![],
        cur: None,
        plain: false,
        fast: vec![],
        tables: HashMap::new(),
        table_init: HashMap::new(),
        ranges: HashMap::new(),
        nonzero_bits: std::collections::HashSet::new(),
    };
    cx.infer(&stmts);
    cx.infer_whole(&stmts);
    cx.find_tables(&stmts);
    cx.find_ranges(&stmts);
    let mut code = cx.program(&stmts, parser.files(), &src);
    // Embedded files become part of the executable.
    let mut entries = vec![];
    for f in embed {
        let full = match std::fs::canonicalize(f) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Couldn't find \"{}\" to embed: {}", f, e);
                return 1;
            }
        };
        let name = full.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let mut text = full.to_string_lossy().replace('\\', "/");
        if let Some(rest) = text.strip_prefix("//?/") {
            text = rest.to_string();
        }
        entries.push(format!("({:?}, include_bytes!({:?}) as &[u8])", name, text));
    }
    code = code.replace(
        "fn body() -> R<()> {\n",
        &format!(
            "static EMBED: &[(&str, &[u8])] = &[{}];\n\nfn body() -> R<()> {{\n    embed(EMBED);\n",
            entries.join(", ")
        ),
    );

    let source = PathBuf::from(path);
    let stem = exe_stem(path);
    let (dependency, work) = match find_source() {
        Some(root) => (
            format!("{{ path = {:?} }}", root.to_string_lossy().replace('\\', "/")),
            root.join("target"),
        ),
        None => {
            let Some(cache) = cache_dir() else {
                eprintln!("Couldn't find a folder to build in. Set ANTARES_HOME to the Antares source folder.");
                return 1;
            };
            eprintln!(
                "Using the Antares {} source from {} (set ANTARES_HOME to use a local copy instead).",
                env!("CARGO_PKG_VERSION"),
                REPOSITORY
            );
            (format!("{{ git = {:?}, tag = \"v{}\" }}", REPOSITORY, env!("CARGO_PKG_VERSION")), cache)
        }
    };
    let dir = work.join("aot").join(&stem);
    let manifest = format!(
        "[package]\nname = \"{stem}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\nantares = {dependency}\n\n[profile.release]\nopt-level = 3\nlto = \"fat\"\ncodegen-units = 1\npanic = \"abort\"\nstrip = true\n\n[workspace]\n",
    );
    if std::fs::create_dir_all(dir.join("src")).is_err()
        || std::fs::write(dir.join("Cargo.toml"), manifest).is_err()
        || std::fs::write(dir.join("src").join("main.rs"), &code).is_err()
    {
        eprintln!("Couldn't write the build files to {}", dir.display());
        return 1;
    }
    if !write_icon_files(&dir, icon) {
        return 1;
    }
    eprintln!("Compiling {} (this takes a little while the first time)...", path);
    let target = work.join("aot-target");
    let result = std::process::Command::new("cargo")
        .args(["build", "--release", "--quiet", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(&target)
        .env("RUSTFLAGS", format!("-C target-cpu={}", cpu.unwrap_or("native")))
        .output();
    let output = match result {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Couldn't run cargo: {}", e);
            eprintln!("`antares build` needs Rust. Install it from https://rustup.rs and try again.");
            return 1;
        }
    };
    if !output.status.success() {
        eprintln!("The compiler could not build this program. This is a problem in Antares, not in your program.");
        eprintln!("Generated code: {}", dir.join("src").join("main.rs").display());
        let text = String::from_utf8_lossy(&output.stderr);
        for line in text.lines().take(60) {
            eprintln!("{}", line);
        }
        return 1;
    }
    let exe = format!("{}{}", stem, std::env::consts::EXE_SUFFIX);
    let built = target.join("release").join(&exe);
    let dest = match out {
        Some(o) => PathBuf::from(o),
        None => source.with_file_name(&exe),
    };
    if let Err(e) = std::fs::copy(&built, &dest) {
        eprintln!("Built, but couldn't copy the program to {}: {}", dest.display(), e);
        return 1;
    }
    eprintln!("Built {}", dest.display());
    0
}
