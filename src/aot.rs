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
        _ => code,
    };
    match to {
        Ty::Num => format!("as_num({}, 0)?", as_dyn),
        Ty::Bool => format!("as_bool({}, 0)?", as_dyn),
        Ty::Arr => format!("as_arr({}, 0)?", as_dyn),
        Ty::Int => format!("as_int({}, 0)?", as_dyn),
        Ty::IArr => format!("as_iarr({}, 0)?", as_dyn),
        Ty::Whole => format!("(as_num({}, 0)? as i64)", as_dyn),
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
    matches!(name, "map" | "filter" | "reduce")
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
        "int" | "popcount" | "lowest_bit" | "highest_bit" | "div" | "rand_bits" => Int,
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
}

/// What an innermost counting loop does with rows (see `Cx::fast_rows`).
struct FastScan {
    ok: bool,
    /// Variables used in some other way than `row[counter + constant]`.
    bad: BTreeSet<usize>,
    /// Row variable: lowest and highest constant added to the counter, and
    /// whether the loop writes to it.
    rows: std::collections::BTreeMap<usize, (i64, i64, bool)>,
}

impl FastScan {
    fn note(&mut self, r: usize, off: i64, write: bool) {
        let e = self.rows.entry(r).or_insert((off, off, false));
        e.0 = e.0.min(off);
        e.1 = e.1.max(off);
        e.2 |= write;
    }
}

/// Where a variable gets a value from (for finding whole-number variables).
enum Src {
    /// An assignment, or an input of a function called by name.
    Expr(*const Expr, usize),
    /// The variable of an `each` loop over this.
    Each(*const Expr, usize),
    /// Anything else.
    Other,
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

    /// The value of an expression that is a known number when compiling.
    fn const_of(&self, e: &Expr, sc: usize) -> Option<f64> {
        match e {
            Expr::Num(n) => Some(*n),
            Expr::Neg(a, _) => self.const_of(a, sc).map(|x| -x),
            Expr::Var { depth, idx, .. } if self.resolve(sc, *depth) == 0 => self.consts.get(&(*idx as usize)).copied(),
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

    fn tail_ty(&self, stmts: &[Stmt], sc: usize) -> Ty {
        match stmts.last() {
            None => Ty::Unit,
            Some(Stmt::Expr(e)) => self.ty(e, sc),
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
                        matches!(self.ty(ie, sc), Num | ILit | Int | Bot)
                            && matches!(self.var_ty(t, *idx as usize), Arr | IArr | Bot)
                            && matches!(self.ty(value, sc), Num | ILit | Int | Bot)
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
                        self.ty(e, sc)
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
            Stmt::Descend { loss, rate, .. } => {
                self.walk_expr(loss, sc);
                self.walk_expr(rate, sc);
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
                    parts.push(self.depth_arg().to_string());
                    let call = self.static_call(s, format!("f{}({})", s, parts.join(", ")));
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
                if want == Ty::Num || want == Ty::Int {
                    if let Expr::Var { depth, idx: vi, .. } = &**base {
                        if let Some((tb, int)) = self.table_of(sc, *depth, *vi) {
                            let k = if int { "i" } else { "a" };
                            let code = match self.table_pos(&idx[0], sc, *line) {
                                Ok((ic, num)) => format!("tget_{}(&*std::ptr::addr_of!({}), {}, {}, {})?", k, tb, ic, num, line),
                                Err(fc) => format!("tget_{}f(&*std::ptr::addr_of!({}), {}, {})?", k, tb, fc, line),
                            };
                            return (code, want);
                        }
                    }
                    if let Expr::Var { depth, idx: vi, .. } = &**base {
                        let t = self.resolve(sc, *depth);
                        if let (Some(p), Some(ic)) = (self.fast_ptr(t, *vi as usize), self.icode(&idx[0], sc, true)) {
                            return (format!("(*{}.add(({}) as usize))", p, ic), want);
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
                    AskKind::Text => "Text",
                    AskKind::Number => "Number",
                    AskKind::YesNo => "YesNo",
                    AskKind::List => "List",
                };
                let m = self.dyn_code(model, sc);
                let p = self.dyn_code(prompt, sc);
                (format!("{{ let m_ = {}; let p_ = {}; ask(&m_, &p_, AskKind::{}, {})? }}", m, p, k, line), Ty::Dyn)
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
        }
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
                Stmt::Descend { loss, rate, .. } => {
                    self.fast_scan_expr(loss, sc, v, f);
                    self.fast_scan_expr(rate, sc, v, f);
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
    fn fast_rows(&self, body: &[Stmt], sc: usize, v: usize) -> Vec<(usize, i64, i64, bool)> {
        let mut f = FastScan { ok: true, bad: BTreeSet::new(), rows: Default::default() };
        self.fast_scan_block(body, sc, v, &mut f);
        if !f.ok {
            return vec![];
        }
        f.rows.iter().filter(|(r, _)| !f.bad.contains(r)).map(|(r, (lo, hi, w))| (*r, *lo, *hi, *w)).collect()
    }

    /// The loop body over `rows` with direct pointers, and the code that gets
    /// those pointers (to run after the loop's checks have passed).
    fn fast_body(&mut self, body: &[Stmt], sc: usize, k: usize, rows: &[(usize, i64, i64, bool)]) -> (String, String) {
        let mut setup = String::new();
        let before = self.fast.len();
        for &(r, _, _, write) in rows {
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

    /// The length of row variable `r` of scope `sc`.
    fn row_len(&self, sc: usize, r: usize) -> String {
        if self.var_ty(sc, r) == Ty::IArr {
            format!("({}.len())", self.place(sc, r))
        } else {
            format!("alen(&{})", self.place(sc, r))
        }
    }

    fn fast_ptr(&self, t: usize, i: usize) -> Option<String> {
        self.fast.iter().find(|(r, _)| *r == (t, i)).map(|(_, p)| p.clone())
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
                    out.push(((self.resolve(sc, *depth), *idx as usize), Src::Other));
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
                        self.whole_sources_expr(e, sc, out);
                    }
                }
                Stmt::Repeat { count, body, .. } => {
                    // the counter of `repeat ... as i` counts in steps of 1
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
                Stmt::Descend { loss, rate, .. } => {
                    self.whole_sources_expr(loss, sc, out);
                    self.whole_sources_expr(rate, sc, out);
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
                                out.push(((s, i), Src::Expr(a, sc)));
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
                    Src::Expr(e, sc) => self.counts(unsafe { &*e }, sc, &w),
                    Src::Each(iter, sc) => match self.range_parts(unsafe { &*iter }, sc) {
                        Some((start, _, step)) => {
                            step == step.trunc() && step.abs() <= 64.0 && start.is_none_or(|x| self.counts(x, sc, &w))
                        }
                        None => false,
                    },
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
                        bad.insert(*idx as usize);
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
                for a in args {
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
    fn find_tables(&mut self, top: &[Stmt]) {
        let n = self.scopes.len();
        let mut first: HashMap<usize, (usize, *const Expr)> = HashMap::new();
        for (k, st) in top.iter().enumerate() {
            if let Stmt::Assign { idx, value } = st {
                first.entry(*idx as usize).or_insert((k, value as *const Expr));
            }
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
            if let Stmt::Assign { idx, value: Expr::Call(f, args, _) } = st {
                let i = *idx as usize;
                let kind = match (self.callee(f, 0), &args[..]) {
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
                        && self.gcount[i] == 1
                        && first.get(&i).map(|f| f.0) == Some(k)
                        && !bad.contains(&i)
                        && !self.scopes[0].captured[i]
                        && !used_before
                    {
                        self.tables.insert(i, (len as usize, int));
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
                    // a fixed table: it already holds zeros
                    return String::new();
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
                if let (Some((tb, int)), Some(ie)) = (self.table_of(sc, *depth, *idx), self.fast_set(path)) {
                    let vc = if int { self.int_code(value, sc, *line) } else { self.num_code(value, sc) };
                    let k = if int { "i" } else { "a" };
                    return match self.table_pos(ie, sc, *line) {
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
                if let (Some(p), Some(ie)) = (self.fast_ptr(t, i), self.fast_set(path)) {
                    if let Some(ic) = self.icode(ie, sc, true) {
                        let vc = if self.var_ty(t, i) == Ty::IArr { self.int_code(value, sc, *line) } else { self.num_code(value, sc) };
                        return format!("{{ let v_ = {}; let i_ = {}; *{}.add(i_ as usize) = v_; }}\n", vc, ic, p);
                    }
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
                if !rows.is_empty() && rows.iter().all(|r| r.1 >= -1) {
                    let conds: Vec<String> =
                        rows.iter().map(|&(r, _, hi, _)| format!("(n{} as i128) + {} < ({} as i128)", k, hi, self.row_len(sc, r))).collect();
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
                        .map(|&(r, lo, hi, _)| {
                            format!("x{k}.wrapping_add({lo}) >= 0 && e{k} + {hi:?}f64 <= ({len} as f64)", k = k, lo = lo, hi = hi as f64, len = self.row_len(sc, r))
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
            Stmt::Descend { loss, rate, adam, line } => {
                let l = self.dyn_code(loss, sc);
                let r = self.dyn_code(rate, sc);
                format!("{{ let l_ = {}; let r_ = {}; descend(&l_, &r_, {}, {})?; }}\n", l, r, adam, line)
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
        out.push_str(&self.block(&def.body, s, true));
        self.loops = saved.0;
        self.try_depth = saved.1;
        self.ret = saved.2;
        self.cur = saved.3;
        self.plain = saved.4;
        out
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
            "{{ {clones}rust_fn({name:?}, {n}, move |args: Vec<Value>| -> R<Value> {{ unsafe {{\n{params}{body}}} }}) }}",
            clones = clones,
            name = self.scopes[s].name,
            n = n,
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
                    "(*GV.as_mut_ptr()).v{gi} = rust_fn({name:?}, {n}, |args: Vec<Value>| -> R<Value> {{ let mut a_ = args.into_iter(); let r_ = unsafe {{ {call} }}; Ok({conv}) }});\n",
                    gi = gi,
                    name = self.scopes[*s].name,
                    n = n,
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
                tables.push_str(&format!("static mut TB{}: [{}; {}] = [{}; {}];   // {}\n", i, ty, len, zero, len, self.gnames.get(i).map(|s| s.as_str()).unwrap_or("")));
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

pub fn build(path: &str, out: Option<&str>, cpu: Option<&str>, embed: &[String]) -> i32 {
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
    };
    cx.infer(&stmts);
    cx.infer_whole(&stmts);
    cx.find_tables(&stmts);
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
    let stem: String = source
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "program".into())
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect();
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
