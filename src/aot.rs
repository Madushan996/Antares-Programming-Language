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
        Ty::Int => "i64",
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
        Ty::Int => "0i64",
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
        _ => {}
    }
    let as_dyn = match from {
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
                self.var_ty(t, i)
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
            format!("(*G).v{}", idx)
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
                return (format!("(*G).v{}.clone()", idx), Ty::Dyn);
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
                    let code = format!("bcall({}, vec![{}], {})?", i, parts.join(", "), line);
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
                        let (c, t) = self.expr(a, sc);
                        parts.push(conv(c, t, self.var_ty_param(s, i)));
                    }
                    if self.scopes[s].ret == Ty::Unit {
                        (format!("{{ f{}({})?; Value::Nothing }}", s, parts.join(", ")), Ty::Dyn)
                    } else {
                        (format!("f{}({})?", s, parts.join(", ")), self.scopes[s].ret)
                    }
                }
                Callee::Dynamic => {
                    let fc = self.dyn_code(f, sc);
                    let code = format!("{{ let f_ = {}; call(&f_, vec![{}], {})? }}", fc, self.dyn_list(args, sc), line);
                    (code, Ty::Dyn)
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
                    let it = self.ty(&idx[0], sc);
                    // which reader: (row kind, position kind)
                    let (ic, reader) = if want == Ty::Int {
                        (self.int_code(&idx[0], sc, *line), "iget")
                    } else if it == Ty::Int {
                        (self.int_code(&idx[0], sc, *line), "aget_i")
                    } else {
                        (self.expr(&idx[0], sc).0, "aget")
                    };
                    // A variable is read in place; no copy of the handle is needed.
                    if let Expr::Var { depth, idx: vi, .. } = &**base {
                        let t = self.resolve(sc, *depth);
                        if !(t != 0 && self.scopes[t].captured[*vi as usize]) && !(t == 0 && self.consts.contains_key(&(*vi as usize))) {
                            let p = self.place(t, *vi as usize);
                            return (format!("{{ let i_ = {}; {}(&{}, i_, {})? }}", ic, reader, p, line), want);
                        }
                    }
                    let (bc, _) = self.expr(base, sc);
                    return (format!("{{ let a_ = {}; let i_ = {}; {}(&a_, i_, {})? }}", bc, ic, reader, line), want);
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

    fn ret_str(&self) -> &'static str {
        match self.ret {
            None => "()",
            Some(t) => rust_ty(t),
        }
    }

    fn emit_return(&self, code: &str) -> String {
        if self.try_depth > 0 {
            format!("return Ok(Flow::Ret({}));\n", code)
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
                let (c, t) = self.expr(value, sc);
                self.assign(sc, i, conv(c, t, self.var_ty(sc, i)))
            }
            Stmt::AssignOuter { depth, idx, value } => {
                let t = self.resolve(sc, *depth);
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
                if self.var_ty(t, i) == Ty::Arr {
                    if let Some(ie) = self.fast_set(path) {
                        let vc = self.num_code(value, sc);
                        let (ic, writer) = if self.ty(ie, sc) == Ty::Int {
                            (self.int_code(ie, sc, *line), "aset_i")
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
                        let ic = self.int_code(ie, sc, *line);
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
                    Some(v) => {
                        let vt = self.var_ty(sc, *v as usize);
                        self.assign(sc, *v as usize, conv(format!("i{}", k), Ty::Num, vt))
                    }
                    None => String::new(),
                };
                self.loops.push((k, self.try_depth));
                let b = self.block(body, sc, false);
                self.loops.pop();
                format!(
                    "{{ let n{k} = {n}; let mut i{k} = 0.0f64;\n'l{k}: while i{k} < n{k} {{\ni{k} += 1.0;\n{set}{b}}} }}\n",
                    k = k,
                    n = n,
                    set = set,
                    b = b
                )
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
    fn fn_body(&mut self, s: usize, ret: Ty) -> String {
        let ret = if ret == Ty::Unit { None } else { Some(ret) };
        let def: &FnDef = unsafe { &*self.scopes[s].def };
        let saved = (std::mem::take(&mut self.loops), self.try_depth, self.ret);
        self.try_depth = 0;
        self.ret = ret;
        let mut out = format!("let _d = enter({:?})?;\n", self.scopes[s].name);
        out.push_str(&self.locals(s));
        out.push_str(&self.block(&def.body, s, true));
        self.loops = saved.0;
        self.try_depth = saved.1;
        self.ret = saved.2;
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
        let body = self.fn_body(s, Ty::Dyn);
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
    fn static_item(&mut self, s: usize) {
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
        let body = self.fn_body(s, ret);
        let item = format!(
            "unsafe fn f{}({}) -> R<{}> {{\n{}{}}}\n\n",
            s,
            sig.join(", "),
            rust_ty(ret),
            cells,
            body
        );
        self.items.push_str(&item);
    }

    fn program(&mut self, top: &[Stmt], files: &[(String, String)], main_src: &str) -> String {
        let mut statics: Vec<(usize, usize)> = self.static_fn.iter().map(|(a, b)| (*a, *b)).collect();
        statics.sort();
        for (_, s) in &statics {
            self.static_item(*s);
        }
        self.ret = None;
        let body = self.block(top, 0, false);

        // functions that are also passed around as values get a wrapper in their global slot
        let mut init = String::new();
        for (gi, s) in &statics {
            if self.scopes[*s].escapes {
                let n = self.scopes[*s].nparams;
                let args: Vec<String> = (0..n).map(|_| "a_.next().unwrap()".to_string()).collect();
                init.push_str(&format!(
                    "(*G).v{gi} = rust_fn({name:?}, {n}, |args: Vec<Value>| -> R<Value> {{ let mut a_ = args.into_iter(); let r_ = unsafe {{ f{s}({args})? }}; Ok({conv}) }});\n",
                    gi = gi,
                    name = self.scopes[*s].name,
                    n = n,
                    s = s,
                    args = args.join(", "),
                    conv = conv("r_".into(), self.scopes[*s].ret, Ty::Dyn)
                ));
            }
        }

        let g = &self.scopes[0];
        let mut fields = String::new();
        let mut inits = String::new();
        for i in 0..g.vars.len() {
            if self.is_builtin(i) {
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
            "// Generated by `antares build`. Do not edit.\n#![allow(warnings)]\nuse antares::rt::*;\n\nstruct Globals {{\n{fields}}}\n\nstatic mut G: *mut Globals = std::ptr::null_mut();\nstatic mut S: *const Vec<Value> = std::ptr::null();\nstatic mut A: *const Vec<T> = std::ptr::null();\n\n#[inline(always)]\nunsafe fn sc(i: usize) -> Value {{ (&*S)[i].clone() }}\n\n#[inline(always)]\nunsafe fn ac(i: usize) -> T {{ (&*A)[i].clone() }}\n\n{items}unsafe fn main_body() -> R<()> {{\nlet _d = enter(\"program\")?;\n{init}{body}Ok(())\n}}\n\nfn body() -> R<()> {{\n    let mut g = Globals {{ {inits} }};\n    let strs: Vec<Value> = vec![{strs}];\n    let arrs: Vec<T> = vec![{arrs}];\n    unsafe {{\n        G = &mut g;\n        S = &strs;\n        A = &arrs;\n        main_body()\n    }}\n}}\n\nstatic FILES: &[(&str, &str)] = &[{files}];\nstatic MAIN_SRC: &str = {main:?};\n\nfn main() {{\n    start(FILES, MAIN_SRC, body)\n}}\n",
            fields = fields,
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
    };
    cx.infer(&stmts);
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
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = root.join("target").join("aot").join(&stem);
    let manifest = format!(
        "[package]\nname = \"{stem}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\nantares = {{ path = {root:?} }}\n\n[profile.release]\nopt-level = 3\nlto = \"fat\"\ncodegen-units = 1\npanic = \"abort\"\nstrip = true\n\n[workspace]\n",
        stem = stem,
        root = root.to_string_lossy().replace('\\', "/")
    );
    if std::fs::create_dir_all(dir.join("src")).is_err()
        || std::fs::write(dir.join("Cargo.toml"), manifest).is_err()
        || std::fs::write(dir.join("src").join("main.rs"), &code).is_err()
    {
        eprintln!("Couldn't write the build files to {}", dir.display());
        return 1;
    }
    eprintln!("Compiling {} (this takes a little while the first time)...", path);
    let target = root.join("target").join("aot-target");
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
            eprintln!("Couldn't run cargo (is Rust installed?): {}", e);
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
