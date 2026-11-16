//! The fast engine. Each function is translated once into a flat list of
//! small instructions, which a single tight loop then runs.
//!
//! Common operations (numbers, comparisons, variables, calls, loops, list
//! indexing) have direct instructions with a quick path for plain numbers.
//! Anything unusual falls back to the tree evaluator in `interp.rs` on the
//! original syntax node, so both engines always agree on results and errors.

use crate::ast::*;
use crate::interp::{binop, index_value, iter_items, set_index, Ctrl, Env, Interp};
use crate::value::{pack_list, Closure, Value};
use std::rc::Rc;

const NONE: u32 = u32::MAX;

#[derive(Clone, Copy)]
pub enum Op {
    Num(f64),
    Bool(bool),
    Nothing,
    /// Push a constant (text) from the function's constant list.
    Const(u32),
    /// name[index] = value, with value and index on the stack.
    SetIndex { node: *const Stmt, idx: u32, depth: u32 },
    /// Evaluate a syntax node with the tree evaluator and push the result.
    Ast(*const Expr),
    /// Run a statement with the tree evaluator.
    StmtAst(*const Stmt),
    LoadLocal(u32, *const Expr),
    Load(u32, u32, *const Expr),
    Store(u32),
    StoreUp(u32, u32),
    Pop,
    PopN(u32),
    Bin(BinOp, u32),
    /// top = top (op) constant
    BinConst(BinOp, u32, f64),
    /// push local (op) constant; the node is the fallback
    LocalBinConst(*const Expr, f64, u32, BinOp),
    Neg(u32),
    Not(u32),
    AndJump(u32, u32),
    OrJump(u32, u32),
    ToBool(u32),
    Jump(u32),
    JumpIfFalse(u32, u32),
    Call(u32, u32),
    /// Call a function held in a variable, without pushing it first.
    CallVar { node: *const Expr, idx: u32, depth: u16, n: u16 },
    Index(u32, u32),
    Field(*const Expr),
    MakeList(u32),
    RepeatInit(u32),
    RepeatNext(u32, u32),
    EachInit(u32),
    EachNext(u32, u32),
    PushHandler(u32, u32),
    PopHandler,
    Return,
}

#[derive(Default)]
pub struct Code {
    pub ops: Vec<Op>,
    pub consts: Vec<Value>,
}

struct LoopCtx {
    head: u32,
    breaks: Vec<usize>,
    slots: u32,
    handler_depth: u32,
}

struct Compiler {
    ops: Vec<Op>,
    consts: Vec<Value>,
    loops: Vec<LoopCtx>,
    handler_depth: u32,
}

pub fn compile(body: &[Stmt]) -> Code {
    let mut c = Compiler { ops: Vec::new(), consts: Vec::new(), loops: Vec::new(), handler_depth: 0 };
    c.block(body, true);
    c.ops.push(Op::Return);
    for i in 0..c.ops.len() {
        if let Op::Jump(mut t) = c.ops[i] {
            let mut hops = 0;
            while let Op::Jump(next) = c.ops[t as usize] {
                t = next;
                hops += 1;
                if hops > 16 {
                    break;
                }
            }
            c.ops[i] = match c.ops[t as usize] {
                Op::Return => Op::Return,
                _ => Op::Jump(t),
            };
        }
    }
    Code { ops: c.ops, consts: c.consts }
}

impl Compiler {
    fn emit(&mut self, op: Op) -> usize {
        self.ops.push(op);
        self.ops.len() - 1
    }

    fn here(&self) -> u32 {
        self.ops.len() as u32
    }

    fn patch(&mut self, at: usize) {
        let target = self.here();
        match &mut self.ops[at] {
            Op::Jump(t)
            | Op::JumpIfFalse(t, _)
            | Op::AndJump(t, _)
            | Op::OrJump(t, _)
            | Op::PushHandler(t, _) => *t = target,
            Op::RepeatNext(_, t) | Op::EachNext(_, t) => *t = target,
            _ => {}
        }
    }

    fn block(&mut self, stmts: &[Stmt], want: bool) {
        match stmts.split_last() {
            None => {
                if want {
                    self.emit(Op::Nothing);
                }
            }
            Some((last, rest)) => {
                for s in rest {
                    self.stmt(s, false);
                }
                self.stmt(last, want);
            }
        }
    }

    fn leave_handlers(&mut self, down_to: u32) {
        for _ in down_to..self.handler_depth {
            self.emit(Op::PopHandler);
        }
    }

    fn finish_loop(&mut self) {
        let ctx = self.loops.pop().unwrap();
        for b in ctx.breaks {
            self.patch(b);
        }
    }

    fn stmt(&mut self, s: &Stmt, want: bool) {
        let mut pushed = false;
        match s {
            Stmt::Expr(e) => {
                self.expr(e);
                if !want {
                    self.emit(Op::Pop);
                }
                pushed = true;
            }
            Stmt::Assign { idx, value } => {
                self.expr(value);
                self.emit(Op::Store(*idx));
            }
            Stmt::AssignOuter { depth, idx, value } => {
                self.expr(value);
                self.emit(Op::StoreUp(*depth, *idx));
            }
            Stmt::When { arms, otherwise } => {
                let mut ends = vec![];
                for (cond, body, line) in arms {
                    self.expr(cond);
                    let j = self.emit(Op::JumpIfFalse(0, *line));
                    self.block(body, want);
                    ends.push(self.emit(Op::Jump(0)));
                    self.patch(j);
                }
                match otherwise {
                    Some(body) => self.block(body, want),
                    None => {
                        if want {
                            self.emit(Op::Nothing);
                        }
                    }
                }
                for e in ends {
                    self.patch(e);
                }
                pushed = true;
            }
            Stmt::Repeat { count, var, body, line } => {
                self.expr(count);
                self.emit(Op::RepeatInit(*line));
                let head = self.emit(Op::RepeatNext(var.unwrap_or(NONE), 0));
                self.loops.push(LoopCtx {
                    head: head as u32,
                    breaks: vec![head],
                    slots: 2,
                    handler_depth: self.handler_depth,
                });
                self.block(body, false);
                self.emit(Op::Jump(head as u32));
                self.finish_loop();
            }
            Stmt::Each { var, iter, body, line } => {
                self.expr(iter);
                self.emit(Op::EachInit(*line));
                let head = self.emit(Op::EachNext(*var, 0));
                self.loops.push(LoopCtx {
                    head: head as u32,
                    breaks: vec![head],
                    slots: 2,
                    handler_depth: self.handler_depth,
                });
                self.block(body, false);
                self.emit(Op::Jump(head as u32));
                self.finish_loop();
            }
            Stmt::While { cond, body, line } => {
                let head = self.here();
                self.expr(cond);
                let exit = self.emit(Op::JumpIfFalse(0, *line));
                self.loops.push(LoopCtx {
                    head,
                    breaks: vec![exit],
                    slots: 0,
                    handler_depth: self.handler_depth,
                });
                self.block(body, false);
                self.emit(Op::Jump(head));
                self.finish_loop();
            }
            Stmt::Give(e) => {
                match e {
                    Some(e) => self.expr(e),
                    None => {
                        self.emit(Op::Nothing);
                    }
                }
                self.emit(Op::Return);
            }
            Stmt::Stop(_) | Stmt::Skip(_) => match self.loops.last() {
                None => {
                    self.emit(Op::StmtAst(s));
                }
                Some(ctx) => {
                    let (head, slots, hd) = (ctx.head, ctx.slots, ctx.handler_depth);
                    self.leave_handlers(hd);
                    if let Stmt::Stop(_) = s {
                        if slots > 0 {
                            self.emit(Op::PopN(slots));
                        }
                        let j = self.emit(Op::Jump(0));
                        self.loops.last_mut().unwrap().breaks.push(j);
                    } else {
                        self.emit(Op::Jump(head));
                    }
                }
            },
            Stmt::Try { body, var, handler } => {
                let h = self.emit(Op::PushHandler(0, var.unwrap_or(NONE)));
                self.handler_depth += 1;
                self.block(body, want);
                self.emit(Op::PopHandler);
                self.handler_depth -= 1;
                let j = self.emit(Op::Jump(0));
                self.patch(h);
                self.block(handler, want);
                self.patch(j);
                pushed = true;
            }
            Stmt::SetPath { depth, idx, path, value, .. }
                if path.len() == 1
                    && matches!(&path[0], Access::Index(i) if i.len() == 1 && !matches!(i[0], Expr::Range(..))) =>
            {
                self.expr(value);
                if let Access::Index(i) = &path[0] {
                    self.expr(&i[0]);
                }
                self.emit(Op::SetIndex { node: s, idx: *idx, depth: *depth });
            }
            Stmt::Learn { .. } | Stmt::SetPath { .. } | Stmt::Show(_) | Stmt::Descend { .. } => {
                self.emit(Op::StmtAst(s));
            }
        }
        if want && !pushed {
            self.emit(Op::Nothing);
        }
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Num(n) => {
                self.emit(Op::Num(*n));
            }
            Expr::Bool(b) => {
                self.emit(Op::Bool(*b));
            }
            Expr::Nothing => {
                self.emit(Op::Nothing);
            }
            Expr::Str(text) => {
                self.consts.push(Value::Str(text.clone()));
                self.emit(Op::Const(self.consts.len() as u32 - 1));
            }
            Expr::Var { depth, idx, .. } => {
                if *depth == 0 {
                    self.emit(Op::LoadLocal(*idx, e));
                } else {
                    self.emit(Op::Load(*depth, *idx, e));
                }
            }
            Expr::Binary(op, a, b, line) => {
                if let Expr::Num(c) = **b {
                    if let Expr::Var { depth: 0, idx, .. } = **a {
                        self.emit(Op::LocalBinConst(e, c, idx, *op));
                    } else {
                        self.expr(a);
                        self.emit(Op::BinConst(*op, *line, c));
                    }
                } else {
                    self.expr(a);
                    self.expr(b);
                    self.emit(Op::Bin(*op, *line));
                }
            }
            Expr::Neg(a, line) => {
                self.expr(a);
                self.emit(Op::Neg(*line));
            }
            Expr::Not(a, line) => {
                self.expr(a);
                self.emit(Op::Not(*line));
            }
            Expr::And(a, b, line) => {
                self.expr(a);
                let j = self.emit(Op::AndJump(0, *line));
                self.expr(b);
                self.emit(Op::ToBool(*line));
                self.patch(j);
            }
            Expr::Or(a, b, line) => {
                self.expr(a);
                let j = self.emit(Op::OrJump(0, *line));
                self.expr(b);
                self.emit(Op::ToBool(*line));
                self.patch(j);
            }
            Expr::Call(f, args, line) => {
                if let Expr::Var { depth, idx, .. } = **f {
                    if depth <= u16::MAX as u32 && args.len() <= u16::MAX as usize {
                        for a in args {
                            self.expr(a);
                        }
                        self.emit(Op::CallVar { node: e, idx, depth: depth as u16, n: args.len() as u16 });
                        return;
                    }
                }
                self.expr(f);
                for a in args {
                    self.expr(a);
                }
                self.emit(Op::Call(args.len() as u32, *line));
            }
            Expr::Index(base, idx, line) => {
                if idx.iter().any(|i| matches!(i, Expr::Range(..))) {
                    self.emit(Op::Ast(e));
                } else {
                    self.expr(base);
                    for i in idx {
                        self.expr(i);
                    }
                    self.emit(Op::Index(idx.len() as u32, *line));
                }
            }
            Expr::Field(base, _, _) => {
                self.expr(base);
                self.emit(Op::Field(e));
            }
            Expr::List(items) => {
                for i in items {
                    self.expr(i);
                }
                self.emit(Op::MakeList(items.len() as u32));
            }
            _ => {
                self.emit(Op::Ast(e));
            }
        }
    }
}

/// Arithmetic and comparison on two plain numbers.
#[inline(always)]
fn fast(op: BinOp, p: f64, q: f64) -> Option<Value> {
    Some(match op {
        BinOp::Add => Value::Num(p + q),
        BinOp::Sub => Value::Num(p - q),
        BinOp::Mul => Value::Num(p * q),
        BinOp::Div => Value::Num(p / q),
        BinOp::Lt => Value::Bool(p < q),
        BinOp::Gt => Value::Bool(p > q),
        BinOp::Le => Value::Bool(p <= q),
        BinOp::Ge => Value::Bool(p >= q),
        BinOp::Eq => Value::Bool(p == q),
        BinOp::Ne => Value::Bool(p != q),
        BinOp::Mod => Value::Num(p.rem_euclid(q)),
        BinOp::Pow => Value::Num(p.powf(q)),
        _ => return None,
    })
}

/// The most common number helpers, answered directly.
#[inline]
fn fast_native(name: &str, args: &[Value]) -> Option<Value> {
    match args {
        [Value::Num(x)] => Some(Value::Num(match name {
            "abs" => x.abs(),
            "floor" => x.floor(),
            "trunc" => x.trunc(),
            "round" => x.round(),
            _ => return None,
        })),
        [Value::Num(x), Value::Num(y)] => Some(Value::Num(match name {
            "min" => x.min(*y),
            "max" => x.max(*y),
            _ => return None,
        })),
        _ => None,
    }
}

pub(crate) type Handler = (u32, u32, usize, usize);

/// A paused caller, waiting for the function it called to finish.
pub(crate) struct Frame {
    ops: *const [Op],
    pc: usize,
    env: Rc<Env>,
    keep: Option<Rc<Closure>>,
    handlers_base: usize,
    stack_base: usize,
    consts: *const Vec<Value>,
}

impl Interp {
    pub(crate) fn exec_code(&mut self, code: &Code, env: &Rc<Env>) -> Result<Value, Ctrl> {
        let base = self.stack.len();
        let r = self.run_loop(code, env);
        self.stack.truncate(base);
        r
    }

    fn run_loop(&mut self, code: &Code, env: &Rc<Env>) -> Result<Value, Ctrl> {
        // Calls between Antares functions do not re-enter this function:
        // the caller is parked on `self.frames` and the loop carries on in
        // the callee. `entry` marks where this invocation's frames begin.
        let entry = self.frames.len();
        // The instruction list is owned by the function being run, which
        // `keep` (or our caller, for the first one) holds alive.
        let mut ops: &[Op] = unsafe { &*(code.ops.as_slice() as *const [Op]) };
        let mut consts: *const Vec<Value> = &code.consts;
        let mut pc = 0usize;
        let mut env: Rc<Env> = env.clone();
        let mut keep: Option<Rc<Closure>> = None;
        let mut hbase = self.handlers.len();
        let mut sbase = self.stack.len();
        // The current frame's variables. Accessed directly for speed; each
        // access is momentary and never held across a call, and everything
        // runs on one thread, so it cannot overlap another borrow.
        let mut locals: *mut Vec<Value> = env.vars.as_ptr();
        macro_rules! local {
            ($i:expr) => {
                unsafe { &mut (&mut *locals)[$i as usize] }
            };
        }
        loop {
            let op = ops[pc];
            pc += 1;
            let res: Result<(), Ctrl> = 'op: {
                macro_rules! tri {
                    ($e:expr) => {
                        match $e {
                            Ok(v) => v,
                            Err(e) => break 'op Err(e),
                        }
                    };
                }
                // A comparison that feeds straight into a conditional jump
                // takes the jump here, saving one instruction.
                macro_rules! push_or_branch {
                    ($v:expr) => {{
                        let v = $v;
                        if let Value::Bool(b) = v {
                            if let Op::JumpIfFalse(t, _) = ops[pc] {
                                pc = if b { pc + 1 } else { t as usize };
                                break 'op Ok(());
                            }
                        }
                        self.stack.push(v);
                        break 'op Ok(());
                    }};
                }
                match op {
                    Op::Num(n) => self.stack.push(Value::Num(n)),
                    Op::Bool(b) => self.stack.push(Value::Bool(b)),
                    Op::Nothing => self.stack.push(Value::Nothing),
                    Op::Const(i) => {
                        let v = unsafe { (&*consts)[i as usize].clone() };
                        self.stack.push(v);
                    }
                    Op::SetIndex { node, idx, depth } => {
                        let key = self.stack.pop().unwrap();
                        let v = self.stack.pop().unwrap();
                        let slot: &mut Value = if depth == 0 {
                            local!(idx)
                        } else {
                            let mut home: &Env = &env;
                            for _ in 0..depth {
                                home = home.parent.as_ref().unwrap();
                            }
                            unsafe { &mut (&mut *home.vars.as_ptr())[idx as usize] }
                        };
                        // Quick path: a whole-number position in a list or a
                        // plain row of numbers that nothing else is sharing.
                        if let Value::Num(i) = &key {
                            let i = *i;
                            if i >= 0.0 && i == i.trunc() {
                                let k = i as usize;
                                match slot {
                                    Value::List(rc) if k < rc.len() => {
                                        Rc::make_mut(rc)[k] = v;
                                        break 'op Ok(());
                                    }
                                    Value::Tensor(t) if t.shape.len() == 1 && k < t.shape[0] && !t.requires => {
                                        if let (Some(tn), Value::Num(x)) = (Rc::get_mut(t), &v) {
                                            tn.host_get_mut()[k] = *x;
                                            break 'op Ok(());
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        let (name, line) = match unsafe { &*node } {
                            Stmt::SetPath { name, line, .. } => (name, *line),
                            _ => unreachable!(),
                        };
                        if let Value::Unset = slot {
                            break 'op Err(Ctrl::Err(format!("'{}' doesn't have a value yet", name), line));
                        }
                        // Take the value out while changing it so it is not
                        // copied, then put it back.
                        let mut target = std::mem::replace(slot, Value::Nothing);
                        let r = set_index(&mut target, key, v);
                        *slot = target;
                        tri!(r.map_err(|m| Ctrl::Err(m, line)));
                    }
                    Op::Ast(e) => {
                        let v = tri!(self.eval(unsafe { &*e }, &env));
                        self.stack.push(v);
                    }
                    Op::StmtAst(s) => {
                        tri!(self.exec(unsafe { &*s }, &env));
                    }
                    Op::LoadLocal(i, e) => {
                        let v = local!(i).clone();
                        if let Value::Unset = v {
                            tri!(self.eval(unsafe { &*e }, &env));
                        }
                        self.stack.push(v);
                    }
                    Op::Load(depth, i, e) => {
                        let mut home: &Env = &env;
                        for _ in 0..depth {
                            home = home.parent.as_ref().unwrap();
                        }
                        let v = home.vars.borrow()[i as usize].clone();
                        if let Value::Unset = v {
                            tri!(self.eval(unsafe { &*e }, &env));
                        }
                        self.stack.push(v);
                    }
                    Op::Store(i) => {
                        let v = self.stack.pop().unwrap();
                        *local!(i) = v;
                    }
                    Op::StoreUp(depth, i) => {
                        let v = self.stack.pop().unwrap();
                        let mut home: &Env = &env;
                        for _ in 0..depth {
                            home = home.parent.as_ref().unwrap();
                        }
                        unsafe { (&mut *home.vars.as_ptr())[i as usize] = v };
                    }
                    Op::Pop => {
                        self.stack.pop();
                    }
                    Op::PopN(n) => {
                        let len = self.stack.len();
                        self.stack.truncate(len - n as usize);
                    }
                    Op::Bin(op, line) => {
                        let len = self.stack.len();
                        if let (Value::Num(p), Value::Num(q)) = (&self.stack[len - 2], &self.stack[len - 1]) {
                            if let Some(v) = fast(op, *p, *q) {
                                self.stack.truncate(len - 2);
                                push_or_branch!(v);
                            }
                        }
                        let y = self.stack.pop().unwrap();
                        let x = self.stack.pop().unwrap();
                        let v = tri!(binop(op, &x, &y).map_err(|m| Ctrl::Err(m, line)));
                        self.stack.push(v);
                    }
                    Op::BinConst(op, line, c) => {
                        let len = self.stack.len();
                        if let Value::Num(p) = &self.stack[len - 1] {
                            if let Some(v) = fast(op, *p, c) {
                                self.stack.truncate(len - 1);
                                push_or_branch!(v);
                            }
                        }
                        let x = self.stack.pop().unwrap();
                        let v = tri!(binop(op, &x, &Value::Num(c)).map_err(|m| Ctrl::Err(m, line)));
                        self.stack.push(v);
                    }
                    Op::LocalBinConst(node, c, i, op) => {
                        if let Value::Num(p) = local!(i) {
                            if let Some(v) = fast(op, *p, c) {
                                push_or_branch!(v);
                            }
                        }
                        let v = tri!(self.eval(unsafe { &*node }, &env));
                        self.stack.push(v);
                    }
                    Op::Neg(line) => {
                        let v = self.stack.pop().unwrap();
                        let r = match v {
                            Value::Num(n) => Value::Num(-n),
                            other => tri!(binop(BinOp::Sub, &Value::Num(0.0), &other).map_err(|_| Ctrl::Err(
                                format!("can't put a minus in front of a {}", other.kind()),
                                line
                            ))),
                        };
                        self.stack.push(r);
                    }
                    Op::Not(line) => {
                        let v = self.stack.pop().unwrap();
                        let b = match v {
                            Value::Bool(b) => b,
                            other => tri!(Interp::not_yes_no(other, line)),
                        };
                        self.stack.push(Value::Bool(!b));
                    }
                    Op::AndJump(t, line) | Op::OrJump(t, line) => {
                        let v = self.stack.pop().unwrap();
                        let b = match v {
                            Value::Bool(b) => b,
                            other => tri!(Interp::not_yes_no(other, line)),
                        };
                        let short = matches!(op, Op::OrJump(..));
                        if b == short {
                            self.stack.push(Value::Bool(b));
                            pc = t as usize;
                        }
                    }
                    Op::ToBool(line) => {
                        let v = self.stack.pop().unwrap();
                        let b = match v {
                            Value::Bool(b) => b,
                            other => tri!(Interp::not_yes_no(other, line)),
                        };
                        self.stack.push(Value::Bool(b));
                    }
                    Op::Jump(t) => pc = t as usize,
                    Op::JumpIfFalse(t, line) => {
                        let v = self.stack.pop().unwrap();
                        let b = match v {
                            Value::Bool(b) => b,
                            other => tri!(Interp::not_yes_no(other, line)),
                        };
                        if !b {
                            pc = t as usize;
                        }
                    }
                    Op::CallVar { node, idx, depth, n } => {
                        let n = n as usize;
                        let callee = if depth == 0 {
                            local!(idx).clone()
                        } else {
                            let mut home: &Env = &env;
                            for _ in 0..depth {
                                home = home.parent.as_ref().unwrap();
                            }
                            let v = home.vars.borrow()[idx as usize].clone();
                            v
                        };
                        let (fexpr, line) = match unsafe { &*node } {
                            Expr::Call(f, _, l) => (&**f, *l),
                            _ => unreachable!(),
                        };
                        let start = self.stack.len() - n;
                        match callee {
                            Value::Fn(c) => {
                                if n != c.def.nparams {
                                    break 'op Err(Interp::wrong_count(&c.def, n, line));
                                }
                                if self.depth >= crate::interp::MAX_DEPTH {
                                    break 'op Err(Ctrl::Err(
                                        format!(
                                            "'{}' keeps calling itself without ending ({} levels deep)",
                                            c.def.name,
                                            crate::interp::MAX_DEPTH
                                        ),
                                        line,
                                    ));
                                }
                                let mut frame = self.take_frame();
                                {
                                    let e = Rc::get_mut(&mut frame).unwrap();
                                    let vars = e.vars.get_mut();
                                    vars.extend(self.stack.drain(start..));
                                    if vars.len() != c.def.nslots {
                                        vars.resize(c.def.nslots, Value::Unset);
                                    }
                                    e.parent = Some(c.env.clone());
                                }
                                let callee_code = c.def.code.get_or_init(|| compile(&c.def.body));
                                let new_ops: &[Op] = unsafe { &*(callee_code.ops.as_slice() as *const [Op]) };
                                let new_consts: *const Vec<Value> = &callee_code.consts;
                                self.depth += 1;
                                self.frames.push(Frame {
                                    ops: ops as *const [Op],
                                    pc,
                                    env: std::mem::replace(&mut env, frame),
                                    keep: std::mem::replace(&mut keep, Some(c)),
                                    handlers_base: hbase,
                                    stack_base: sbase,
                                    consts,
                                });
                                ops = new_ops;
                                consts = new_consts;
                                pc = 0;
                                locals = env.vars.as_ptr();
                                hbase = self.handlers.len();
                                sbase = self.stack.len();
                            }
                            Value::Unset => {
                                tri!(self.eval(fexpr, &env));
                            }
                            callee => {
                                if let Value::Native(name, _) = &callee {
                                    if let Some(v) = fast_native(name, &self.stack[start..]) {
                                        self.stack.truncate(start);
                                        self.stack.push(v);
                                        break 'op Ok(());
                                    }
                                }
                                let args: Vec<Value> = self.stack.drain(start..).collect();
                                let v = tri!(self.call(&callee, args, line));
                                self.stack.push(v);
                            }
                        }
                    }
                    Op::Call(n, line) => {
                        let start = self.stack.len() - n as usize;
                        let args: Vec<Value> = self.stack.drain(start..).collect();
                        let callee = self.stack.pop().unwrap();
                        let v = tri!(self.call(&callee, args, line));
                        self.stack.push(v);
                    }
                    Op::Index(n, line) => {
                        let len = self.stack.len();
                        if n == 1 {
                            if let Value::Num(i) = &self.stack[len - 1] {
                                let i = *i;
                                if i >= 0.0 && i == i.trunc() {
                                    let k = i as usize;
                                    let hit = match &self.stack[len - 2] {
                                        Value::List(l) if k < l.len() => Some(l[k].clone()),
                                        Value::Tensor(t) if !t.requires && t.shape.len() == 1 && k < t.shape[0] => {
                                            Some(Value::Num(t.vals()[k]))
                                        }
                                        _ => None,
                                    };
                                    if let Some(v) = hit {
                                        self.stack[len - 2] = v;
                                        self.stack.truncate(len - 1);
                                        break 'op Ok(());
                                    }
                                }
                            }
                        }
                        let idx: Vec<Value> = self.stack.drain(len - n as usize..).collect();
                        let b = self.stack.pop().unwrap();
                        let v = tri!(index_value(&b, &idx).map_err(|m| Ctrl::Err(m, line)));
                        self.stack.push(v);
                    }
                    Op::Field(node) => {
                        let (name, line) = match unsafe { &*node } {
                            Expr::Field(_, name, l) => (name, *l),
                            _ => unreachable!(),
                        };
                        let b = self.stack.pop().unwrap();
                        let found = match &b {
                            Value::Map(m) => m.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone()),
                            _ => None,
                        };
                        match found {
                            Some(v) => self.stack.push(v),
                            None => {
                                let msg = match &b {
                                    Value::Map(_) => format!("this map has no '{}'", name),
                                    other => format!(
                                        "only maps have named parts, and this is a {} (to call a function on a value, write value -> {}())",
                                        other.kind(),
                                        name
                                    ),
                                };
                                break 'op Err(Ctrl::Err(msg, line));
                            }
                        }
                    }
                    Op::MakeList(n) => {
                        let len = self.stack.len();
                        let items: Vec<Value> = self.stack.drain(len - n as usize..).collect();
                        self.stack.push(pack_list(items));
                    }
                    Op::RepeatInit(line) => {
                        let c = self.stack.pop().unwrap();
                        let n = match c.scalar() {
                            Some(n) if n >= 0.0 => (n as u64) as f64,
                            _ => break 'op Err(Ctrl::Err("`repeat` needs a count that is zero or more".into(), line)),
                        };
                        self.stack.push(Value::Num(n));
                        self.stack.push(Value::Num(0.0));
                    }
                    Op::RepeatNext(var, exit) => {
                        let len = self.stack.len();
                        if let (Value::Num(n), Value::Num(i)) = (&self.stack[len - 2], &self.stack[len - 1]) {
                            let (n, i) = (*n, *i);
                            if i >= n {
                                self.stack.truncate(len - 2);
                                pc = exit as usize;
                            } else {
                                self.stack[len - 1] = Value::Num(i + 1.0);
                                if var != NONE {
                                    *local!(var) = Value::Num(i + 1.0);
                                }
                            }
                        }
                    }
                    Op::EachInit(line) => {
                        let it = self.stack.pop().unwrap();
                        let list = match it {
                            Value::List(_) => it,
                            Value::Tensor(ref t) if t.shape.len() == 1 && !t.requires => it,
                            other => Value::List(Rc::new(tri!(iter_items(&other).map_err(|m| Ctrl::Err(m, line))))),
                        };
                        self.stack.push(list);
                        self.stack.push(Value::Num(0.0));
                    }
                    Op::EachNext(var, exit) => {
                        let len = self.stack.len();
                        let next = match (&self.stack[len - 2], &self.stack[len - 1]) {
                            (Value::List(l), Value::Num(i)) => l.get(*i as usize).cloned(),
                            (Value::Tensor(t), Value::Num(i)) => t.vals().get(*i as usize).map(|x| Value::Num(*x)),
                            _ => None,
                        };
                        match next {
                            Some(item) => {
                                if let Value::Num(i) = &mut self.stack[len - 1] {
                                    *i += 1.0;
                                }
                                *local!(var) = item;
                            }
                            None => {
                                self.stack.truncate(len - 2);
                                pc = exit as usize;
                            }
                        }
                    }
                    Op::PushHandler(hpc, var) => self.handlers.push((hpc, var, self.stack.len(), self.depth)),
                    Op::PopHandler => {
                        self.handlers.pop();
                    }
                    Op::Return => {
                        let v = self.stack.pop().unwrap_or(Value::Nothing);
                        self.stack.truncate(sbase);
                        self.handlers.truncate(hbase);
                        if self.frames.len() == entry {
                            return Ok(v);
                        }
                        // Back to the caller parked on the frame list.
                        let f = self.frames.pop().unwrap();
                        let finished = std::mem::replace(&mut env, f.env);
                        self.release_frame(finished);
                        keep = f.keep;
                        ops = unsafe { &*f.ops };
                        consts = f.consts;
                        pc = f.pc;
                        hbase = f.handlers_base;
                        sbase = f.stack_base;
                        locals = env.vars.as_ptr();
                        self.depth -= 1;
                        self.stack.push(v);
                    }
                }
                Ok(())
            };
            if let Err(e) = res {
                let mut e = e;
                // Unwind: a `try` in the current function catches the
                // problem; otherwise leave this function and ask its caller.
                loop {
                    match e {
                        Ctrl::Err(msg, _) if self.handlers.len() > hbase => {
                            let (hpc, var, height, depth) = self.handlers.pop().unwrap();
                            self.stack.truncate(height);
                            self.depth = depth;
                            if var != NONE {
                                *local!(var) = Value::Str(msg.into());
                            }
                            pc = hpc as usize;
                            break;
                        }
                        other => {
                            self.handlers.truncate(hbase);
                            if self.frames.len() == entry {
                                return Err(other);
                            }
                            let f = self.frames.pop().unwrap();
                            let finished = std::mem::replace(&mut env, f.env);
                            self.release_frame(finished);
                            keep = f.keep;
                            ops = unsafe { &*f.ops };
                            consts = f.consts;
                            hbase = f.handlers_base;
                            sbase = f.stack_base;
                            locals = env.vars.as_ptr();
                            self.depth -= 1;
                            e = match other {
                                Ctrl::Stop(l) => Ctrl::Err("`stop` only works inside a loop".into(), l),
                                Ctrl::Skip(l) => Ctrl::Err("`skip` only works inside a loop".into(), l),
                                x => x,
                            };
                        }
                    }
                }
            }
        }
    }
}
