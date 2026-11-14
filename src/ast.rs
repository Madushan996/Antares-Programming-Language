use std::rc::Rc;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    MatMul,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

#[derive(Clone, Copy, PartialEq)]
pub enum AskKind {
    Text,
    Number,
    YesNo,
    List,
}

pub enum Expr {
    Num(f64),
    /// A 64-bit whole number.
    Int(i64),
    /// ~x: every bit flipped.
    BitNot(Box<Expr>, u32),
    Str(Rc<str>),
    Interp(Vec<Expr>),
    Bool(bool),
    Nothing,
    Var { depth: u32, idx: u32, name: Rc<str>, line: u32 },
    List(Vec<Expr>),
    Map(Vec<(Rc<str>, Expr)>),
    Neg(Box<Expr>, u32),
    Not(Box<Expr>, u32),
    Binary(BinOp, Box<Expr>, Box<Expr>, u32),
    And(Box<Expr>, Box<Expr>, u32),
    Or(Box<Expr>, Box<Expr>, u32),
    Call(Box<Expr>, Vec<Expr>, u32),
    Index(Box<Expr>, Vec<Expr>, u32),
    Field(Box<Expr>, Rc<str>, u32),
    /// `start:end` inside [ ]; either side may be left out.
    Range(Option<Box<Expr>>, Option<Box<Expr>>, u32),
    Lambda(Rc<FnDef>),
    Ask { model: Box<Expr>, prompt: Box<Expr>, kind: AskKind, line: u32 },
    ModelNew { spec: Box<Expr>, opts: Option<Box<Expr>>, line: u32 },
}

pub enum Access {
    Index(Vec<Expr>),
    Field(Rc<str>),
}

pub enum Stmt {
    Assign { idx: u32, value: Expr },
    /// `change name = value`: replaces a variable that lives outside the current function.
    AssignOuter { depth: u32, idx: u32, value: Expr },
    Learn { idx: u32, value: Expr, line: u32 },
    SetPath { depth: u32, idx: u32, name: Rc<str>, path: Vec<Access>, value: Expr, line: u32 },
    Show(Vec<Expr>),
    Give(Option<Expr>),
    Repeat { count: Expr, var: Option<u32>, body: Vec<Stmt>, line: u32 },
    Each { var: u32, iter: Expr, body: Vec<Stmt>, line: u32 },
    While { cond: Expr, body: Vec<Stmt>, line: u32 },
    When { arms: Vec<(Expr, Vec<Stmt>, u32)>, otherwise: Option<Vec<Stmt>> },
    Descend { loss: Expr, rate: Expr, adam: bool, line: u32 },
    Try { body: Vec<Stmt>, var: Option<u32>, handler: Vec<Stmt> },
    Stop(u32),
    Skip(u32),
    Expr(Expr),
}

pub struct FnDef {
    pub name: Rc<str>,
    pub nparams: usize,
    pub nslots: usize,
    pub body: Vec<Stmt>,
    /// The body translated for the fast engine, made on first call.
    pub code: std::cell::OnceCell<crate::vm::Code>,
}
