//! Turns tokens into a tree, and resolves every name to a slot as it goes,
//! so the interpreter never looks a variable up by name.

use crate::ast::*;
use crate::lexer::{lex, StrPart, Tok, Token};
use std::path::PathBuf;
use std::rc::Rc;

pub struct PErr {
    pub msg: String,
    pub line: u32,
}

type PR<T> = Result<T, PErr>;

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
    scopes: Vec<Vec<String>>,
    assigned: Vec<bool>,
    forward: Vec<(usize, String, u32)>,
    /// Files brought in with `use`: (name shown in errors, source text).
    files: Vec<(String, String)>,
    loaded: Vec<PathBuf>,
    dirs: Vec<PathBuf>,
    /// Top-level variables given a range with `within`: (slot, lowest, highest, line).
    ranged: Vec<(u32, Expr, Expr, u32)>,
}

/// A copy of a range bound: a number, a variable, or arithmetic on them, with
/// variables `extra` scopes further out (for using it inside a function).
fn copy_bound(e: &Expr, extra: u32) -> Option<Expr> {
    Some(match e {
        Expr::Num(n) => Expr::Num(*n),
        Expr::Int(n) => Expr::Int(*n),
        Expr::Var { depth, idx, name, line } => Expr::Var { depth: depth + extra, idx: *idx, name: name.clone(), line: *line },
        Expr::Neg(a, l) => Expr::Neg(Box::new(copy_bound(a, extra)?), *l),
        Expr::Binary(op, a, b, l) => Expr::Binary(*op, Box::new(copy_bound(a, extra)?), Box::new(copy_bound(b, extra)?), *l),
        Expr::Call(f, args, l) => Expr::Call(
            Box::new(copy_bound(f, extra)?),
            args.iter().map(|a| copy_bound(a, extra)).collect::<Option<Vec<_>>>()?,
            *l,
        ),
        _ => return None,
    })
}

/// A copy of the place in `place += value` (a name, or part of a named list
/// or map at positions that are names, numbers, text or arithmetic on them),
/// so it can be read as well as given a value. Positions that call
/// something aren't copied: working them out twice could do something twice.
fn copy_place(e: &Expr) -> Option<Expr> {
    Some(match e {
        Expr::Num(n) => Expr::Num(*n),
        Expr::Int(n) => Expr::Int(*n),
        Expr::Str(t) => Expr::Str(t.clone()),
        Expr::Var { depth, idx, name, line } => Expr::Var { depth: *depth, idx: *idx, name: name.clone(), line: *line },
        Expr::Neg(a, l) => Expr::Neg(Box::new(copy_place(a)?), *l),
        Expr::Binary(op, a, b, l) => Expr::Binary(*op, Box::new(copy_place(a)?), Box::new(copy_place(b)?), *l),
        Expr::Field(b, name, l) => Expr::Field(Box::new(copy_place(b)?), name.clone(), *l),
        Expr::Index(b, idx, l) => Expr::Index(
            Box::new(copy_place(b)?),
            idx.iter().map(|x| if matches!(x, Expr::Range(..)) { None } else { copy_place(x) }).collect::<Option<Vec<_>>>()?,
            *l,
        ),
        _ => return None,
    })
}

/// The arithmetic of `+=` `-=` `*=` `/=`.
fn compound_op(t: &Tok) -> Option<BinOp> {
    match t {
        Tok::PlusAssign => Some(BinOp::Add),
        Tok::MinusAssign => Some(BinOp::Sub),
        Tok::StarAssign => Some(BinOp::Mul),
        Tok::SlashAssign => Some(BinOp::Div),
        _ => None,
    }
}

/// The line an expression is on, if it records one.
fn expr_line(e: &Expr) -> Option<u32> {
    match e {
        Expr::Var { line, .. } | Expr::Call(_, _, line) | Expr::Index(_, _, line) | Expr::Binary(_, _, _, line) => Some(*line),
        Expr::Neg(_, line) | Expr::Not(_, line) | Expr::BitNot(_, line) | Expr::Field(_, _, line) => Some(*line),
        Expr::And(_, _, line) | Expr::Or(_, _, line) => Some(*line),
        _ => None,
    }
}

fn describe(t: &Tok) -> String {
    match t {
        Tok::Newline => "the end of the line".into(),
        Tok::Eof => "the end of the file".into(),
        Tok::Num(n) => format!("the number {}", n),
        Tok::Str(_) => "some text".into(),
        Tok::Ident(s) => format!("'{}'", s),
        Tok::LParen => "'('".into(),
        Tok::RParen => "')'".into(),
        Tok::LBrack => "'['".into(),
        Tok::RBrack => "']'".into(),
        Tok::LBrace => "'{'".into(),
        Tok::RBrace => "'}'".into(),
        Tok::Comma => "','".into(),
        Tok::Colon => "':'".into(),
        Tok::Assign => "'='".into(),
        Tok::Arrow => "'->'".into(),
        Tok::FatArrow => "'=>'".into(),
        other => format!("'{}'", format!("{:?}", other).to_lowercase()),
    }
}

fn distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i];
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur.push((prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost));
        }
        prev = cur;
    }
    prev[b.len()]
}

impl Parser {
    pub fn new(builtins: &[&str]) -> Self {
        Parser {
            toks: vec![],
            pos: 0,
            scopes: vec![builtins.iter().map(|s| s.to_string()).collect()],
            assigned: vec![true; builtins.len()],
            forward: vec![],
            files: vec![],
            loaded: vec![],
            dirs: vec![PathBuf::new()],
            ranged: vec![],
        }
    }

    /// Tell the parser which file is being run, so `use` paths are found
    /// next to it and it is never loaded twice.
    pub fn set_main(&mut self, path: &str) {
        let p = PathBuf::from(path);
        if let Ok(c) = std::fs::canonicalize(&p) {
            self.dirs = vec![c.parent().map(|d| d.to_path_buf()).unwrap_or_default()];
            self.loaded.push(c);
        }
    }

    /// Name and source of a file loaded with `use` (ids start at 1).
    pub fn file(&self, id: u32) -> Option<(&str, &str)> {
        self.files.get(id as usize - 1).map(|(n, s)| (n.as_str(), s.as_str()))
    }

    /// The names of the global variables, by slot.
    pub fn global_names(&self) -> &[String] {
        &self.scopes[0]
    }

    /// Every file brought in with `use`: (name, source).
    pub fn files(&self) -> &[(String, String)] {
        &self.files
    }

    pub fn global_count(&self) -> usize {
        self.scopes[0].len()
    }

    pub fn parse(&mut self, toks: Vec<Token>) -> PR<Vec<Stmt>> {
        let saved = self.scopes[0].len();
        let saved_assigned = self.assigned.clone();
        let saved_loaded = self.loaded.len();
        let saved_ranged = self.ranged.len();
        self.dirs.truncate(1);
        self.toks = toks;
        self.pos = 0;
        self.forward.clear();
        self.scopes.truncate(1);
        let mut r = self.program();
        if let Ok(stmts) = &mut r {
            if self.ranged.len() > saved_ranged {
                if let Err(e) = self.keep_in_range(stmts, 0) {
                    r = Err(e);
                }
            }
        }
        if r.is_ok() {
            for (idx, name, line) in &self.forward {
                if !self.assigned[*idx] {
                    r = Err(self.unknown(name, *line));
                    break;
                }
            }
        }
        if r.is_err() {
            self.scopes.truncate(1);
            self.scopes[0].truncate(saved);
            self.assigned = saved_assigned;
            self.loaded.truncate(saved_loaded);
            self.ranged.truncate(saved_ranged);
        }
        r
    }

    fn unknown(&self, name: &str, line: u32) -> PErr {
        let mut best: Option<(&str, usize)> = None;
        for (i, cand) in self.scopes[0].iter().enumerate() {
            if !self.assigned[i] || cand == name {
                continue;
            }
            let d = distance(name, cand);
            if d <= 2 && d < name.chars().count() && best.map_or(true, |(_, bd)| d < bd) {
                best = Some((cand, d));
            }
        }
        let msg = match best {
            Some((c, _)) => format!("I don't know what '{}' is. Did you mean '{}'?", name, c),
            None => format!("I don't know what '{}' is. It was never given a value", name),
        };
        PErr { msg, line }
    }

    // ----- token helpers -----

    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }

    fn peek_at(&self, n: usize) -> &Tok {
        &self.toks[(self.pos + n).min(self.toks.len() - 1)].tok
    }

    fn line(&self) -> u32 {
        self.toks[self.pos].line
    }

    fn advance(&mut self) -> Tok {
        let t = self.toks[self.pos].tok.clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == t {
            self.advance();
            true
        } else {
            false
        }
    }

    fn err<T>(&self, msg: String) -> PR<T> {
        Err(PErr { msg, line: self.line() })
    }

    fn expect(&mut self, t: Tok, what: &str) -> PR<()> {
        if self.eat(&t) {
            Ok(())
        } else {
            self.err(format!("I expected {} here, but found {}", what, describe(self.peek())))
        }
    }

    fn ident(&mut self, what: &str) -> PR<String> {
        match self.peek().clone() {
            Tok::Ident(s) => {
                self.advance();
                Ok(s)
            }
            other => self.err(format!("I expected {} here, but found {}", what, describe(&other))),
        }
    }

    fn skip_nl(&mut self) {
        while *self.peek() == Tok::Newline {
            self.pos += 1;
        }
    }

    // ----- scopes -----

    fn declare(&mut self, name: &str) -> u32 {
        let global = self.scopes.len() == 1;
        let scope = self.scopes.last_mut().unwrap();
        let idx = match scope.iter().position(|n| n == name) {
            Some(i) => i,
            None => {
                scope.push(name.to_string());
                if global {
                    self.assigned.push(true);
                }
                scope.len() - 1
            }
        };
        if global {
            self.assigned[idx] = true;
        }
        idx as u32
    }

    fn resolve(&mut self, name: &str, line: u32) -> (u32, u32) {
        let n = self.scopes.len();
        for (k, scope) in self.scopes.iter().enumerate().rev() {
            if let Some(i) = scope.iter().position(|s| s == name) {
                return ((n - 1 - k) as u32, i as u32);
            }
        }
        // Not known yet: it may be defined further down the file.
        self.scopes[0].push(name.to_string());
        self.assigned.push(false);
        let idx = self.scopes[0].len() - 1;
        self.forward.push((idx, name.to_string(), line));
        ((n - 1) as u32, idx as u32)
    }

    // ----- statements -----

    fn program(&mut self) -> PR<Vec<Stmt>> {
        let mut stmts = vec![];
        self.top_level(&mut stmts)?;
        Ok(stmts)
    }

    fn top_level(&mut self, stmts: &mut Vec<Stmt>) -> PR<()> {
        loop {
            self.skip_nl();
            match self.peek() {
                Tok::Eof => return Ok(()),
                Tok::Use => self.import(stmts)?,
                _ => stmts.push(self.statement()?),
            }
            self.end_stmt()?;
        }
    }

    /// `use "file"`: run another file's top level here, once.
    fn import(&mut self, stmts: &mut Vec<Stmt>) -> PR<()> {
        let line = self.line();
        self.advance();
        let rel = match self.advance() {
            Tok::Str(parts) if parts.len() == 1 && matches!(parts[0], StrPart::Lit(_)) => match &parts[0] {
                StrPart::Lit(s) => s.clone(),
                _ => unreachable!(),
            },
            other => {
                return Err(PErr {
                    msg: format!("after `use` I expected a file name in quotes, but found {}", describe(&other)),
                    line,
                })
            }
        };
        let mut path = self.dirs.last().cloned().unwrap_or_default().join(&rel);
        if path.extension().is_none() {
            path.set_extension("ant");
        }
        let shown = if std::path::Path::new(&rel).extension().is_none() { format!("{}.ant", rel) } else { rel.clone() };
        let canon = std::fs::canonicalize(&path)
            .map_err(|_| PErr { msg: format!("I couldn't find the file \"{}\" to use", shown), line })?;
        if self.loaded.contains(&canon) {
            return Ok(());
        }
        let src = std::fs::read_to_string(&canon)
            .map_err(|e| PErr { msg: format!("I couldn't read \"{}\": {}", shown, e), line })?;
        self.loaded.push(canon.clone());
        self.files.push((shown, src.clone()));
        let id = self.files.len() as u32;
        let toks = lex(&src, (id << 20) | 1).map_err(|(msg, line)| PErr { msg, line })?;
        let saved_toks = std::mem::replace(&mut self.toks, toks);
        let saved_pos = self.pos;
        self.pos = 0;
        self.dirs.push(canon.parent().map(|d| d.to_path_buf()).unwrap_or_default());
        let r = self.top_level(stmts);
        self.dirs.pop();
        self.toks = saved_toks;
        self.pos = saved_pos;
        r
    }

    fn end_stmt(&mut self) -> PR<()> {
        match self.peek() {
            Tok::Newline | Tok::Eof | Tok::RBrace => Ok(()),
            other => self.err(format!(
                "I expected the line to end here, but found {}",
                describe(other)
            )),
        }
    }

    fn block(&mut self) -> PR<Vec<Stmt>> {
        self.expect(Tok::LBrace, "'{' to start a block")?;
        let mut stmts = vec![];
        loop {
            self.skip_nl();
            if self.eat(&Tok::RBrace) {
                break;
            }
            if *self.peek() == Tok::Eof {
                return self.err("a block is missing its closing '}'".into());
            }
            stmts.push(self.statement()?);
            self.end_stmt()?;
        }
        Ok(stmts)
    }

    fn function(&mut self, name: &str, params: Vec<String>, ranges: Vec<Option<(Expr, Expr, u32)>>) -> PR<Rc<FnDef>> {
        let nparams = params.len();
        // inputs with a range are checked first: x = within(x, lo, hi, "x")
        let nest = self.scopes.len() as u32;
        let mut checks = vec![];
        for (k, r) in ranges.iter().enumerate() {
            if let Some((lo, hi, line)) = r {
                let v = Expr::Var { depth: 0, idx: k as u32, name: params[k].as_str().into(), line: *line };
                checks.push(Stmt::Assign { idx: k as u32, value: self.within_call_at(v, lo, hi, nest, &params[k], *line) });
            }
        }
        self.scopes.push(params);
        let body = if self.eat(&Tok::Assign) {
            vec![Stmt::Expr(self.expr()?)]
        } else if *self.peek() == Tok::LBrace {
            self.block()?
        } else {
            return self.err(format!(
                "after the name and inputs, a `to` needs '= ...' or a '{{ ... }}' block, but I found {}",
                describe(self.peek())
            ));
        };
        let nslots = self.scopes.pop().unwrap().len();
        let body = if checks.is_empty() {
            body
        } else {
            // a body `= value` gives that value back after the checks
            let mut b = checks;
            b.extend(body);
            b
        };
        Ok(Rc::new(FnDef { name: name.into(), nparams, nslots, body, code: Default::default() }))
    }

    /// within(value, lo, hi, "name") for code inside a function `nest` scopes
    /// below the top level, with `lo` and `hi` written one scope further out.
    fn within_call_at(&self, value: Expr, lo: &Expr, hi: &Expr, nest: u32, name: &str, line: u32) -> Expr {
        let args = vec![value, copy_bound(lo, 1).unwrap(), copy_bound(hi, 1).unwrap(), Expr::Str(name.into())];
        Expr::Call(Box::new(self.within_fn(nest, line)), args, line)
    }

    /// `within lowest to highest` after a value or an input name, if there.
    fn within_range(&mut self) -> PR<Option<(Expr, Expr)>> {
        if !matches!(self.peek(), Tok::Ident(w) if w == "within") {
            return Ok(None);
        }
        self.advance();
        let lo = self.expr()?;
        self.expect(Tok::To, "`to` between the lowest and highest number (for example: within 0 to 63)")?;
        let hi = self.expr()?;
        if copy_bound(&lo, 0).is_none() || copy_bound(&hi, 0).is_none() {
            return self.err("the lowest and highest numbers after `within` must be numbers, names of numbers, or arithmetic on them".into());
        }
        Ok(Some((lo, hi)))
    }

    /// The slot of the `within` built-in, as seen from `nest` functions deep.
    fn within_fn(&self, nest: u32, line: u32) -> Expr {
        let idx = self.scopes[0].iter().position(|n| n == "within").unwrap_or(0) as u32;
        Expr::Var { depth: nest, idx, name: "within".into(), line }
    }

    /// within(value, lo, hi, "name"), for code `nest` functions deep.
    fn within_call(&self, value: Expr, lo: &Expr, hi: &Expr, extra: u32, name: &str, line: u32) -> Expr {
        let args = vec![value, copy_bound(lo, extra).unwrap(), copy_bound(hi, extra).unwrap(), Expr::Str(name.into())];
        Expr::Call(Box::new(self.within_fn(extra, line)), args, line)
    }

    /// Makes every value given to a variable with a range (a top-level
    /// `name = ... within lo to hi`) go through within(value, lo, hi, "name"):
    /// assignments, `change`, and changing a part (`name[i] = value`).
    /// `nest` is how many functions deep `stmts` are.
    fn keep_in_range(&self, stmts: &mut [Stmt], nest: u32) -> PR<()> {
        for st in stmts.iter_mut() {
            let (target, value, line) = match st {
                Stmt::Assign { idx, value } => ((nest == 0).then_some(*idx), Some(value), None),
                Stmt::AssignOuter { depth, idx, value } => ((*depth == nest).then_some(*idx), Some(value), None),
                Stmt::SetPath { depth, idx, value, line, .. } => ((*depth == nest).then_some(*idx), Some(value), Some(*line)),
                Stmt::Learn { idx, value, line } => {
                    if nest == 0 && self.ranged.iter().any(|r| r.0 == *idx) {
                        return Err(PErr { msg: "a variable with a range can't be learned".into(), line: *line });
                    }
                    (None, Some(value), None)
                }
                _ => (None, None, None),
            };
            if let Some(value) = value {
                self.keep_in_range_expr(value, nest)?;
                if let Some(r) = target.and_then(|t| self.ranged.iter().find(|r| r.0 == t)) {
                    let line = line.or_else(|| expr_line(value)).unwrap_or(r.3);
                    let name = self.scopes[0][r.0 as usize].clone();
                    let v = std::mem::replace(value, Expr::Nothing);
                    *value = self.within_call(v, &r.1, &r.2, nest, &name, line);
                }
            }
            let counter = |v: &Option<u32>, line: u32| -> PR<()> {
                match v {
                    Some(v) if nest == 0 && self.ranged.iter().any(|r| r.0 == *v) => {
                        Err(PErr { msg: format!("'{}' has a range, so it can't count the rounds of a loop or hold a problem", self.scopes[0][*v as usize]), line })
                    }
                    _ => Ok(()),
                }
            };
            match st {
                Stmt::SetPath { path, .. } => {
                    for a in path {
                        if let Access::Index(es) = a {
                            for e in es {
                                self.keep_in_range_expr(e, nest)?;
                            }
                        }
                    }
                }
                Stmt::Show(es) => {
                    for e in es {
                        self.keep_in_range_expr(e, nest)?;
                    }
                }
                Stmt::Give(Some(e)) | Stmt::Expr(e) => self.keep_in_range_expr(e, nest)?,
                Stmt::Repeat { count, var, body, line } => {
                    counter(var, *line)?;
                    self.keep_in_range_expr(count, nest)?;
                    self.keep_in_range(body, nest)?;
                }
                Stmt::Each { var, iter, body, line } => {
                    counter(&Some(*var), *line)?;
                    self.keep_in_range_expr(iter, nest)?;
                    self.keep_in_range(body, nest)?;
                }
                Stmt::While { cond, body, .. } => {
                    self.keep_in_range_expr(cond, nest)?;
                    self.keep_in_range(body, nest)?;
                }
                Stmt::When { arms, otherwise } => {
                    for (c, b, _) in arms.iter_mut() {
                        self.keep_in_range_expr(c, nest)?;
                        self.keep_in_range(b, nest)?;
                    }
                    if let Some(b) = otherwise {
                        self.keep_in_range(b, nest)?;
                    }
                }
                Stmt::Descend { loss, rate, .. } => {
                    self.keep_in_range_expr(loss, nest)?;
                    self.keep_in_range_expr(rate, nest)?;
                }
                Stmt::Try { body, var, handler } => {
                    counter(var, 0)?;
                    self.keep_in_range(body, nest)?;
                    self.keep_in_range(handler, nest)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn keep_in_range_expr(&self, e: &mut Expr, nest: u32) -> PR<()> {
        match e {
            Expr::Lambda(def) => {
                if let Some(def) = Rc::get_mut(def) {
                    self.keep_in_range(&mut def.body, nest + 1)?;
                }
            }
            Expr::Interp(p) | Expr::List(p) => {
                for x in p {
                    self.keep_in_range_expr(x, nest)?;
                }
            }
            Expr::Map(m) => {
                for (_, x) in m {
                    self.keep_in_range_expr(x, nest)?;
                }
            }
            Expr::Neg(a, _) | Expr::Not(a, _) | Expr::Field(a, _, _) | Expr::BitNot(a, _) => self.keep_in_range_expr(a, nest)?,
            Expr::Binary(_, a, b, _) | Expr::And(a, b, _) | Expr::Or(a, b, _) => {
                self.keep_in_range_expr(a, nest)?;
                self.keep_in_range_expr(b, nest)?;
            }
            Expr::Call(f, args, _) => {
                self.keep_in_range_expr(f, nest)?;
                for a in args {
                    self.keep_in_range_expr(a, nest)?;
                }
            }
            Expr::Index(b, idx, _) => {
                self.keep_in_range_expr(b, nest)?;
                for x in idx {
                    self.keep_in_range_expr(x, nest)?;
                }
            }
            Expr::Range(a, z, _) => {
                for x in [a, z].into_iter().flatten() {
                    self.keep_in_range_expr(x, nest)?;
                }
            }
            Expr::Ask { model, prompt, .. } => {
                self.keep_in_range_expr(model, nest)?;
                self.keep_in_range_expr(prompt, nest)?;
            }
            Expr::ModelNew { spec, opts, .. } => {
                self.keep_in_range_expr(spec, nest)?;
                if let Some(o) = opts {
                    self.keep_in_range_expr(o, nest)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn params(&mut self) -> PR<Vec<String>> {
        Ok(self.params_with_ranges()?.0)
    }

    /// Input names, and the range given to each with `within`, if any.
    fn params_with_ranges(&mut self) -> PR<(Vec<String>, Vec<Option<(Expr, Expr, u32)>>)> {
        let mut params = vec![];
        let mut ranges = vec![];
        if self.eat(&Tok::LParen) {
            while !self.eat(&Tok::RParen) {
                let p = self.ident("an input name")?;
                if params.contains(&p) {
                    return self.err(format!("the input name '{}' is used twice", p));
                }
                params.push(p);
                let line = self.line();
                ranges.push(self.within_range()?.map(|(lo, hi)| (lo, hi, line)));
                if !self.eat(&Tok::Comma) {
                    self.expect(Tok::RParen, "')'")?;
                    break;
                }
            }
        }
        Ok((params, ranges))
    }

    fn statement(&mut self) -> PR<Stmt> {
        let line = self.line();
        match self.peek().clone() {
            Tok::Learn => {
                self.advance();
                let name = self.ident("a name after `learn`")?;
                self.expect(Tok::Assign, "'='")?;
                let value = self.expr()?;
                let idx = self.declare(&name);
                Ok(Stmt::Learn { idx, value, line })
            }
            Tok::Model => {
                self.advance();
                let name = self.ident("a name after `model`")?;
                self.expect(Tok::Assign, "'='")?;
                let spec = Box::new(self.expr()?);
                let opts = if self.eat(&Tok::With) { Some(Box::new(self.expr()?)) } else { None };
                let idx = self.declare(&name);
                Ok(Stmt::Assign { idx, value: Expr::ModelNew { spec, opts, line } })
            }
            Tok::To => {
                self.advance();
                let name = self.ident("a name after `to`")?;
                let idx = self.declare(&name);
                let (params, ranges) = self.params_with_ranges()?;
                let def = self.function(&name, params, ranges)?;
                Ok(Stmt::Assign { idx, value: Expr::Lambda(def) })
            }
            Tok::Show => {
                self.advance();
                let mut items = vec![self.expr()?];
                while self.eat(&Tok::Comma) {
                    items.push(self.expr()?);
                }
                Ok(Stmt::Show(items))
            }
            Tok::Give => {
                self.advance();
                if matches!(self.peek(), Tok::Newline | Tok::Eof | Tok::RBrace) {
                    Ok(Stmt::Give(None))
                } else {
                    Ok(Stmt::Give(Some(self.expr()?)))
                }
            }
            Tok::Repeat => {
                self.advance();
                let count = self.expr()?;
                let var = if self.eat(&Tok::As) {
                    let n = self.ident("a counter name after `as`")?;
                    Some(self.declare(&n))
                } else {
                    None
                };
                let body = self.block()?;
                Ok(Stmt::Repeat { count, var, body, line })
            }
            Tok::Each => {
                self.advance();
                let n = self.ident("a name after `each`")?;
                self.expect(Tok::In, "`in`")?;
                let iter = self.expr()?;
                let var = self.declare(&n);
                let body = self.block()?;
                Ok(Stmt::Each { var, iter, body, line })
            }
            Tok::While => {
                self.advance();
                let cond = self.expr()?;
                let body = self.block()?;
                Ok(Stmt::While { cond, body, line })
            }
            Tok::When => {
                self.advance();
                let mut arms = vec![];
                let cond = self.expr()?;
                let body = self.block()?;
                arms.push((cond, body, line));
                let mut otherwise = None;
                loop {
                    let save = self.pos;
                    self.skip_nl();
                    if !self.eat(&Tok::Else) {
                        self.pos = save;
                        break;
                    }
                    if self.eat(&Tok::When) {
                        let l = self.line();
                        let cond = self.expr()?;
                        let body = self.block()?;
                        arms.push((cond, body, l));
                    } else {
                        otherwise = Some(self.block()?);
                        break;
                    }
                }
                Ok(Stmt::When { arms, otherwise })
            }
            Tok::Descend => {
                self.advance();
                let loss = self.expr()?;
                self.expect(Tok::By, "`by` and a rate (for example: descend loss by 0.01)")?;
                let rate = self.expr()?;
                let mut adam = false;
                if self.eat(&Tok::Using) {
                    let m = self.ident("`adam` or `sgd` after `using`")?;
                    adam = match m.as_str() {
                        "adam" => true,
                        "sgd" => false,
                        other => {
                            return self.err(format!(
                                "I know the methods `adam` and `sgd`, but not '{}'",
                                other
                            ))
                        }
                    };
                }
                Ok(Stmt::Descend { loss, rate, adam, line })
            }
            Tok::Change => {
                self.advance();
                let name = self.ident("a name after `change`")?;
                let op = compound_op(self.peek());
                if op.is_some() {
                    self.advance();
                } else {
                    self.expect(Tok::Assign, "'=' (or += -= *= /=)")?;
                }
                let value = self.expr()?;
                let (depth, idx) = self.resolve(&name, line);
                // change x += v is change x = x + v
                let value = match op {
                    Some(op) => Expr::Binary(op, Box::new(Expr::Var { depth, idx, name: name.as_str().into(), line }), Box::new(value), line),
                    None => value,
                };
                if depth == 0 {
                    let idx = self.declare(&name);
                    Ok(Stmt::Assign { idx, value })
                } else {
                    Ok(Stmt::AssignOuter { depth, idx, value })
                }
            }
            Tok::Use => self.err("`use` belongs at the top level of a file, not inside a block".into()),
            Tok::Try => {
                self.advance();
                let body = self.block()?;
                self.skip_nl();
                if !self.eat(&Tok::Else) {
                    return self.err("a `try { ... }` block must be followed by `else { ... }`, saying what to do if it fails".into());
                }
                let var = match self.peek().clone() {
                    Tok::Ident(n) => {
                        self.advance();
                        Some(self.declare(&n))
                    }
                    _ => None,
                };
                let handler = self.block()?;
                Ok(Stmt::Try { body, var, handler })
            }
            Tok::Stop => {
                self.advance();
                Ok(Stmt::Stop(line))
            }
            Tok::Skip => {
                self.advance();
                Ok(Stmt::Skip(line))
            }
            Tok::Ident(name) if compound_op(self.peek_at(1)).is_some() => {
                // x += v is x = x + v
                self.advance();
                let op = compound_op(&self.advance()).unwrap();
                let (depth, idx) = self.resolve(&name, line);
                let old = Expr::Var { depth, idx, name: name.as_str().into(), line };
                let value = Expr::Binary(op, Box::new(old), Box::new(self.expr()?), line);
                let idx = self.declare(&name);
                Ok(Stmt::Assign { idx, value })
            }
            Tok::Ident(name) if *self.peek_at(1) == Tok::Assign => {
                self.advance();
                self.advance();
                let value = self.expr()?;
                let range = self.within_range()?;
                let idx = self.declare(&name);
                if let Some((lo, hi)) = range {
                    if self.scopes.len() != 1 {
                        return self.err("`within` gives a range to a top-level variable or to a function's input; inside a function, use within(x, lowest, highest)".into());
                    }
                    if self.ranged.iter().any(|r| r.0 == idx) {
                        return self.err(format!("'{}' already has a range", name));
                    }
                    self.ranged.push((idx, lo, hi, line));
                }
                Ok(Stmt::Assign { idx, value })
            }
            _ => {
                let e = self.expr()?;
                let op = compound_op(self.peek());
                if *self.peek() == Tok::Assign || op.is_some() {
                    // a[i] += v is a[i] = a[i] + v
                    let old = match op {
                        Some(_) => match copy_place(&e) {
                            Some(c) => Some(c),
                            None => {
                                return self.err(
                                    "with += -= *= /= the place must be simple (no calls in its positions); write it out, as in xs[i] = xs[i] + 1".into(),
                                )
                            }
                        },
                        None => None,
                    };
                    let mut path = vec![];
                    let mut cur = e;
                    loop {
                        match cur {
                            Expr::Index(base, idx, _) => {
                                path.push(Access::Index(idx));
                                cur = *base;
                            }
                            Expr::Field(base, name, _) => {
                                path.push(Access::Field(name));
                                cur = *base;
                            }
                            Expr::Var { depth, idx, name, .. } => {
                                path.reverse();
                                self.advance();
                                let value = self.expr()?;
                                let value = match (op, old) {
                                    (Some(op), Some(old)) => Expr::Binary(op, Box::new(old), Box::new(value), line),
                                    _ => value,
                                };
                                return Ok(Stmt::SetPath { depth, idx, name, path, value, line });
                            }
                            _ => {
                                return self.err(
                                    "only a name, or part of a named list or map, can be given a value with '='"
                                        .into(),
                                )
                            }
                        }
                    }
                }
                Ok(Stmt::Expr(e))
            }
        }
    }

    // ----- expressions -----

    pub fn expr(&mut self) -> PR<Expr> {
        let mut left = self.or_expr()?;
        while *self.peek() == Tok::Arrow {
            let line = self.line();
            self.advance();
            let target = self.postfix()?;
            left = match target {
                Expr::Call(f, mut args, l) => {
                    args.insert(0, left);
                    Expr::Call(f, args, l)
                }
                other => Expr::Call(Box::new(other), vec![left], line),
            };
        }
        Ok(left)
    }

    fn or_expr(&mut self) -> PR<Expr> {
        let mut left = self.and_expr()?;
        while *self.peek() == Tok::Or {
            let line = self.line();
            self.advance();
            let right = self.and_expr()?;
            left = Expr::Or(Box::new(left), Box::new(right), line);
        }
        Ok(left)
    }

    fn and_expr(&mut self) -> PR<Expr> {
        let mut left = self.not_expr()?;
        while *self.peek() == Tok::And {
            let line = self.line();
            self.advance();
            let right = self.not_expr()?;
            left = Expr::And(Box::new(left), Box::new(right), line);
        }
        Ok(left)
    }

    fn not_expr(&mut self) -> PR<Expr> {
        if *self.peek() == Tok::Not {
            let line = self.line();
            self.advance();
            let e = self.not_expr()?;
            return Ok(Expr::Not(Box::new(e), line));
        }
        self.comparison()
    }

    /// & | xor: looser than shifts and arithmetic, tighter than comparisons,
    /// so `a & b != 0` means `(a & b) != 0`.
    fn bits(&mut self) -> PR<Expr> {
        let mut left = self.shift()?;
        loop {
            let op = match self.peek() {
                Tok::Amp => BinOp::BitAnd,
                Tok::Pipe => BinOp::BitOr,
                Tok::Xor => BinOp::BitXor,
                _ => return Ok(left),
            };
            let line = self.line();
            self.advance();
            let right = self.shift()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right), line);
        }
    }

    fn shift(&mut self) -> PR<Expr> {
        let mut left = self.additive()?;
        loop {
            let op = match self.peek() {
                Tok::Shl => BinOp::Shl,
                Tok::Shr => BinOp::Shr,
                _ => return Ok(left),
            };
            let line = self.line();
            self.advance();
            let right = self.additive()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right), line);
        }
    }

    fn comparison(&mut self) -> PR<Expr> {
        let left = self.bits()?;
        let op = match self.peek() {
            Tok::Eq => BinOp::Eq,
            Tok::Ne => BinOp::Ne,
            Tok::Lt => BinOp::Lt,
            Tok::Gt => BinOp::Gt,
            Tok::Le => BinOp::Le,
            Tok::Ge => BinOp::Ge,
            _ => return Ok(left),
        };
        let line = self.line();
        self.advance();
        let right = self.bits()?;
        Ok(Expr::Binary(op, Box::new(left), Box::new(right), line))
    }

    fn additive(&mut self) -> PR<Expr> {
        let mut left = self.multiplicative()?;
        loop {
            let op = match self.peek() {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => return Ok(left),
            };
            let line = self.line();
            self.advance();
            let right = self.multiplicative()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right), line);
        }
    }

    fn multiplicative(&mut self) -> PR<Expr> {
        let mut left = self.unary()?;
        loop {
            let op = match self.peek() {
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                Tok::Percent => BinOp::Mod,
                Tok::At => BinOp::MatMul,
                _ => return Ok(left),
            };
            let line = self.line();
            self.advance();
            let right = self.unary()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right), line);
        }
    }

    fn unary(&mut self) -> PR<Expr> {
        if *self.peek() == Tok::Tilde {
            let line = self.line();
            self.advance();
            let e = self.unary()?;
            return Ok(Expr::BitNot(Box::new(e), line));
        }
        if *self.peek() == Tok::Minus {
            let line = self.line();
            self.advance();
            let e = self.unary()?;
            return Ok(Expr::Neg(Box::new(e), line));
        }
        self.power()
    }

    fn power(&mut self) -> PR<Expr> {
        let base = self.postfix()?;
        if *self.peek() == Tok::Caret {
            let line = self.line();
            self.advance();
            let exp = self.unary()?;
            return Ok(Expr::Binary(BinOp::Pow, Box::new(base), Box::new(exp), line));
        }
        Ok(base)
    }

    fn postfix(&mut self) -> PR<Expr> {
        let mut e = self.primary()?;
        loop {
            let line = self.line();
            match self.peek() {
                Tok::LParen => {
                    self.advance();
                    let args = self.expr_list(Tok::RParen, "')'")?;
                    e = Expr::Call(Box::new(e), args, line);
                }
                Tok::LBrack => {
                    self.advance();
                    let mut idx = vec![];
                    loop {
                        if *self.peek() == Tok::RBrack {
                            break;
                        }
                        let start = if *self.peek() == Tok::Colon { None } else { Some(Box::new(self.expr()?)) };
                        if self.eat(&Tok::Colon) {
                            let end = if matches!(self.peek(), Tok::Comma | Tok::RBrack) {
                                None
                            } else {
                                Some(Box::new(self.expr()?))
                            };
                            idx.push(Expr::Range(start, end, line));
                        } else {
                            idx.push(*start.unwrap());
                        }
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    self.expect(Tok::RBrack, "']'")?;
                    if idx.is_empty() {
                        return self.err("an index is missing between [ and ]".into());
                    }
                    e = Expr::Index(Box::new(e), idx, line);
                }
                Tok::Dot => {
                    self.advance();
                    let name = self.ident("a field name after '.'")?;
                    e = Expr::Field(Box::new(e), name.into(), line);
                }
                _ => return Ok(e),
            }
        }
    }

    fn expr_list(&mut self, close: Tok, what: &str) -> PR<Vec<Expr>> {
        let mut items = vec![];
        if self.eat(&close) {
            return Ok(items);
        }
        loop {
            items.push(self.expr()?);
            if self.eat(&Tok::Comma) {
                if self.eat(&close) {
                    break;
                }
                continue;
            }
            self.expect(close.clone(), what)?;
            break;
        }
        Ok(items)
    }

    fn lambda_ahead(&self) -> bool {
        let mut k = 1;
        if *self.peek_at(k) != Tok::RParen {
            loop {
                if !matches!(self.peek_at(k), Tok::Ident(_)) {
                    return false;
                }
                k += 1;
                match self.peek_at(k) {
                    Tok::Comma => k += 1,
                    Tok::RParen => break,
                    _ => return false,
                }
            }
        }
        *self.peek_at(k + 1) == Tok::FatArrow
    }

    fn lambda(&mut self, params: Vec<String>) -> PR<Expr> {
        let nparams = params.len();
        self.scopes.push(params);
        let body = self.expr();
        let nslots = self.scopes.pop().unwrap().len();
        let body = vec![Stmt::Expr(body?)];
        Ok(Expr::Lambda(Rc::new(FnDef { name: "function".into(), nparams, nslots, body, code: Default::default() })))
    }

    fn interpolation(&mut self, code: &str, line: u32) -> PR<Expr> {
        let toks = lex(code, line).map_err(|(msg, line)| PErr { msg, line })?;
        let saved_toks = std::mem::replace(&mut self.toks, toks);
        let saved_pos = self.pos;
        self.pos = 0;
        let r = self.expr().and_then(|e| {
            self.skip_nl();
            if *self.peek() == Tok::Eof {
                Ok(e)
            } else {
                self.err(format!("inside {{ }} in text I found something extra: {}", describe(self.peek())))
            }
        });
        self.toks = saved_toks;
        self.pos = saved_pos;
        r
    }

    fn primary(&mut self) -> PR<Expr> {
        let line = self.line();
        match self.peek().clone() {
            Tok::Num(n) => {
                self.advance();
                Ok(Expr::Num(n))
            }
            Tok::Int(n) => {
                self.advance();
                Ok(Expr::Int(n))
            }
            Tok::Str(parts) => {
                self.advance();
                if parts.len() == 1 {
                    if let StrPart::Lit(s) = &parts[0] {
                        return Ok(Expr::Str(s.as_str().into()));
                    }
                }
                let mut out = vec![];
                for p in parts {
                    out.push(match p {
                        StrPart::Lit(s) => Expr::Str(s.as_str().into()),
                        StrPart::Code(code) => self.interpolation(&code, line)?,
                    });
                }
                Ok(Expr::Interp(out))
            }
            Tok::Yes => {
                self.advance();
                Ok(Expr::Bool(true))
            }
            Tok::No => {
                self.advance();
                Ok(Expr::Bool(false))
            }
            Tok::Nothing => {
                self.advance();
                Ok(Expr::Nothing)
            }
            Tok::Ident(name) => {
                self.advance();
                if self.eat(&Tok::FatArrow) {
                    return self.lambda(vec![name]);
                }
                let (depth, idx) = self.resolve(&name, line);
                Ok(Expr::Var { depth, idx, name: name.into(), line })
            }
            Tok::LParen => {
                if self.lambda_ahead() {
                    let params = self.params()?;
                    self.expect(Tok::FatArrow, "'=>'")?;
                    return self.lambda(params);
                }
                self.advance();
                let e = self.expr()?;
                self.expect(Tok::RParen, "')'")?;
                Ok(e)
            }
            Tok::LBrack => {
                self.advance();
                let items = self.expr_list(Tok::RBrack, "']'")?;
                Ok(Expr::List(items))
            }
            Tok::LBrace => {
                self.advance();
                let mut entries: Vec<(Rc<str>, Expr)> = vec![];
                loop {
                    self.skip_nl();
                    if self.eat(&Tok::RBrace) {
                        break;
                    }
                    let key: Rc<str> = match self.advance() {
                        Tok::Ident(s) => s.into(),
                        Tok::Str(parts) if parts.len() == 1 => match &parts[0] {
                            StrPart::Lit(s) => s.as_str().into(),
                            _ => return self.err("a map key must be plain text".into()),
                        },
                        other => {
                            return self.err(format!(
                                "I expected a key name in this map, but found {}",
                                describe(&other)
                            ))
                        }
                    };
                    self.expect(Tok::Colon, "':' after the key")?;
                    let v = self.expr()?;
                    entries.push((key, v));
                    self.skip_nl();
                    if !self.eat(&Tok::Comma) {
                        self.skip_nl();
                        self.expect(Tok::RBrace, "',' or '}'")?;
                        break;
                    }
                }
                Ok(Expr::Map(entries))
            }
            Tok::Ask => {
                self.advance();
                let model = Box::new(self.postfix()?);
                let prompt = Box::new(self.additive()?);
                let mut kind = AskKind::Text;
                if self.eat(&Tok::As) {
                    kind = match self.advance() {
                        Tok::Yes => {
                            self.expect(Tok::Slash, "'/no' (write: as yes/no)")?;
                            self.expect(Tok::No, "'no' (write: as yes/no)")?;
                            AskKind::YesNo
                        }
                        Tok::Ident(s) if s == "number" => AskKind::Number,
                        Tok::Ident(s) if s == "text" => AskKind::Text,
                        Tok::Ident(s) if s == "list" => AskKind::List,
                        other => {
                            return self.err(format!(
                                "after `as` I expected text, number, list or yes/no, but found {}",
                                describe(&other)
                            ))
                        }
                    };
                }
                Ok(Expr::Ask { model, prompt, kind, line })
            }
            other => self.err(format!("I didn't expect {} here", describe(&other))),
        }
    }
}
