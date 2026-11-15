#[derive(Clone, Debug, PartialEq)]
pub enum StrPart {
    Lit(String),
    Code(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Num(f64),
    /// A whole number written in hexadecimal, like 0xFF.
    Int(i64),
    Str(Vec<StrPart>),
    Ident(String),
    // keywords
    Learn,
    Model,
    To,
    Show,
    Give,
    Repeat,
    As,
    Each,
    In,
    While,
    When,
    Else,
    Descend,
    By,
    Using,
    Stop,
    Skip,
    Ask,
    And,
    Or,
    Not,
    Yes,
    No,
    Nothing,
    With,
    Use,
    Try,
    Change,
    Xor,
    // symbols
    LParen,
    RParen,
    LBrack,
    RBrack,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Dot,
    Assign,
    /// `+=` `-=` `*=` `/=`: change a value by an amount.
    PlusAssign,
    MinusAssign,
    StarAssign,
    SlashAssign,
    Arrow,
    FatArrow,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    At,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    Amp,
    Pipe,
    Tilde,
    Shl,
    Shr,
    Newline,
    Eof,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    pub line: u32,
}

fn keyword(s: &str) -> Option<Tok> {
    Some(match s {
        "learn" => Tok::Learn,
        "model" => Tok::Model,
        "to" => Tok::To,
        "show" => Tok::Show,
        "give" => Tok::Give,
        "repeat" => Tok::Repeat,
        "as" => Tok::As,
        "each" => Tok::Each,
        "in" => Tok::In,
        "while" => Tok::While,
        "when" => Tok::When,
        "else" => Tok::Else,
        "descend" => Tok::Descend,
        "by" => Tok::By,
        "using" => Tok::Using,
        "stop" => Tok::Stop,
        "skip" => Tok::Skip,
        "ask" => Tok::Ask,
        "and" => Tok::And,
        "or" => Tok::Or,
        "not" => Tok::Not,
        "yes" => Tok::Yes,
        "no" => Tok::No,
        "nothing" => Tok::Nothing,
        "with" => Tok::With,
        "use" => Tok::Use,
        "try" => Tok::Try,
        "change" => Tok::Change,
        "xor" => Tok::Xor,
        _ => return None,
    })
}

/// A line that ends with one of these continues on the next line.
fn continues(t: &Tok) -> bool {
    matches!(
        t,
        Tok::Plus
            | Tok::Minus
            | Tok::Star
            | Tok::Slash
            | Tok::Percent
            | Tok::Caret
            | Tok::At
            | Tok::Eq
            | Tok::Ne
            | Tok::Lt
            | Tok::Gt
            | Tok::Le
            | Tok::Ge
            | Tok::Amp
            | Tok::Pipe
            | Tok::Xor
            | Tok::Shl
            | Tok::Shr
            | Tok::Assign
            | Tok::PlusAssign
            | Tok::MinusAssign
            | Tok::StarAssign
            | Tok::SlashAssign
            | Tok::Arrow
            | Tok::FatArrow
            | Tok::Comma
            | Tok::Colon
            | Tok::And
            | Tok::Or
            | Tok::Not
    )
}

pub fn lex(src: &str, first_line: u32) -> Result<Vec<Token>, (String, u32)> {
    let c: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut line = first_line;
    let mut depth = 0i32;
    let mut out: Vec<Token> = Vec::new();
    macro_rules! push {
        ($t:expr) => {
            out.push(Token { tok: $t, line })
        };
    }
    while i < c.len() {
        let ch = c[i];
        let next = if i + 1 < c.len() { c[i + 1] } else { '\0' };
        match ch {
            ' ' | '\t' | '\r' => i += 1,
            '\n' => {
                if depth == 0 {
                    push!(Tok::Newline);
                }
                line += 1;
                i += 1;
            }
            '#' => {
                while i < c.len() && c[i] != '\n' {
                    i += 1;
                }
            }
            '"' => {
                let start_line = line;
                i += 1;
                let mut parts = Vec::new();
                let mut cur = String::new();
                loop {
                    if i >= c.len() {
                        return Err(("this text is missing its closing \"".into(), start_line));
                    }
                    match c[i] {
                        '"' => {
                            i += 1;
                            break;
                        }
                        '\\' => {
                            let e = if i + 1 < c.len() { c[i + 1] } else { '\\' };
                            cur.push(match e {
                                'n' => '\n',
                                't' => '\t',
                                other => other,
                            });
                            i += 2;
                        }
                        '{' => {
                            let mut j = i + 1;
                            let mut d = 1;
                            while j < c.len() {
                                if c[j] == '{' {
                                    d += 1;
                                } else if c[j] == '}' {
                                    d -= 1;
                                    if d == 0 {
                                        break;
                                    }
                                }
                                j += 1;
                            }
                            if j >= c.len() {
                                return Err((
                                    "a { inside text needs a matching } (write \\{ for a plain brace)"
                                        .into(),
                                    line,
                                ));
                            }
                            if !cur.is_empty() {
                                parts.push(StrPart::Lit(std::mem::take(&mut cur)));
                            }
                            parts.push(StrPart::Code(c[i + 1..j].iter().collect()));
                            i = j + 1;
                        }
                        other => {
                            if other == '\n' {
                                line += 1;
                            }
                            cur.push(other);
                            i += 1;
                        }
                    }
                }
                if !cur.is_empty() || parts.is_empty() {
                    parts.push(StrPart::Lit(cur));
                }
                out.push(Token { tok: Tok::Str(parts), line: start_line });
            }
            '0' if next == 'x' || next == 'X' => {
                let mut j = i + 2;
                let mut digits = String::new();
                while j < c.len() && (c[j].is_ascii_hexdigit() || c[j] == '_') {
                    if c[j] != '_' {
                        digits.push(c[j]);
                    }
                    j += 1;
                }
                match u64::from_str_radix(&digits, 16) {
                    Ok(n) => push!(Tok::Int(n as i64)),
                    Err(_) => return Err(("this hexadecimal number is empty or too long (at most 16 digits)".into(), line)),
                }
                i = j;
            }
            '0'..='9' => {
                let mut s = String::new();
                while i < c.len() && (c[i].is_ascii_digit() || c[i] == '_') {
                    if c[i] != '_' {
                        s.push(c[i]);
                    }
                    i += 1;
                }
                if i + 1 < c.len() && c[i] == '.' && c[i + 1].is_ascii_digit() {
                    s.push('.');
                    i += 1;
                    while i < c.len() && c[i].is_ascii_digit() {
                        s.push(c[i]);
                        i += 1;
                    }
                }
                if i < c.len() && (c[i] == 'e' || c[i] == 'E') {
                    let mut j = i + 1;
                    if j < c.len() && (c[j] == '+' || c[j] == '-') {
                        j += 1;
                    }
                    if j < c.len() && c[j].is_ascii_digit() {
                        s.extend(c[i..j].iter());
                        i = j;
                        while i < c.len() && c[i].is_ascii_digit() {
                            s.push(c[i]);
                            i += 1;
                        }
                    }
                }
                match s.parse::<f64>() {
                    Ok(n) => push!(Tok::Num(n)),
                    Err(_) => return Err((format!("'{}' isn't a number I can read", s), line)),
                }
            }
            _ if ch.is_alphabetic() || ch == '_' => {
                let mut s = String::new();
                while i < c.len() && (c[i].is_alphanumeric() || c[i] == '_') {
                    s.push(c[i]);
                    i += 1;
                }
                push!(keyword(&s).unwrap_or(Tok::Ident(s)));
            }
            _ => {
                let (t, n) = match (ch, next) {
                    ('-', '>') => (Tok::Arrow, 2),
                    ('=', '>') => (Tok::FatArrow, 2),
                    ('=', '=') => (Tok::Eq, 2),
                    ('!', '=') => (Tok::Ne, 2),
                    ('<', '<') => (Tok::Shl, 2),
                    ('>', '>') => (Tok::Shr, 2),
                    ('<', '=') => (Tok::Le, 2),
                    ('>', '=') => (Tok::Ge, 2),
                    ('+', '=') => (Tok::PlusAssign, 2),
                    ('-', '=') => (Tok::MinusAssign, 2),
                    ('*', '=') => (Tok::StarAssign, 2),
                    ('/', '=') => (Tok::SlashAssign, 2),
                    ('(', _) => (Tok::LParen, 1),
                    (')', _) => (Tok::RParen, 1),
                    ('[', _) => (Tok::LBrack, 1),
                    (']', _) => (Tok::RBrack, 1),
                    ('{', _) => (Tok::LBrace, 1),
                    ('}', _) => (Tok::RBrace, 1),
                    (',', _) => (Tok::Comma, 1),
                    (':', _) => (Tok::Colon, 1),
                    ('.', _) => (Tok::Dot, 1),
                    ('=', _) => (Tok::Assign, 1),
                    ('+', _) => (Tok::Plus, 1),
                    ('-', _) => (Tok::Minus, 1),
                    ('*', _) => (Tok::Star, 1),
                    ('/', _) => (Tok::Slash, 1),
                    ('%', _) => (Tok::Percent, 1),
                    ('^', _) => (Tok::Caret, 1),
                    ('@', _) => (Tok::At, 1),
                    ('&', _) => (Tok::Amp, 1),
                    ('|', _) => (Tok::Pipe, 1),
                    ('~', _) => (Tok::Tilde, 1),
                    ('<', _) => (Tok::Lt, 1),
                    ('>', _) => (Tok::Gt, 1),
                    _ => return Err((format!("I don't understand the symbol '{}'", ch), line)),
                };
                match t {
                    Tok::LParen | Tok::LBrack => depth += 1,
                    Tok::RParen | Tok::RBrack => depth = (depth - 1).max(0),
                    _ => {}
                }
                push!(t);
                i += n;
            }
        }
    }
    push!(Tok::Newline);
    push!(Tok::Eof);

    // Join lines: a line starting with `->`, or a previous line ending in an
    // operator, continues the same statement.
    let mut res: Vec<Token> = Vec::with_capacity(out.len());
    for (k, t) in out.iter().enumerate() {
        if t.tok == Tok::Newline {
            let next = out[k + 1..].iter().find(|t| t.tok != Tok::Newline).map(|t| &t.tok);
            if matches!(next, Some(Tok::Arrow)) {
                continue;
            }
            if let Some(prev) = res.last() {
                if continues(&prev.tok) {
                    continue;
                }
            }
        }
        res.push(t.clone());
    }
    Ok(res)
}
