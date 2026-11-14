//! A tiny JSON reader and writer (no dependencies).

pub enum J {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    pub fn get(&self, key: &str) -> Option<&J> {
        match self {
            J::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn str(&self) -> Option<&str> {
        match self {
            J::Str(s) => Some(s),
            _ => None,
        }
    }
}

pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

struct P {
    c: Vec<char>,
    i: usize,
}

impl P {
    fn ws(&mut self) {
        while self.i < self.c.len() && self.c[self.i].is_whitespace() {
            self.i += 1;
        }
    }

    fn lit(&mut self, word: &str, v: J) -> Result<J, String> {
        let end = self.i + word.len();
        if end <= self.c.len() && self.c[self.i..end].iter().collect::<String>() == word {
            self.i = end;
            Ok(v)
        } else {
            Err(format!("unexpected text in JSON at position {}", self.i))
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.c.len() {
            return Err("JSON ends inside a \\u escape".into());
        }
        let s: String = self.c[self.i..self.i + 4].iter().collect();
        self.i += 4;
        u32::from_str_radix(&s, 16).map_err(|_| "bad \\u escape in JSON".to_string())
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1;
        let mut out = String::new();
        loop {
            if self.i >= self.c.len() {
                return Err("JSON text is missing its closing quote".into());
            }
            let ch = self.c[self.i];
            self.i += 1;
            match ch {
                '"' => return Ok(out),
                '\\' => {
                    let e = *self.c.get(self.i).ok_or("JSON ends inside an escape")?;
                    self.i += 1;
                    match e {
                        'n' => out.push('\n'),
                        't' => out.push('\t'),
                        'r' => out.push('\r'),
                        'b' => out.push('\u{8}'),
                        'f' => out.push('\u{c}'),
                        'u' => {
                            let mut cp = self.hex4()?;
                            if (0xD800..0xDC00).contains(&cp)
                                && self.c.get(self.i) == Some(&'\\')
                                && self.c.get(self.i + 1) == Some(&'u')
                            {
                                self.i += 2;
                                let lo = self.hex4()?;
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo.wrapping_sub(0xDC00) & 0x3FF);
                            }
                            out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                        }
                        other => out.push(other),
                    }
                }
                other => out.push(other),
            }
        }
    }

    fn value(&mut self) -> Result<J, String> {
        self.ws();
        let ch = *self.c.get(self.i).ok_or("JSON ended too early")?;
        match ch {
            '{' => {
                self.i += 1;
                let mut m = vec![];
                loop {
                    self.ws();
                    match self.c.get(self.i) {
                        Some('}') => {
                            self.i += 1;
                            return Ok(J::Obj(m));
                        }
                        Some(',') => self.i += 1,
                        Some('"') => {
                            let k = self.string()?;
                            self.ws();
                            if self.c.get(self.i) != Some(&':') {
                                return Err("JSON object is missing a ':'".into());
                            }
                            self.i += 1;
                            let v = self.value()?;
                            m.push((k, v));
                        }
                        _ => return Err("JSON object is not closed properly".into()),
                    }
                }
            }
            '[' => {
                self.i += 1;
                let mut a = vec![];
                loop {
                    self.ws();
                    match self.c.get(self.i) {
                        Some(']') => {
                            self.i += 1;
                            return Ok(J::Arr(a));
                        }
                        Some(',') => self.i += 1,
                        Some(_) => a.push(self.value()?),
                        None => return Err("JSON list is not closed".into()),
                    }
                }
            }
            '"' => Ok(J::Str(self.string()?)),
            't' => self.lit("true", J::Bool(true)),
            'f' => self.lit("false", J::Bool(false)),
            'n' => self.lit("null", J::Null),
            _ => {
                let start = self.i;
                while self.i < self.c.len()
                    && (self.c[self.i].is_ascii_digit() || "+-.eE".contains(self.c[self.i]))
                {
                    self.i += 1;
                }
                let s: String = self.c[start..self.i].iter().collect();
                s.parse::<f64>()
                    .map(J::Num)
                    .map_err(|_| format!("unexpected text in JSON at position {}", start))
            }
        }
    }
}

pub fn parse(s: &str) -> Result<J, String> {
    let mut p = P { c: s.chars().collect(), i: 0 };
    let v = p.value()?;
    p.ws();
    if p.i < p.c.len() {
        return Err(format!("extra text after the JSON value at position {}", p.i));
    }
    Ok(v)
}
