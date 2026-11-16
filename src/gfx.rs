//! Graphics for games: a window to draw pictures in, and the keyboard and
//! mouse.
//!
//! Antares draws every picture itself into a `Canvas` (a row of 0xRRGGBB
//! pixels) and the window only shows the finished picture, scaled up so
//! pixel art stays sharp. Drawing works the same on every system; opening a
//! window is Windows-only for now (`win` below), using only what Windows
//! itself provides.
//!
//! Everything runs on the program's own thread: the window belongs to it,
//! and its messages are handled whenever the program calls `window_open`,
//! `frame`, `key` or `wait`.

use crate::builtins::{num, s, text, want};
use crate::interp::Interp;
use crate::value::Value;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::time::{Duration, Instant};

type Res = Result<Value, String>;

// ---------- the picture ----------

/// A picture: `w` by `h` pixels, each 0xRRGGBB, row by row from the top.
pub struct Canvas {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u32>,
}

impl Canvas {
    pub fn new(w: usize, h: usize) -> Canvas {
        Canvas { w, h, px: vec![0; w * h] }
    }

    pub fn fill(&mut self, c: u32) {
        self.px.fill(c);
    }

    pub fn put(&mut self, x: i64, y: i64, c: u32) {
        if x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h {
            self.px[y as usize * self.w + x as usize] = c;
        }
    }

    pub fn get(&self, x: i64, y: i64) -> Option<u32> {
        (x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h).then(|| self.px[y as usize * self.w + x as usize])
    }

    /// A filled rectangle, its top-left corner at (x, y).
    pub fn rect(&mut self, x: i64, y: i64, w: i64, h: i64, c: u32) {
        let (x0, y0) = (x.max(0), y.max(0));
        let (x1, y1) = (x.saturating_add(w).min(self.w as i64), y.saturating_add(h).min(self.h as i64));
        for yy in y0..y1 {
            let row = yy as usize * self.w;
            self.px[row + x0 as usize..row + x1.max(x0) as usize].fill(c);
        }
    }

    /// A filled circle centred on (cx, cy).
    pub fn circle(&mut self, cx: i64, cy: i64, r: f64, c: u32) {
        if !(r >= 0.0) {
            return;
        }
        let ri = r.floor().min(1e9) as i64;
        let limit = (r + 0.5) * (r + 0.5);
        // only the rows inside the picture
        for dy in (-ri).max(-cy)..=ri.min(self.h as i64 - 1 - cy) {
            let half = (limit - (dy as f64) * (dy as f64)).max(0.0).sqrt().floor() as i64;
            let half = half.min(ri);
            self.rect(cx - half, cy + dy, 2 * half + 1, 1, c);
        }
    }

    /// A line from (x0, y0) to (x1, y1), both ends included.
    pub fn line(&mut self, x0: i64, y0: i64, x1: i64, y1: i64, c: u32) {
        // Ends far outside the picture are first moved to its edge, so a line
        // a million pixels long costs no more than one across the picture.
        let far = |v: i64, n: usize| v < -1 || v > n as i64;
        let (mut x0, mut y0, x1, y1) = if far(x0, self.w) || far(x1, self.w) || far(y0, self.h) || far(y1, self.h) {
            let (fx0, fy0, dxf, dyf) = (x0 as f64, y0 as f64, (x1 - x0) as f64, (y1 - y0) as f64);
            let (mut t0, mut t1) = (0.0f64, 1.0f64);
            let edges = [(-dxf, fx0 + 1.0), (dxf, self.w as f64 - fx0), (-dyf, fy0 + 1.0), (dyf, self.h as f64 - fy0)];
            for (p, q) in edges {
                if p == 0.0 {
                    if q < 0.0 {
                        return;
                    }
                } else if p < 0.0 {
                    t0 = t0.max(q / p);
                } else {
                    t1 = t1.min(q / p);
                }
            }
            if t0 > t1 {
                return;
            }
            let at = |t: f64| ((fx0 + t * dxf).round() as i64, (fy0 + t * dyf).round() as i64);
            let ((a, b), (c2, d)) = (at(t0), at(t1));
            (a, b, c2, d)
        } else {
            (x0, y0, x1, y1)
        };
        let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
        let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
        let mut err = dx + dy;
        loop {
            self.put(x0, y0, c);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    /// Text in the built-in font, its top-left corner at (x, y). Each
    /// character is 6 pixels wide and each line 8 pixels tall, times `size`.
    pub fn text(&mut self, t: &str, x: i64, y: i64, c: u32, size: i64) {
        let (mut cx, mut cy) = (x, y);
        for ch in t.chars() {
            if ch == '\n' {
                cx = x;
                cy += 8 * size;
                continue;
            }
            let glyph = glyph(ch);
            for (row, bits) in glyph.iter().enumerate() {
                for (col, b) in bits.bytes().enumerate() {
                    if b == b'#' {
                        self.rect(cx + col as i64 * size, cy + row as i64 * size, size, size, c);
                    }
                }
            }
            cx += 6 * size;
        }
    }

    /// Pixel art: each row a list of colours, `None` showing what's behind.
    pub fn sprite(&mut self, rows: &[Vec<Option<u32>>], x: i64, y: i64, size: i64) {
        for (r, row) in rows.iter().enumerate() {
            for (k, c) in row.iter().enumerate() {
                if let Some(c) = c {
                    self.rect(x + k as i64 * size, y + r as i64 * size, size, size, *c);
                }
            }
        }
    }

    /// The picture as a PNG file.
    pub fn png(&self) -> Vec<u8> {
        let mut raw = Vec::with_capacity(self.h * (1 + 3 * self.w));
        for row in self.px.chunks(self.w.max(1)) {
            raw.push(0); // no filter
            for p in row {
                raw.extend_from_slice(&[(p >> 16) as u8, (p >> 8) as u8, *p as u8]);
            }
        }
        // zlib data made of uncompressed ("stored") blocks
        let mut z = vec![0x78, 0x01];
        let mut blocks = raw.chunks(65535).peekable();
        if blocks.peek().is_none() {
            z.extend_from_slice(&[1, 0, 0, 0xFF, 0xFF]);
        }
        while let Some(b) = blocks.next() {
            z.push(if blocks.peek().is_none() { 1 } else { 0 });
            z.extend_from_slice(&(b.len() as u16).to_le_bytes());
            z.extend_from_slice(&(!(b.len() as u16)).to_le_bytes());
            z.extend_from_slice(b);
        }
        z.extend_from_slice(&adler32(&raw).to_be_bytes());

        let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
        let mut header = vec![];
        header.extend_from_slice(&(self.w as u32).to_be_bytes());
        header.extend_from_slice(&(self.h as u32).to_be_bytes());
        header.extend_from_slice(&[8, 2, 0, 0, 0]); // 8 bits, RGB
        chunk(&mut out, b"IHDR", &header);
        chunk(&mut out, b"IDAT", &z);
        chunk(&mut out, b"IEND", &[]);
        out
    }
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in data.chunks(5552) {
        for &x in chunk {
            a += x as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

// ---------- colours ----------

const COLOURS: &[(&str, u32)] = &[
    ("black", 0x000000),
    ("white", 0xFFFFFF),
    ("gray", 0x808080),
    ("grey", 0x808080),
    ("lightgray", 0xC0C0C0),
    ("lightgrey", 0xC0C0C0),
    ("darkgray", 0x404040),
    ("darkgrey", 0x404040),
    ("silver", 0xC0C0C0),
    ("red", 0xFF0000),
    ("maroon", 0x800000),
    ("orange", 0xFFA500),
    ("gold", 0xFFD700),
    ("yellow", 0xFFFF00),
    ("olive", 0x808000),
    ("lime", 0x00FF00),
    ("green", 0x008000),
    ("teal", 0x008080),
    ("cyan", 0x00FFFF),
    ("skyblue", 0x87CEEB),
    ("blue", 0x0000FF),
    ("navy", 0x000080),
    ("purple", 0x800080),
    ("magenta", 0xFF00FF),
    ("pink", 0xFFC0CB),
    ("brown", 0xA52A2A),
];

/// A colour given as a name ("red"), "#rgb", "#rrggbb", or [red, green, blue]
/// with each part from 0 to 255.
pub fn colour(v: &Value, name: &str) -> Result<u32, String> {
    let bad = |what: String| {
        format!("{} needs a colour: a name like \"red\", a code like \"#ff8800\", or [red, green, blue] from 0 to 255, not {}", name, what)
    };
    match v {
        Value::Str(t) => {
            let t = t.trim().to_ascii_lowercase();
            if let Some(hex) = t.strip_prefix('#') {
                let digits: Option<Vec<u32>> = hex.chars().map(|c| c.to_digit(16)).collect();
                return match digits.as_deref() {
                    Some([r, g, b]) => Ok((r * 17) << 16 | (g * 17) << 8 | b * 17),
                    Some(d) if d.len() == 6 => Ok(d.iter().fold(0, |acc, x| acc << 4 | x)),
                    _ => Err(bad(format!("\"{}\"", t))),
                };
            }
            let key: String = t.chars().filter(|c| !c.is_whitespace()).collect();
            COLOURS.iter().find(|(n, _)| *n == key).map(|(_, c)| *c).ok_or_else(|| bad(format!("\"{}\" (a colour I don't know)", t)))
        }
        Value::List(_) | Value::Tensor(_) | Value::Ints(_) => {
            let parts: Vec<f64> = match v {
                Value::List(l) => l.iter().map(|x| x.scalar()).collect::<Option<Vec<f64>>>().ok_or_else(|| bad("a list with something other than numbers".into()))?,
                Value::Tensor(t) => t.data.borrow().clone(),
                Value::Ints(n) => n.iter().map(|x| *x as f64).collect(),
                _ => unreachable!(),
            };
            if parts.len() != 3 {
                return Err(bad(format!("a list of {} numbers", parts.len())));
            }
            let part = |x: f64| x.round().clamp(0.0, 255.0) as u32;
            Ok(part(parts[0]) << 16 | part(parts[1]) << 8 | part(parts[2]))
        }
        other => Err(bad(format!("a {}", other.kind()))),
    }
}

// ---------- the font ----------

/// The built-in font: 5 by 7 pixels for each character from ' ' to '~'.
#[rustfmt::skip]
const FONT: [[&str; 7]; 95] = [
    [".....",".....",".....",".....",".....",".....","....."], // space
    ["..#..","..#..","..#..","..#..","..#..",".....","..#.."], // !
    [".#.#.",".#.#.",".....",".....",".....",".....","....."], // "
    [".#.#.",".#.#.","#####",".#.#.","#####",".#.#.",".#.#."], // #
    ["..#..",".####","#.#..",".###.","..#.#","####.","..#.."], // $
    ["##...","##..#","...#.","..#..",".#...","#..##","...##"], // %
    [".##..","#..#.","#.#..",".#...","#.#.#","#..#.",".##.#"], // &
    ["..#..","..#..",".....",".....",".....",".....","....."], // '
    ["...#.","..#..",".#...",".#...",".#...","..#..","...#."], // (
    [".#...","..#..","...#.","...#.","...#.","..#..",".#..."], // )
    [".....","..#..","#.#.#",".###.","#.#.#","..#..","....."], // *
    [".....","..#..","..#..","#####","..#..","..#..","....."], // +
    [".....",".....",".....",".....","..##.","..#..",".#..."], // ,
    [".....",".....",".....","#####",".....",".....","....."], // -
    [".....",".....",".....",".....",".....",".##..",".##.."], // .
    [".....","....#","...#.","..#..",".#...","#....","....."], // /
    [".###.","#...#","#..##","#.#.#","##..#","#...#",".###."], // 0
    ["..#..",".##..","..#..","..#..","..#..","..#..",".###."], // 1
    [".###.","#...#","....#","...#.","..#..",".#...","#####"], // 2
    ["#####","...#.","..#..","...#.","....#","#...#",".###."], // 3
    ["...#.","..##.",".#.#.","#..#.","#####","...#.","...#."], // 4
    ["#####","#....","####.","....#","....#","#...#",".###."], // 5
    ["..##.",".#...","#....","####.","#...#","#...#",".###."], // 6
    ["#####","....#","...#.","..#..",".#...",".#...",".#..."], // 7
    [".###.","#...#","#...#",".###.","#...#","#...#",".###."], // 8
    [".###.","#...#","#...#",".####","....#","...#.",".##.."], // 9
    [".....",".##..",".##..",".....",".##..",".##..","....."], // :
    [".....",".##..",".##..",".....",".##..","..#..",".#..."], // ;
    ["...#.","..#..",".#...","#....",".#...","..#..","...#."], // <
    [".....",".....","#####",".....","#####",".....","....."], // =
    [".#...","..#..","...#.","....#","...#.","..#..",".#..."], // >
    [".###.","#...#","....#","...#.","..#..",".....","..#.."], // ?
    [".###.","#...#","....#",".##.#","#.#.#","#.#.#",".###."], // @
    [".###.","#...#","#...#","#####","#...#","#...#","#...#"], // A
    ["####.","#...#","#...#","####.","#...#","#...#","####."], // B
    [".###.","#...#","#....","#....","#....","#...#",".###."], // C
    ["###..","#..#.","#...#","#...#","#...#","#..#.","###.."], // D
    ["#####","#....","#....","####.","#....","#....","#####"], // E
    ["#####","#....","#....","####.","#....","#....","#...."], // F
    [".###.","#...#","#....","#.###","#...#","#...#",".####"], // G
    ["#...#","#...#","#...#","#####","#...#","#...#","#...#"], // H
    [".###.","..#..","..#..","..#..","..#..","..#..",".###."], // I
    ["..###","...#.","...#.","...#.","...#.","#..#.",".##.."], // J
    ["#...#","#..#.","#.#..","##...","#.#..","#..#.","#...#"], // K
    ["#....","#....","#....","#....","#....","#....","#####"], // L
    ["#...#","##.##","#.#.#","#.#.#","#...#","#...#","#...#"], // M
    ["#...#","#...#","##..#","#.#.#","#..##","#...#","#...#"], // N
    [".###.","#...#","#...#","#...#","#...#","#...#",".###."], // O
    ["####.","#...#","#...#","####.","#....","#....","#...."], // P
    [".###.","#...#","#...#","#...#","#.#.#","#..#.",".##.#"], // Q
    ["####.","#...#","#...#","####.","#.#..","#..#.","#...#"], // R
    [".####","#....","#....",".###.","....#","....#","####."], // S
    ["#####","..#..","..#..","..#..","..#..","..#..","..#.."], // T
    ["#...#","#...#","#...#","#...#","#...#","#...#",".###."], // U
    ["#...#","#...#","#...#","#...#","#...#",".#.#.","..#.."], // V
    ["#...#","#...#","#...#","#.#.#","#.#.#","#.#.#",".#.#."], // W
    ["#...#","#...#",".#.#.","..#..",".#.#.","#...#","#...#"], // X
    ["#...#","#...#",".#.#.","..#..","..#..","..#..","..#.."], // Y
    ["#####","....#","...#.","..#..",".#...","#....","#####"], // Z
    [".###.",".#...",".#...",".#...",".#...",".#...",".###."], // [
    [".....","#....",".#...","..#..","...#.","....#","....."], // backslash
    [".###.","...#.","...#.","...#.","...#.","...#.",".###."], // ]
    ["..#..",".#.#.","#...#",".....",".....",".....","....."], // ^
    [".....",".....",".....",".....",".....",".....","#####"], // _
    [".#...","..#..",".....",".....",".....",".....","....."], // `
    [".....",".....",".###.","....#",".####","#...#",".####"], // a
    ["#....","#....","#.##.","##..#","#...#","#...#","####."], // b
    [".....",".....",".###.","#....","#....","#...#",".###."], // c
    ["....#","....#",".##.#","#..##","#...#","#...#",".####"], // d
    [".....",".....",".###.","#...#","#####","#....",".###."], // e
    ["..##.",".#..#",".#...","###..",".#...",".#...",".#..."], // f
    [".....",".####","#...#","#...#",".####","....#",".###."], // g
    ["#....","#....","#.##.","##..#","#...#","#...#","#...#"], // h
    ["..#..",".....",".##..","..#..","..#..","..#..",".###."], // i
    ["...#.",".....","..##.","...#.","...#.","#..#.",".##.."], // j
    ["#....","#....","#..#.","#.#..","##...","#.#..","#..#."], // k
    [".##..","..#..","..#..","..#..","..#..","..#..",".###."], // l
    [".....",".....","##.#.","#.#.#","#.#.#","#...#","#...#"], // m
    [".....",".....","#.##.","##..#","#...#","#...#","#...#"], // n
    [".....",".....",".###.","#...#","#...#","#...#",".###."], // o
    [".....",".....","####.","#...#","####.","#....","#...."], // p
    [".....",".....",".##.#","#..##",".####","....#","....#"], // q
    [".....",".....","#.##.","##..#","#....","#....","#...."], // r
    [".....",".....",".####","#....",".###.","....#","####."], // s
    [".#...",".#...","###..",".#...",".#...",".#..#","..##."], // t
    [".....",".....","#...#","#...#","#...#","#..##",".##.#"], // u
    [".....",".....","#...#","#...#","#...#",".#.#.","..#.."], // v
    [".....",".....","#...#","#...#","#.#.#","#.#.#",".#.#."], // w
    [".....",".....","#...#",".#.#.","..#..",".#.#.","#...#"], // x
    [".....",".....","#...#","#...#",".####","....#",".###."], // y
    [".....",".....","#####","...#.","..#..",".#...","#####"], // z
    ["...#.","..#..","..#..",".#...","..#..","..#..","...#."], // {
    ["..#..","..#..","..#..","..#..","..#..","..#..","..#.."], // |
    [".#...","..#..","..#..","...#.","..#..","..#..",".#..."], // }
    [".....",".....",".#...","#.#.#","...#.",".....","....."], // ~
];

fn glyph(c: char) -> &'static [&'static str; 7] {
    let i = c as u32;
    if (32..127).contains(&i) {
        &FONT[(i - 32) as usize]
    } else {
        &FONT[('?' as u32 - 32) as usize]
    }
}

// ---------- the window and its input ----------

/// What the keyboard and mouse are doing, as the window reports it.
#[derive(Default)]
struct Input {
    /// Keys (and "click", "right click") held down now.
    held: Vec<&'static str>,
    /// Keys pressed since the last frame.
    pressed: Vec<&'static str>,
    /// Keys pressed (with repeats), for `key()`.
    typed: VecDeque<&'static str>,
    mouse_x: i64,
    mouse_y: i64,
}

// (only the window code reports keys)
#[cfg_attr(not(windows), allow(dead_code))]
impl Input {
    fn down(&mut self, k: &'static str, repeat: bool) {
        if !repeat {
            if !self.held.contains(&k) {
                self.held.push(k);
            }
            if !self.pressed.contains(&k) {
                self.pressed.push(k);
            }
        }
        if self.typed.len() < 64 && k != "click" && k != "right click" {
            self.typed.push_back(k);
        }
    }

    fn up(&mut self, k: &'static str) {
        self.held.retain(|h| *h != k);
    }
}

struct Screen {
    canvas: Canvas,
    #[cfg_attr(not(windows), allow(dead_code))]
    scale: usize,
    input: Input,
    /// When the next frame is due.
    next: Instant,
    closed: bool,
    #[cfg(windows)]
    hwnd: win::Hwnd,
}

thread_local! {
    static SCREEN: RefCell<Option<Screen>> = const { RefCell::new(None) };
}

/// The names `held` and `pressed` know.
fn key_name(k: &str) -> Option<&'static str> {
    const NAMED: &[&str] = &[
        "up", "down", "left", "right", "space", "enter", "escape", "backspace", "tab", "shift", "ctrl", "alt", "delete", "click",
        "right click", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
    ];
    const CHARS: &[&str] = &[
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ];
    let k = k.to_ascii_lowercase();
    NAMED.iter().chain(CHARS).find(|n| **n == k).copied()
}

/// Runs `f` on the open window, or explains that there isn't one.
fn with_screen<R>(name: &str, f: impl FnOnce(&mut Screen) -> R) -> Result<R, String> {
    SCREEN.with(|s| match s.borrow_mut().as_mut() {
        Some(sc) => Ok(f(sc)),
        None => Err(format!("{} draws in a window; open one first with window(width, height)", name)),
    })
}

/// Whether a window is open (and not closed by the player).
pub fn window_is_open() -> bool {
    SCREEN.with(|s| s.borrow().as_ref().is_some_and(|sc| !sc.closed))
}

/// Handles the window's waiting messages (keys, mouse, closing).
fn pump() {
    #[cfg(windows)]
    win::pump();
}

/// For `key()`: the next key pressed in the window, when a window is open.
pub fn next_key() -> Option<Option<String>> {
    if !window_is_open() {
        return None;
    }
    pump();
    SCREEN.with(|s| s.borrow_mut().as_mut().map(|sc| sc.input.typed.pop_front().map(|k| k.to_string())))
}

/// For `wait()` while a window is open: waits, but keeps the window answering.
pub fn wait(secs: f64) {
    let end = Instant::now() + Duration::from_secs_f64(secs);
    loop {
        pump();
        let now = Instant::now();
        if now >= end {
            break;
        }
        std::thread::sleep((end - now).min(Duration::from_millis(5)));
    }
}

/// Tells the player about a problem that stopped the program, in a message
/// box if the program's console was hidden when its window opened.
pub fn tell_problem(msg: &str) {
    #[cfg(windows)]
    win::tell_problem(msg);
    #[cfg(not(windows))]
    let _ = msg;
}

/// Whole numbers for positions: fractions are rounded down.
fn pos(v: &Value, name: &str) -> Result<i64, String> {
    let x = num(v, name)?;
    if !x.is_finite() {
        return Err(format!("{} needs an ordinary number, not {}", name, x));
    }
    Ok(x.floor() as i64)
}

fn opt_colour(a: &[Value], i: usize, name: &str, default: u32) -> Result<u32, String> {
    match a.get(i) {
        Some(v) => colour(v, name),
        None => Ok(default),
    }
}

fn opt_size(a: &[Value], i: usize, name: &str) -> Result<i64, String> {
    match a.get(i) {
        None => Ok(1),
        Some(v) => {
            let n = num(v, name)?;
            if n != n.trunc() || !(1.0..=64.0).contains(&n) {
                return Err(format!("{} needs a size that is a whole number from 1 to 64, not {}", name, v.show()));
            }
            Ok(n as i64)
        }
    }
}

fn inputs(a: &[Value], lo: usize, hi: usize, name: &str, what: &str) -> Result<(), String> {
    if a.len() < lo || a.len() > hi {
        return Err(format!("{} takes {}, but was given {} input{}", name, what, a.len(), if a.len() == 1 { "" } else { "s" }));
    }
    Ok(())
}

const WHITE: u32 = 0xFFFFFF;

// ---------- the built-in functions ----------

/// window(width, height [, title [, scale]])
pub fn b_window(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 2, 4, "window", "a width and a height, and optionally a title and a scale")?;
    let whole = |v: &Value, what: &str, hi: f64| -> Result<usize, String> {
        let n = num(v, "window")?;
        if n != n.trunc() || n < 1.0 || n > hi {
            return Err(format!("window needs a {} that is a whole number from 1 to {}, not {}", what, hi, v.show()));
        }
        Ok(n as usize)
    };
    let w = whole(&a[0], "width", 4096.0)?;
    let h = whole(&a[1], "height", 4096.0)?;
    let title = match a.get(2) {
        Some(v) => text(v, "window")?.to_string(),
        None => "Antares".to_string(),
    };
    let scale = match a.get(3) {
        Some(v) => Some(whole(v, "scale", 32.0)?),
        None => None,
    };
    if SCREEN.with(|s| s.borrow().as_ref().is_some_and(|sc| !sc.closed)) {
        return Err("a window is already open; a program has one window".into());
    }
    open(w, h, &title, scale)?;
    Ok(Value::Nothing)
}

#[cfg(windows)]
fn open(w: usize, h: usize, title: &str, scale: Option<usize>) -> Result<(), String> {
    let (hwnd, scale) = win::open(w, h, title, scale)?;
    SCREEN.with(|s| {
        *s.borrow_mut() = Some(Screen { canvas: Canvas::new(w, h), scale, input: Input::default(), next: Instant::now(), closed: false, hwnd })
    });
    win::show(hwnd);
    pump();
    Ok(())
}

#[cfg(not(windows))]
fn open(_: usize, _: usize, _: &str, _: Option<usize>) -> Result<(), String> {
    Err("graphics windows work on Windows for now; macOS and Linux are coming".into())
}

/// window_open(): yes until the player closes the window.
pub fn b_window_open(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 0, "window_open")?;
    pump();
    SCREEN.with(|s| match s.borrow().as_ref() {
        Some(sc) => Ok(Value::Bool(!sc.closed)),
        None => Err("window_open says whether the window is still open; open one first with window(width, height)".into()),
    })
}

/// frame(): shows the picture, then waits for the next of 60 frames a second.
pub fn b_frame(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 0, "frame")?;
    let closed = with_screen("frame", |sc| sc.closed)?;
    if closed {
        // Closing a game's window ends the game, even in a `while yes` loop.
        use std::io::Write;
        let _ = std::io::stdout().flush();
        std::process::exit(0);
    }
    #[cfg(windows)]
    with_screen("frame", |sc| win::present(sc.hwnd, &sc.canvas, sc.scale))?;
    let next = with_screen("frame", |sc| {
        sc.input.pressed.clear();
        let step = Duration::from_secs_f64(1.0 / 60.0);
        let now = Instant::now();
        // after a slow frame, start counting again instead of rushing to catch up
        sc.next = if sc.next + step < now { now } else { sc.next + step };
        sc.next
    })?;
    loop {
        let now = Instant::now();
        if now >= next {
            break;
        }
        let left = next - now;
        if left > Duration::from_millis(2) {
            std::thread::sleep(left - Duration::from_millis(1));
        } else {
            std::thread::yield_now();
        }
    }
    pump();
    Ok(Value::Nothing)
}

/// fill([colour]): the whole picture in one colour (black if none is given).
pub fn b_fill(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 0, 1, "fill", "a colour")?;
    let c = opt_colour(&a, 0, "fill", 0)?;
    with_screen("fill", |sc| sc.canvas.fill(c))?;
    Ok(Value::Nothing)
}

/// rect(x, y, width, height [, colour])
pub fn b_rect(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 4, 5, "rect", "x, y, a width and a height, and optionally a colour")?;
    let (x, y, w, h) = (pos(&a[0], "rect")?, pos(&a[1], "rect")?, pos(&a[2], "rect")?, pos(&a[3], "rect")?);
    let c = opt_colour(&a, 4, "rect", WHITE)?;
    with_screen("rect", |sc| sc.canvas.rect(x, y, w, h, c))?;
    Ok(Value::Nothing)
}

/// circle(x, y, radius [, colour])
pub fn b_circle(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 3, 4, "circle", "x, y and a radius, and optionally a colour")?;
    let (x, y, r) = (pos(&a[0], "circle")?, pos(&a[1], "circle")?, num(&a[2], "circle")?);
    let c = opt_colour(&a, 3, "circle", WHITE)?;
    with_screen("circle", |sc| sc.canvas.circle(x, y, r, c))?;
    Ok(Value::Nothing)
}

/// line(x1, y1, x2, y2 [, colour])
pub fn b_line(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 4, 5, "line", "x1, y1, x2 and y2, and optionally a colour")?;
    let (x0, y0, x1, y1) = (pos(&a[0], "line")?, pos(&a[1], "line")?, pos(&a[2], "line")?, pos(&a[3], "line")?);
    let c = opt_colour(&a, 4, "line", WHITE)?;
    with_screen("line", |sc| sc.canvas.line(x0, y0, x1, y1, c))?;
    Ok(Value::Nothing)
}

/// pixel(x, y [, colour])
pub fn b_pixel(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 2, 3, "pixel", "x and y, and optionally a colour")?;
    let (x, y) = (pos(&a[0], "pixel")?, pos(&a[1], "pixel")?);
    let c = opt_colour(&a, 2, "pixel", WHITE)?;
    with_screen("pixel", |sc| sc.canvas.put(x, y, c))?;
    Ok(Value::Nothing)
}

/// label(text, x, y [, colour [, size]])
pub fn b_label(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 3, 5, "label", "some text, x and y, and optionally a colour and a size")?;
    let t = a[0].show();
    let (x, y) = (pos(&a[1], "label")?, pos(&a[2], "label")?);
    let c = opt_colour(&a, 3, "label", WHITE)?;
    let size = opt_size(&a, 4, "label")?;
    with_screen("label", |sc| sc.canvas.text(&t, x, y, c, size))?;
    Ok(Value::Nothing)
}

/// sprite(rows, x, y, colours [, size]): pixel art written as text, one
/// character per pixel; `colours` maps characters to colours, and any other
/// character shows what's behind.
pub fn b_sprite(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 4, 5, "sprite", "rows of text, x, y and a map of colours, and optionally a size")?;
    let Value::List(rows) = &a[0] else {
        return Err(format!("sprite needs its picture as a list of text rows, like [\".r.\", \"rrr\"], not a {}", a[0].kind()));
    };
    let (x, y) = (pos(&a[1], "sprite")?, pos(&a[2], "sprite")?);
    let Value::Map(map) = &a[3] else {
        return Err(format!("sprite needs a map from characters to colours, like {{r: \"red\"}}, not a {}", a[3].kind()));
    };
    let mut colours: Vec<(char, u32)> = vec![];
    for (k, v) in map.iter() {
        let mut cs = k.chars();
        let (Some(ch), None) = (cs.next(), cs.next()) else {
            return Err(format!("sprite's colours are named by one character each, not \"{}\"", k));
        };
        colours.push((ch, colour(v, "sprite")?));
    }
    let size = opt_size(&a, 4, "sprite")?;
    let mut pixels = vec![];
    for r in rows.iter() {
        let row = text(r, "sprite").map_err(|_| format!("each row of a sprite is text, not a {}", r.kind()))?;
        pixels.push(row.chars().map(|ch| colours.iter().find(|(k, _)| *k == ch).map(|(_, c)| *c)).collect::<Vec<_>>());
    }
    with_screen("sprite", |sc| sc.canvas.sprite(&pixels, x, y, size))?;
    Ok(Value::Nothing)
}

fn key_input(a: &[Value], name: &str) -> Result<&'static str, String> {
    want(a, 1, name)?;
    let k = text(&a[0], name)?;
    key_name(&k).ok_or_else(|| {
        format!("{} doesn't know a key called \"{}\"; keys are named like \"left\", \"space\", \"enter\", \"a\", \"7\" or \"click\"", name, k)
    })
}

/// held(key): yes while the key (or "click" / "right click") is held down.
pub fn b_held(_: &mut Interp, a: Vec<Value>) -> Res {
    let k = key_input(&a, "held")?;
    Ok(Value::Bool(with_screen("held", |sc| sc.input.held.contains(&k))?))
}

/// pressed(key): yes if the key went down since the last frame.
pub fn b_pressed(_: &mut Interp, a: Vec<Value>) -> Res {
    let k = key_input(&a, "pressed")?;
    Ok(Value::Bool(with_screen("pressed", |sc| sc.input.pressed.contains(&k))?))
}

/// mouse(): {x, y, left, right}, in the window's own pixels.
pub fn b_mouse(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 0, "mouse")?;
    let (x, y, l, r) = with_screen("mouse", |sc| {
        (sc.input.mouse_x, sc.input.mouse_y, sc.input.held.contains(&"click"), sc.input.held.contains(&"right click"))
    })?;
    Ok(Value::Map(Rc::new(vec![
        ("x".into(), Value::Num(x as f64)),
        ("y".into(), Value::Num(y as f64)),
        ("left".into(), Value::Bool(l)),
        ("right".into(), Value::Bool(r)),
    ])))
}

/// screenshot(file): saves the picture as a PNG file.
pub fn b_screenshot(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "screenshot")?;
    let file = text(&a[0], "screenshot")?;
    let png = with_screen("screenshot", |sc| sc.canvas.png())?;
    std::fs::write(&*file, png).map_err(|e| format!("couldn't save the screenshot to \"{}\": {}", file, e))?;
    Ok(s(&*file))
}

// ---------- Windows ----------

#[cfg(windows)]
mod win {
    use super::{key_name, Canvas, SCREEN};
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, Ordering};

    pub type Hwnd = *mut c_void;
    type Handle = *mut c_void;

    #[repr(C)]
    struct WndClassExW {
        size: u32,
        style: u32,
        wnd_proc: extern "system" fn(Hwnd, u32, usize, isize) -> isize,
        cls_extra: i32,
        wnd_extra: i32,
        instance: Handle,
        icon: Handle,
        cursor: Handle,
        background: Handle,
        menu_name: *const u16,
        class_name: *const u16,
        icon_small: Handle,
    }

    #[repr(C)]
    struct Msg {
        hwnd: Hwnd,
        message: u32,
        wparam: usize,
        lparam: isize,
        time: u32,
        pt: [i32; 2],
        private: u32,
    }

    #[repr(C)]
    struct Rect {
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    }

    #[repr(C)]
    struct BitmapInfo {
        size: u32,
        width: i32,
        height: i32,
        planes: u16,
        bit_count: u16,
        compression: u32,
        size_image: u32,
        x_ppm: i32,
        y_ppm: i32,
        clr_used: u32,
        clr_important: u32,
        colors: [u32; 3],
    }

    #[link(name = "user32")]
    unsafe extern "system" {
        fn RegisterClassExW(class: *const WndClassExW) -> u16;
        fn CreateWindowExW(ex_style: u32, class: *const u16, title: *const u16, style: u32, x: i32, y: i32, w: i32, h: i32, parent: Hwnd, menu: Handle, instance: Handle, param: *mut c_void) -> Hwnd;
        fn DefWindowProcW(hwnd: Hwnd, msg: u32, wparam: usize, lparam: isize) -> isize;
        fn ShowWindow(hwnd: Hwnd, cmd: i32) -> i32;
        fn SetForegroundWindow(hwnd: Hwnd) -> i32;
        fn DestroyWindow(hwnd: Hwnd) -> i32;
        fn PeekMessageW(msg: *mut Msg, hwnd: Hwnd, min: u32, max: u32, remove: u32) -> i32;
        fn TranslateMessage(msg: *const Msg) -> i32;
        fn DispatchMessageW(msg: *const Msg) -> isize;
        fn GetDC(hwnd: Hwnd) -> Handle;
        fn ReleaseDC(hwnd: Hwnd, dc: Handle) -> i32;
        fn ValidateRect(hwnd: Hwnd, rect: *const Rect) -> i32;
        fn AdjustWindowRect(rect: *mut Rect, style: u32, menu: i32) -> i32;
        fn GetSystemMetrics(index: i32) -> i32;
        fn LoadCursorW(instance: Handle, name: *const u16) -> Handle;
        fn LoadIconW(instance: Handle, name: *const u16) -> Handle;
        fn SetProcessDPIAware() -> i32;
        fn MessageBoxW(hwnd: Hwnd, text: *const u16, caption: *const u16, kind: u32) -> i32;
    }

    #[link(name = "gdi32")]
    unsafe extern "system" {
        fn StretchDIBits(dc: Handle, x: i32, y: i32, w: i32, h: i32, sx: i32, sy: i32, sw: i32, sh: i32, bits: *const c_void, info: *const BitmapInfo, usage: u32, rop: u32) -> i32;
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleW(name: *const u16) -> Handle;
        fn GetConsoleProcessList(list: *mut u32, count: u32) -> u32;
        fn GetConsoleWindow() -> Hwnd;
    }

    #[link(name = "winmm")]
    unsafe extern "system" {
        fn timeBeginPeriod(ms: u32) -> u32;
    }

    const WS_CAPTION: u32 = 0x00C0_0000;
    const WS_SYSMENU: u32 = 0x0008_0000;
    const WS_MINIMIZEBOX: u32 = 0x0002_0000;
    const STYLE: u32 = WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;

    const WM_DESTROY: u32 = 0x0002;
    const WM_KILLFOCUS: u32 = 0x0008;
    const WM_PAINT: u32 = 0x000F;
    const WM_CLOSE: u32 = 0x0010;
    const WM_ERASEBKGND: u32 = 0x0014;
    const WM_KEYDOWN: u32 = 0x0100;
    const WM_KEYUP: u32 = 0x0101;
    const WM_SYSKEYDOWN: u32 = 0x0104;
    const WM_SYSKEYUP: u32 = 0x0105;
    const WM_SYSCHAR: u32 = 0x0106;
    const WM_MOUSEMOVE: u32 = 0x0200;
    const WM_LBUTTONDOWN: u32 = 0x0201;
    const WM_LBUTTONUP: u32 = 0x0202;
    const WM_RBUTTONDOWN: u32 = 0x0204;
    const WM_RBUTTONUP: u32 = 0x0205;

    /// Whether the console was hidden because it belonged to this program alone.
    static HID_CONSOLE: AtomicBool = AtomicBool::new(false);

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain([0]).collect()
    }

    fn id(n: u16) -> *const u16 {
        n as usize as *const u16
    }

    /// The Antares name of a Windows key code.
    fn vk_name(vk: usize) -> Option<&'static str> {
        let name = match vk {
            0x25 => "left",
            0x26 => "up",
            0x27 => "right",
            0x28 => "down",
            0x20 => "space",
            0x0D => "enter",
            0x1B => "escape",
            0x08 => "backspace",
            0x09 => "tab",
            0x10 => "shift",
            0x11 => "ctrl",
            0x12 => "alt",
            0x2E => "delete",
            0x30..=0x39 | 0x41..=0x5A => return key_name(&((vk as u8 as char).to_ascii_lowercase()).to_string()),
            0x70..=0x7B => return key_name(&format!("f{}", vk - 0x6F)),
            _ => return None,
        };
        Some(name)
    }

    extern "system" fn wnd_proc(hwnd: Hwnd, msg: u32, wparam: usize, lparam: isize) -> isize {
        // Messages can arrive while the program is busy with the screen (some
        // are sent during window calls); those just get the usual handling.
        let handled = SCREEN.with(|s| {
            let Ok(mut guard) = s.try_borrow_mut() else { return None };
            let sc = guard.as_mut()?;
            if sc.hwnd != hwnd {
                return None;
            }
            match msg {
                WM_KEYDOWN | WM_SYSKEYDOWN | WM_KEYUP | WM_SYSKEYUP => {
                    if let Some(k) = vk_name(wparam) {
                        if msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN {
                            sc.input.down(k, lparam & (1 << 30) != 0);
                        } else {
                            sc.input.up(k);
                        }
                    }
                    // Alt+F4 and the like still work, but Alt or F10 on its
                    // own would open the window menu and pause the game.
                    (msg == WM_KEYDOWN || msg == WM_KEYUP || wparam == 0x12 || wparam == 0x79).then_some(0)
                }
                WM_MOUSEMOVE | WM_LBUTTONDOWN | WM_LBUTTONUP | WM_RBUTTONDOWN | WM_RBUTTONUP => {
                    let (x, y) = ((lparam & 0xFFFF) as i16 as i64, ((lparam >> 16) & 0xFFFF) as i16 as i64);
                    let scale = sc.scale as i64;
                    sc.input.mouse_x = x.div_euclid(scale);
                    sc.input.mouse_y = y.div_euclid(scale);
                    match msg {
                        WM_LBUTTONDOWN => sc.input.down("click", false),
                        WM_LBUTTONUP => sc.input.up("click"),
                        WM_RBUTTONDOWN => sc.input.down("right click", false),
                        WM_RBUTTONUP => sc.input.up("right click"),
                        _ => {}
                    }
                    Some(0)
                }
                WM_KILLFOCUS => {
                    sc.input.held.clear();
                    None
                }
                WM_PAINT => {
                    present(hwnd, &sc.canvas, sc.scale);
                    unsafe { ValidateRect(hwnd, std::ptr::null()) };
                    Some(0)
                }
                WM_CLOSE => {
                    sc.closed = true;
                    sc.input.held.clear();
                    Some(-1)
                }
                _ => None,
            }
        });
        match (msg, handled) {
            (WM_CLOSE, Some(-1)) => {
                unsafe { DestroyWindow(hwnd) };
                0
            }
            (WM_ERASEBKGND, _) => 1,
            (WM_SYSCHAR, _) => 0, // Alt+letter: no menu to open, so no beep
            (WM_DESTROY, _) => 0,
            (_, Some(r)) => r,
            _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
        }
    }

    /// Opens the window (not shown yet). Gives the window and the scale used.
    pub fn open(w: usize, h: usize, title: &str, scale: Option<usize>) -> Result<(Hwnd, usize), String> {
        static REGISTERED: AtomicBool = AtomicBool::new(false);
        let class = wide("AntaresWindow");
        unsafe {
            SetProcessDPIAware();
            timeBeginPeriod(1);
            let instance = GetModuleHandleW(std::ptr::null());
            if !REGISTERED.load(Ordering::Relaxed) {
                // A program built with --icon shows its icon on the window too.
                let icon = LoadIconW(instance, id(1));
                let wc = WndClassExW {
                    size: std::mem::size_of::<WndClassExW>() as u32,
                    style: 0,
                    wnd_proc,
                    cls_extra: 0,
                    wnd_extra: 0,
                    instance,
                    icon,
                    cursor: LoadCursorW(std::ptr::null_mut(), id(32512)), // IDC_ARROW
                    background: std::ptr::null_mut(),
                    menu_name: std::ptr::null(),
                    class_name: class.as_ptr(),
                    icon_small: icon,
                };
                if RegisterClassExW(&wc) == 0 {
                    return Err("couldn't open a window".into());
                }
                REGISTERED.store(true, Ordering::Relaxed);
            }
            let (sw, sh) = (GetSystemMetrics(0).max(320) as usize, GetSystemMetrics(1).max(240) as usize);
            // as big as fits comfortably on the screen, in whole steps so pixels stay square
            let scale = scale.unwrap_or_else(|| (sw * 3 / 4 / w).min(sh * 3 / 4 / h).clamp(1, 32));
            let mut r = Rect { left: 0, top: 0, right: (w * scale) as i32, bottom: (h * scale) as i32 };
            AdjustWindowRect(&mut r, STYLE, 0);
            let (ow, oh) = (r.right - r.left, r.bottom - r.top);
            let (x, y) = (((sw as i32 - ow) / 2).max(0), ((sh as i32 - oh) / 2).max(0));
            let t = wide(title);
            let hwnd = CreateWindowExW(0, class.as_ptr(), t.as_ptr(), STYLE, x, y, ow, oh, std::ptr::null_mut(), std::ptr::null_mut(), instance, std::ptr::null_mut());
            if hwnd.is_null() {
                return Err("couldn't open a window".into());
            }
            // A game started by double-clicking it gets a console of its own,
            // which only gets in the way: hide it.
            let mut ids = [0u32; 4];
            if GetConsoleProcessList(ids.as_mut_ptr(), 4) == 1 {
                let console = GetConsoleWindow();
                if !console.is_null() {
                    ShowWindow(console, 0);
                    HID_CONSOLE.store(true, Ordering::Relaxed);
                }
            }
            Ok((hwnd, scale))
        }
    }

    pub fn show(hwnd: Hwnd) {
        unsafe {
            ShowWindow(hwnd, 5); // SW_SHOW
            SetForegroundWindow(hwnd);
        }
    }

    pub fn pump() {
        unsafe {
            let mut msg: Msg = std::mem::zeroed();
            while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, 1) != 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    }

    /// Shows the picture in the window, scaled up.
    pub fn present(hwnd: Hwnd, c: &Canvas, scale: usize) {
        let info = BitmapInfo {
            size: 40,
            width: c.w as i32,
            height: -(c.h as i32), // rows from the top
            planes: 1,
            bit_count: 32,
            compression: 0, // BI_RGB
            size_image: 0,
            x_ppm: 0,
            y_ppm: 0,
            clr_used: 0,
            clr_important: 0,
            colors: [0; 3],
        };
        unsafe {
            let dc = GetDC(hwnd);
            if dc.is_null() {
                return;
            }
            StretchDIBits(dc, 0, 0, (c.w * scale) as i32, (c.h * scale) as i32, 0, 0, c.w as i32, c.h as i32, c.px.as_ptr().cast(), &info, 0, 0x00CC_0020);
            ReleaseDC(hwnd, dc);
        }
    }

    pub fn tell_problem(msg: &str) {
        if HID_CONSOLE.load(Ordering::Relaxed) {
            let (text, caption) = (wide(msg), wide("Problem"));
            unsafe { MessageBoxW(std::ptr::null_mut(), text.as_ptr(), caption.as_ptr(), 0x10) }; // MB_ICONERROR
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn font_glyphs_are_5_by_7() {
        for (i, g) in FONT.iter().enumerate() {
            for row in g {
                assert_eq!(row.len(), 5, "glyph {}", (i as u8 + 32) as char);
                assert!(row.bytes().all(|b| b == b'#' || b == b'.'), "glyph {}", (i as u8 + 32) as char);
            }
        }
        assert_eq!(glyph('A')[3], "#####");
        assert_eq!(glyph('é'), glyph('?'));
    }

    #[test]
    fn colours() {
        assert_eq!(colour(&Value::Str("red".into()), "t"), Ok(0xFF0000));
        assert_eq!(colour(&Value::Str("Light Gray".into()), "t"), Ok(0xC0C0C0));
        assert_eq!(colour(&Value::Str("#f80".into()), "t"), Ok(0xFF8800));
        assert_eq!(colour(&Value::Str("#123abc".into()), "t"), Ok(0x123ABC));
        let rgb = Value::List(Rc::new(vec![Value::Num(255.0), Value::Num(136.4), Value::Num(-3.0)]));
        assert_eq!(colour(&rgb, "t"), Ok(0xFF8800));
        assert!(colour(&Value::Str("redd".into()), "t").is_err());
        assert!(colour(&Value::Str("#12".into()), "t").is_err());
        assert!(colour(&Value::Num(3.0), "t").is_err());
    }

    #[test]
    fn shapes_stay_inside_the_picture() {
        let mut c = Canvas::new(10, 8);
        c.rect(-5, -5, 100, 100, 1);
        assert!(c.px.iter().all(|p| *p == 1));
        c.rect(8, 6, 5, 5, 2);
        assert_eq!(c.get(9, 7), Some(2));
        assert_eq!(c.get(7, 7), Some(1));
        c.rect(3, 3, -2, 4, 9); // nothing for a negative width
        assert!(!c.px.contains(&9));
        c.line(-1_000_000_000, 4, 1_000_000_000, 4, 3); // far ends: clipped, so quick
        assert!((0..10).all(|x| c.get(x, 4) == Some(3)));
        c.line(0, 0, 9, 7, 4);
        assert_eq!((c.get(0, 0), c.get(9, 7)), (Some(4), Some(4)));
        c.circle(5, 4, 2.0, 5);
        assert_eq!((c.get(5, 4), c.get(3, 4), c.get(7, 4), c.get(5, 2), c.get(5, 6)), (Some(5), Some(5), Some(5), Some(5), Some(5)));
        assert_ne!(c.get(3, 2), Some(5)); // corners stay outside
        c.circle(1000, 1000, 1e9, 6); // huge circles far away finish quickly enough to test
    }

    #[test]
    fn circle_of_radius_0_is_one_pixel() {
        let mut c = Canvas::new(5, 5);
        c.circle(2, 2, 0.0, 7);
        assert_eq!(c.px.iter().filter(|p| **p == 7).count(), 1);
    }

    #[test]
    fn text_and_sprites() {
        let mut c = Canvas::new(20, 10);
        c.text("I", 0, 0, 1, 1);
        assert_eq!((c.get(1, 0), c.get(2, 3), c.get(0, 0)), (Some(1), Some(1), Some(0)));
        c.text("I", 0, 0, 2, 2); // twice the size
        assert_eq!((c.get(2, 0), c.get(3, 1), c.get(4, 6)), (Some(2), Some(2), Some(2)));
        let mut c = Canvas::new(4, 4);
        c.fill(9);
        c.sprite(&[vec![Some(1), None], vec![None, Some(2)]], 1, 1, 1);
        assert_eq!((c.get(1, 1), c.get(2, 1), c.get(2, 2)), (Some(1), Some(9), Some(2)));
    }

    #[test]
    fn png_is_well_formed() {
        let mut c = Canvas::new(3, 2);
        c.fill(0x112233);
        let png = c.png();
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(u32::from_be_bytes(png[16..20].try_into().unwrap()), 3);
        assert_eq!(&png[png.len() - 8..png.len() - 4], b"IEND");
        assert_eq!(crc32(b"IEND"), 0xAE42_6082);
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }
}
