//! Graphics for games: a window to draw pictures in, pictures to draw, and
//! the keyboard and mouse.
//!
//! Antares draws every picture itself (raster.rs does the drawing: smooth
//! edges, see-through colours, gradients, pictures, shadows and effects)
//! and the window only shows the finished picture, scaled to fit. Drawing
//! works the same on every system, so pictures can be made and saved
//! anywhere; opening a window is Windows-only for now (`win` below), using
//! only what Windows itself provides, and so are smooth fonts.
//!
//! Everything runs on the program's own thread: the window belongs to it,
//! and its messages are handled whenever the program calls `window_open`,
//! `frame`, `key` or `wait`.

use crate::builtins::{num, s, text, want};
use crate::interp::Interp;
use crate::raster::{self, Blend, Canvas, Mask, Paint, Rgba};
use crate::value::Value;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::time::{Duration, Instant};

type Res = Result<Value, String>;

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

const COLOUR_HELP: &str = "a name like \"red\", a code like \"#ff8800\", or [red, green, blue] from 0 to 255 (with a fourth part from 0 to 1 to make it see-through)";

/// A colour given as a name ("red", "transparent"), "#rgb", "#rgba",
/// "#rrggbb", "#rrggbbaa", or [red, green, blue] from 0 to 255 with an
/// optional fourth part, the alpha, from 0 (see-through) to 1 (solid).
pub fn colour(v: &Value, name: &str) -> Result<Rgba, String> {
    let bad = |what: String| format!("{} needs a colour: {}, not {}", name, COLOUR_HELP, what);
    match v {
        Value::Str(t) => {
            let t = t.trim().to_ascii_lowercase();
            if let Some(hex) = t.strip_prefix('#') {
                let d: Option<Vec<f64>> = hex.chars().map(|c| c.to_digit(16).map(|x| x as f64)).collect();
                return match d.as_deref() {
                    Some([r, g, b]) => Ok(Rgba::new(r * 17.0, g * 17.0, b * 17.0, 1.0)),
                    Some([r, g, b, a]) => Ok(Rgba::new(r * 17.0, g * 17.0, b * 17.0, a * 17.0 / 255.0)),
                    Some(d) if d.len() == 6 || d.len() == 8 => {
                        let byte = |i: usize| d[i] * 16.0 + d[i + 1];
                        let a = if d.len() == 8 { byte(6) / 255.0 } else { 1.0 };
                        Ok(Rgba::new(byte(0), byte(2), byte(4), a))
                    }
                    _ => Err(bad(format!("\"{}\"", t))),
                };
            }
            let key: String = t.chars().filter(|c| !c.is_whitespace()).collect();
            if key == "transparent" || key == "clear" || key == "none" {
                return Ok(Rgba::CLEAR);
            }
            COLOURS
                .iter()
                .find(|(n, _)| *n == key)
                .map(|(_, c)| Rgba::new((c >> 16) as f64, ((c >> 8) & 255) as f64, (c & 255) as f64, 1.0))
                .ok_or_else(|| bad(format!("\"{}\" (a colour I don't know)", t)))
        }
        Value::List(_) | Value::Tensor(_) | Value::Ints(_) => {
            let parts: Vec<f64> = match v {
                Value::List(l) => l.iter().map(|x| x.scalar()).collect::<Option<Vec<f64>>>().ok_or_else(|| bad("a list with something other than numbers".into()))?,
                Value::Tensor(t) => t.vals().clone(),
                Value::Ints(n) => n.iter().map(|x| *x as f64).collect(),
                _ => unreachable!(),
            };
            match parts[..] {
                [r, g, b] => Ok(Rgba::new(r, g, b, 1.0)),
                [r, g, b, a] => {
                    if !(0.0..=1.0).contains(&a) {
                        return Err(format!("{}: the fourth part of a colour, its alpha, goes from 0 (see-through) to 1 (solid), not {}", name, crate::value::fmt_num(a)));
                    }
                    Ok(Rgba::new(r, g, b, a))
                }
                _ => Err(bad(format!("a list of {} numbers", parts.len()))),
            }
        }
        other => Err(bad(format!("a {}", other.kind()))),
    }
}

/// A colour as a list: [red, green, blue], with the alpha as a fourth part
/// if it isn't solid.
fn colour_value(c: Rgba) -> Value {
    let (r, g, b, a) = c.straight();
    let mut parts = vec![Value::Num(r), Value::Num(g), Value::Num(b)];
    if a < 1.0 {
        parts.push(Value::Num((a * 1000.0).round() / 1000.0));
    }
    crate::value::pack_list(parts)
}

fn field<'a>(m: &'a [(Rc<str>, Value)], key: &str) -> Option<&'a Value> {
    m.iter().find(|(k, _)| &**k == key).map(|(_, v)| v)
}

fn numbers(v: &Value, n: usize, name: &str, what: &str) -> Result<Vec<f64>, String> {
    let parts: Option<Vec<f64>> = match v {
        Value::List(l) => l.iter().map(|x| x.scalar()).collect(),
        Value::Tensor(t) => Some(t.vals().clone()),
        Value::Ints(i) => Some(i.iter().map(|x| *x as f64).collect()),
        _ => None,
    };
    match parts {
        Some(p) if p.len() == n => Ok(p),
        _ => Err(format!("{} needs {} as a list of {} numbers, not {}", name, what, n, v.show())),
    }
}

fn stops(v: &Value, name: &str) -> Result<Vec<Rgba>, String> {
    // colours all given as lists of numbers make a table of numbers
    if let Value::Tensor(t) = v {
        if t.shape.len() == 2 && t.shape[0] > 0 && (t.shape[1] == 3 || t.shape[1] == 4) {
            let rows: Vec<Value> = t.vals().chunks(t.shape[1]).map(|c| Value::List(Rc::new(c.iter().map(|x| Value::Num(*x)).collect()))).collect();
            return rows.iter().map(|c| colour(c, name)).collect();
        }
    }
    let Value::List(l) = v else {
        return Err(format!("{} needs a list of colours, like [\"skyblue\", \"navy\"], not {}", name, v.show()));
    };
    if l.is_empty() {
        return Err(format!("{} needs at least one colour", name));
    }
    l.iter().map(|c| colour(c, name)).collect()
}

/// What fills a shape: a colour, or a gradient made by `gradient` or
/// `radial`.
fn paint(v: &Value, name: &str) -> Result<Paint<'static>, String> {
    if let Value::Map(m) = v {
        let kind = field(m, "gradient").map(|k| k.show());
        let colours = field(m, "colours").ok_or_else(|| format!("{} needs a colour or a gradient, not a map without colours", name))?;
        let cs = stops(colours, name)?;
        return match kind.as_deref() {
            Some("linear") => {
                let from = numbers(field(m, "from").unwrap_or(&Value::Nothing), 2, name, "a gradient's start")?;
                let to = numbers(field(m, "to").unwrap_or(&Value::Nothing), 2, name, "a gradient's end")?;
                Ok(Paint::linear(from[0], from[1], to[0], to[1], &cs))
            }
            Some("radial") => {
                let c = numbers(field(m, "centre").unwrap_or(&Value::Nothing), 2, name, "a gradient's centre")?;
                let r = num(field(m, "radius").unwrap_or(&Value::Nothing), name)?;
                Ok(Paint::radial(c[0], c[1], r, &cs))
            }
            _ => Err(format!("{} needs a colour or a gradient made by gradient(...) or radial(...)", name)),
        };
    }
    Ok(Paint::solid(colour(v, name)?))
}

fn opt_paint(a: &[Value], i: usize, name: &str) -> Result<Paint<'static>, String> {
    match a.get(i) {
        Some(v) => paint(v, name),
        None => Ok(Paint::solid(WHITE)),
    }
}

const WHITE: Rgba = Rgba { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
const BLACK: Rgba = Rgba { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };

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

/// Text in the built-in font as a mask: each character 6 by 8 pixels (5 by 7
/// and a gap), times `size`.
fn pixel_text(t: &str, size: usize) -> (usize, usize, Vec<f32>) {
    let lines: Vec<&str> = t.split('\n').collect();
    let w = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0) * 6 * size;
    let h = lines.len() * 8 * size;
    let mut cov = vec![0f32; w * h];
    for (li, line) in lines.iter().enumerate() {
        for (ci, ch) in line.chars().enumerate() {
            for (row, bits) in glyph(ch).iter().enumerate() {
                for (col, b) in bits.bytes().enumerate() {
                    if b == b'#' {
                        for dy in 0..size {
                            let y = li * 8 * size + row * size + dy;
                            let x = ci * 6 * size + col * size;
                            cov[y * w + x..y * w + x + size].fill(1.0);
                        }
                    }
                }
            }
        }
    }
    (w, h, cov)
}

/// A font chosen with `font`: its name, size in pixels, and style.
#[derive(Clone, PartialEq)]
struct FontSpec {
    name: String,
    size: f64,
    bold: bool,
    italic: bool,
}

type TextImage = Rc<(usize, usize, Vec<f32>)>;

thread_local! {
    /// Text already drawn, kept so labels that don't change each frame are
    /// quick.
    static TEXT_CACHE: RefCell<HashMap<(String, u32, bool, bool, String), TextImage>> = RefCell::new(HashMap::new());
}

/// Text in a font, as coverage: width, height, and how much of each pixel
/// the letters cover.
fn font_text(spec: &FontSpec, size: f64, t: &str) -> TextImage {
    let key = (spec.name.clone(), (size * 16.0).round() as u32, spec.bold, spec.italic, t.to_string());
    if let Some(hit) = TEXT_CACHE.with(|c| c.borrow().get(&key).cloned()) {
        return hit;
    }
    #[cfg(windows)]
    let made = win::render_text(&spec.name, size, spec.bold, spec.italic, t);
    #[cfg(not(windows))]
    let made = pixel_text(t, ((size / 8.0).round() as usize).max(1));
    let made = Rc::new(made);
    TEXT_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() > 2000 {
            c.clear();
        }
        c.insert(key, made.clone());
    });
    made
}

// ---------- drawing state ----------

/// How shapes are drawn: set by `blend`, `smooth`, `shadow` and `font`, and
/// kept until changed.
#[derive(Clone)]
struct Style {
    blend: Blend,
    smooth: bool,
    /// Offset across, offset down, softness, colour.
    shadow: Option<(f64, f64, f64, Rgba)>,
    font: Option<FontSpec>,
}

thread_local! {
    static STYLE: RefCell<Style> = const { RefCell::new(Style { blend: Blend::Normal, smooth: true, shadow: None, font: None }) };
    /// Pictures made by `picture`, by number.
    static PICTURES: RefCell<Vec<Canvas>> = const { RefCell::new(vec![]) };
    /// Picture files already read, by file name, so reading one in a loop
    /// costs nothing.
    static LOADED: RefCell<HashMap<String, usize>> = RefCell::new(HashMap::new());
    /// The picture being drawn on (`canvas`), or none for the window.
    static TARGET: Cell<Option<usize>> = const { Cell::new(None) };
    /// How many times each picture has been drawn on, so a copy kept on the
    /// graphics card (for 3D) is sent again when it has changed.
    static VERSIONS: RefCell<Vec<u64>> = const { RefCell::new(vec![]) };
}

fn style() -> Style {
    STYLE.with(|s| s.borrow().clone())
}

/// Runs `f` on the picture being drawn on: the window's, or one chosen
/// with `canvas`.
pub(crate) fn with_target<R>(name: &str, f: impl FnOnce(&mut Canvas) -> R) -> Result<R, String> {
    if let Some(id) = TARGET.with(|t| t.get()) {
        VERSIONS.with(|v| v.borrow_mut()[id] += 1);
        return PICTURES.with(|p| Ok(f(&mut p.borrow_mut()[id])));
    }
    SCREEN.with(|s| match s.borrow_mut().as_mut() {
        Some(sc) => Ok(f(&mut sc.canvas)),
        None => Err(format!("{} draws in a window; open one first with window(width, height), or draw on a picture with canvas(picture)", name)),
    })
}

/// Draws a mask with a paint, and its shadow first if shadows are on.
fn draw_mask(c: &mut Canvas, m: &Mask, p: &Paint, st: &Style) {
    if let Some((dx, dy, soft, col)) = st.shadow {
        let sm = m.alpha_of(p).shadow(dx.round() as i64, dy.round() as i64, soft);
        c.paint_mask(&sm, &Paint::solid(col), Blend::Normal);
    }
    c.paint_mask(m, p, st.blend);
}

/// The area shapes may cover: the picture, and around it as far as a shadow
/// could reach into it.
fn reach(c: &Canvas, st: &Style) -> (f64, f64, f64, f64) {
    let (w, h) = (c.w as f64, c.h as f64);
    match st.shadow {
        Some((dx, dy, soft, _)) => {
            let pad = 3.0 * (soft / 2.0).round().max(1.0) + 1.0;
            ((-dx).min(0.0) - pad, (-dy).min(0.0) - pad, w + dx.max(0.0) + pad, h + dy.max(0.0) + pad)
        }
        None => (0.0, 0.0, w, h),
    }
}

/// Draws a shape made of outlines.
fn shape(name: &str, outlines: Vec<Vec<(f64, f64)>>, p: Paint) -> Res {
    let st = style();
    with_target(name, |c| {
        if let Some(m) = raster::fill_in(&outlines, reach(c, &st), st.smooth) {
            draw_mask(c, &m, &p, &st);
        }
    })?;
    Ok(Value::Nothing)
}

// ---------- the window and its input ----------

/// What the keyboard and mouse are doing, as the window reports it.
#[derive(Default)]
struct Input {
    /// Keys (and "click", "right click") held down now.
    held: Vec<&'static str>,
    /// Keys pressed since the last frame.
    pressed: Vec<&'static str>,
    /// How many of `pressed` the program has had a frame to see. The next
    /// `frame()` forgets those, and keeps any that came in after (read by
    /// `window_open()`, say, while the program was busy drawing).
    seen: usize,
    /// Keys pressed (with repeats), for `key()`.
    typed: VecDeque<&'static str>,
    mouse_x: f64,
    mouse_y: f64,
}

// (only the window code reports keys)
#[cfg_attr(not(windows), allow(dead_code))]
impl Input {
    fn down(&mut self, k: &'static str, repeat: bool) {
        if !repeat {
            if !self.held.contains(&k) {
                self.held.push(k);
            }
            // a key already seen and pressed again counts again
            if !self.pressed[self.seen..].contains(&k) {
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

/// Where the picture goes in the window: how much it is scaled, and how far
/// in from the top-left corner (in fullscreen, where black bars fill the
/// rest).
#[derive(Clone, Copy)]
#[cfg_attr(not(windows), allow(dead_code))]
struct View {
    zoom: f64,
    x: i32,
    y: i32,
    /// The window's inside, when fullscreen.
    full: Option<(i32, i32)>,
}

#[cfg_attr(not(windows), allow(dead_code))]
struct Screen {
    canvas: Canvas,
    view: View,
    /// The scale in a window, kept for leaving fullscreen.
    window_zoom: f64,
    /// Where the window was before going fullscreen.
    saved: [i32; 4],
    input: Input,
    /// When the next frame is due.
    next: Instant,
    /// When the last frame was shown.
    last: Instant,
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
        None => Err(format!("{} works with a window; open one first with window(width, height)", name)),
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

/// A position or size: any ordinary number.
fn coord(v: &Value, name: &str) -> Result<f64, String> {
    let x = num(v, name)?;
    if !x.is_finite() {
        return Err(format!("{} needs an ordinary number, not {}", name, x));
    }
    Ok(x)
}

/// Whole numbers, for the classic look: fractions are rounded down.
fn pos(v: &Value, name: &str) -> Result<i64, String> {
    Ok(coord(v, name)?.floor() as i64)
}

fn opt_size(a: &[Value], i: usize, name: &str) -> Result<usize, String> {
    match a.get(i) {
        None => Ok(1),
        Some(v) => {
            let n = num(v, name)?;
            if n != n.trunc() || !(1.0..=64.0).contains(&n) {
                return Err(format!("{} needs a size that is a whole number from 1 to 64, not {}", name, v.show()));
            }
            Ok(n as usize)
        }
    }
}

fn inputs(a: &[Value], lo: usize, hi: usize, name: &str, what: &str) -> Result<(), String> {
    if a.len() < lo || a.len() > hi {
        return Err(format!("{} takes {}, but was given {} input{}", name, what, a.len(), if a.len() == 1 { "" } else { "s" }));
    }
    Ok(())
}

fn yes_no(v: &Value, name: &str) -> Result<bool, String> {
    match v {
        Value::Bool(b) => Ok(*b),
        _ => Err(format!("{} needs yes or no, not {}", name, v.show())),
    }
}

// ---------- the built-in functions: the window ----------

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
    let w = whole(&a[0], "width", 8192.0)?;
    let h = whole(&a[1], "height", 8192.0)?;
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
    let (hwnd, zoom) = win::open(w, h, title, scale)?;
    let now = Instant::now();
    SCREEN.with(|s| {
        *s.borrow_mut() = Some(Screen {
            canvas: Canvas::opaque(w, h),
            view: View { zoom, x: 0, y: 0, full: None },
            window_zoom: zoom,
            saved: [0; 4],
            input: Input::default(),
            next: now,
            last: now,
            closed: false,
            hwnd,
        })
    });
    win::show(hwnd);
    pump();
    Ok(())
}

#[cfg(not(windows))]
fn open(_: usize, _: usize, _: &str, _: Option<usize>) -> Result<(), String> {
    Err("graphics windows work on Windows for now; macOS and Linux are coming (pictures made with picture(width, height) work everywhere)".into())
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

/// frame(): shows the picture, then waits for the next of 60 frames a
/// second. Gives the seconds since the last frame, for smooth movement.
pub fn b_frame(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 0, "frame")?;
    let closed = with_screen("frame", |sc| sc.closed)?;
    if !closed {
        // a 3D scene described this frame, and not yet drawn
        crate::gfx3d::flush()?;
    }
    if closed {
        // Closing a game's window ends the game, even in a `while yes` loop.
        use std::io::Write;
        let _ = std::io::stdout().flush();
        std::process::exit(0);
    }
    #[cfg(windows)]
    with_screen("frame", |sc| win::present(sc.hwnd, &sc.canvas, sc.view))?;
    let next = with_screen("frame", |sc| {
        // forget the presses the program has had a frame to see, but not
        // ones that came in since
        let seen = sc.input.seen.min(sc.input.pressed.len());
        sc.input.pressed.drain(..seen);
        sc.input.seen = 0;
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
    let dt = with_screen("frame", |sc| {
        // everything pressed so far is the program's to see this frame
        sc.input.seen = sc.input.pressed.len();
        let now = Instant::now();
        let dt = (now - sc.last).as_secs_f64();
        sc.last = now;
        dt.min(0.25)
    })?;
    Ok(Value::Num(dt))
}

/// fullscreen([yes or no]): fills the screen with the window, or goes back
/// to a window; with nothing, says whether it is fullscreen.
pub fn b_fullscreen(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 0, 1, "fullscreen", "yes or no")?;
    let Some(v) = a.first() else {
        return Ok(Value::Bool(with_screen("fullscreen", |sc| sc.view.full.is_some())?));
    };
    let on = yes_no(v, "fullscreen")?;
    let crisp = !style().smooth;
    #[cfg(windows)]
    with_screen("fullscreen", |sc| win::fullscreen(sc, on, crisp))?;
    #[cfg(not(windows))]
    let _ = (on, crisp, with_screen("fullscreen", |_| ())?);
    Ok(Value::Nothing)
}

// ---------- the built-in functions: shapes ----------

/// fill([colour]): paints the whole picture (black if no colour is given).
/// A see-through colour paints over what is there, and "transparent"
/// empties a picture.
pub fn b_fill(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 0, 1, "fill", "a colour")?;
    let p = match a.first() {
        Some(v) => paint(v, "fill")?,
        None => Paint::solid(BLACK),
    };
    let st = style();
    with_target("fill", |c| match p {
        Paint::Solid(0) => c.clear(0),
        _ => c.fill(&p, st.blend),
    })?;
    Ok(Value::Nothing)
}

/// rect(x, y, width, height [, colour [, radius]]): radius rounds the corners.
pub fn b_rect(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 4, 6, "rect", "x, y, a width and a height, and optionally a colour and a corner radius")?;
    let st = style();
    let p = opt_paint(&a, 4, "rect")?;
    let radius = match a.get(5) {
        Some(v) => coord(v, "rect")?,
        None => 0.0,
    };
    let [mut x, mut y, mut w, mut h] = [coord(&a[0], "rect")?, coord(&a[1], "rect")?, coord(&a[2], "rect")?, coord(&a[3], "rect")?];
    if !st.smooth {
        [x, y, w, h] = [x.floor(), y.floor(), w.floor(), h.floor()];
    }
    if w <= 0.0 || h <= 0.0 {
        return Ok(Value::Nothing);
    }
    shape("rect", vec![raster::rect(x, y, w, h, radius)], p)
}

/// circle(x, y, radius [, colour])
pub fn b_circle(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 3, 4, "circle", "x, y and a radius, and optionally a colour")?;
    let (x, y, r) = (coord(&a[0], "circle")?, coord(&a[1], "circle")?, coord(&a[2], "circle")?);
    let p = opt_paint(&a, 3, "circle")?;
    if r < 0.0 {
        return Ok(Value::Nothing);
    }
    if !style().smooth {
        // the classic look: centred on a pixel, 2 r + 1 pixels across
        return shape("circle", vec![pixel_circle(x.floor(), y.floor(), r)], p);
    }
    shape("circle", vec![raster::ellipse(x, y, r, r)], p)
}

/// The outline of a circle drawn the classic way: whole rows of pixels.
fn pixel_circle(cx: f64, cy: f64, r: f64) -> Vec<(f64, f64)> {
    let ri = r.floor().min(1e6) as i64;
    let limit = (r + 0.5) * (r + 0.5);
    let half = |dy: i64| ((limit - (dy * dy) as f64).max(0.0).sqrt().floor() as i64).min(ri) as f64;
    let mut right = vec![];
    let mut left = vec![];
    for dy in -ri..=ri {
        let hw = half(dy);
        let (top, bot) = (cy + dy as f64, cy + dy as f64 + 1.0);
        right.push((cx + hw + 1.0, top));
        right.push((cx + hw + 1.0, bot));
        left.push((cx - hw, top));
        left.push((cx - hw, bot));
    }
    left.reverse();
    right.extend(left);
    right
}

/// ellipse(x, y, radius across, radius down [, colour])
pub fn b_ellipse(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 4, 5, "ellipse", "x, y, a radius across and a radius down, and optionally a colour")?;
    let (x, y, rx, ry) = (coord(&a[0], "ellipse")?, coord(&a[1], "ellipse")?, coord(&a[2], "ellipse")?, coord(&a[3], "ellipse")?);
    let p = opt_paint(&a, 4, "ellipse")?;
    if rx <= 0.0 || ry <= 0.0 {
        return Ok(Value::Nothing);
    }
    shape("ellipse", vec![raster::ellipse(x, y, rx, ry)], p)
}

/// ring(x, y, radius, thickness [, colour]): the outline of a circle.
pub fn b_ring(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 4, 5, "ring", "x, y, a radius and a thickness, and optionally a colour")?;
    let (x, y, r, t) = (coord(&a[0], "ring")?, coord(&a[1], "ring")?, coord(&a[2], "ring")?, coord(&a[3], "ring")?);
    let p = opt_paint(&a, 4, "ring")?;
    if r <= 0.0 || t <= 0.0 {
        return Ok(Value::Nothing);
    }
    // the thickness is centred on the radius
    let (outer, inner) = (r + t / 2.0, (r - t / 2.0).max(0.0));
    let mut outlines = vec![raster::clockwise(raster::ellipse(x, y, outer, outer))];
    if inner > 0.0 {
        outlines.push(raster::anticlockwise(raster::ellipse(x, y, inner, inner)));
    }
    shape("ring", outlines, p)
}

/// The points of a polygon or path: [[x, y], [x, y], ...].
fn points(v: &Value, name: &str) -> Result<Vec<(f64, f64)>, String> {
    let bad = || format!("{} needs a list of points, like [[10, 10], [50, 20], [30, 40]], not {}", name, v.show());
    let rows: Vec<Vec<f64>> = match v {
        Value::List(l) => l.iter().map(|p| numbers(p, 2, name, "each point").map_err(|_| bad())).collect::<Result<_, _>>()?,
        Value::Tensor(t) if t.shape.len() == 2 && t.shape[1] == 2 => t.vals().chunks(2).map(|c| c.to_vec()).collect(),
        _ => return Err(bad()),
    };
    if rows.iter().flatten().any(|x| !x.is_finite()) {
        return Err(format!("{} needs ordinary numbers for its points", name));
    }
    Ok(rows.into_iter().map(|p| (p[0], p[1])).collect())
}

/// polygon(points [, colour]): a filled shape with corners at the points.
pub fn b_polygon(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 1, 2, "polygon", "a list of points, and optionally a colour")?;
    let pts = points(&a[0], "polygon")?;
    let p = opt_paint(&a, 1, "polygon")?;
    if pts.len() < 3 {
        return Ok(Value::Nothing);
    }
    shape("polygon", vec![pts], p)
}

/// Lines run through the middle of pixels, so a line at whole numbers is
/// sharp.
fn centred(pts: &[(f64, f64)]) -> Vec<(f64, f64)> {
    pts.iter().map(|(x, y)| (x + 0.5, y + 0.5)).collect()
}

/// line(x1, y1, x2, y2 [, colour [, width]])
pub fn b_line(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 4, 6, "line", "x1, y1, x2 and y2, and optionally a colour and a width")?;
    let c = [coord(&a[0], "line")?, coord(&a[1], "line")?, coord(&a[2], "line")?, coord(&a[3], "line")?];
    let p = opt_paint(&a, 4, "line")?;
    let width = match a.get(5) {
        Some(v) => coord(v, "line")?,
        None => 1.0,
    };
    let st = style();
    if !st.smooth && width <= 1.0 && st.shadow.is_none() {
        // the classic one-pixel line
        let [x0, y0, x1, y1] = c.map(|v| v.floor() as i64);
        with_target("line", |cv| {
            let col = p.at(x0 as f64 + 0.5, y0 as f64 + 0.5);
            cv.hard_line(x0, y0, x1, y1, col, st.blend);
        })?;
        return Ok(Value::Nothing);
    }
    shape("line", raster::stroke(&centred(&[(c[0], c[1]), (c[2], c[3])]), width, false), p)
}

/// path(points [, colour [, width]]): lines joining the points; it closes
/// into a loop when the last point is the first.
pub fn b_path(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 1, 3, "path", "a list of points, and optionally a colour and a width")?;
    let mut pts = points(&a[0], "path")?;
    let p = opt_paint(&a, 1, "path")?;
    let width = match a.get(2) {
        Some(v) => coord(v, "path")?,
        None => 1.0,
    };
    let closed = pts.len() > 2 && pts.first() == pts.last();
    if closed {
        pts.pop();
    }
    if pts.is_empty() {
        return Ok(Value::Nothing);
    }
    shape("path", raster::stroke(&centred(&pts), width, closed), p)
}

/// pixel(x, y [, colour])
pub fn b_pixel(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 2, 3, "pixel", "x and y, and optionally a colour")?;
    let (x, y) = (pos(&a[0], "pixel")?, pos(&a[1], "pixel")?);
    let p = opt_paint(&a, 2, "pixel")?;
    let mode = style().blend;
    with_target("pixel", |c| c.put(x, y, p.at(x as f64 + 0.5, y as f64 + 0.5), mode))?;
    Ok(Value::Nothing)
}

/// colour_at(x, y): the colour of a pixel, as [red, green, blue] (with a
/// fourth part if it is see-through).
pub fn b_colour_at(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 2, "colour_at")?;
    let (x, y) = (pos(&a[0], "colour_at")?, pos(&a[1], "colour_at")?);
    let p = with_target("colour_at", |c| c.get(x, y))?;
    Ok(match p {
        Some(p) => colour_value(Rgba::from_px(p)),
        None => Value::Nothing,
    })
}

// ---------- the built-in functions: colours and gradients ----------

/// gradient(x1, y1, x2, y2, colours): colours changing smoothly along a line,
/// to use anywhere a colour goes.
pub fn b_gradient(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 5, "gradient")?;
    let c = [coord(&a[0], "gradient")?, coord(&a[1], "gradient")?, coord(&a[2], "gradient")?, coord(&a[3], "gradient")?];
    stops(&a[4], "gradient")?;
    let pair = |x: f64, y: f64| crate::value::pack_list(vec![Value::Num(x), Value::Num(y)]);
    Ok(Value::Map(Rc::new(vec![
        ("gradient".into(), s("linear")),
        ("from".into(), pair(c[0], c[1])),
        ("to".into(), pair(c[2], c[3])),
        ("colours".into(), a[4].clone()),
    ])))
}

/// radial(x, y, radius, colours): colours changing smoothly outwards from a
/// centre: glows, lights, spheres.
pub fn b_radial(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 4, "radial")?;
    let (x, y, r) = (coord(&a[0], "radial")?, coord(&a[1], "radial")?, coord(&a[2], "radial")?);
    stops(&a[3], "radial")?;
    Ok(Value::Map(Rc::new(vec![
        ("gradient".into(), s("radial")),
        ("centre".into(), crate::value::pack_list(vec![Value::Num(x), Value::Num(y)])),
        ("radius".into(), Value::Num(r)),
        ("colours".into(), a[3].clone()),
    ])))
}

/// hsl(hue, saturation, lightness [, alpha]): a colour from its hue (0 to
/// 360 degrees round the colour wheel), saturation and lightness (0 to 1).
pub fn b_hsl(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 3, 4, "hsl", "a hue, a saturation and a lightness, and optionally an alpha")?;
    let h = num(&a[0], "hsl")?.rem_euclid(360.0) / 60.0;
    let (sat, l) = (num(&a[1], "hsl")?.clamp(0.0, 1.0), num(&a[2], "hsl")?.clamp(0.0, 1.0));
    let alpha = match a.get(3) {
        Some(v) => num(v, "hsl")?.clamp(0.0, 1.0),
        None => 1.0,
    };
    let c = (1.0 - (2.0 * l - 1.0).abs()) * sat;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as i64 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    let k = |v: f64| ((v + m) * 255.0).round();
    Ok(colour_value(Rgba::new(k(r), k(g), k(b), alpha)))
}

/// mix(colour1, colour2, amount): a colour between two, from 0 (the first)
/// to 1 (the second).
pub fn b_mix(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 3, "mix")?;
    let (p, q) = (colour(&a[0], "mix")?, colour(&a[1], "mix")?);
    let t = num(&a[2], "mix")?.clamp(0.0, 1.0) as f32;
    let m = Rgba { r: p.r + (q.r - p.r) * t, g: p.g + (q.g - p.g) * t, b: p.b + (q.b - p.b) * t, a: p.a + (q.a - p.a) * t };
    Ok(colour_value(m))
}

// ---------- the built-in functions: how things are drawn ----------

/// blend(mode): how what is drawn next combines with what is there:
/// "normal", "add" (light: glows, fire), "multiply" (darken) or "screen"
/// (lighten softly). blend() goes back to normal.
pub fn b_blend(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 0, 1, "blend", "a mode: \"normal\", \"add\", \"multiply\" or \"screen\"")?;
    let mode = match a.first() {
        None => Blend::Normal,
        Some(v) => match &*text(v, "blend")?.to_ascii_lowercase() {
            "normal" => Blend::Normal,
            "add" | "light" => Blend::Add,
            "multiply" => Blend::Multiply,
            "screen" => Blend::Screen,
            other => return Err(format!("blend knows \"normal\", \"add\", \"multiply\" and \"screen\", not \"{}\"", other)),
        },
    };
    STYLE.with(|s| s.borrow_mut().blend = mode);
    Ok(Value::Nothing)
}

/// smooth(yes or no): smooth edges (the default), or hard pixel edges for
/// the classic pixel-art look.
pub fn b_smooth(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "smooth")?;
    let on = yes_no(&a[0], "smooth")?;
    STYLE.with(|s| s.borrow_mut().smooth = on);
    Ok(Value::Nothing)
}

/// shadow(across, down, softness [, colour]): what is drawn next casts a
/// soft shadow; shadow() turns shadows off.
pub fn b_shadow(_: &mut Interp, a: Vec<Value>) -> Res {
    if a.is_empty() {
        STYLE.with(|s| s.borrow_mut().shadow = None);
        return Ok(Value::Nothing);
    }
    inputs(&a, 3, 4, "shadow", "how far across and down it falls and its softness, and optionally a colour (or nothing, to turn shadows off)")?;
    let (dx, dy, soft) = (coord(&a[0], "shadow")?, coord(&a[1], "shadow")?, coord(&a[2], "shadow")?);
    if !(0.0..=200.0).contains(&soft) {
        return Err(format!("shadow needs a softness from 0 to 200, not {}", a[2].show()));
    }
    let col = match a.get(3) {
        Some(v) => colour(v, "shadow")?,
        None => Rgba::new(0.0, 0.0, 0.0, 0.5),
    };
    STYLE.with(|s| s.borrow_mut().shadow = Some((dx.clamp(-500.0, 500.0), dy.clamp(-500.0, 500.0), soft, col)));
    Ok(Value::Nothing)
}

/// font(name [, size [, style]]): the font labels are written in, like
/// font("Segoe UI", 24, "bold"); font() goes back to the built-in pixel font.
pub fn b_font(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 0, 3, "font", "a font name, and optionally a size and a style (\"bold\", \"italic\" or \"bold italic\")")?;
    let Some(n) = a.first() else {
        STYLE.with(|s| s.borrow_mut().font = None);
        return Ok(Value::Nothing);
    };
    let name = text(n, "font")?.trim().to_string();
    if name.is_empty() || name.eq_ignore_ascii_case("pixel") {
        STYLE.with(|s| s.borrow_mut().font = None);
        return Ok(Value::Nothing);
    }
    let size = match a.get(1) {
        Some(v) => font_size(v, "font")?,
        None => 16.0,
    };
    let st = match a.get(2) {
        Some(v) => text(v, "font")?.to_ascii_lowercase(),
        None => "".into(),
    };
    let words: Vec<&str> = st.split_whitespace().collect();
    if let Some(w) = words.iter().find(|w| !["bold", "italic", "normal", "regular"].contains(w)) {
        return Err(format!("font knows the styles \"bold\", \"italic\" and \"bold italic\", not \"{}\"", w));
    }
    let spec = FontSpec { name, size, bold: words.contains(&"bold"), italic: words.contains(&"italic") };
    STYLE.with(|s| s.borrow_mut().font = Some(spec));
    Ok(Value::Nothing)
}

fn font_size(v: &Value, name: &str) -> Result<f64, String> {
    let n = num(v, name)?;
    if !(1.0..=1000.0).contains(&n) {
        return Err(format!("{} needs a font size from 1 to 1000 pixels, not {}", name, v.show()));
    }
    Ok(n)
}

/// The mask of some text in the current font, its top-left corner at (x, y).
fn text_mask(t: &str, x: f64, y: f64, size: Option<&Value>, name: &str) -> Result<Mask, String> {
    let (w, h, cov) = text_image(t, size, name)?;
    Ok(Mask { x: x.round() as i64, y: y.round() as i64, w, h, cov })
}

fn text_image(t: &str, size: Option<&Value>, name: &str) -> Result<(usize, usize, Vec<f32>), String> {
    match style().font {
        None => {
            let size = match size {
                Some(v) => opt_size(std::slice::from_ref(v), 0, name)?,
                None => 1,
            };
            Ok(pixel_text(t, size))
        }
        Some(spec) => {
            let px = match size {
                Some(v) => font_size(v, name)?,
                None => spec.size,
            };
            let img = font_text(&spec, px, t);
            Ok((img.0, img.1, img.2.clone()))
        }
    }
}

/// label(text, x, y [, colour [, size]]): text, its top-left corner at x, y.
/// In the built-in pixel font, size makes it that many times as big; in a
/// font chosen with `font`, it is the size in pixels.
pub fn b_label(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 3, 5, "label", "some text, x and y, and optionally a colour and a size")?;
    let t = a[0].show();
    let (x, y) = (coord(&a[1], "label")?, coord(&a[2], "label")?);
    let p = opt_paint(&a, 3, "label")?;
    let m = text_mask(&t, x, y, a.get(4), "label")?;
    let st = style();
    with_target("label", |c| draw_mask(c, &m, &p, &st))?;
    Ok(Value::Nothing)
}

/// measure(text [, size]): {width, height}, the size the text takes as a
/// label, to centre it or fit it in a box.
pub fn b_measure(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 1, 2, "measure", "some text, and optionally a size")?;
    let (w, h, _) = text_image(&a[0].show(), a.get(1), "measure")?;
    Ok(Value::Map(Rc::new(vec![("width".into(), Value::Num(w as f64)), ("height".into(), Value::Num(h as f64))])))
}

// ---------- the built-in functions: pictures ----------

/// A picture as a value: {picture, width, height}.
fn picture_value(id: usize, w: usize, h: usize) -> Value {
    Value::Map(Rc::new(vec![("picture".into(), Value::Num(id as f64)), ("width".into(), Value::Num(w as f64)), ("height".into(), Value::Num(h as f64))]))
}

pub(crate) fn picture_id(v: &Value, name: &str) -> Result<usize, String> {
    if let Value::Map(m) = v {
        if let Some(Value::Num(n)) = field(m, "picture") {
            let id = *n as usize;
            if PICTURES.with(|p| id < p.borrow().len()) && *n == id as f64 {
                return Ok(id);
            }
        }
    }
    Err(format!("{} needs a picture made by picture(file) or picture(width, height), not {}", name, v.show()))
}

fn add_picture(c: Canvas) -> Result<usize, String> {
    PICTURES.with(|p| {
        let mut p = p.borrow_mut();
        if p.len() >= 100_000 {
            return Err("there are already 100000 pictures: make pictures once, before the game loop, and draw them many times".to_string());
        }
        p.push(c);
        VERSIONS.with(|v| v.borrow_mut().push(1));
        Ok(p.len() - 1)
    })
}

/// A picture's version (how many times it has been drawn on) and size.
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn picture_version(id: usize) -> (u64, usize, usize) {
    let v = VERSIONS.with(|v| v.borrow()[id]);
    PICTURES.with(|p| (v, p.borrow()[id].w, p.borrow()[id].h))
}

/// A copy of a picture's pixels.
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn picture_pixels(id: usize) -> Vec<u32> {
    PICTURES.with(|p| p.borrow()[id].px.clone())
}

/// The size of the picture being drawn on.
pub(crate) fn target_size(name: &str) -> Result<(usize, usize), String> {
    with_target(name, |c| (c.w, c.h))
}

/// Puts a rendered 3D picture over the one being drawn on (or in place of
/// it, when the sky filled every pixel).
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn paint_rendered(w: usize, h: usize, px: Vec<u32>, opaque: bool) -> Result<(), String> {
    with_target("render", |c| {
        if c.w != w || c.h != h {
            return;
        }
        if opaque {
            c.px = px;
        } else {
            let src = Canvas { w, h, px };
            c.blit(&src, [0.0, 0.0, w as f64, h as f64], 0, 0);
        }
    })
}

/// picture(file): reads a PNG file; picture(width, height): a new,
/// see-through picture to draw on with canvas.
pub fn b_picture(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 1, 2, "picture", "a PNG file's name, or a width and a height")?;
    if a.len() == 2 {
        let whole = |v: &Value| -> Result<usize, String> {
            let n = num(v, "picture")?;
            if n != n.trunc() || !(1.0..=16384.0).contains(&n) {
                return Err(format!("picture needs a width and height that are whole numbers from 1 to 16384, not {}", v.show()));
            }
            Ok(n as usize)
        };
        let (w, h) = (whole(&a[0])?, whole(&a[1])?);
        let id = add_picture(Canvas::new(w, h))?;
        return Ok(picture_value(id, w, h));
    }
    let file = text(&a[0], "picture")?.to_string();
    if let Some(id) = LOADED.with(|l| l.borrow().get(&file).copied()) {
        let (w, h) = PICTURES.with(|p| (p.borrow()[id].w, p.borrow()[id].h));
        return Ok(picture_value(id, w, h));
    }
    let owned;
    let data: &[u8] = match crate::builtins::embedded_file(&file) {
        Some(Some(b)) => b,
        Some(None) => return Err(format!("picture couldn't find \"{}\": no file of that name was embedded", file)),
        None => {
            owned = std::fs::read(&file).map_err(|e| format!("couldn't read the picture \"{}\": {}", file, e))?;
            &owned
        }
    };
    if data.starts_with(&[0xFF, 0xD8]) {
        return Err(format!("\"{}\" is a JPEG file; picture reads PNG files, so save it as PNG first (any image editor can)", file));
    }
    let (w, h, px) = crate::png::decode(data).map_err(|e| format!("couldn't read the picture \"{}\": {}", file, e))?;
    let id = add_picture(Canvas { w, h, px })?;
    LOADED.with(|l| l.borrow_mut().insert(file, id));
    Ok(picture_value(id, w, h))
}

/// canvas(picture): what is drawn next goes on the picture; canvas() goes
/// back to the window.
pub fn b_canvas(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 0, 1, "canvas", "a picture (or nothing, for the window)")?;
    let target = match a.first() {
        Some(v) => Some(picture_id(v, "canvas")?),
        None => None,
    };
    TARGET.with(|t| t.set(target));
    Ok(Value::Nothing)
}

/// The options `draw` knows.
const DRAW_OPTIONS: &[&str] = &["scale", "width", "height", "angle", "alpha", "flip", "part", "tint", "smooth"];

/// draw(picture, x, y [, options]): a picture, its top-left corner at x, y.
/// Options: scale, width, height, angle (degrees, turning clockwise about its
/// middle), alpha (0 to 1), flip (yes, "vertical" or "both"), part ([x, y,
/// width, height], a frame of a sprite sheet), tint (a colour it is
/// multiplied by) and smooth (no keeps pixels sharp when scaled).
pub fn b_draw_picture(a: Vec<Value>) -> Res {
    inputs(&a, 3, 4, "draw", "a picture, x and y, and optionally a map of options")?;
    let id = picture_id(&a[0], "draw")?;
    let (mut x, mut y) = (coord(&a[1], "draw")?, coord(&a[2], "draw")?);
    let st = style();
    let (pw, ph) = PICTURES.with(|p| (p.borrow()[id].w as f64, p.borrow()[id].h as f64));
    let mut part = [0.0, 0.0, pw, ph];
    let (mut scale, mut dw, mut dh, mut angle, mut alpha) = (1.0, None, None, 0.0f64, 1.0f64);
    let (mut flip_x, mut flip_y, mut tint, mut nearest) = (false, false, None, !st.smooth);
    if let Some(o) = a.get(3) {
        let Value::Map(m) = o else {
            return Err(format!("draw's fourth input is a map of options, like {{angle: 45, scale: 2}}, not {}", o.show()));
        };
        for (k, v) in m.iter() {
            match &**k {
                "scale" => scale = coord(v, "draw")?,
                "width" => dw = Some(coord(v, "draw")?),
                "height" => dh = Some(coord(v, "draw")?),
                "angle" => angle = coord(v, "draw")?,
                "alpha" => alpha = num(v, "draw")?.clamp(0.0, 1.0),
                "flip" => match v {
                    Value::Bool(b) => flip_x = *b,
                    _ => match &*text(v, "draw")?.to_ascii_lowercase() {
                        "horizontal" | "across" => flip_x = true,
                        "vertical" | "down" => flip_y = true,
                        "both" => (flip_x, flip_y) = (true, true),
                        other => return Err(format!("draw's flip is yes, \"vertical\" or \"both\", not \"{}\"", other)),
                    },
                },
                "part" => {
                    let p = numbers(v, 4, "draw", "part")?;
                    let (px0, py0) = (p[0].clamp(0.0, pw), p[1].clamp(0.0, ph));
                    part = [px0.floor(), py0.floor(), (px0 + p[2]).clamp(px0, pw).floor(), (py0 + p[3]).clamp(py0, ph).floor()];
                }
                "tint" => tint = Some(colour(v, "draw")?),
                "smooth" => nearest = !yes_no(v, "draw")?,
                other => return Err(format!("draw doesn't know the option \"{}\"; it knows {}", other, DRAW_OPTIONS.join(", "))),
            }
        }
    }
    let (sw, sh) = (part[2] - part[0], part[3] - part[1]);
    let dw = dw.unwrap_or(sw * scale);
    let dh = dh.unwrap_or(sh * scale);
    if sw <= 0.0 || sh <= 0.0 || dw <= 0.0 || dh <= 0.0 || alpha <= 0.0 {
        return Ok(Value::Nothing);
    }
    if !st.smooth {
        (x, y) = (x.floor(), y.floor());
    }
    // Drawing a picture on itself: draw a copy. Otherwise the picture is
    // lent out of the list while it is drawn.
    let target = TARGET.with(|t| t.get());
    let src = if target == Some(id) {
        PICTURES.with(|p| p.borrow()[id].clone())
    } else {
        PICTURES.with(|p| std::mem::replace(&mut p.borrow_mut()[id], Canvas::new(0, 0)))
    };
    let result = (|| {
        let plain = angle == 0.0 && !flip_x && !flip_y && tint.is_none() && alpha >= 1.0 && dw == sw && dh == sh && x == x.floor() && y == y.floor();
        if plain && st.shadow.is_none() && st.blend == Blend::Normal {
            return with_target("draw", |c| c.blit(&src, part, x as i64, y as i64));
        }
        let (cx, cy) = (x + dw / 2.0, y + dh / 2.0);
        let (sin, cos) = angle.to_radians().sin_cos();
        let corner = |lx: f64, ly: f64| (cx + lx * cos - ly * sin, cy + lx * sin + ly * cos);
        let (hw, hh) = (dw / 2.0, dh / 2.0);
        let quad = vec![corner(-hw, -hh), corner(hw, -hh), corner(hw, hh), corner(-hw, hh)];
        // from a point on the target back to the picture
        let (fx, fy) = (if flip_x { -1.0 } else { 1.0 }, if flip_y { -1.0 } else { 1.0 });
        let (kx, ky) = (sw / dw, sh / dh);
        let inv = [
            kx * fx * cos,
            kx * fx * sin,
            part[0] + kx * hw - kx * fx * (cx * cos + cy * sin),
            -ky * fy * sin,
            ky * fy * cos,
            part[1] + ky * hh - ky * fy * (-cx * sin + cy * cos),
        ];
        let p = Paint::Image { src: &src, inv, clip: part, alpha: (alpha * 255.0).round() as u32, tint: tint.map(|t| t.to_px()), nearest };
        with_target("draw", |c| {
            if let Some(m) = raster::fill_in(&[quad], reach(c, &st), true) {
                draw_mask(c, &m, &p, &st);
            }
        })
    })();
    if target != Some(id) {
        PICTURES.with(|p| p.borrow_mut()[id] = src);
    }
    result?;
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
    let (mut x, mut y) = (coord(&a[1], "sprite")?, coord(&a[2], "sprite")?);
    let Value::Map(map) = &a[3] else {
        return Err(format!("sprite needs a map from characters to colours, like {{r: \"red\"}}, not a {}", a[3].kind()));
    };
    let mut colours: Vec<(char, u32)> = vec![];
    for (k, v) in map.iter() {
        let mut cs = k.chars();
        let (Some(ch), None) = (cs.next(), cs.next()) else {
            return Err(format!("sprite's colours are named by one character each, not \"{}\"", k));
        };
        colours.push((ch, colour(v, "sprite")?.to_px()));
    }
    let size = opt_size(&a, 4, "sprite")? as f64;
    let mut lines = vec![];
    for r in rows.iter() {
        let row = text(r, "sprite").map_err(|_| format!("each row of a sprite is text, not a {}", r.kind()))?;
        lines.push(row.chars().map(|ch| colours.iter().find(|(k, _)| *k == ch).map_or(0, |(_, c)| *c)).collect::<Vec<u32>>());
    }
    let (w, h) = (lines.iter().map(|l| l.len()).max().unwrap_or(0), lines.len());
    if w == 0 || h == 0 {
        return Ok(Value::Nothing);
    }
    let mut pic = Canvas::new(w, h);
    for (r, l) in lines.iter().enumerate() {
        pic.px[r * w..r * w + l.len()].copy_from_slice(l);
    }
    let st = style();
    if !st.smooth {
        (x, y) = (x.floor(), y.floor());
    }
    let (dw, dh) = (w as f64 * size, h as f64 * size);
    let quad = vec![(x, y), (x + dw, y), (x + dw, y + dh), (x, y + dh)];
    let inv = [1.0 / size, 0.0, -x / size, 0.0, 1.0 / size, -y / size];
    let p = Paint::Image { src: &pic, inv, clip: [0.0, 0.0, w as f64, h as f64], alpha: 255, tint: None, nearest: true };
    with_target("sprite", |c| {
        if let Some(m) = raster::fill_in(&[quad], reach(c, &st), st.smooth) {
            draw_mask(c, &m, &p, &st);
        }
    })?;
    Ok(Value::Nothing)
}

/// screenshot(file): saves the picture being drawn on (the window's, or one
/// chosen with canvas) as a PNG file.
pub fn b_screenshot(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "screenshot")?;
    let file = text(&a[0], "screenshot")?;
    crate::gfx3d::flush()?;
    let png = with_target("screenshot", |c| c.png())?;
    std::fs::write(&*file, png).map_err(|e| format!("couldn't save the screenshot to \"{}\": {}", file, e))?;
    Ok(s(&*file))
}

// ---------- the built-in functions: effects ----------

/// blur(radius [, x, y, width, height]): softens the picture, or part of it
/// (frosted glass, depth of field).
pub fn b_blur(_: &mut Interp, a: Vec<Value>) -> Res {
    if a.len() != 1 && a.len() != 5 {
        return Err(format!("blur takes a radius, and optionally x, y, a width and a height, but was given {} inputs", a.len()));
    }
    let r = coord(&a[0], "blur")?;
    if !(0.0..=500.0).contains(&r) {
        return Err(format!("blur needs a radius from 0 to 500, not {}", a[0].show()));
    }
    let area = if a.len() == 5 { Some([pos(&a[1], "blur")?, pos(&a[2], "blur")?, pos(&a[3], "blur")?, pos(&a[4], "blur")?]) } else { None };
    with_target("blur", |c| {
        let [x, y, w, h] = area.unwrap_or([0, 0, c.w as i64, c.h as i64]);
        c.blur(x, y, w, h, r)
    })?;
    Ok(Value::Nothing)
}

/// glow(amount [, radius [, threshold]]): bright parts glow, as light does
/// in a camera.
pub fn b_glow(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 1, 3, "glow", "an amount, and optionally a radius and a threshold")?;
    let amount = num(&a[0], "glow")?.clamp(0.0, 10.0);
    let radius = match a.get(1) {
        Some(v) => coord(v, "glow")?.clamp(1.0, 500.0),
        None => 16.0,
    };
    let threshold = match a.get(2) {
        Some(v) => num(v, "glow")?.clamp(0.0, 1.0),
        None => 0.6,
    };
    with_target("glow", |c| c.glow(amount, radius, threshold))?;
    Ok(Value::Nothing)
}

/// vignette(amount): darkens the edges and corners, from 0 (not at all) to 1.
pub fn b_vignette(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "vignette")?;
    let amount = num(&a[0], "vignette")?;
    with_target("vignette", |c| c.vignette(amount))?;
    Ok(Value::Nothing)
}

// ---------- the built-in functions: input ----------

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
        ("x".into(), Value::Num(x.floor())),
        ("y".into(), Value::Num(y.floor())),
        ("left".into(), Value::Bool(l)),
        ("right".into(), Value::Bool(r)),
    ])))
}

// ---------- Windows ----------

#[cfg(windows)]
mod win {
    use super::{key_name, Canvas, Screen, View, SCREEN};
    use std::cell::RefCell;
    use std::collections::HashMap;
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

    #[repr(C)]
    struct Size {
        cx: i32,
        cy: i32,
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
        fn FillRect(dc: Handle, rect: *const Rect, brush: Handle) -> i32;
        fn SetWindowLongPtrW(hwnd: Hwnd, index: i32, value: isize) -> isize;
        fn SetWindowPos(hwnd: Hwnd, after: Hwnd, x: i32, y: i32, cx: i32, cy: i32, flags: u32) -> i32;
        fn GetWindowRect(hwnd: Hwnd, rect: *mut Rect) -> i32;
        fn InvalidateRect(hwnd: Hwnd, rect: *const Rect, erase: i32) -> i32;
    }

    #[link(name = "gdi32")]
    unsafe extern "system" {
        fn StretchDIBits(dc: Handle, x: i32, y: i32, w: i32, h: i32, sx: i32, sy: i32, sw: i32, sh: i32, bits: *const c_void, info: *const BitmapInfo, usage: u32, rop: u32) -> i32;
        fn SetStretchBltMode(dc: Handle, mode: i32) -> i32;
        fn SetBrushOrgEx(dc: Handle, x: i32, y: i32, old: *mut c_void) -> i32;
        fn GetStockObject(kind: i32) -> Handle;
        fn CreateCompatibleDC(dc: Handle) -> Handle;
        fn DeleteDC(dc: Handle) -> i32;
        fn CreateDIBSection(dc: Handle, info: *const BitmapInfo, usage: u32, bits: *mut *mut c_void, section: Handle, offset: u32) -> Handle;
        fn SelectObject(dc: Handle, obj: Handle) -> Handle;
        fn DeleteObject(obj: Handle) -> i32;
        fn CreateFontW(h: i32, w: i32, esc: i32, orient: i32, weight: i32, italic: u32, underline: u32, strike: u32, charset: u32, out_prec: u32, clip_prec: u32, quality: u32, pitch: u32, face: *const u16) -> Handle;
        fn SetTextColor(dc: Handle, colour: u32) -> u32;
        fn SetBkMode(dc: Handle, mode: i32) -> i32;
        fn TextOutW(dc: Handle, x: i32, y: i32, text: *const u16, n: i32) -> i32;
        fn GetTextExtentPoint32W(dc: Handle, text: *const u16, n: i32, size: *mut Size) -> i32;
        fn GdiFlush() -> i32;
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
    const WS_POPUP: u32 = 0x8000_0000;
    const WS_VISIBLE: u32 = 0x1000_0000;
    const STYLE: u32 = WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    const GWL_STYLE: i32 = -16;
    const SWP_FRAMECHANGED: u32 = 0x0020;
    const SWP_SHOWWINDOW: u32 = 0x0040;

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
                    let (x, y) = ((lparam & 0xFFFF) as i16 as f64, ((lparam >> 16) & 0xFFFF) as i16 as f64);
                    let v = sc.view;
                    sc.input.mouse_x = (x - v.x as f64) / v.zoom;
                    sc.input.mouse_y = (y - v.y as f64) / v.zoom;
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
                    present(hwnd, &sc.canvas, sc.view);
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

    fn screen_size() -> (usize, usize) {
        unsafe { (GetSystemMetrics(0).max(320) as usize, GetSystemMetrics(1).max(240) as usize) }
    }

    /// Opens the window (not shown yet). Gives the window and its scale.
    pub fn open(w: usize, h: usize, title: &str, scale: Option<usize>) -> Result<(Hwnd, f64), String> {
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
            let (sw, sh) = screen_size();
            // As big as fits comfortably on the screen, in whole steps so
            // pixels stay square; a picture too big for the screen is shrunk
            // smoothly to fit.
            let zoom = match scale {
                Some(s) => s as f64,
                None => {
                    let whole = (sw * 3 / 4 / w).min(sh * 3 / 4 / h).clamp(1, 32) as f64;
                    let fit = (sw as f64 * 0.9 / w as f64).min(sh as f64 * 0.85 / h as f64);
                    if fit < 1.0 { fit } else { whole }
                }
            };
            let (cw, ch) = ((w as f64 * zoom).round() as i32, (h as f64 * zoom).round() as i32);
            let mut r = Rect { left: 0, top: 0, right: cw, bottom: ch };
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
            Ok((hwnd, zoom))
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

    /// Fills the screen with the window, or puts it back as it was.
    pub fn fullscreen(sc: &mut Screen, on: bool, crisp: bool) {
        if on == sc.view.full.is_some() {
            return;
        }
        let hwnd = sc.hwnd;
        unsafe {
            if on {
                let mut r = Rect { left: 0, top: 0, right: 0, bottom: 0 };
                GetWindowRect(hwnd, &mut r);
                sc.saved = [r.left, r.top, r.right - r.left, r.bottom - r.top];
                let (sw, sh) = screen_size();
                let (w, h) = (sc.canvas.w as f64, sc.canvas.h as f64);
                let mut zoom = (sw as f64 / w).min(sh as f64 / h);
                // pixel art stays sharp in whole steps
                if crisp && zoom >= 1.0 {
                    zoom = zoom.floor();
                }
                let (dw, dh) = ((w * zoom).round() as i32, (h * zoom).round() as i32);
                sc.view = View { zoom, x: (sw as i32 - dw) / 2, y: (sh as i32 - dh) / 2, full: Some((sw as i32, sh as i32)) };
                SetWindowLongPtrW(hwnd, GWL_STYLE, (WS_POPUP | WS_VISIBLE) as isize);
                SetWindowPos(hwnd, std::ptr::null_mut(), 0, 0, sw as i32, sh as i32, SWP_FRAMECHANGED | SWP_SHOWWINDOW);
            } else {
                sc.view = View { zoom: sc.window_zoom, x: 0, y: 0, full: None };
                SetWindowLongPtrW(hwnd, GWL_STYLE, (STYLE | WS_VISIBLE) as isize);
                let [x, y, w, h] = sc.saved;
                SetWindowPos(hwnd, std::ptr::null_mut(), x, y, w, h, SWP_FRAMECHANGED | SWP_SHOWWINDOW);
            }
            InvalidateRect(hwnd, std::ptr::null(), 0);
        }
    }

    /// Shows the picture in the window, scaled.
    pub fn present(hwnd: Hwnd, c: &Canvas, v: View) {
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
        let (dw, dh) = ((c.w as f64 * v.zoom).round() as i32, (c.h as f64 * v.zoom).round() as i32);
        unsafe {
            let dc = GetDC(hwnd);
            if dc.is_null() {
                return;
            }
            if let Some((fw, fh)) = v.full {
                // black bars around the picture
                let black = GetStockObject(4); // BLACK_BRUSH
                for r in [
                    Rect { left: 0, top: 0, right: fw, bottom: v.y },
                    Rect { left: 0, top: v.y + dh, right: fw, bottom: fh },
                    Rect { left: 0, top: v.y, right: v.x, bottom: v.y + dh },
                    Rect { left: v.x + dw, top: v.y, right: fw, bottom: v.y + dh },
                ] {
                    if r.right > r.left && r.bottom > r.top {
                        FillRect(dc, &r, black);
                    }
                }
            }
            // whole steps copy pixels; other sizes are smoothed
            if v.zoom == v.zoom.floor() {
                SetStretchBltMode(dc, 3); // COLORONCOLOR
            } else {
                SetStretchBltMode(dc, 4); // HALFTONE
                SetBrushOrgEx(dc, 0, 0, std::ptr::null_mut());
            }
            StretchDIBits(dc, v.x, v.y, dw, dh, 0, 0, c.w as i32, c.h as i32, c.px.as_ptr().cast(), &info, 0, 0x00CC_0020);
            ReleaseDC(hwnd, dc);
        }
    }

    pub fn tell_problem(msg: &str) {
        if HID_CONSOLE.load(Ordering::Relaxed) {
            let (text, caption) = (wide(msg), wide("Problem"));
            unsafe { MessageBoxW(std::ptr::null_mut(), text.as_ptr(), caption.as_ptr(), 0x10) }; // MB_ICONERROR
        }
    }

    thread_local! {
        /// Fonts made so far, kept for reuse.
        static FONTS: RefCell<HashMap<(String, i32, bool, bool), Handle>> = RefCell::new(HashMap::new());
    }

    fn font(name: &str, size: f64, bold: bool, italic: bool) -> Handle {
        let key = (name.to_string(), (size * 4.0).round() as i32, bold, italic);
        if let Some(f) = FONTS.with(|f| f.borrow().get(&key).copied()) {
            return f;
        }
        let face = wide(name);
        // a negative height asks for the size of the letters themselves
        let f = unsafe { CreateFontW(-(size.round() as i32).max(1), 0, 0, 0, if bold { 700 } else { 400 }, italic as u32, 0, 0, 1, 0, 0, 4, 0, face.as_ptr()) };
        FONTS.with(|fs| fs.borrow_mut().insert(key, f));
        f
    }

    /// Text in a Windows font, smoothed: its width, height, and how much of
    /// each pixel the letters cover.
    pub fn render_text(name: &str, size: f64, bold: bool, italic: bool, t: &str) -> (usize, usize, Vec<f32>) {
        let lines: Vec<Vec<u16>> = t.split('\n').map(|l| l.encode_utf16().collect()).collect();
        unsafe {
            let dc = CreateCompatibleDC(std::ptr::null_mut());
            if dc.is_null() {
                return (0, 0, vec![]);
            }
            SelectObject(dc, font(name, size, bold, italic));
            let mut sz = Size { cx: 0, cy: 0 };
            let probe: Vec<u16> = "Ag".encode_utf16().collect();
            GetTextExtentPoint32W(dc, probe.as_ptr(), 2, &mut sz);
            let line_h = sz.cy.max(1);
            let mut widths = vec![];
            for l in &lines {
                GetTextExtentPoint32W(dc, l.as_ptr(), l.len() as i32, &mut sz);
                widths.push(if l.is_empty() { 0 } else { sz.cx });
            }
            // room for slanted letters that lean past their box
            let extra = if italic { (size / 3.0).ceil() as i32 } else { (size / 10.0).ceil() as i32 };
            let w = widths.iter().copied().max().unwrap_or(0) + extra;
            let h = line_h * lines.len() as i32;
            if w <= extra || h <= 0 {
                DeleteDC(dc);
                return (0, h.max(0) as usize, vec![]);
            }
            let info = BitmapInfo {
                size: 40,
                width: w,
                height: -h,
                planes: 1,
                bit_count: 32,
                compression: 0,
                size_image: 0,
                x_ppm: 0,
                y_ppm: 0,
                clr_used: 0,
                clr_important: 0,
                colors: [0; 3],
            };
            let mut bits: *mut c_void = std::ptr::null_mut();
            let bmp = CreateDIBSection(dc, &info, 0, &mut bits, std::ptr::null_mut(), 0);
            if bmp.is_null() || bits.is_null() {
                DeleteDC(dc);
                return (0, 0, vec![]);
            }
            let old = SelectObject(dc, bmp);
            SetTextColor(dc, 0x00FF_FFFF);
            SetBkMode(dc, 1); // TRANSPARENT
            for (i, l) in lines.iter().enumerate() {
                if !l.is_empty() {
                    TextOutW(dc, 0, i as i32 * line_h, l.as_ptr(), l.len() as i32);
                }
            }
            GdiFlush();
            let n = (w * h) as usize;
            let px = std::slice::from_raw_parts(bits as *const u32, n);
            let cov = px.iter().map(|p| ((p >> 8) & 255) as f32 / 255.0).collect();
            SelectObject(dc, old);
            DeleteObject(bmp);
            DeleteDC(dc);
            (w as usize, h as usize, cov)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(c: Rgba) -> u32 {
        c.to_px() & 0xFF_FFFF
    }

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
        let c = |t: &str| colour(&Value::Str(t.into()), "t");
        assert_eq!(c("red").map(rgb), Ok(0xFF0000));
        assert_eq!(c("Light Gray").map(rgb), Ok(0xC0C0C0));
        assert_eq!(c("#f80").map(rgb), Ok(0xFF8800));
        assert_eq!(c("#123abc").map(rgb), Ok(0x123ABC));
        assert_eq!(c("#ff000080").map(|x| x.to_px()), Ok(0x8080_0000));
        assert_eq!(c("transparent"), Ok(Rgba::CLEAR));
        let list = |v: Vec<f64>| Value::List(Rc::new(v.into_iter().map(Value::Num).collect()));
        assert_eq!(colour(&list(vec![255.0, 136.4, -3.0]), "t").map(rgb), Ok(0xFF8800));
        assert_eq!(colour(&list(vec![255.0, 0.0, 0.0, 0.5]), "t").map(|x| x.to_px() >> 24), Ok(128));
        assert!(colour(&list(vec![255.0, 0.0, 0.0, 128.0]), "t").unwrap_err().contains("alpha"));
        assert!(c("redd").is_err());
        assert!(c("#12").is_err());
        assert!(colour(&Value::Num(3.0), "t").is_err());
    }

    #[test]
    fn pixel_text_masks() {
        let (w, h, cov) = pixel_text("I", 1);
        assert_eq!((w, h), (6, 8));
        assert_eq!((cov[1], cov[3 * 6 + 2], cov[0]), (1.0, 1.0, 0.0));
        let (w, h, _) = pixel_text("ab\ncde", 2);
        assert_eq!((w, h), (36, 32));
    }

    #[test]
    fn classic_circles() {
        // radius 2: 5 pixels across, corners left out
        let m = raster::fill(&[pixel_circle(5.0, 4.0, 2.0)], 10, 8, true).unwrap();
        let at = |x: i64, y: i64| m.cov[((y - m.y) * m.w as i64 + x - m.x) as usize];
        assert_eq!((at(5, 4), at(3, 4), at(7, 4), at(5, 2), at(5, 6)), (1.0, 1.0, 1.0, 1.0, 1.0));
        assert_eq!(at(3, 2), 0.0);
        // radius 0: one pixel
        let m = raster::fill(&[pixel_circle(2.0, 2.0, 0.0)], 5, 5, true).unwrap();
        assert_eq!(m.cov.iter().filter(|v| **v == 1.0).count(), 1);
    }
}
