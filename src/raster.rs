//! Drawing with smooth edges.
//!
//! Every shape is first turned into an outline (one or more lists of
//! points). The outline becomes a `Mask`: how much of each pixel the shape
//! covers, from 0 to 1, worked out exactly from the area inside the outline
//! (the method font renderers use). The mask is then blended into the
//! picture with a `Paint` (a colour, a gradient or a picture), so every
//! shape gets smooth edges, see-through colours and gradients the same way.
//!
//! Pixels are stored premultiplied, as 0xAARRGGBB: red, green and blue are
//! already multiplied by the alpha (how opaque the pixel is). That makes
//! blending simple and keeps see-through edges free of dark fringes. The
//! blending itself is done on whole numbers, two colour parts at a time,
//! which is several times quicker than with fractions.

use std::rc::Rc;

/// A colour with its alpha, premultiplied, each part from 0 to 1.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Rgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Rgba {
    pub const CLEAR: Rgba = Rgba { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

    /// From red, green and blue (0 to 255) and alpha (0 to 1).
    pub fn new(r: f64, g: f64, b: f64, a: f64) -> Rgba {
        let a = a.clamp(0.0, 1.0) as f32;
        let p = |x: f64| (x.clamp(0.0, 255.0) / 255.0) as f32 * a;
        Rgba { r: p(r), g: p(g), b: p(b), a }
    }

    #[inline]
    pub fn from_px(p: u32) -> Rgba {
        const K: f32 = 1.0 / 255.0;
        Rgba { r: ((p >> 16) & 255) as f32 * K, g: ((p >> 8) & 255) as f32 * K, b: (p & 255) as f32 * K, a: (p >> 24) as f32 * K }
    }

    #[inline]
    pub fn to_px(self) -> u32 {
        let q = |x: f32| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
        let a = q(self.a);
        // premultiplied parts never exceed the alpha
        q(self.a) << 24 | q(self.r).min(a) << 16 | q(self.g).min(a) << 8 | q(self.b).min(a)
    }

    #[inline]
    fn lerp(self, o: Rgba, t: f32) -> Rgba {
        Rgba { r: self.r + (o.r - self.r) * t, g: self.g + (o.g - self.g) * t, b: self.b + (o.b - self.b) * t, a: self.a + (o.a - self.a) * t }
    }

    /// Red, green and blue from 0 to 255 and the alpha from 0 to 1, no
    /// longer premultiplied.
    pub fn straight(self) -> (f64, f64, f64, f64) {
        if self.a <= 0.0 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let u = |x: f32| ((x / self.a).clamp(0.0, 1.0) * 255.0).round() as f64;
        (u(self.r), u(self.g), u(self.b), self.a as f64)
    }
}

/// How a new colour combines with what is already there.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Blend {
    /// Paints over, letting through what see-through colours don't cover.
    Normal,
    /// Adds light: glows, fire, lasers.
    Add,
    /// Darkens, like tinted glass or a shadow map.
    Multiply,
    /// Lightens softly.
    Screen,
}

// ---------- quick rounding ----------

// `floor` and `ceil` are calls into the maths library on processors without
// SSE4.1 (which plain x86-64 builds must allow for); turning into a whole
// number is one instruction, so these are much quicker in the inner loops.

#[inline]
fn floor_i(x: f64) -> i64 {
    let i = x as i64;
    if (i as f64) > x { i - 1 } else { i }
}

#[inline]
fn floor32(x: f32) -> f32 {
    let i = x as i32 as f32;
    if i > x { i - 1.0 } else { i }
}

#[inline]
fn ceil32(x: f32) -> f32 {
    let i = x as i32 as f32;
    if i < x { i + 1.0 } else { i }
}

// ---------- blending whole-number pixels ----------

/// All four parts of a pixel times k / 255.
#[inline]
pub fn mul8(c: u32, k: u32) -> u32 {
    let rb = (c & 0x00FF_00FF) * k + 0x0080_0080;
    let rb = ((rb + ((rb >> 8) & 0x00FF_00FF)) >> 8) & 0x00FF_00FF;
    let ag = ((c >> 8) & 0x00FF_00FF) * k + 0x0080_0080;
    let ag = (ag + ((ag >> 8) & 0x00FF_00FF)) & 0xFF00_FF00;
    rb | ag
}

/// Between two pixels: t from 0 (a) to 256 (b).
#[inline]
fn lerp8(a: u32, b: u32, t: u32) -> u32 {
    let it = 256 - t;
    let rb = (((a & 0x00FF_00FF) * it + (b & 0x00FF_00FF) * t) >> 8) & 0x00FF_00FF;
    let ag = (((a >> 8) & 0x00FF_00FF) * it + ((b >> 8) & 0x00FF_00FF) * t) & 0xFF00_FF00;
    rb | ag
}

/// Each part of one pixel times the same part of another, over 255.
#[inline]
fn mul_parts(c: u32, t: u32) -> u32 {
    let mut out = 0;
    for shift in [0, 8, 16, 24] {
        let v = ((c >> shift) & 255) * ((t >> shift) & 255);
        out |= ((v + 128 + ((v + 128) >> 8)) >> 8) << shift;
    }
    out
}

/// `s` painted over `d`.
#[inline]
fn over(d: u32, s: u32) -> u32 {
    s.wrapping_add(mul8(d, 255 - (s >> 24)))
}

/// `s` added to `d`, each part stopping at 255.
#[inline]
fn add_sat(d: u32, s: u32) -> u32 {
    let mut out = 0;
    for shift in [0, 8, 16, 24] {
        out |= (((d >> shift) & 255) + ((s >> shift) & 255)).min(255) << shift;
    }
    out
}

#[inline]
fn blend_px(d: u32, s: u32, mode: Blend) -> u32 {
    match mode {
        Blend::Normal => {
            if s >> 24 == 255 {
                s
            } else {
                over(d, s)
            }
        }
        Blend::Add => add_sat(d, s),
        Blend::Multiply | Blend::Screen => {
            let (d, s) = (Rgba::from_px(d), Rgba::from_px(s));
            let r = if mode == Blend::Multiply {
                let (ks, kd) = (1.0 - s.a, 1.0 - d.a);
                Rgba {
                    r: s.r * d.r + s.r * kd + d.r * ks,
                    g: s.g * d.g + s.g * kd + d.g * ks,
                    b: s.b * d.b + s.b * kd + d.b * ks,
                    a: s.a + d.a * ks,
                }
            } else {
                Rgba { r: s.r + d.r - s.r * d.r, g: s.g + d.g - s.g * d.g, b: s.b + d.b - s.b * d.b, a: s.a + d.a - s.a * d.a }
            };
            r.to_px()
        }
    }
}

// ---------- the picture ----------

/// A picture: `w` by `h` pixels, each premultiplied 0xAARRGGBB, row by row
/// from the top.
#[derive(Clone)]
pub struct Canvas {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u32>,
}

impl Canvas {
    /// A see-through picture.
    pub fn new(w: usize, h: usize) -> Canvas {
        Canvas { w, h, px: vec![0; w * h] }
    }

    /// An opaque black picture (the screen).
    pub fn opaque(w: usize, h: usize) -> Canvas {
        Canvas { w, h, px: vec![0xFF00_0000; w * h] }
    }

    pub fn get(&self, x: i64, y: i64) -> Option<u32> {
        (x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h).then(|| self.px[y as usize * self.w + x as usize])
    }

    /// Blends one pixel.
    #[inline]
    pub fn put(&mut self, x: i64, y: i64, s: u32, mode: Blend) {
        if x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h {
            let i = y as usize * self.w + x as usize;
            self.px[i] = blend_px(self.px[i], s, mode);
        }
    }

    /// The whole picture painted over.
    pub fn fill(&mut self, paint: &Paint, mode: Blend) {
        if let Paint::Solid(c) = paint {
            if mode == Blend::Normal && c >> 24 == 255 {
                self.px.fill(*c);
                return;
            }
        }
        let w = self.w;
        let mut span = vec![0u32; w];
        for (y, row) in self.px.chunks_mut(w.max(1)).enumerate() {
            paint.span(0, y as i64, &mut span);
            if mode == Blend::Normal {
                for (p, &s) in row.iter_mut().zip(&span) {
                    *p = if s >> 24 == 255 { s } else { over(*p, s) };
                }
            } else {
                for (p, &s) in row.iter_mut().zip(&span) {
                    *p = blend_px(*p, s, mode);
                }
            }
        }
    }

    /// The whole picture made one colour, replacing what was there (even
    /// with a see-through colour).
    pub fn clear(&mut self, c: u32) {
        self.px.fill(c);
    }

    /// A one-pixel line from (x0, y0) to (x1, y1), both ends included, with
    /// no smoothing: the classic look.
    pub fn hard_line(&mut self, x0: i64, y0: i64, x1: i64, y1: i64, c: u32, mode: Blend) {
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
            self.put(x0, y0, c, mode);
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

    /// Copies part of a picture ([left, top, right, bottom]) with its
    /// top-left corner at (x, y), blending see-through pixels: the quick way
    /// to draw a picture that isn't turned, scaled or faded.
    pub fn blit(&mut self, src: &Canvas, part: [f64; 4], x: i64, y: i64) {
        let [l, t, r, b] = part.map(|v| v as i64);
        let x0 = x.max(0);
        let x1 = (x + r - l).min(self.w as i64);
        if x1 <= x0 {
            return;
        }
        for sy in t..b {
            let dy = y + sy - t;
            if dy < 0 || dy >= self.h as i64 {
                continue;
            }
            let srow = sy as usize * src.w + (l + x0 - x) as usize;
            let drow = dy as usize * self.w + x0 as usize;
            let n = (x1 - x0) as usize;
            for (d, &p) in self.px[drow..drow + n].iter_mut().zip(&src.px[srow..srow + n]) {
                let a = p >> 24;
                if a == 255 {
                    *d = p;
                } else if p != 0 {
                    *d = over(*d, p);
                }
            }
        }
    }

    /// Blends a mask into the picture with a paint.
    pub fn paint_mask(&mut self, m: &Mask, paint: &Paint, mode: Blend) {
        let mut span = vec![0u32; m.w];
        let (c0, c1) = ((-m.x).max(0) as usize, ((self.w as i64 - m.x).max(0) as usize).min(m.w));
        if c1 <= c0 {
            return;
        }
        for r in 0..m.h {
            let y = m.y + r as i64;
            if y < 0 || y >= self.h as i64 {
                continue;
            }
            let row = &m.cov[r * m.w..(r + 1) * m.w];
            // only the covered part of the row
            let Some(a) = row[c0..c1].iter().position(|v| *v > 0.0) else { continue };
            let b = c1 - row[c0..c1].iter().rev().position(|v| *v > 0.0).unwrap_or(0);
            let a = c0 + a;
            let colours = &mut span[a..b];
            paint.span(m.x + a as i64, y, colours);
            let base = y as usize * self.w;
            let dst = &mut self.px[base + (m.x + a as i64) as usize..base + (m.x + b as i64) as usize];
            for ((d, &cov), &s) in dst.iter_mut().zip(&row[a..b]).zip(colours.iter()) {
                if cov <= 0.0 {
                    continue;
                }
                let s = if cov >= 1.0 { s } else { mul8(s, (cov * 255.0 + 0.5) as u32) };
                *d = blend_px(*d, s, mode);
            }
        }
    }

    /// Softens part of the picture (or all of it): `radius` is about how far
    /// each pixel spreads.
    pub fn blur(&mut self, x: i64, y: i64, w: i64, h: i64, radius: f64) {
        let (x0, y0) = (x.clamp(0, self.w as i64) as usize, y.clamp(0, self.h as i64) as usize);
        let (x1, y1) = (x.saturating_add(w).clamp(0, self.w as i64) as usize, y.saturating_add(h).clamp(0, self.h as i64) as usize);
        if x1 <= x0 || y1 <= y0 || radius <= 0.0 {
            return;
        }
        let (bw, bh) = (x1 - x0, y1 - y0);
        // big blurs are worked out at half or a quarter of the size: the
        // result looks the same, and it is many times quicker
        let f = if radius >= 12.0 { 4 } else if radius >= 5.0 { 2 } else { 1 };
        let (sw, sh) = (bw.div_ceil(f), bh.div_ceil(f));
        let mut planes = vec![vec![0f32; sw * sh]; 4];
        let mut counts = vec![0f32; sw * sh];
        for yy in 0..bh {
            let row = &self.px[(y0 + yy) * self.w + x0..(y0 + yy) * self.w + x1];
            let srow = (yy / f) * sw;
            for (xx, p) in row.iter().enumerate() {
                let i = srow + xx / f;
                planes[0][i] += ((p >> 16) & 255) as f32;
                planes[1][i] += ((p >> 8) & 255) as f32;
                planes[2][i] += (p & 255) as f32;
                planes[3][i] += (p >> 24) as f32;
                counts[i] += 1.0;
            }
        }
        for p in planes.iter_mut() {
            p.iter_mut().zip(&counts).for_each(|(v, n)| *v /= n);
        }
        let r = box_radius(radius / f as f64);
        for p in planes.iter_mut() {
            gaussian(p, sw, sh, r);
        }
        let q = |v: f32| (v + 0.5).clamp(0.0, 255.0) as u32;
        let place = |i: usize, n: usize| {
            let v = ((i as f32 + 0.5) / f as f32 - 0.5).clamp(0.0, (n - 1) as f32);
            let a = v as usize;
            (a, (a + 1).min(n - 1), v - a as f32)
        };
        let cols: Vec<(usize, usize, f32)> = (0..bw).map(|x| place(x, sw)).collect();
        let mut rows = vec![vec![0f32; sw]; 4];
        for yy in 0..bh {
            let (r0, r1, ty) = place(yy, sh);
            for (row, p) in rows.iter_mut().zip(&planes) {
                for (i, r) in row.iter_mut().enumerate() {
                    let (a, b) = (p[r0 * sw + i], p[r1 * sw + i]);
                    *r = a + (b - a) * ty;
                }
            }
            let dst = &mut self.px[(y0 + yy) * self.w + x0..(y0 + yy) * self.w + x1];
            for (d, &(c0, c1, tx)) in dst.iter_mut().zip(&cols) {
                let s = |r: &[f32]| q(r[c0] + (r[c1] - r[c0]) * tx);
                let a = s(&rows[3]);
                *d = a << 24 | s(&rows[0]).min(a) << 16 | s(&rows[1]).min(a) << 8 | s(&rows[2]).min(a);
            }
        }
    }

    /// Bright parts of the picture glow: they are picked out, blurred and
    /// added back, like light bleeding in a camera. The work is shared out
    /// between the processor's cores, a band of rows each.
    pub fn glow(&mut self, amount: f64, radius: f64, threshold: f64) {
        if amount <= 0.0 || self.w == 0 || self.h == 0 {
            return;
        }
        // worked out at a quarter of the size, which is much faster and
        // looks the same once blurred
        let f = 4usize;
        let (w, h) = (self.w, self.h);
        let (sw, sh) = (w.div_ceil(f), h.div_ceil(f));
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 8);
        let mut planes = vec![vec![0f32; sw * sh]; 3];
        let th = (threshold.clamp(0.0, 0.99) * 255.0) as i32;
        let gain = 1.0 / (255 - th) as f32;
        {
            let band = sh.div_ceil(threads).max(1);
            let (a, rest) = planes.split_at_mut(1);
            let (b, c) = rest.split_at_mut(1);
            let px = &self.px;
            std::thread::scope(|s| {
                for (((pr, pg), pb), rows) in a[0].chunks_mut(sw * band).zip(b[0].chunks_mut(sw * band)).zip(c[0].chunks_mut(sw * band)).zip(px.chunks(w * f * band)) {
                    s.spawn(move || {
                        for (ry, row) in rows.chunks(w).enumerate() {
                            let srow = (ry / f) * sw;
                            for (x, p) in row.iter().enumerate() {
                                let (r, g, b) = (((p >> 16) & 255) as i32, ((p >> 8) & 255) as i32, (p & 255) as i32);
                                let lum = (54 * r + 183 * g + 19 * b) >> 8;
                                if lum > th {
                                    let k = (lum - th) as f32 * gain;
                                    let i = srow + x / f;
                                    pr[i] += r as f32 * k;
                                    pg[i] += g as f32 * k;
                                    pb[i] += b as f32 * k;
                                }
                            }
                        }
                    });
                }
            });
        }
        let norm = amount as f32 / (f * f) as f32;
        let r = box_radius(radius / f as f64);
        for p in planes.iter_mut() {
            p.iter_mut().for_each(|v| *v *= norm);
            gaussian(p, sw, sh, r);
        }
        // added back with bilinear smoothing, so no blocks show
        let place = |i: usize, n: usize| {
            let v = ((i as f32 + 0.5) / f as f32 - 0.5).clamp(0.0, (n - 1) as f32);
            let a = v as usize;
            (a, (a + 1).min(n - 1), v - a as f32)
        };
        let cols: Vec<(usize, usize, f32)> = (0..w).map(|x| place(x, sw)).collect();
        let band = h.div_ceil(threads).max(1);
        let (planes, cols) = (&planes, &cols);
        std::thread::scope(|s| {
            for (bi, chunk) in self.px.chunks_mut(w * band).enumerate() {
                s.spawn(move || {
                    let mut rows = vec![vec![0f32; sw]; 3];
                    for (ry, line) in chunk.chunks_mut(w).enumerate() {
                        // the two rows around this one, mixed, then across
                        let (y0, y1, ty) = place(bi * band + ry, sh);
                        for (row, p) in rows.iter_mut().zip(planes) {
                            for (i, r) in row.iter_mut().enumerate() {
                                let (a, b) = (p[y0 * sw + i], p[y1 * sw + i]);
                                *r = a + (b - a) * ty;
                            }
                        }
                        for (px, &(x0, x1, tx)) in line.iter_mut().zip(cols) {
                            let s = |r: &[f32]| (r[x0] + (r[x1] - r[x0]) * tx).clamp(0.0, 255.0) as u32;
                            let add = s(&rows[0]) << 16 | s(&rows[1]) << 8 | s(&rows[2]);
                            if add != 0 {
                                *px = add_sat(*px, add);
                            }
                        }
                    }
                });
            }
        });
    }

    /// Darkens the edges and corners, like an old camera lens.
    pub fn vignette(&mut self, amount: f64) {
        let amt = amount.clamp(0.0, 1.0);
        if amt <= 0.0 || self.w == 0 {
            return;
        }
        // how much each pixel is darkened, worked out once for each size
        thread_local! {
            static TABLE: std::cell::RefCell<(usize, usize, u32, Vec<u8>)> = const { std::cell::RefCell::new((0, 0, 0, vec![])) };
        }
        let key = (amt * 1000.0).round() as u32;
        TABLE.with(|t| {
            let mut t = t.borrow_mut();
            if (t.0, t.1, t.2) != (self.w, self.h, key) {
                let (w, h) = (self.w as f32, self.h as f32);
                let mut k = Vec::with_capacity(self.w * self.h);
                for y in 0..self.h {
                    let dy2 = ((y as f32 + 0.5) / h * 2.0 - 1.0).powi(2);
                    for x in 0..self.w {
                        let dx2 = ((x as f32 + 0.5) / w * 2.0 - 1.0).powi(2);
                        let d = ((dx2 + dy2) * 0.5).sqrt();
                        // nothing near the middle, then smoothly darker
                        let s = ((d - 0.35) / 0.65).clamp(0.0, 1.0);
                        k.push(((1.0 - amt as f32 * s * s * (3.0 - 2.0 * s)) * 255.0 + 0.5) as u8);
                    }
                }
                *t = (self.w, self.h, key, k);
            }
            for (p, &k) in self.px.iter_mut().zip(&t.3) {
                if k < 255 {
                    *p = (mul8(*p, k as u32) & 0x00FF_FFFF) | (*p & 0xFF00_0000);
                }
            }
        });
    }

    /// The picture as a PNG file (with its see-through parts, if it has any).
    pub fn png(&self) -> Vec<u8> {
        let alpha = self.px.iter().any(|p| p >> 24 != 255);
        let bpp = if alpha { 4 } else { 3 };
        let mut raw = Vec::with_capacity(self.h * (1 + bpp * self.w));
        for row in self.px.chunks(self.w.max(1)) {
            raw.push(0); // no filter
            for p in row {
                if alpha {
                    let (r, g, b, a) = Rgba::from_px(*p).straight();
                    raw.extend_from_slice(&[r as u8, g as u8, b as u8, (p >> 24) as u8]);
                    let _ = a;
                } else {
                    raw.extend_from_slice(&[(p >> 16) as u8, (p >> 8) as u8, *p as u8]);
                }
            }
        }
        crate::png::encode(self.w, self.h, alpha, &raw)
    }
}

// ---------- paints ----------

/// What fills a shape. Colours are premultiplied pixels.
pub enum Paint<'a> {
    Solid(u32),
    /// Colours along a line: the colour at a point is found from how far
    /// along (x0, y0) + t (dx, dy) it is.
    Linear { x0: f64, y0: f64, dx: f64, dy: f64, lut: Rc<Vec<u32>> },
    /// Colours outwards from a centre.
    Radial { cx: f64, cy: f64, r: f64, lut: Rc<Vec<u32>> },
    /// A picture, placed by `inv`: from a point in the target to a point in
    /// the picture, (u, v) = (a x + b y + c, d x + e y + f). `clip` is the
    /// part used (left, top, right, bottom), `alpha` from 0 to 255.
    Image { src: &'a Canvas, inv: [f64; 6], clip: [f64; 4], alpha: u32, tint: Option<u32>, nearest: bool },
}

/// 256 colours along a gradient, from colours spaced evenly.
fn gradient_lut(stops: &[Rgba]) -> Rc<Vec<u32>> {
    let n = stops.len();
    let lut = (0..256)
        .map(|i| {
            if n == 1 {
                return stops[0].to_px();
            }
            let t = i as f32 / 255.0 * (n - 1) as f32;
            let k = (t.floor() as usize).min(n - 2);
            stops[k].lerp(stops[k + 1], t - k as f32).to_px()
        })
        .collect();
    Rc::new(lut)
}

impl Paint<'_> {
    pub fn solid(c: Rgba) -> Paint<'static> {
        Paint::Solid(c.to_px())
    }

    pub fn linear(x0: f64, y0: f64, x1: f64, y1: f64, stops: &[Rgba]) -> Paint<'static> {
        let (dx, dy) = (x1 - x0, y1 - y0);
        let len2 = (dx * dx + dy * dy).max(1e-12);
        Paint::Linear { x0, y0, dx: dx / len2, dy: dy / len2, lut: gradient_lut(stops) }
    }

    pub fn radial(cx: f64, cy: f64, r: f64, stops: &[Rgba]) -> Paint<'static> {
        Paint::Radial { cx, cy, r: r.max(1e-9), lut: gradient_lut(stops) }
    }

    /// The colours of `out.len()` pixels in row `y`, starting at column `x`.
    pub fn span(&self, x: i64, y: i64, out: &mut [u32]) {
        let (fx, fy) = (x as f64 + 0.5, y as f64 + 0.5);
        match self {
            Paint::Solid(c) => out.fill(*c),
            Paint::Linear { x0, y0, dx, dy, lut } => {
                let mut t = (fx - x0) * dx + (fy - y0) * dy;
                for o in out.iter_mut() {
                    *o = lut[(t.clamp(0.0, 1.0) * 255.0 + 0.5) as usize];
                    t += dx;
                }
            }
            Paint::Radial { cx, cy, r, lut } => {
                let dy2 = (fy - cy) * (fy - cy);
                let k = 1.0 / r;
                for (i, o) in out.iter_mut().enumerate() {
                    let dx = fx + i as f64 - cx;
                    *o = lut[(((dx * dx + dy2).sqrt() * k).min(1.0) * 255.0 + 0.5) as usize];
                }
            }
            Paint::Image { src, inv, clip, alpha, tint, nearest } => {
                let mut u = inv[0] * fx + inv[1] * fy + inv[2];
                let mut v = inv[3] * fx + inv[4] * fy + inv[5];
                for o in out.iter_mut() {
                    let mut c = sample(src, u, v, clip, *nearest);
                    if let Some(t) = tint {
                        c = mul_parts(c, *t);
                    }
                    if *alpha < 255 {
                        c = mul8(c, *alpha);
                    }
                    *o = c;
                    u += inv[0];
                    v += inv[3];
                }
            }
        }
    }

    /// The colour at a point, as a premultiplied pixel.
    #[inline]
    pub fn at(&self, x: f64, y: f64) -> u32 {
        match self {
            Paint::Solid(c) => *c,
            Paint::Linear { x0, y0, dx, dy, lut } => {
                let t = ((x - x0) * dx + (y - y0) * dy).clamp(0.0, 1.0);
                lut[(t * 255.0 + 0.5) as usize]
            }
            Paint::Radial { cx, cy, r, lut } => {
                let t = (((x - cx) * (x - cx) + (y - cy) * (y - cy)).sqrt() / r).clamp(0.0, 1.0);
                lut[(t * 255.0 + 0.5) as usize]
            }
            Paint::Image { src, inv, clip, alpha, tint, nearest } => {
                let u = inv[0] * x + inv[1] * y + inv[2];
                let v = inv[3] * x + inv[4] * y + inv[5];
                let mut c = sample(src, u, v, clip, *nearest);
                if let Some(t) = tint {
                    c = mul_parts(c, *t);
                }
                if *alpha < 255 {
                    c = mul8(c, *alpha);
                }
                c
            }
        }
    }
}

/// The colour of a picture at (u, v), kept inside `clip` (left, top, right,
/// bottom) so a frame of a sprite sheet doesn't bleed into the next.
#[inline]
fn sample(src: &Canvas, u: f64, v: f64, clip: &[f64; 4], nearest: bool) -> u32 {
    let (l, t, r, b) = (clip[0] as i64, clip[1] as i64, clip[2] as i64 - 1, clip[3] as i64 - 1);
    let px = |x: i64, y: i64| src.px[y.clamp(t, b) as usize * src.w + x.clamp(l, r) as usize];
    if nearest {
        return px(floor_i(u), floor_i(v));
    }
    let (u, v) = (u - 0.5, v - 0.5);
    let (x0, y0) = (floor_i(u), floor_i(v));
    let (tx, ty) = (((u - x0 as f64) * 256.0) as u32, ((v - y0 as f64) * 256.0) as u32);
    let top = lerp8(px(x0, y0), px(x0 + 1, y0), tx);
    let bot = lerp8(px(x0, y0 + 1), px(x0 + 1, y0 + 1), tx);
    lerp8(top, bot, ty)
}

// ---------- masks ----------

/// How much of each pixel in a rectangle a shape covers, from 0 to 1.
pub struct Mask {
    pub x: i64,
    pub y: i64,
    pub w: usize,
    pub h: usize,
    pub cov: Vec<f32>,
}

impl Mask {
    /// The mask's coverage times the paint's alpha: what a shadow is cast by.
    pub fn alpha_of(&self, paint: &Paint) -> Mask {
        let mut cov = self.cov.clone();
        if let Paint::Solid(c) = paint {
            let a = (c >> 24) as f32 / 255.0;
            cov.iter_mut().for_each(|v| *v *= a);
        } else {
            for r in 0..self.h {
                for c in 0..self.w {
                    let i = r * self.w + c;
                    if cov[i] > 0.0 {
                        cov[i] *= (paint.at((self.x + c as i64) as f64 + 0.5, (self.y + r as i64) as f64 + 0.5) >> 24) as f32 / 255.0;
                    }
                }
            }
        }
        Mask { x: self.x, y: self.y, w: self.w, h: self.h, cov }
    }

    /// A soft copy of the mask, moved by (dx, dy): a shadow.
    pub fn shadow(&self, dx: i64, dy: i64, softness: f64) -> Mask {
        let r = if softness > 0.0 { box_radius(softness) } else { 0 };
        let pad = 3 * r;
        let (w, h) = (self.w + 2 * pad, self.h + 2 * pad);
        let mut cov = vec![0f32; w * h];
        for row in 0..self.h {
            let at = (row + pad) * w + pad;
            cov[at..at + self.w].copy_from_slice(&self.cov[row * self.w..(row + 1) * self.w]);
        }
        if r > 0 {
            gaussian(&mut cov, w, h, r);
        }
        Mask { x: self.x - pad as i64 + dx, y: self.y - pad as i64 + dy, w, h, cov }
    }
}

/// The mask of a shape made of outlines, inside a `cw` by `ch` picture.
/// Where outlines overlap, the shape is simply covered, and outlines going
/// the opposite way cut holes (as in a ring). With `smooth` off, each pixel
/// is either covered or not: hard pixel edges.
pub fn fill(outlines: &[Vec<(f64, f64)>], cw: usize, ch: usize, smooth: bool) -> Option<Mask> {
    fill_in(outlines, (0.0, 0.0, cw as f64, ch as f64), smooth)
}

/// The mask of a shape, kept within `clip` (left, top, right, bottom).
pub fn fill_in(outlines: &[Vec<(f64, f64)>], clip: (f64, f64, f64, f64), smooth: bool) -> Option<Mask> {
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for o in outlines {
        for &(x, y) in o {
            if !x.is_finite() || !y.is_finite() {
                return None;
            }
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
    }
    let bx0 = x0.floor().max(clip.0.floor());
    let by0 = y0.floor().max(clip.1.floor());
    let bx1 = x1.ceil().min(clip.2.ceil());
    let by1 = y1.ceil().min(clip.3.ceil());
    if bx1 <= bx0 || by1 <= by0 {
        return None;
    }
    let (w, h) = ((bx1 - bx0) as usize, (by1 - by0) as usize);
    let mut acc = Acc { w, h, stride: w + 2, a: vec![0f32; (w + 2) * h] };
    for o in outlines {
        let n = o.len();
        if n < 2 {
            continue;
        }
        for i in 0..n {
            let (p, q) = (o[i], o[(i + 1) % n]);
            acc.edge((p.0 - bx0, p.1 - by0), (q.0 - bx0, q.1 - by0));
        }
    }
    let mut cov = vec![0f32; w * h];
    for r in 0..h {
        let mut sum = 0f32;
        for c in 0..w {
            sum += acc.a[r * acc.stride + c];
            let v = sum.abs().min(1.0);
            cov[r * w + c] = if smooth {
                if v < 1.0 / 512.0 {
                    0.0
                } else {
                    v
                }
            } else if v >= 0.5 {
                1.0
            } else {
                0.0
            };
        }
    }
    Some(Mask { x: bx0 as i64, y: by0 as i64, w, h, cov })
}

/// Signed areas, added up along each row to give coverage.
struct Acc {
    w: usize,
    h: usize,
    stride: usize,
    a: Vec<f32>,
}

impl Acc {
    /// Adds an edge of an outline: cut to the rows of the mask, and split
    /// where it crosses the left and right sides, so the parts outside can be
    /// moved onto the sides (an edge left of the mask still covers what is
    /// right of it).
    fn edge(&mut self, p: (f64, f64), q: (f64, f64)) {
        let (h, w) = (self.h as f64, self.w as f64);
        if p.1 == q.1 || (p.1 <= 0.0 && q.1 <= 0.0) || (p.1 >= h && q.1 >= h) {
            return;
        }
        // keep the part between y = 0 and y = h
        let (dx, dy) = (q.0 - p.0, q.1 - p.1);
        let ty = |y: f64| (y - p.1) / dy;
        let (mut t0, mut t1) = (ty(0.0), ty(h));
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        let (t0, t1) = (t0.max(0.0), t1.min(1.0));
        if t1 <= t0 {
            return;
        }
        // split where it crosses x = 0 and x = w
        let mut ts = [t0, t1, t0, t0];
        let mut n = 2;
        if dx != 0.0 {
            for side in [0.0, w] {
                let t = (side - p.0) / dx;
                if t > t0 && t < t1 {
                    ts[n] = t;
                    n += 1;
                }
            }
        }
        let ts = &mut ts[..n];
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for k in 0..n - 1 {
            let (ta, tb) = (ts[k], ts[k + 1]);
            let at = |t: f64| ((p.0 + dx * t).clamp(0.0, w) as f32, (p.1 + dy * t).clamp(0.0, h) as f32);
            self.line(at(ta), at(tb));
        }
    }

    /// One edge inside the mask (after `edge` has cut it to fit).
    fn line(&mut self, p0: (f32, f32), p1: (f32, f32)) {
        if (p0.1 - p1.1).abs() <= f32::EPSILON {
            return;
        }
        let (dir, p0, p1) = if p0.1 < p1.1 { (1.0f32, p0, p1) } else { (-1.0, p1, p0) };
        let dxdy = (p1.0 - p0.0) / (p1.1 - p0.1);
        let mut x = p0.0;
        let y0 = p0.1.max(0.0) as usize;
        let yend = (self.h).min(ceil32(p1.1) as usize);
        for y in y0..yend {
            let row = y * self.stride;
            let dy = ((y + 1) as f32).min(p1.1) - (y as f32).max(p0.1);
            if dy <= 0.0 {
                continue;
            }
            let xnext = x + dxdy * dy;
            let d = dy * dir;
            let (xa, xb) = if x < xnext { (x, xnext) } else { (xnext, x) };
            let xa = xa.max(0.0);
            let xb = xb.max(xa);
            let xaf = floor32(xa);
            let xai = xaf as usize;
            let xbc = ceil32(xb);
            let xbi = xbc as usize;
            if xbi <= xai + 1 {
                // within one pixel
                let xmf = 0.5 * (xa + xb) - xaf;
                self.a[row + xai] += d - d * xmf;
                self.a[row + xai + 1] += d * xmf;
            } else {
                let s = 1.0 / (xb - xa);
                let x0f = xa - xaf;
                let a0 = 0.5 * s * (1.0 - x0f) * (1.0 - x0f);
                let x1f = xb - xbc + 1.0;
                let am = 0.5 * s * x1f * x1f;
                self.a[row + xai] += d * a0;
                if xbi == xai + 2 {
                    self.a[row + xai + 1] += d * (1.0 - a0 - am);
                } else {
                    let a1 = s * (1.5 - x0f);
                    self.a[row + xai + 1] += d * (a1 - a0);
                    for xi in xai + 2..xbi - 1 {
                        self.a[row + xi] += d * s;
                    }
                    let a2 = a1 + (xbi - xai - 3) as f32 * s;
                    self.a[row + xbi - 1] += d * (1.0 - a2 - am);
                }
                self.a[row + xbi] += d * am;
            }
            x = xnext;
        }
    }
}

// ---------- outlines of shapes ----------

/// Enough points for a smooth curve of this radius (off by under a
/// twentieth of a pixel).
fn steps(r: f64) -> usize {
    if r <= 0.5 {
        return 8;
    }
    let n = std::f64::consts::PI / (1.0 - 0.05 / r).clamp(-1.0, 1.0).acos();
    (n.ceil() as usize).clamp(8, 1024)
}

/// An ellipse, going clockwise on the screen.
pub fn ellipse(cx: f64, cy: f64, rx: f64, ry: f64) -> Vec<(f64, f64)> {
    let n = steps(rx.abs().max(ry.abs()));
    // the corners pushed out a little, so the shape has the ellipse's area
    let step = std::f64::consts::TAU / n as f64;
    let k = (step / step.sin()).sqrt();
    (0..n)
        .map(|i| {
            let t = i as f64 * step;
            (cx + k * rx * t.cos(), cy + k * ry * t.sin())
        })
        .collect()
}

/// A rectangle with corners rounded by `radius`, going clockwise.
pub fn rect(x: f64, y: f64, w: f64, h: f64, radius: f64) -> Vec<(f64, f64)> {
    let r = radius.clamp(0.0, w.min(h) / 2.0);
    if r <= 0.0 {
        return vec![(x, y), (x + w, y), (x + w, y + h), (x, y + h)];
    }
    let n = (steps(r) / 4).max(2);
    let mut pts = vec![];
    let corners = [(x + w - r, y + r, -90.0f64), (x + w - r, y + h - r, 0.0), (x + r, y + h - r, 90.0), (x + r, y + r, 180.0)];
    for (cx, cy, start) in corners {
        for i in 0..=n {
            let t = (start + 90.0 * i as f64 / n as f64).to_radians();
            pts.push((cx + r * t.cos(), cy + r * t.sin()));
        }
    }
    pts
}

/// Twice the signed area of an outline (positive going clockwise on the
/// screen, where y points down).
fn area2(o: &[(f64, f64)]) -> f64 {
    let n = o.len();
    (0..n).map(|i| o[i].0 * o[(i + 1) % n].1 - o[(i + 1) % n].0 * o[i].1).sum()
}

/// The outline going clockwise.
pub fn clockwise(mut o: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    if area2(&o) < 0.0 {
        o.reverse();
    }
    o
}

/// The outline going anticlockwise (to cut a hole).
pub fn anticlockwise(mut o: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    if area2(&o) > 0.0 {
        o.reverse();
    }
    o
}

/// The outlines of a line `width` wide through `points`, with round ends
/// and joins; `closed` joins the last point back to the first.
pub fn stroke(points: &[(f64, f64)], width: f64, closed: bool) -> Vec<Vec<(f64, f64)>> {
    let hw = width / 2.0;
    let mut out = vec![];
    if hw <= 0.0 || points.is_empty() {
        return out;
    }
    let n = points.len();
    let segs = if closed && n > 2 { n } else { n - 1 };
    for i in 0..segs {
        let (a, b) = (points[i], points[(i + 1) % n]);
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len = (dx * dx + dy * dy).sqrt();
        if len == 0.0 {
            continue;
        }
        let (nx, ny) = (-dy / len * hw, dx / len * hw);
        out.push(clockwise(vec![(a.0 + nx, a.1 + ny), (b.0 + nx, b.1 + ny), (b.0 - nx, b.1 - ny), (a.0 - nx, a.1 - ny)]));
    }
    // round joins and ends (very thin lines don't need them)
    if width > 1.5 || segs == 0 {
        for p in points {
            out.push(ellipse(p.0, p.1, hw, hw));
        }
    }
    out
}

// ---------- blurring ----------

/// The box size that, used three times, blurs by about `radius`.
fn box_radius(radius: f64) -> usize {
    ((radius / 2.0).round() as usize).clamp(1, 512)
}

/// Three box blurs one after another, across then down: very close to a
/// Gaussian blur, and quick at any size.
fn gaussian(p: &mut [f32], w: usize, h: usize, r: usize) {
    if w == 0 || h == 0 {
        return;
    }
    let mut tmp = vec![0f32; w];
    let mut out = vec![0f32; w * h];
    let mut acc = vec![0f32; w];
    let k = 1.0 / (2 * r + 1) as f32;
    for _ in 0..3 {
        for y in 0..h {
            box_line(&mut p[y * w..(y + 1) * w], &mut tmp, r);
        }
        // down the columns, a whole row at a time (much kinder to the cache
        // than going down one column after another)
        acc.fill(0.0);
        for y in 0..r.min(h) {
            acc.iter_mut().zip(&p[y * w..(y + 1) * w]).for_each(|(a, v)| *a += v);
        }
        for y in 0..h {
            if y + r < h {
                acc.iter_mut().zip(&p[(y + r) * w..(y + r + 1) * w]).for_each(|(a, v)| *a += v);
            }
            out[y * w..(y + 1) * w].iter_mut().zip(&acc).for_each(|(o, a)| *o = a * k);
            if y >= r {
                acc.iter_mut().zip(&p[(y - r) * w..(y - r + 1) * w]).for_each(|(a, v)| *a -= v);
            }
        }
        p.copy_from_slice(&out);
    }
}

/// Each value replaced by the average of those within `r` of it (counting
/// what lies past the ends as 0, so shapes fade out at the edges).
fn box_line(v: &mut [f32], tmp: &mut [f32], r: usize) {
    let n = v.len();
    let k = 1.0 / (2 * r + 1) as f32;
    let mut sum = 0f32;
    for x in v.iter().take(r.min(n)) {
        sum += x;
    }
    for i in 0..n {
        if i + r < n {
            sum += v[i + r];
        }
        tmp[i] = sum * k;
        if i >= r {
            sum -= v[i - r];
        }
    }
    v.copy_from_slice(&tmp[..n]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total(m: &Mask) -> f64 {
        m.cov.iter().map(|v| *v as f64).sum()
    }

    #[test]
    fn coverage_is_the_area() {
        // a whole-pixel square covers exactly its pixels
        let m = fill(&[rect(2.0, 3.0, 4.0, 5.0, 0.0)], 20, 20, true).unwrap();
        assert!((total(&m) - 20.0).abs() < 1e-3);
        assert!(m.cov.iter().all(|v| *v == 0.0 || (*v - 1.0).abs() < 1e-5));
        // half a pixel
        let m = fill(&[rect(2.5, 3.0, 1.0, 1.0, 0.0)], 20, 20, true).unwrap();
        assert!((total(&m) - 1.0).abs() < 1e-4);
        // a circle's area
        let m = fill(&[ellipse(10.0, 10.0, 6.0, 6.0)], 20, 20, true).unwrap();
        assert!((total(&m) - std::f64::consts::PI * 36.0).abs() < 0.5);
        // a ring: outer minus inner
        let ring = [clockwise(ellipse(10.0, 10.0, 8.0, 8.0)), anticlockwise(ellipse(10.0, 10.0, 4.0, 4.0))];
        let m = fill(&ring, 20, 20, true).unwrap();
        assert!((total(&m) - std::f64::consts::PI * 48.0).abs() < 1.0);
    }

    #[test]
    fn shapes_off_the_picture() {
        // partly off every side: only the part on the picture counts
        let m = fill(&[rect(-5.0, -5.0, 10.0, 10.0, 0.0)], 20, 20, true).unwrap();
        assert!((total(&m) - 25.0).abs() < 1e-3);
        let m = fill(&[rect(15.0, 15.0, 10.0, 10.0, 0.0)], 20, 20, true).unwrap();
        assert!((total(&m) - 25.0).abs() < 1e-3);
        assert!(fill(&[rect(30.0, 30.0, 4.0, 4.0, 0.0)], 20, 20, true).is_none());
        // huge shapes cost no more than the picture
        let m = fill(&[rect(-1e7, -1e7, 2e7, 2e7, 0.0)], 20, 20, true).unwrap();
        assert!((total(&m) - 400.0).abs() < 1e-2);
    }

    #[test]
    fn blending() {
        let mut c = Canvas::opaque(4, 4);
        c.fill(&Paint::solid(Rgba::new(255.0, 255.0, 255.0, 1.0)), Blend::Normal);
        c.fill(&Paint::solid(Rgba::new(0.0, 0.0, 0.0, 0.5)), Blend::Normal);
        assert_eq!(c.px[0], 0xFF7F_7F7F);
        c.fill(&Paint::solid(Rgba::new(255.0, 0.0, 0.0, 1.0)), Blend::Multiply);
        assert_eq!(c.px[0], 0xFF7F_0000);
        c.fill(&Paint::solid(Rgba::new(0.0, 255.0, 0.0, 1.0)), Blend::Add);
        assert_eq!(c.px[0], 0xFF7F_FF00);
    }

    #[test]
    fn whole_number_blending_matches() {
        // mul8 is exact rounding of c * k / 255
        for c in [0u32, 1, 127, 128, 254, 255] {
            for k in [0u32, 1, 128, 254, 255] {
                assert_eq!(mul8(c, k), (c * k + 127) / 255, "{} {}", c, k);
            }
        }
        // see-through over see-through stays premultiplied
        let d = Rgba::new(10.0, 200.0, 30.0, 0.7).to_px();
        let s = Rgba::new(250.0, 20.0, 90.0, 0.4).to_px();
        let o = over(d, s);
        let a = o >> 24;
        assert!(((o >> 16) & 255) <= a && ((o >> 8) & 255) <= a && (o & 255) <= a);
        assert_eq!(lerp8(0xFF00_00FF, 0xFFFF_0000, 128), 0xFF7F_007F);
    }
}

