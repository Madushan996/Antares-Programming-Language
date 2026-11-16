//! PNG files: reading them (for `picture`) and writing them (for
//! `screenshot`), with no libraries. Reading needs `inflate`, the
//! decompression PNG uses (the same as zip and gzip).

// ---------- inflate ----------

struct Bits<'a> {
    d: &'a [u8],
    pos: usize,
    buf: u64,
    cnt: u32,
}

const CUT: &str = "the file ends too soon";

impl Bits<'_> {
    fn fill(&mut self) {
        while self.cnt <= 56 && self.pos < self.d.len() {
            self.buf |= (self.d[self.pos] as u64) << self.cnt;
            self.pos += 1;
            self.cnt += 8;
        }
    }

    fn take(&mut self, n: u32) -> Result<u32, String> {
        if n == 0 {
            return Ok(0);
        }
        if self.cnt < n {
            self.fill();
            if self.cnt < n {
                return Err(CUT.into());
            }
        }
        let v = (self.buf & ((1u64 << n) - 1)) as u32;
        self.buf >>= n;
        self.cnt -= n;
        Ok(v)
    }

    /// Skips to the next whole byte.
    fn align(&mut self) {
        let k = self.cnt % 8;
        self.buf >>= k;
        self.cnt -= k;
    }
}

/// A Huffman code as a table: the next `bits` bits (read backwards, as
/// inflate stores them) give the symbol and how many bits it used.
struct Code {
    bits: u32,
    table: Vec<(u16, u8)>,
}

impl Code {
    fn new(lengths: &[u8]) -> Result<Code, String> {
        let max = lengths.iter().copied().max().unwrap_or(0) as u32;
        if max == 0 {
            // a code with no symbols: decoding with it is a problem
            return Ok(Code { bits: 1, table: vec![(0, 0); 2] });
        }
        let mut count = [0u32; 16];
        for &l in lengths {
            count[l as usize] += 1;
        }
        count[0] = 0;
        let mut next = [0u32; 16];
        let mut code = 0;
        for l in 1..16 {
            code = (code + count[l - 1]) << 1;
            next[l] = code;
        }
        let size = 1usize << max;
        let mut table = vec![(0u16, 0u8); size];
        for (sym, &l) in lengths.iter().enumerate() {
            if l == 0 {
                continue;
            }
            let c = next[l as usize];
            next[l as usize] += 1;
            if c >= 1 << l {
                return Err("its compressed data is damaged (a bad code)".into());
            }
            let rev = c.reverse_bits() >> (32 - l as u32);
            let mut i = rev as usize;
            while i < size {
                table[i] = (sym as u16, l);
                i += 1 << l;
            }
        }
        Ok(Code { bits: max, table })
    }

    fn read(&self, b: &mut Bits) -> Result<u16, String> {
        if b.cnt < self.bits {
            b.fill();
        }
        let (sym, len) = self.table[(b.buf & ((1u64 << self.bits) - 1)) as usize];
        if len == 0 {
            return Err("its compressed data is damaged".into());
        }
        if (len as u32) > b.cnt {
            return Err(CUT.into());
        }
        b.buf >>= len;
        b.cnt -= len as u32;
        Ok(sym)
    }
}

const LEN_BASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE: [u16; 30] =
    [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

/// Decompresses raw deflate data; `limit` stops a damaged or hostile file
/// from asking for endless memory.
pub fn inflate(data: &[u8], limit: usize) -> Result<Vec<u8>, String> {
    let mut b = Bits { d: data, pos: 0, buf: 0, cnt: 0 };
    let mut out: Vec<u8> = Vec::with_capacity(limit.min(1 << 24));
    loop {
        let last = b.take(1)?;
        match b.take(2)? {
            0 => {
                b.align();
                let len = b.take(16)?;
                let nlen = b.take(16)?;
                if len != !nlen & 0xFFFF {
                    return Err("its compressed data is damaged".into());
                }
                for _ in 0..len {
                    out.push(b.take(8)? as u8);
                }
            }
            kind @ (1 | 2) => {
                let (lit, dist) = if kind == 1 {
                    let mut l = [8u8; 288];
                    l[144..256].fill(9);
                    l[256..280].fill(7);
                    (Code::new(&l)?, Code::new(&[5u8; 30])?)
                } else {
                    let hlit = b.take(5)? as usize + 257;
                    let hdist = b.take(5)? as usize + 1;
                    let hclen = b.take(4)? as usize + 4;
                    const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
                    let mut cl = [0u8; 19];
                    for &o in ORDER.iter().take(hclen) {
                        cl[o] = b.take(3)? as u8;
                    }
                    let clc = Code::new(&cl)?;
                    let mut lens = vec![];
                    while lens.len() < hlit + hdist {
                        let sym = clc.read(&mut b)?;
                        match sym {
                            0..=15 => lens.push(sym as u8),
                            16 => {
                                let prev = *lens.last().ok_or("its compressed data is damaged")?;
                                for _ in 0..3 + b.take(2)? {
                                    lens.push(prev);
                                }
                            }
                            17 => lens.extend(std::iter::repeat_n(0, 3 + b.take(3)? as usize)),
                            _ => lens.extend(std::iter::repeat_n(0, 11 + b.take(7)? as usize)),
                        }
                    }
                    if lens.len() > hlit + hdist {
                        return Err("its compressed data is damaged".into());
                    }
                    (Code::new(&lens[..hlit])?, Code::new(&lens[hlit..])?)
                };
                loop {
                    let sym = lit.read(&mut b)? as usize;
                    if sym < 256 {
                        out.push(sym as u8);
                    } else if sym == 256 {
                        break;
                    } else {
                        let i = sym - 257;
                        if i >= 29 {
                            return Err("its compressed data is damaged".into());
                        }
                        let len = LEN_BASE[i] as usize + b.take(LEN_EXTRA[i] as u32)? as usize;
                        let ds = dist.read(&mut b)? as usize;
                        if ds >= 30 {
                            return Err("its compressed data is damaged".into());
                        }
                        let d = DIST_BASE[ds] as usize + b.take(DIST_EXTRA[ds] as u32)? as usize;
                        if d > out.len() {
                            return Err("its compressed data is damaged".into());
                        }
                        let start = out.len() - d;
                        if d >= len {
                            out.extend_from_within(start..start + len);
                        } else {
                            for k in 0..len {
                                out.push(out[start + k]);
                            }
                        }
                    }
                    if out.len() > limit {
                        return Err("it holds more data than its size says".into());
                    }
                }
            }
            _ => return Err("its compressed data is damaged (an unknown block)".into()),
        }
        if out.len() > limit {
            return Err("it holds more data than its size says".into());
        }
        if last == 1 {
            return Ok(out);
        }
    }
}

// ---------- reading PNG files ----------

fn be32(d: &[u8]) -> u32 {
    u32::from_be_bytes([d[0], d[1], d[2], d[3]])
}

/// A PNG file's pixels: width, height, and premultiplied 0xAARRGGBB pixels.
pub fn decode(data: &[u8]) -> Result<(usize, usize, Vec<u32>), String> {
    if data.len() < 8 || &data[..8] != b"\x89PNG\r\n\x1a\n" {
        return Err("it isn't a PNG file".into());
    }
    let mut pos = 8;
    let (mut w, mut h, mut depth, mut ctype, mut interlace) = (0usize, 0usize, 0u8, 0u8, 0u8);
    let mut palette: Vec<[u8; 3]> = vec![];
    let mut trns: Vec<u8> = vec![];
    let mut idat: Vec<u8> = vec![];
    let mut seen_header = false;
    while pos + 8 <= data.len() {
        let len = be32(&data[pos..]) as usize;
        let kind = &data[pos + 4..pos + 8];
        let body = data.get(pos + 8..pos + 8 + len).ok_or(CUT)?;
        match kind {
            b"IHDR" => {
                if len < 13 {
                    return Err("its header is damaged".into());
                }
                w = be32(body) as usize;
                h = be32(&body[4..]) as usize;
                depth = body[8];
                ctype = body[9];
                interlace = body[12];
                seen_header = true;
            }
            b"PLTE" => palette = body.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
            b"tRNS" => trns = body.to_vec(),
            b"IDAT" => idat.extend_from_slice(body),
            b"IEND" => break,
            _ => {}
        }
        pos += 12 + len;
    }
    if !seen_header {
        return Err("its header is missing".into());
    }
    if w == 0 || h == 0 || w > 16384 || h > 16384 {
        return Err(format!("it is {} by {} pixels; pictures can be up to 16384 by 16384", w, h));
    }
    let channels = match ctype {
        0 => 1,
        2 => 3,
        3 => 1,
        4 => 2,
        6 => 4,
        _ => return Err("it uses a colour type PNG doesn't have".into()),
    };
    let ok_depth = match ctype {
        0 => [1, 2, 4, 8, 16].contains(&depth),
        3 => [1, 2, 4, 8].contains(&depth),
        _ => [8, 16].contains(&depth),
    };
    if !ok_depth {
        return Err(format!("it uses {} bits a sample, which PNG doesn't allow with its colour type", depth));
    }
    if ctype == 3 && palette.is_empty() {
        return Err("its palette is missing".into());
    }
    if idat.len() < 2 || idat[0] & 0x0F != 8 {
        return Err("its compressed data is damaged".into());
    }
    let bits = channels * depth as usize;
    let bpp = bits.div_ceil(8);
    let row_bytes = |pw: usize| (pw * bits).div_ceil(8);
    // how much the pixels take, with each row's filter byte
    let passes: Vec<(usize, usize, usize, usize)> =
        if interlace == 1 { vec![(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)] } else { vec![(0, 0, 1, 1)] };
    let dims = |&(sx, sy, dx, dy): &(usize, usize, usize, usize)| ((w + dx - 1 - sx.min(w)) / dx, (h + dy - 1 - sy.min(h)) / dy);
    let expected: usize = passes.iter().map(|p| {
        let (pw, ph) = dims(p);
        if pw == 0 || ph == 0 { 0 } else { ph * (1 + row_bytes(pw)) }
    }).sum();
    let raw = inflate(&idat[2..], expected)?;
    if raw.len() < expected {
        return Err(CUT.into());
    }

    let gray_key = (ctype == 0 && trns.len() >= 2).then(|| u16::from_be_bytes([trns[0], trns[1]]));
    let rgb_key = (ctype == 2 && trns.len() >= 6).then(|| [u16::from_be_bytes([trns[0], trns[1]]), u16::from_be_bytes([trns[2], trns[3]]), u16::from_be_bytes([trns[4], trns[5]])]);
    let scale = |v: u16| -> u8 {
        match depth {
            1 => (v * 255) as u8,
            2 => (v * 85) as u8,
            4 => (v * 17) as u8,
            8 => v as u8,
            _ => (v >> 8) as u8,
        }
    };

    let mut px = vec![0u32; w * h];
    let mut at = 0;
    for p in &passes {
        let (pw, ph) = dims(p);
        if pw == 0 || ph == 0 {
            continue;
        }
        let rb = row_bytes(pw);
        let mut prev = vec![0u8; rb];
        let mut cur = vec![0u8; rb];
        for j in 0..ph {
            let filter = raw[at];
            cur.copy_from_slice(&raw[at + 1..at + 1 + rb]);
            at += 1 + rb;
            unfilter(filter, &mut cur, &prev, bpp)?;
            for i in 0..pw {
                // one sample of `depth` bits
                let sample = |k: usize| -> u16 {
                    match depth {
                        16 => u16::from_be_bytes([cur[(i * channels + k) * 2], cur[(i * channels + k) * 2 + 1]]),
                        8 => cur[i * channels + k] as u16,
                        _ => {
                            let bit = (i * channels + k) * depth as usize;
                            ((cur[bit / 8] >> (8 - depth as usize - bit % 8)) & ((1u8 << depth) - 1)) as u16
                        }
                    }
                };
                let (r, g, b, a) = match ctype {
                    0 => {
                        let v = sample(0);
                        let g = scale(v);
                        (g, g, g, if gray_key == Some(v) { 0 } else { 255 })
                    }
                    2 => {
                        let (r, g, b) = (sample(0), sample(1), sample(2));
                        let a = if rgb_key == Some([r, g, b]) { 0 } else { 255 };
                        (scale(r), scale(g), scale(b), a)
                    }
                    3 => {
                        let k = sample(0) as usize;
                        let c = palette.get(k).copied().unwrap_or([0, 0, 0]);
                        (c[0], c[1], c[2], trns.get(k).copied().unwrap_or(255))
                    }
                    4 => {
                        let g = scale(sample(0));
                        (g, g, g, scale(sample(1)))
                    }
                    _ => (scale(sample(0)), scale(sample(1)), scale(sample(2)), scale(sample(3))),
                };
                let (x, y) = (p.0 + i * p.2, p.1 + j * p.3);
                let m = |c: u8| (c as u32 * a as u32 + 127) / 255;
                px[y * w + x] = (a as u32) << 24 | m(r) << 16 | m(g) << 8 | m(b);
            }
            std::mem::swap(&mut prev, &mut cur);
        }
    }
    Ok((w, h, px))
}

/// Undoes a row's PNG filter, given the row above.
fn unfilter(filter: u8, cur: &mut [u8], prev: &[u8], bpp: usize) -> Result<(), String> {
    let n = cur.len();
    match filter {
        0 => {}
        1 => {
            for i in bpp..n {
                cur[i] = cur[i].wrapping_add(cur[i - bpp]);
            }
        }
        2 => {
            for i in 0..n {
                cur[i] = cur[i].wrapping_add(prev[i]);
            }
        }
        3 => {
            for i in 0..n {
                let left = if i >= bpp { cur[i - bpp] as u16 } else { 0 };
                cur[i] = cur[i].wrapping_add(((left + prev[i] as u16) / 2) as u8);
            }
        }
        4 => {
            for i in 0..n {
                let a = if i >= bpp { cur[i - bpp] as i16 } else { 0 };
                let b = prev[i] as i16;
                let c = if i >= bpp { prev[i - bpp] as i16 } else { 0 };
                let p = a + b - c;
                let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
                let pred = if pa <= pb && pa <= pc { a } else if pb <= pc { b } else { c };
                cur[i] = cur[i].wrapping_add(pred as u8);
            }
        }
        _ => return Err("its pixel data is damaged (an unknown filter)".into()),
    }
    Ok(())
}

// ---------- writing PNG files ----------

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    !c
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
    b << 16 | a
}

/// A PNG file from rows that each start with a filter byte, then 3 (RGB) or
/// 4 (RGBA) bytes a pixel.
pub fn encode(w: usize, h: usize, alpha: bool, raw: &[u8]) -> Vec<u8> {
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
    z.extend_from_slice(&adler32(raw).to_be_bytes());

    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = vec![];
    header.extend_from_slice(&(w as u32).to_be_bytes());
    header.extend_from_slice(&(h as u32).to_be_bytes());
    header.extend_from_slice(&[8, if alpha { 6 } else { 2 }, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &header);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_what_it_writes() {
        // 2 by 2: red, see-through green / blue, half-see-through white
        let raw = [0, 255, 0, 0, 255, 0, 255, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 128];
        let (w, h, px) = decode(&encode(2, 2, true, &raw)).unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(px[0], 0xFFFF_0000);
        assert_eq!(px[1], 0);
        assert_eq!(px[2], 0xFF00_00FF);
        assert_eq!(px[3], 0x8080_8080);
    }

    #[test]
    fn inflates_compressed_data() {
        // "hello hello hello hello", compressed by zlib (fixed codes, with a
        // repeat)
        let z = [0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x27, 0x01];
        assert_eq!(inflate(&z, 1000).unwrap(), b"hello hello hello hello");
    }

    #[test]
    fn reads_the_test_picture() {
        let data = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/pixel.png")).unwrap();
        let (w, h, px) = decode(&data).unwrap();
        assert!(w > 0 && h > 0 && px.len() == w * h);
    }
}
