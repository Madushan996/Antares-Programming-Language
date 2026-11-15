//! `antares build --pack`: a program you can share, made without Rust.
//!
//! The executable is a copy of this `antares` program with the Antares
//! program attached to its end: its source, the files it brings in with
//! `use`, and any files given with `--embed`. When it starts, `main` finds
//! the attachment and runs it with the same engine as `antares run`, so it
//! behaves (and runs at the speed) of an interpreted program.
//!
//! The attachment is a list of entries followed by its length (8 bytes) and
//! `MAGIC`. Each entry is a kind (1 byte), a name (4-byte length and bytes)
//! and data (8-byte length and bytes). Lengths are little-endian.

use crate::builtins;
use crate::lexer;
use crate::parser::Parser;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 16] = b"\0ANTARES-PACKED\0";

const MAIN: u8 = 0;
const FILE: u8 = 1;
const ALIAS: u8 = 2;
const EMBEDDED: u8 = 3;

/// A program attached to an executable.
pub struct Packed {
    /// The main file's name, and its source.
    pub main: String,
    pub main_src: String,
    /// Files brought in with `use`: (name relative to the main file's folder, source).
    pub files: Vec<(String, String)>,
    /// How each path written after `use` was resolved: (written, file).
    pub aliases: Vec<(String, String)>,
    /// Files given with `--embed`: (file name, contents).
    pub embedded: Vec<(String, Vec<u8>)>,
}

fn put(out: &mut Vec<u8>, kind: u8, name: &str, data: &[u8]) {
    out.push(kind);
    out.extend_from_slice(&(name.len() as u32).to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(data);
}

fn encode(p: &Packed) -> Vec<u8> {
    let mut out = vec![];
    put(&mut out, MAIN, &p.main, p.main_src.as_bytes());
    for (name, src) in &p.files {
        put(&mut out, FILE, name, src.as_bytes());
    }
    for (written, file) in &p.aliases {
        put(&mut out, ALIAS, written, file.as_bytes());
    }
    for (name, data) in &p.embedded {
        put(&mut out, EMBEDDED, name, data);
    }
    let len = out.len() as u64;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(MAGIC);
    out
}

fn decode(b: &[u8]) -> Option<Packed> {
    let mut p = Packed { main: String::new(), main_src: String::new(), files: vec![], aliases: vec![], embedded: vec![] };
    let mut at: usize = 0;
    let mut take = |n: usize| -> Option<&[u8]> {
        let s = b.get(at..at.checked_add(n)?)?;
        at += n;
        Some(s)
    };
    let mut seen_main = false;
    while let Some(kind) = take(1) {
        let kind = kind[0];
        let n = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
        let name = String::from_utf8(take(n)?.to_vec()).ok()?;
        let n = usize::try_from(u64::from_le_bytes(take(8)?.try_into().ok()?)).ok()?;
        let data = take(n)?.to_vec();
        let text = || String::from_utf8(data.clone()).ok();
        match kind {
            MAIN => {
                p.main = name;
                p.main_src = text()?;
                seen_main = true;
            }
            FILE => p.files.push((name, text()?)),
            ALIAS => p.aliases.push((name, text()?)),
            EMBEDDED => p.embedded.push((name, data)),
            _ => return None,
        }
    }
    seen_main.then_some(p)
}

/// The program attached to the running executable, if there is one.
pub fn find() -> Option<Packed> {
    let exe = std::env::current_exe().ok()?;
    let mut f = std::fs::File::open(exe).ok()?;
    let tail = (8 + MAGIC.len()) as u64;
    let size = f.seek(SeekFrom::End(0)).ok()?;
    if size < tail {
        return None;
    }
    f.seek(SeekFrom::Start(size - tail)).ok()?;
    let mut end = [0u8; 24];
    f.read_exact(&mut end).ok()?;
    if &end[8..] != MAGIC {
        return None;
    }
    let len = u64::from_le_bytes(end[..8].try_into().ok()?);
    let start = (size - tail).checked_sub(len)?;
    f.seek(SeekFrom::Start(start)).ok()?;
    let mut body = vec![0u8; usize::try_from(len).ok()?];
    f.read_exact(&mut body).ok()?;
    decode(&body)
}

/// Builds `path` into an executable by attaching it to a copy of this
/// `antares` program. `out`, `embed` and `icon` are as for `aot::build`.
pub fn build(path: &str, out: Option<&str>, embed: &[String], icon: Option<&str>) -> i32 {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Couldn't open \"{}\": {}", path, e);
            return 1;
        }
    };
    // Parse it now: mistakes show up here rather than when someone runs it,
    // and this finds every file brought in with `use`.
    let b = builtins::all();
    let names: Vec<&str> = b.iter().map(|(n, _)| *n).collect();
    let mut parser = Parser::new(&names);
    parser.set_main(path);
    if let Err((msg, line)) = lexer::lex(&src, 1).and_then(|toks| parser.parse(toks).map(|_| ()).map_err(|e| (e.msg, e.line))) {
        let (file, line) = (line >> 20, line & 0xFFFFF);
        let place = if file > 0 { parser.file(file).map(|(n, _)| format!(" of {}", n)).unwrap_or_default() } else { String::new() };
        eprintln!("\nProblem on line {}{}: {}", line, place, msg);
        return 1;
    }
    let (files, aliases) = parser.uses();
    let main = std::fs::canonicalize(path)
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "program.ant".into());
    let mut embedded = vec![];
    for f in embed {
        let name = Path::new(f).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        match std::fs::read(f) {
            Ok(data) => embedded.push((name, data)),
            Err(e) => {
                eprintln!("Couldn't find \"{}\" to embed: {}", f, e);
                return 1;
            }
        }
    }
    let packed = Packed { main, main_src: src, files, aliases: aliases.to_vec(), embedded };

    let me = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Couldn't find the antares program itself to copy: {}", e);
            return 1;
        }
    };
    let dest = match out {
        Some(o) => PathBuf::from(o),
        None => PathBuf::from(path).with_file_name(format!("{}{}", crate::aot::exe_stem(path), std::env::consts::EXE_SUFFIX)),
    };
    if std::fs::canonicalize(&dest).ok() == std::fs::canonicalize(&me).ok() {
        eprintln!("The program can't be written over antares itself; choose another name.");
        return 1;
    }
    if let Err(e) = std::fs::copy(&me, &dest) {
        eprintln!("Couldn't write {}: {}", dest.display(), e);
        return 1;
    }
    let fail = |msg: String| {
        let _ = std::fs::remove_file(&dest);
        eprintln!("{}", msg);
        1
    };
    // The icon goes in first: changing resources rewrites the file.
    if let Some(icon) = icon {
        let data = match std::fs::read(icon) {
            Ok(d) => d,
            Err(e) => return fail(format!("Couldn't open the icon \"{}\": {}", icon, e)),
        };
        let Some((pictures, group)) = crate::aot::ico_parts(&data) else {
            return fail(format!("\"{}\" isn't an .ico file.", icon));
        };
        if let Err(e) = set_icon(&dest, &pictures, &group) {
            return fail(e);
        }
    }
    let attached = std::fs::OpenOptions::new().append(true).open(&dest).and_then(|mut f| f.write_all(&encode(&packed)));
    if let Err(e) = attached {
        return fail(format!("Couldn't write {}: {}", dest.display(), e));
    }
    eprintln!("Built {}", dest.display());
    0
}

#[cfg(windows)]
fn set_icon(exe: &Path, pictures: &[&[u8]], group: &[u8]) -> Result<(), String> {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn BeginUpdateResourceW(file: *const u16, delete_existing: i32) -> *mut c_void;
        fn UpdateResourceW(update: *mut c_void, kind: *const u16, name: *const u16, language: u16, data: *const c_void, size: u32) -> i32;
        fn EndUpdateResourceW(update: *mut c_void, discard: i32) -> i32;
    }
    // resource types and names given as numbers, as MAKEINTRESOURCE does
    let id = |n: u16| n as usize as *const u16;
    const ENGLISH_US: u16 = 0x0409;
    let wide: Vec<u16> = exe.as_os_str().encode_wide().chain([0]).collect();
    unsafe {
        let h = BeginUpdateResourceW(wide.as_ptr(), 0);
        if h.is_null() {
            return Err(format!("Couldn't add the icon to {}", exe.display()));
        }
        let mut ok = true;
        for (k, picture) in pictures.iter().enumerate() {
            ok &= UpdateResourceW(h, id(3), id(k as u16 + 1), ENGLISH_US, picture.as_ptr().cast(), picture.len() as u32) != 0;
        }
        ok &= UpdateResourceW(h, id(14), id(1), ENGLISH_US, group.as_ptr().cast(), group.len() as u32) != 0;
        if EndUpdateResourceW(h, if ok { 0 } else { 1 }) == 0 || !ok {
            return Err(format!("Couldn't add the icon to {}", exe.display()));
        }
    }
    Ok(())
}

#[cfg(not(windows))]
fn set_icon(_: &Path, _: &[&[u8]], _: &[u8]) -> Result<(), String> {
    eprintln!("Note: --icon only applies to Windows programs, so this one is built without it.");
    Ok(())
}
