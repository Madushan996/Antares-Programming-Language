//! Sound for games: sound files, tones, tunes written as text, ready-made
//! effects and looping music.
//!
//! Antares mixes every sound itself: a background thread adds up whatever
//! is playing and hands the result to the speakers, 44,100 numbers a
//! second. Sounds are made the same way on every system; sending them to
//! the speakers is Windows-only for now (`win` below, using only what
//! Windows provides). Elsewhere, and on a computer with no sound device,
//! the sound functions do nothing, so a game still runs, silently.

use crate::builtins::{num, text, want};
use crate::interp::Interp;
use crate::value::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

type Res = Result<Value, String>;

/// Samples a second.
pub const RATE: usize = 44_100;

// ---------- the mixer ----------

#[cfg_attr(not(windows), allow(dead_code))]
struct Voice {
    samples: Arc<Vec<f32>>,
    pos: usize,
    gain: f32,
    /// Background music: loops, and there is at most one.
    music: bool,
}

struct Mixer {
    voices: Vec<Voice>,
    volume: f32,
}

fn mixer() -> &'static Mutex<Mixer> {
    static MIXER: OnceLock<Mutex<Mixer>> = OnceLock::new();
    MIXER.get_or_init(|| Mutex::new(Mixer { voices: vec![], volume: 1.0 }))
}

/// The speakers: not used yet, opening (which can take a second, so it
/// happens in the background), ready, or not available.
const UNUSED: u8 = 0;
const OPENING: u8 = 1;
const READY: u8 = 2;
const NONE: u8 = 3;
static OUTPUT: AtomicU8 = AtomicU8::new(UNUSED);

/// Starts the speaker thread the first time it is needed, without waiting
/// for it. False if there is no way to play sound here.
fn ensure_output() -> bool {
    if OUTPUT.compare_exchange(UNUSED, OPENING, Ordering::AcqRel, Ordering::Acquire).is_ok() {
        #[cfg(windows)]
        let started = std::thread::Builder::new().name("antares-sound".into()).spawn(win::run).is_ok();
        #[cfg(not(windows))]
        let started = false;
        if !started {
            no_output();
        }
    }
    OUTPUT.load(Ordering::Acquire) != NONE
}

/// There's no way to play sound: sounds waiting for the speakers are dropped.
fn no_output() {
    OUTPUT.store(NONE, Ordering::Release);
    mixer().lock().unwrap_or_else(|e| e.into_inner()).voices.clear();
}

/// Fills `out` with the next stretch of mixed sound, from -1 to 1.
#[cfg_attr(not(windows), allow(dead_code))]
fn mix(out: &mut [f32]) {
    out.fill(0.0);
    let mut m = mixer().lock().unwrap_or_else(|e| e.into_inner());
    for v in m.voices.iter_mut() {
        let n = v.samples.len();
        if n == 0 {
            continue;
        }
        for o in out.iter_mut() {
            if v.pos >= n {
                if v.music {
                    v.pos = 0;
                } else {
                    break;
                }
            }
            *o += v.samples[v.pos] * v.gain;
            v.pos += 1;
        }
    }
    m.voices.retain(|v| v.music || v.pos < v.samples.len());
    let vol = m.volume;
    for o in out.iter_mut() {
        // loud mixes are squeezed smoothly instead of crackling
        let x = *o * vol;
        *o = if x.abs() <= 0.8 { x } else { x.signum() * (0.8 + 0.2 * ((x.abs() - 0.8) / 0.2).tanh()) };
    }
}

fn start(samples: Arc<Vec<f32>>, gain: f32, music: bool) {
    if !ensure_output() {
        return;
    }
    let mut m = mixer().lock().unwrap_or_else(|e| e.into_inner());
    if music {
        m.voices.retain(|v| !v.music);
    }
    m.voices.push(Voice { samples, pos: 0, gain, music });
}

/// At the end of a program, lets the sounds still playing finish (not the
/// music), so `beep(...)` as a program's last line is heard.
pub fn finish() {
    if playing_count() == 0 {
        return;
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    while OUTPUT.load(Ordering::Acquire) == OPENING && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    if OUTPUT.load(Ordering::Acquire) != READY {
        return;
    }
    while Instant::now() < deadline && playing_count() > 0 {
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn playing_count() -> usize {
    mixer().lock().unwrap_or_else(|e| e.into_inner()).voices.iter().filter(|v| !v.music).count()
}

// ---------- making sounds ----------

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Wave {
    Square,
    Sine,
    Triangle,
    Saw,
    Noise,
}

fn wave(v: Option<&Value>, name: &str) -> Result<Wave, String> {
    let Some(v) = v else { return Ok(Wave::Square) };
    Ok(match &*text(v, name)? {
        "square" => Wave::Square,
        "sine" => Wave::Sine,
        "triangle" => Wave::Triangle,
        "saw" => Wave::Saw,
        "noise" => Wave::Noise,
        other => return Err(format!("{} makes \"square\", \"sine\", \"triangle\", \"saw\" or \"noise\" waves, not \"{}\"", name, other)),
    })
}

/// A little random-number maker for noise, the same every time.
struct Noise(u32);

impl Noise {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// A tone sliding from `f0` to `f1` hertz over `secs` seconds, with short
/// fades at both ends so it doesn't click.
fn tone(f0: f64, f1: f64, secs: f64, w: Wave, amp: f32) -> Vec<f32> {
    let n = (secs.max(0.0) * RATE as f64).round() as usize;
    let mut out = Vec::with_capacity(n);
    let mut phase = 0.0f64;
    let mut noise = Noise(0x9E37_79B9);
    let mut held = 0.0f32;
    for i in 0..n {
        let t = i as f64 / n.max(1) as f64;
        let f = f0 + (f1 - f0) * t;
        let before = phase;
        phase = (phase + f / RATE as f64).fract();
        let s = match w {
            Wave::Square => {
                if phase < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            Wave::Sine => (phase * std::f64::consts::TAU).sin() as f32,
            Wave::Triangle => (1.0 - 4.0 * (phase - 0.5).abs()) as f32,
            Wave::Saw => (2.0 * phase - 1.0) as f32,
            // noise at the given pitch: a new random value each cycle-step
            Wave::Noise => {
                if phase < before || i == 0 || (phase * 8.0).floor() != (before * 8.0).floor() {
                    held = noise.next();
                }
                held
            }
        };
        out.push(s as f32 * amp);
    }
    fade(&mut out);
    out
}

/// Short fades in and out, so sounds start and stop without a click.
fn fade(s: &mut [f32]) {
    let n = s.len();
    let edge = (RATE / 200).min(n / 2); // 5 ms
    for i in 0..edge {
        let g = i as f32 / edge as f32;
        s[i] *= g;
        s[n - 1 - i] *= g;
    }
}

/// How loud each kind of wave is, so they sound about equally loud.
fn amp(w: Wave) -> f32 {
    match w {
        Wave::Square | Wave::Saw | Wave::Noise => 0.22,
        Wave::Sine | Wave::Triangle => 0.4,
    }
}

/// A note name like "C", "F#", "Bb3" or "A5" as hertz (octave 4 if none is given).
fn note_hz(name: &str) -> Option<f64> {
    let mut cs = name.chars().peekable();
    let base = match cs.next()?.to_ascii_uppercase() {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => return None,
    };
    let mut semi = base;
    match cs.peek() {
        Some('#') => {
            semi += 1;
            cs.next();
        }
        Some('b') => {
            semi -= 1;
            cs.next();
        }
        _ => {}
    }
    let rest: String = cs.collect();
    let octave: i32 = if rest.is_empty() { 4 } else { rest.parse().ok().filter(|o| (0..=8).contains(o))? };
    let midi = 12 * (octave + 1) + semi;
    Some(440.0 * 2f64.powf((midi - 69) as f64 / 12.0))
}

/// A tune written as text: notes like "C", "F#", "Bb3" or "A5" (octave 4
/// if none is given) and "-" for a rest, each lasting `step` seconds, or
/// several steps with ":n" ("C:2").
pub fn tune_samples(notes: &str, step: f64, w: Wave, name: &str) -> Result<Vec<f32>, String> {
    let mut out = vec![];
    for token in notes.split_whitespace() {
        let (note, steps) = match token.split_once(':') {
            Some((n, k)) => {
                let k: f64 = k.parse().ok().filter(|k: &f64| *k > 0.0 && k.is_finite()).ok_or_else(|| {
                    format!("{}: in \"{}\", what follows : is how many steps the note lasts, like \"C:2\"", name, token)
                })?;
                (n, k)
            }
            None => (token, 1.0),
        };
        let secs = step * steps;
        if note == "-" || note == "." {
            out.extend(std::iter::repeat_n(0.0, (secs * RATE as f64).round() as usize));
            continue;
        }
        let hz = note_hz(note).ok_or_else(|| {
            format!("{} doesn't know the note \"{}\"; notes are written like \"C\", \"F#\", \"Bb3\" or \"A5\", and \"-\" is a rest", name, token)
        })?;
        // a small gap between notes keeps repeated notes apart
        let sound = (secs * 0.9).max(0.0);
        out.extend(tone(hz, hz, sound, w, amp(w)));
        out.extend(std::iter::repeat_n(0.0, ((secs - sound) * RATE as f64).round() as usize));
    }
    Ok(out)
}

fn noise_burst(secs: f64, amp: f32, pitch: f64) -> Vec<f32> {
    let mut s = tone(pitch, pitch * 0.3, secs, Wave::Noise, amp);
    let n = s.len();
    for (i, x) in s.iter_mut().enumerate() {
        let t = i as f32 / n.max(1) as f32;
        *x *= (1.0 - t) * (1.0 - t);
    }
    s
}

fn join(parts: Vec<Vec<f32>>) -> Vec<f32> {
    parts.concat()
}

fn overlay(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
    let (mut long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    for (x, y) in long.iter_mut().zip(short) {
        *x += y;
    }
    long
}

const EFFECTS: &[&str] = &["coin", "jump", "hit", "laser", "explosion", "powerup", "select"];

/// The ready-made effects.
pub fn effect_samples(name: &str) -> Option<Vec<f32>> {
    let sq = amp(Wave::Square);
    Some(match name {
        "coin" => join(vec![tone(988.0, 988.0, 0.06, Wave::Square, sq), tone(1319.0, 1319.0, 0.22, Wave::Square, sq)]),
        "jump" => tone(280.0, 760.0, 0.16, Wave::Square, sq),
        "hit" => overlay(noise_burst(0.14, 0.35, 3000.0), tone(160.0, 60.0, 0.14, Wave::Square, sq)),
        "laser" => tone(1500.0, 220.0, 0.18, Wave::Saw, amp(Wave::Saw)),
        "explosion" => noise_burst(0.7, 0.5, 1200.0),
        "powerup" => join(["C5", "E5", "G5", "C6", "E6", "G6"].iter().map(|n| tone(note_hz(n).unwrap(), note_hz(n).unwrap(), 0.06, Wave::Square, sq)).collect()),
        "select" => tone(880.0, 880.0, 0.05, Wave::Square, sq),
        _ => return None,
    })
}

// ---------- sound files ----------

/// A WAV file's sound as numbers from -1 to 1, one channel, at `RATE`.
pub fn decode_wav(b: &[u8]) -> Result<Vec<f32>, String> {
    let u16_at = |i: usize| b.get(i..i + 2).map(|s| u16::from_le_bytes([s[0], s[1]]));
    let u32_at = |i: usize| b.get(i..i + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]));
    if b.get(0..4) != Some(b"RIFF") || b.get(8..12) != Some(b"WAVE") {
        return Err("it isn't a WAV file".into());
    }
    let (mut fmt, mut data) = (None, None);
    let mut at = 12;
    while at + 8 <= b.len() {
        let id = &b[at..at + 4];
        let len = u32_at(at + 4).unwrap() as usize;
        let body = at + 8;
        let end = body.saturating_add(len).min(b.len());
        match id {
            b"fmt " => fmt = Some(body),
            b"data" => data = Some((body, end)),
            _ => {}
        }
        at = body.saturating_add(len).saturating_add(len & 1);
    }
    let (Some(f), Some((d0, d1))) = (fmt, data) else {
        return Err("its WAV data is incomplete".into());
    };
    let mut format = u16_at(f).ok_or("its WAV data is incomplete")?;
    let channels = u16_at(f + 2).ok_or("its WAV data is incomplete")? as usize;
    let rate = u32_at(f + 4).ok_or("its WAV data is incomplete")? as usize;
    let bits = u16_at(f + 14).ok_or("its WAV data is incomplete")? as usize;
    if format == 0xFFFE {
        // WAVE_FORMAT_EXTENSIBLE: the real format is at the start of the sub-format
        format = u16_at(f + 24).ok_or("its WAV data is incomplete")?;
    }
    if channels == 0 || rate == 0 {
        return Err("its WAV data is incomplete".into());
    }
    let bytes = bits / 8;
    let read: fn(&[u8]) -> f32 = match (format, bits) {
        (1, 8) => |s| (s[0] as f32 - 128.0) / 128.0,
        (1, 16) => |s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0,
        (1, 24) => |s| (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) as f32 / 8_388_608.0,
        (1, 32) => |s| i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32 / 2_147_483_648.0,
        (3, 32) => |s| f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
        _ => return Err(format!("it is a kind of WAV file Antares can't play (format {}, {} bits); save it as 16-bit PCM", format, bits)),
    };
    let frame = bytes * channels;
    let mono: Vec<f32> = b[d0..d1]
        .chunks_exact(frame)
        .map(|fr| fr.chunks_exact(bytes).map(read).sum::<f32>() / channels as f32)
        .collect();
    if rate == RATE {
        return Ok(mono);
    }
    // to RATE samples a second, by straight lines between samples
    let n = (mono.len() as f64 * RATE as f64 / rate as f64).round() as usize;
    let step = rate as f64 / RATE as f64;
    Ok((0..n)
        .map(|i| {
            let x = i as f64 * step;
            let k = x as usize;
            let t = (x - k as f64) as f32;
            let a = mono.get(k).copied().unwrap_or(0.0);
            let b = mono.get(k + 1).copied().unwrap_or(a);
            a + (b - a) * t
        })
        .collect())
}

thread_local! {
    /// Sound files already read, by name, so playing one every frame is cheap.
    static FILES: std::cell::RefCell<HashMap<String, Arc<Vec<f32>>>> = std::cell::RefCell::new(HashMap::new());
}

fn load(file: &str, name: &str) -> Result<Arc<Vec<f32>>, String> {
    if let Some(s) = FILES.with(|f| f.borrow().get(file).cloned()) {
        return Ok(s);
    }
    let bytes = match crate::builtins::embedded_file(file) {
        Some(Some(b)) => b.to_vec(),
        Some(None) => return Err(format!("{}: no file was embedded as \"{}\"", name, file)),
        None => std::fs::read(file).map_err(|e| format!("{} couldn't open the sound \"{}\": {}", name, file, e))?,
    };
    let samples = Arc::new(decode_wav(&bytes).map_err(|e| format!("{} can't play \"{}\": {}", name, file, e))?);
    FILES.with(|f| f.borrow_mut().insert(file.to_string(), samples.clone()));
    Ok(samples)
}

fn level(v: Option<&Value>, name: &str) -> Result<f32, String> {
    match v {
        None => Ok(1.0),
        Some(v) => {
            let x = num(v, name)?;
            if !(0.0..=1.0).contains(&x) {
                return Err(format!("{} needs a volume from 0 (silent) to 1 (full), not {}", name, v.show()));
            }
            Ok(x as f32)
        }
    }
}

fn inputs(a: &[Value], lo: usize, hi: usize, name: &str, what: &str) -> Result<(), String> {
    if a.len() < lo || a.len() > hi {
        return Err(format!("{} takes {}, but was given {} input{}", name, what, a.len(), if a.len() == 1 { "" } else { "s" }));
    }
    Ok(())
}

fn step(v: Option<&Value>, name: &str) -> Result<f64, String> {
    match v {
        None => Ok(0.15),
        Some(v) => {
            let s = num(v, name)?;
            if !(s > 0.0 && s <= 10.0) {
                return Err(format!("{} needs a step from 0 to 10 seconds, not {}", name, v.show()));
            }
            Ok(s)
        }
    }
}

// ---------- the built-in functions ----------

/// sound(file [, volume]): plays a WAV file once.
pub fn b_sound(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 1, 2, "sound", "a WAV file's name, and optionally a volume from 0 to 1")?;
    let file = text(&a[0], "sound")?;
    let gain = level(a.get(1), "sound")?;
    let samples = load(&file, "sound")?;
    start(samples, gain, false);
    Ok(Value::Nothing)
}

/// music(file), music(notes, step [, wave]), or music() to stop.
pub fn b_music(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 0, 3, "music", "a WAV file's name, or a tune and a step, or nothing to stop the music")?;
    let samples = match a.len() {
        0 => {
            mixer().lock().unwrap_or_else(|e| e.into_inner()).voices.retain(|v| !v.music);
            return Ok(Value::Nothing);
        }
        1 if matches!(a[0], Value::Nothing) => {
            mixer().lock().unwrap_or_else(|e| e.into_inner()).voices.retain(|v| !v.music);
            return Ok(Value::Nothing);
        }
        1 => load(&text(&a[0], "music")?, "music")?,
        _ => Arc::new(tune_samples(&text(&a[0], "music")?, step(a.get(1), "music")?, wave(a.get(2), "music")?, "music")?),
    };
    start(samples, 0.7, true);
    Ok(Value::Nothing)
}

/// beep(frequency, seconds [, wave])
pub fn b_beep(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 2, 3, "beep", "a frequency and a length in seconds, and optionally a wave")?;
    let f = num(&a[0], "beep")?;
    let secs = num(&a[1], "beep")?;
    if !(20.0..=20_000.0).contains(&f) {
        return Err(format!("beep needs a frequency from 20 to 20000 hertz, not {}", a[0].show()));
    }
    if !(0.0..=10.0).contains(&secs) {
        return Err(format!("beep needs a length from 0 to 10 seconds, not {}", a[1].show()));
    }
    let w = wave(a.get(2), "beep")?;
    start(Arc::new(tone(f, f, secs, w, amp(w))), 1.0, false);
    Ok(Value::Nothing)
}

/// tune(notes [, step [, wave]]): plays a tune once; gives its length in seconds.
pub fn b_tune(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 1, 3, "tune", "notes as text, and optionally a step in seconds and a wave")?;
    let s = tune_samples(&text(&a[0], "tune")?, step(a.get(1), "tune")?, wave(a.get(2), "tune")?, "tune")?;
    let secs = s.len() as f64 / RATE as f64;
    start(Arc::new(s), 1.0, false);
    Ok(Value::Num(secs))
}

/// effect(name): a ready-made sound effect.
pub fn b_effect(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "effect")?;
    let name = text(&a[0], "effect")?;
    let s = effect_samples(&name).ok_or_else(|| format!("effect doesn't know \"{}\"; the effects are {}", name, EFFECTS.iter().map(|e| format!("\"{}\"", e)).collect::<Vec<_>>().join(", ")))?;
    start(Arc::new(s), 1.0, false);
    Ok(Value::Nothing)
}

/// volume(level): how loud everything is, from 0 to 1.
pub fn b_volume(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "volume")?;
    let v = level(a.first(), "volume")?;
    mixer().lock().unwrap_or_else(|e| e.into_inner()).volume = v;
    Ok(Value::Nothing)
}

/// silence(): stops every sound, and the music.
pub fn b_silence(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 0, "silence")?;
    mixer().lock().unwrap_or_else(|e| e.into_inner()).voices.clear();
    Ok(Value::Nothing)
}

/// playing(): how many sounds are still playing (not counting music).
pub fn b_playing(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 0, "playing")?;
    Ok(Value::Num(playing_count() as f64))
}

// ---------- Windows ----------

#[cfg(windows)]
mod win {
    use super::{mix, mixer, no_output, OUTPUT, RATE, READY};
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};
    use std::ffi::c_void;

    #[repr(C)]
    struct WaveFormat {
        format: u16,
        channels: u16,
        rate: u32,
        bytes_per_sec: u32,
        block_align: u16,
        bits: u16,
        extra: u16,
    }

    #[repr(C)]
    struct WaveHdr {
        data: *mut u8,
        len: u32,
        recorded: u32,
        user: usize,
        flags: u32,
        loops: u32,
        next: *mut WaveHdr,
        reserved: usize,
    }

    #[link(name = "winmm")]
    unsafe extern "system" {
        fn waveOutOpen(out: *mut *mut c_void, device: u32, format: *const WaveFormat, callback: usize, instance: usize, flags: u32) -> u32;
        fn waveOutPrepareHeader(out: *mut c_void, hdr: *mut WaveHdr, size: u32) -> u32;
        fn waveOutWrite(out: *mut c_void, hdr: *mut WaveHdr, size: u32) -> u32;
        fn timeBeginPeriod(ms: u32) -> u32;
    }

    const WAVE_MAPPER: u32 = 0xFFFF_FFFF;
    const WHDR_DONE: u32 = 1;
    /// Samples in each buffer: 20 ms. Windows plays smaller buffers with
    /// gaps between them. Three buffers keep the sound at most 60 ms behind
    /// the program.
    const N: usize = 882;
    const BUFFERS: usize = 3;

    /// The speaker thread: opens the sound device, then keeps the buffers
    /// full while anything is playing.
    pub fn run() {
        let format = WaveFormat { format: 1, channels: 1, rate: RATE as u32, bytes_per_sec: RATE as u32 * 2, block_align: 2, bits: 16, extra: 0 };
        let mut out: *mut c_void = std::ptr::null_mut();
        if unsafe { waveOutOpen(&mut out, WAVE_MAPPER, &format, 0, 0, 0) } != 0 || out.is_null() {
            no_output();
            return;
        }
        unsafe { timeBeginPeriod(1) };
        let size = std::mem::size_of::<WaveHdr>() as u32;
        let mut mixed = vec![0f32; N];
        let mut pcm: Vec<Box<[i16; N]>> = (0..BUFFERS).map(|_| Box::new([0i16; N])).collect();
        let mut hdrs: Vec<Box<WaveHdr>> = pcm
            .iter_mut()
            .map(|b| Box::new(WaveHdr { data: b.as_mut_ptr().cast(), len: (N * 2) as u32, recorded: 0, user: 0, flags: 0, loops: 0, next: std::ptr::null_mut(), reserved: 0 }))
            .collect();
        for h in hdrs.iter_mut() {
            if unsafe { waveOutPrepareHeader(out, &mut **h, size) } != 0 {
                no_output();
                return;
            }
            h.flags |= WHDR_DONE; // free to fill
        }
        OUTPUT.store(READY, Ordering::Release);
        let mut last_sound = Instant::now();
        loop {
            for i in 0..BUFFERS {
                // wait until the device has finished playing this buffer
                while unsafe { std::ptr::read_volatile(&hdrs[i].flags) } & WHDR_DONE == 0 {
                    std::thread::sleep(Duration::from_millis(1));
                }
                // After a second with nothing playing, stop sending silence
                // (it costs time, and can keep a computer from sleeping),
                // until there is something to play again.
                loop {
                    let busy = !mixer().lock().unwrap_or_else(|e| e.into_inner()).voices.is_empty();
                    if busy {
                        last_sound = Instant::now();
                        break;
                    }
                    if last_sound.elapsed() < Duration::from_secs(1) {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
                mix(&mut mixed);
                for (p, m) in pcm[i].iter_mut().zip(&mixed) {
                    *p = (m.clamp(-1.0, 1.0) * 32767.0) as i16;
                }
                unsafe { waveOutWrite(out, &mut *hdrs[i], size) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(format: u16, channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        let mut b = b"RIFF\0\0\0\0WAVE".to_vec();
        b.extend_from_slice(b"fmt \x10\0\0\0");
        b.extend_from_slice(&format.to_le_bytes());
        b.extend_from_slice(&channels.to_le_bytes());
        b.extend_from_slice(&rate.to_le_bytes());
        b.extend_from_slice(&(rate * channels as u32 * bits as u32 / 8).to_le_bytes());
        b.extend_from_slice(&(channels * bits / 8).to_le_bytes());
        b.extend_from_slice(&bits.to_le_bytes());
        b.extend_from_slice(b"LIST\x03\0\0\0abc\0"); // an odd-sized chunk to skip, padded
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(data.len() as u32).to_le_bytes());
        b.extend_from_slice(data);
        b
    }

    #[test]
    fn wav_16_bit_mono() {
        let mut d = vec![];
        for s in [0i16, 16384, -32768, 32767] {
            d.extend_from_slice(&s.to_le_bytes());
        }
        let s = decode_wav(&wav(1, 1, 44100, 16, &d)).unwrap();
        assert_eq!(s.len(), 4);
        assert_eq!((s[0], s[1], s[2]), (0.0, 0.5, -1.0));
    }

    #[test]
    fn wav_8_bit_stereo_is_mixed_to_one_channel() {
        let s = decode_wav(&wav(1, 2, 44100, 8, &[255, 1, 128, 128])).unwrap();
        assert_eq!(s.len(), 2);
        assert!(s[0].abs() < 0.01 && s[1] == 0.0);
    }

    #[test]
    fn wav_other_rates_are_resampled() {
        let d: Vec<u8> = (0..22050).flat_map(|_| 1000i16.to_le_bytes()).collect();
        let s = decode_wav(&wav(1, 1, 22050, 16, &d)).unwrap();
        assert_eq!(s.len(), 44100);
        assert!((s[1000] - 1000.0 / 32768.0).abs() < 1e-6);
    }

    #[test]
    fn wav_float_and_bad_files() {
        let s = decode_wav(&wav(3, 1, 44100, 32, &0.25f32.to_le_bytes())).unwrap();
        assert_eq!(s, vec![0.25]);
        assert!(decode_wav(b"not a wav file at all").is_err());
        assert!(decode_wav(&wav(2, 1, 44100, 4, &[0, 0])).unwrap_err().contains("16-bit PCM"));
    }

    #[test]
    fn notes() {
        assert!((note_hz("A").unwrap() - 440.0).abs() < 1e-9);
        assert!((note_hz("A5").unwrap() - 880.0).abs() < 1e-9);
        assert!((note_hz("C4").unwrap() - 261.6256).abs() < 1e-3);
        assert!((note_hz("C#4").unwrap() - note_hz("Db4").unwrap()).abs() < 1e-9);
        assert!(note_hz("H").is_none() && note_hz("C9").is_none() && note_hz("").is_none());
    }

    #[test]
    fn tunes() {
        let s = tune_samples("C E - G:2", 0.1, Wave::Square, "tune").unwrap();
        assert_eq!(s.len(), (0.5 * RATE as f64).round() as usize);
        // the rest is silent
        assert!(s[(0.25 * RATE as f64) as usize..(0.29 * RATE as f64) as usize].iter().all(|x| *x == 0.0));
        assert!(tune_samples("C X", 0.1, Wave::Square, "tune").unwrap_err().contains("doesn't know the note \"X\""));
        assert!(tune_samples("C:0", 0.1, Wave::Square, "tune").is_err());
    }

    #[test]
    fn tones_and_effects_stay_in_range() {
        for w in [Wave::Square, Wave::Sine, Wave::Triangle, Wave::Saw, Wave::Noise] {
            let s = tone(440.0, 440.0, 0.2, w, amp(w));
            assert_eq!(s.len(), 8820);
            assert!(s.iter().all(|x| x.abs() <= 0.5), "{:?}", w);
            assert_eq!(s[0], 0.0); // fades in
        }
        for e in EFFECTS {
            let s = effect_samples(e).unwrap();
            assert!(!s.is_empty() && s.iter().all(|x| x.abs() <= 1.0), "{}", e);
        }
        assert!(effect_samples("boing").is_none());
    }

    #[test]
    fn mixing_adds_and_drops_finished_sounds() {
        {
            let mut m = mixer().lock().unwrap();
            m.voices.clear();
            m.voices.push(Voice { samples: Arc::new(vec![0.25; 3]), pos: 0, gain: 1.0, music: false });
            m.voices.push(Voice { samples: Arc::new(vec![0.25, -0.25]), pos: 0, gain: 1.0, music: true });
        }
        let mut out = [0.0f32; 5];
        mix(&mut out);
        assert_eq!(out, [0.5, 0.0, 0.5, -0.25, 0.25]);
        let m = mixer().lock().unwrap();
        assert_eq!(m.voices.len(), 1); // the music keeps looping, the sound is done
        assert!(m.voices[0].music);
    }
}
