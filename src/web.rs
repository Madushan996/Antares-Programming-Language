//! Reaching outside the program: running shell commands (`shell`) and
//! fetching from the web (`fetch`, `read_page`), with helpers that turn a web
//! page into readable text (`page_text`) and its links (`page_links`).
//!
//! Web requests go through the `curl` program that comes with Windows, macOS
//! and Linux, as calls to language models do, so Antares needs no networking
//! library of its own.

use crate::builtins::{s, text};
use crate::interp::Interp;
use crate::value::Value;
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::rc::Rc;
use std::time::{Duration, Instant};

type Res = Result<Value, String>;

fn options<'a>(a: &'a [Value], name: &str, example: &str) -> Result<Vec<(&'a str, &'a Value)>, String> {
    if a.is_empty() || a.len() > 2 {
        return Err(format!("{} takes {} and, if you like, a map of options, like {}", name, if name == "shell" { "a command" } else { "a web address" }, example));
    }
    match a.get(1) {
        None | Some(Value::Nothing) => Ok(vec![]),
        Some(Value::Map(m)) => Ok(m.iter().map(|(k, v)| (&**k, v)).collect()),
        Some(other) => Err(format!("the options for {} are a map, like {}, not a {}", name, example, other.kind())),
    }
}

fn seconds(v: &Value, name: &str) -> Result<f64, String> {
    match v {
        Value::Num(n) if *n > 0.0 => Ok(*n),
        _ => Err(format!("the {} option `timeout` is a number of seconds above 0", name)),
    }
}

fn trim_end_lines(t: &str) -> &str {
    t.trim_end_matches(['\n', '\r'])
}

// ---------- shell ----------

/// shell(command [, options]): runs a command in the computer's shell (cmd on
/// Windows, sh elsewhere) and gives {output, errors, code}.
pub fn b_shell(_: &mut Interp, a: Vec<Value>) -> Res {
    let example = "shell(\"git status\", {folder: \"src\", timeout: 30})";
    let opts = options(&a, "shell", example)?;
    let command = text(&a[0], "shell")?;
    let (mut folder, mut input, mut timeout) = (None, None, 120.0);
    for (k, v) in opts {
        match (k, v) {
            ("folder", Value::Str(f)) => folder = Some(f.to_string()),
            ("input", v) => input = Some(v.show()),
            ("timeout", v) => timeout = seconds(v, "shell")?,
            _ => return Err(format!("I don't know the shell option '{}' (or it has the wrong kind of value). Available: folder (text), input (text), timeout (seconds)", k)),
        }
    }
    let (output, errors, code) = run_command(&command, folder.as_deref(), input.as_deref(), timeout)?;
    Ok(Value::Map(Rc::new(vec![
        ("output".into(), s(trim_end_lines(&output))),
        ("errors".into(), s(trim_end_lines(&errors))),
        ("code".into(), Value::Num(code as f64)),
    ])))
}

fn shell_command(command: &str) -> Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let mut c = Command::new("cmd");
        // /S /C "..." hands cmd the command exactly as written
        c.raw_arg(format!("/S /C \"{}\"", command));
        c
    }
    #[cfg(not(windows))]
    {
        use std::os::unix::process::CommandExt;
        let mut c = Command::new("sh");
        c.arg("-c").arg(command);
        // its own process group, so a timeout can stop everything it started
        c.process_group(0);
        c
    }
}

/// Stops a command and every program it started (a game window, a server,
/// a test runner's workers), not just the shell running it.
fn stop_tree(child: &mut std::process::Child) {
    let pid = child.id().to_string();
    #[cfg(windows)]
    let _ = Command::new("taskkill").args(["/T", "/F", "/PID", &pid]).stdout(Stdio::null()).stderr(Stdio::null()).status();
    #[cfg(not(windows))]
    let _ = Command::new("kill").args(["-KILL", &format!("-{}", pid)]).stdout(Stdio::null()).stderr(Stdio::null()).status();
    let _ = child.kill();
    let _ = child.wait();
}

/// Runs a command, giving (output, errors, exit code). A command still running
/// after `timeout` seconds is stopped, along with everything it started, with
/// code -1.
pub fn run_command(command: &str, folder: Option<&str>, input: Option<&str>, timeout: f64) -> Result<(String, String, i64), String> {
    let mut c = shell_command(command);
    if let Some(f) = folder {
        if !std::path::Path::new(f).is_dir() {
            return Err(format!("shell can't run in the folder \"{}\": there is no such folder", f));
        }
        c.current_dir(f);
    }
    c.stdin(if input.is_some() { Stdio::piped() } else { Stdio::null() }).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = c.spawn().map_err(|e| format!("couldn't run the command: {}", e))?;
    if let Some(t) = input {
        let mut stdin = child.stdin.take().unwrap();
        let t = t.to_string();
        // in its own thread, so a command that doesn't read can't block us
        std::thread::spawn(move || {
            let _ = stdin.write_all(t.as_bytes());
        });
    }
    let reader = |mut r: Box<dyn Read + Send>| {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut buf = vec![];
            let _ = r.read_to_end(&mut buf);
            let _ = tx.send(buf);
        });
        rx
    };
    let out = reader(Box::new(child.stdout.take().unwrap()));
    let err = reader(Box::new(child.stderr.take().unwrap()));
    let start = Instant::now();
    let mut stopped = false;
    let code = loop {
        match child.try_wait().map_err(|e| format!("couldn't run the command: {}", e))? {
            Some(status) => break status.code().unwrap_or(-1) as i64,
            None if start.elapsed().as_secs_f64() > timeout => {
                stop_tree(&mut child);
                stopped = true;
                break -1;
            }
            None => std::thread::sleep(Duration::from_millis(5)),
        }
    };
    // A program the command started may still hold the output open; don't
    // wait for it for long once the command itself has finished.
    let wait = Duration::from_millis(if stopped { 200 } else { 2000 });
    let collect = |rx: std::sync::mpsc::Receiver<Vec<u8>>| String::from_utf8_lossy(&rx.recv_timeout(wait).unwrap_or_default()).to_string();
    let output = collect(out);
    let mut errors = collect(err);
    if stopped {
        if !errors.is_empty() && !errors.ends_with('\n') {
            errors.push('\n');
        }
        errors.push_str(&format!("The command was stopped after {} seconds.", crate::value::fmt_num(timeout)));
    }
    Ok((output, errors, code))
}

// ---------- fetch ----------

fn temp_path(what: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let n = COUNT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("antares_{}_{}_{}", std::process::id(), n, what))
}

/// Text made safe to put inside a web address.
pub fn url_encode(t: &str) -> String {
    let mut out = String::new();
    for b in t.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

/// fetch(url [, options]): what a web address sends back, as text.
pub fn b_fetch(_: &mut Interp, a: Vec<Value>) -> Res {
    let example = "fetch(\"https://example.com\", {query: {q: \"moon\"}})";
    let opts = options(&a, "fetch", example)?;
    let mut url = text(&a[0], "fetch")?.to_string();
    let (mut method, mut headers, mut body, mut timeout) = (None, vec![], None, 60.0);
    for (k, v) in opts {
        match (k, v) {
            ("method", Value::Str(m)) => method = Some(m.to_uppercase()),
            ("headers", Value::Map(h)) => {
                for (name, value) in h.iter() {
                    headers.push(format!("{}: {}", name, value.show()));
                }
            }
            ("query", Value::Map(q)) => {
                for (i, (name, value)) in q.iter().enumerate() {
                    let sep = if i == 0 && !url.contains('?') { '?' } else { '&' };
                    url.push_str(&format!("{}{}={}", sep, url_encode(name), url_encode(&value.show())));
                }
            }
            ("body", Value::Str(b)) => body = Some(b.to_string()),
            ("body", v @ (Value::Map(_) | Value::List(_))) => {
                body = Some(crate::builtins::to_json(v)?);
                if !headers.iter().any(|h| h.to_lowercase().starts_with("content-type:")) {
                    headers.push("Content-Type: application/json".into());
                }
            }
            ("timeout", v) => timeout = seconds(v, "fetch")?,
            _ => {
                return Err(format!(
                    "I don't know the fetch option '{}' (or it has the wrong kind of value). Available: method (text), headers (a map), query (a map), body (text, or a map or list sent as JSON), timeout (seconds)",
                    k
                ))
            }
        }
    }
    let got = request(&url, method.as_deref(), &headers, body.as_deref(), timeout)?;
    Ok(s(got))
}

/// Sends a request with curl and gives the body as text. Answers of 400 and
/// above are problems.
pub fn request(url: &str, method: Option<&str>, headers: &[String], body: Option<&str>, timeout: f64) -> Result<String, String> {
    if url.trim().is_empty() {
        return Err("fetch needs a web address, like \"https://example.com\"".into());
    }
    let quote = |x: &str| x.replace('\\', "\\\\").replace('"', "\\\"");
    let out_path = temp_path("page");
    let body_path = temp_path("body");
    let mut config = format!("url = \"{}\"\n", quote(url));
    for h in headers {
        config.push_str(&format!("header = \"{}\"\n", quote(h)));
    }
    if let Some(b) = body {
        std::fs::write(&body_path, b).map_err(|e| format!("couldn't prepare the request: {}", e))?;
        config.push_str(&format!("data-binary = \"@{}\"\n", quote(&body_path.to_string_lossy().replace('\\', "/"))));
    }
    if let Some(m) = method {
        config.push_str(&format!("request = \"{}\"\n", quote(m)));
    }
    config.push_str(&format!("output = \"{}\"\n", quote(&out_path.to_string_lossy().replace('\\', "/"))));
    config.push_str(&format!("user-agent = \"Antares/{}\"\n", env!("CARGO_PKG_VERSION")));
    config.push_str(&format!("write-out = \"%{{http_code}}\"\nlocation\nsilent\nshow-error\nmax-time = {}\n", timeout));
    let result = (|| {
        let mut child = Command::new("curl")
            .args(["--config", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("couldn't start curl to fetch {}: {}", url, e))?;
        child.stdin.take().unwrap().write_all(config.as_bytes()).map_err(|e| format!("couldn't send the request: {}", e))?;
        let out = child.wait_with_output().map_err(|e| format!("couldn't fetch {}: {}", url, e))?;
        if !out.status.success() {
            let why = String::from_utf8_lossy(&out.stderr);
            let why = why.trim().trim_start_matches("curl: ");
            return Err(format!("couldn't fetch {}: {}", url, if why.is_empty() { "the request failed" } else { why }));
        }
        let code: u32 = String::from_utf8_lossy(&out.stdout).trim().parse().unwrap_or(0);
        let page = String::from_utf8_lossy(&std::fs::read(&out_path).unwrap_or_default()).to_string();
        if code >= 400 {
            let gist = page_text(&page);
            let gist: String = gist.chars().take(200).collect();
            let gist = gist.trim();
            return Err(format!("{} answered with error {}{}", url, code, if gist.is_empty() { String::new() } else { format!(": {}", gist) }));
        }
        Ok(page)
    })();
    let _ = std::fs::remove_file(&out_path);
    let _ = std::fs::remove_file(&body_path);
    result
}

/// read_page(url): a web page's readable text, without its markup.
pub fn b_read_page(_: &mut Interp, a: Vec<Value>) -> Res {
    crate::builtins::want(&a, 1, "read_page")?;
    let url = text(&a[0], "read_page")?;
    Ok(s(page_text(&request(&url, None, &[], None, 60.0)?)))
}

// ---------- reading web pages ----------

/// page_text(html): the readable text of a web page.
pub fn b_page_text(_: &mut Interp, a: Vec<Value>) -> Res {
    crate::builtins::want(&a, 1, "page_text")?;
    Ok(s(page_text(&text(&a[0], "page_text")?)))
}

/// page_links(html [, address]): the links on a page, as {text, url} maps,
/// made into full addresses when the page's own address is given.
pub fn b_page_links(_: &mut Interp, a: Vec<Value>) -> Res {
    if a.is_empty() || a.len() > 2 {
        return Err("page_links takes a page's HTML and, if you like, its address, like page_links(html, \"https://example.com/\")".into());
    }
    let html = text(&a[0], "page_links")?;
    let base = match a.get(1) {
        Some(v) => Some(text(v, "page_links")?.to_string()),
        None => None,
    };
    let links = page_links(&html, base.as_deref());
    Ok(Value::List(Rc::new(
        links.into_iter().map(|(t, u)| Value::Map(Rc::new(vec![("text".into(), s(t)), ("url".into(), s(u))]))).collect(),
    )))
}

/// Elements whose insides aren't HTML, skipped to their closing tag.
const RAW_TEXT: [&str; 4] = ["script", "style", "noscript", "template"];
/// Elements that are never part of what a page says.
const SKIPPED: [&str; 8] = ["svg", "iframe", "nav", "aside", "footer", "form", "button", "select"];
const VOID: [&str; 14] = ["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"];
/// Words in a class or id that mark menus and other clutter.
const CLUTTER: [&str; 13] = ["menu", "dropdown", "sidebar", "breadcrumb", "breadcrumbs", "cookie", "cookies", "popup", "navbox", "navigation", "share", "social", "skip"];

/// Whether a start tag marks something hidden, or a menu, banner and the like.
fn clutter(tag: &str) -> bool {
    if attr(tag, "aria-hidden").as_deref() == Some("true") {
        return true;
    }
    if let Some(role) = attr(tag, "role") {
        if ["navigation", "menu", "menubar", "banner", "contentinfo", "search", "dialog"].contains(&role.to_ascii_lowercase().as_str()) {
            return true;
        }
    }
    ["class", "id"].iter().any(|a| attr(tag, a).is_some_and(|v| v.to_ascii_lowercase().split(|c: char| !c.is_ascii_alphanumeric()).any(|w| CLUTTER.contains(&w))))
}

/// The position just after the element whose start tag ends at `after`,
/// counting nested elements of the same name.
fn skip_element(b: &[char], after: usize, name: &str) -> usize {
    let mut depth = 1;
    let mut i = after;
    while let Some(p) = (i..b.len()).find(|&k| b[k] == '<') {
        let (n, closing, end) = tag_at(b, p + 1);
        i = end.max(p + 1);
        if n == name {
            if closing {
                depth -= 1;
                if depth == 0 {
                    return end;
                }
            } else if !(end >= 2 && b[end - 2] == '/') {
                depth += 1;
            }
        }
    }
    b.len()
}
const BLOCKS: [&str; 33] = [
    "p", "div", "br", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "header", "footer", "ul", "ol", "table", "blockquote", "pre", "hr", "title",
    "dd", "dt", "dl", "nav", "main", "aside", "form", "figure", "figcaption", "address", "details", "summary",
];

/// The tag starting at `i` (just after '<'): (name in lower case, closing?,
/// position just after '>').
fn tag_at(b: &[char], i: usize) -> (String, bool, usize) {
    let mut j = i;
    let closing = j < b.len() && b[j] == '/';
    if closing {
        j += 1;
    }
    let mut name = String::new();
    while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == '-') {
        name.push(b[j].to_ascii_lowercase());
        j += 1;
    }
    // to the closing '>', stepping over quoted attribute values
    let mut quote = None;
    while j < b.len() {
        match (quote, b[j]) {
            (None, '>') => return (name, closing, j + 1),
            (None, '"' | '\'') => quote = Some(b[j]),
            (Some(q), c) if c == q => quote = None,
            _ => {}
        }
        j += 1;
    }
    (name, closing, b.len())
}

fn find_from(b: &[char], from: usize, pat: &str) -> Option<usize> {
    let p: Vec<char> = pat.chars().collect();
    (from..b.len().saturating_sub(p.len() - 1)).find(|&i| b[i..i + p.len()].iter().zip(&p).all(|(x, y)| x.to_ascii_lowercase() == *y))
}

/// Replaces &amp;, &#39;, &#x2014; and the like with their characters.
pub fn decode_entities(t: &str) -> String {
    if !t.contains('&') {
        return t.to_string();
    }
    let mut out = String::new();
    let mut rest = t;
    while let Some(p) = rest.find('&') {
        out.push_str(&rest[..p]);
        rest = &rest[p..];
        let end = rest[1..].find(|c: char| !(c.is_ascii_alphanumeric() || c == '#')).map(|e| e + 1).unwrap_or(rest.len());
        let name = &rest[1..end];
        let named = match name {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            "nbsp" => Some(' '),
            "mdash" => Some('\u{2014}'),
            "ndash" => Some('\u{2013}'),
            "hellip" => Some('\u{2026}'),
            "copy" => Some('\u{a9}'),
            "reg" => Some('\u{ae}'),
            "trade" => Some('\u{2122}'),
            "lsquo" => Some('\u{2018}'),
            "rsquo" => Some('\u{2019}'),
            "ldquo" => Some('\u{201c}'),
            "rdquo" => Some('\u{201d}'),
            "bull" => Some('\u{2022}'),
            "middot" => Some('\u{b7}'),
            "euro" => Some('\u{20ac}'),
            "pound" => Some('\u{a3}'),
            "deg" => Some('\u{b0}'),
            "times" => Some('\u{d7}'),
            _ => {
                let n = if let Some(h) = name.strip_prefix("#x").or_else(|| name.strip_prefix("#X")) {
                    u32::from_str_radix(h, 16).ok()
                } else if let Some(d) = name.strip_prefix('#') {
                    d.parse().ok()
                } else {
                    None
                };
                n.and_then(char::from_u32)
            }
        };
        match named {
            Some(c) => {
                out.push(c);
                rest = &rest[end..];
                if rest.starts_with(';') {
                    rest = &rest[1..];
                }
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// The part of a page between `<name ...>` and `</name>`, from the first such
/// tag, or None.
fn element(b: &[char], name: &str) -> Option<String> {
    let mut from = 0;
    while let Some(p) = find_from(b, from, &format!("<{}", name)) {
        let (found, closing, after) = tag_at(b, p + 1);
        if found == name && !closing {
            let end = find_from(b, after, &format!("</{}", name)).unwrap_or(b.len());
            return Some(b[after..end].iter().collect());
        }
        from = p + 1;
    }
    None
}

/// The readable text of a web page: no tags, scripts, styles, menus or
/// footers; only the title and the main part when the page marks one with
/// `<main>`; a line for each paragraph, heading or list item; spaces tidied.
pub fn page_text(html: &str) -> String {
    let all: Vec<char> = html.chars().collect();
    let focused = match element(&all, "main") {
        Some(main) => format!("<title>{}</title>{}", element(&all, "title").unwrap_or_default(), main),
        None => html.to_string(),
    };
    let b: Vec<char> = focused.chars().collect();
    let mut raw = String::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] != '<' {
            let start = i;
            while i < b.len() && b[i] != '<' {
                i += 1;
            }
            let chunk: String = b[start..i].iter().collect();
            let chunk = decode_entities(&chunk);
            for c in chunk.chars() {
                raw.push(if c.is_whitespace() && c != '\u{a0}' { ' ' } else if c == '\u{a0}' { ' ' } else { c });
            }
            continue;
        }
        if b[i..].starts_with(&['<', '!', '-', '-']) {
            i = find_from(&b, i + 4, "-->").map(|e| e + 3).unwrap_or(b.len());
            continue;
        }
        if i + 1 < b.len() && !(b[i + 1].is_ascii_alphabetic() || b[i + 1] == '/' || b[i + 1] == '!') {
            raw.push('<');
            i += 1;
            continue;
        }
        let start = i;
        let (name, closing, after) = tag_at(&b, i + 1);
        i = after;
        if !closing && RAW_TEXT.contains(&name.as_str()) {
            i = find_from(&b, i, &format!("</{}", name)).map(|e| tag_at(&b, e + 1).2).unwrap_or(b.len());
            continue;
        }
        if !closing && !VOID.contains(&name.as_str()) && !(after >= 2 && b[after - 2] == '/') {
            let tag: String = b[start..after].iter().collect();
            if SKIPPED.contains(&name.as_str()) || clutter(&tag) {
                i = skip_element(&b, after, &name);
                continue;
            }
        }
        if name == "li" && !closing {
            raw.push_str("\n- ");
        } else if BLOCKS.contains(&name.as_str()) || name == "li" {
            raw.push('\n');
        } else if name == "td" || name == "th" {
            raw.push(' ');
        }
    }
    // tidy: single spaces, no empty runs of lines
    let mut out = String::new();
    let mut blank = false;
    for line in raw.split('\n') {
        let words: Vec<&str> = line.split(' ').filter(|w| !w.is_empty()).collect();
        if words.is_empty() || words == ["-"] {
            if !out.is_empty() {
                blank = true;
            }
            continue;
        }
        if !out.is_empty() {
            out.push('\n');
            if blank && !words[0].starts_with('-') {
                out.push('\n');
            }
        }
        blank = false;
        out.push_str(&words.join(" "));
    }
    out
}

fn attr(tag: &str, name: &str) -> Option<String> {
    let lower = tag.to_ascii_lowercase();
    let mut from = 0;
    while let Some(p) = lower[from..].find(name).map(|p| p + from) {
        let before_ok = p == 0 || lower.as_bytes()[p - 1].is_ascii_whitespace();
        let rest = lower[p + name.len()..].trim_start();
        if before_ok && rest.starts_with('=') {
            let eq = tag.len() - rest.len();
            let value = tag[eq + 1..].trim_start();
            let v = match value.chars().next() {
                Some(q @ ('"' | '\'')) => value[1..].split(q).next().unwrap_or(""),
                _ => value.split(|c: char| c.is_whitespace() || c == '>').next().unwrap_or(""),
            };
            return Some(decode_entities(v.trim()));
        }
        from = p + name.len();
    }
    None
}

/// The links on a page as (text, address), without repeats.
pub fn page_links(html: &str, base: Option<&str>) -> Vec<(String, String)> {
    let b: Vec<char> = html.chars().collect();
    let mut links: Vec<(String, String)> = vec![];
    let mut i = 0;
    while let Some(p) = find_from(&b, i, "<a") {
        let (name, _, after) = tag_at(&b, p + 1);
        i = after;
        if name != "a" {
            continue;
        }
        let tag: String = b[p..after].iter().collect();
        let Some(href) = attr(&tag, "href") else { continue };
        let lower = href.to_ascii_lowercase();
        if href.is_empty() || href.starts_with('#') || lower.starts_with("javascript:") || lower.starts_with("mailto:") || lower.starts_with("tel:") {
            continue;
        }
        let end = find_from(&b, after, "</a").unwrap_or(b.len());
        let inner: String = b[after..end].iter().collect();
        let label = page_text(&inner).split_whitespace().collect::<Vec<_>>().join(" ");
        let url = match base {
            Some(base) => resolve(base, &href),
            None => href,
        };
        if !links.iter().any(|(_, u)| *u == url) {
            links.push((label, url));
        }
    }
    links
}

/// A link's full address, given the address of the page it is on.
pub fn resolve(base: &str, href: &str) -> String {
    let has_scheme = href.find(':').is_some_and(|c| href[..c].chars().all(|ch| ch.is_ascii_alphanumeric() || "+-.".contains(ch)) && c > 0);
    if has_scheme {
        return href.to_string();
    }
    let scheme_end = match base.find("://") {
        Some(e) => e,
        None => return href.to_string(),
    };
    if let Some(rest) = href.strip_prefix("//") {
        return format!("{}://{}", &base[..scheme_end], rest);
    }
    let after_scheme = scheme_end + 3;
    let host_end = base[after_scheme..].find(['/', '?', '#']).map(|e| e + after_scheme).unwrap_or(base.len());
    let origin = &base[..host_end];
    let path = &base[host_end..];
    let path = path.split(['?', '#']).next().unwrap_or("");
    if href.starts_with('?') {
        return format!("{}{}{}", origin, if path.is_empty() { "/" } else { path }, href);
    }
    let joined = if href.starts_with('/') {
        href.to_string()
    } else {
        let dir = match path.rfind('/') {
            Some(p) => &path[..=p],
            None => "/",
        };
        format!("{}{}", dir, href)
    };
    // tidy "./" and "../"
    let (path_part, tail) = match joined.find(['?', '#']) {
        Some(p) => (&joined[..p], &joined[p..]),
        None => (&joined[..], ""),
    };
    let mut parts: Vec<&str> = vec![];
    let segs: Vec<&str> = path_part.split('/').collect();
    for (k, seg) in segs.iter().enumerate() {
        match *seg {
            "." => {
                if k == segs.len() - 1 {
                    parts.push("");
                }
            }
            ".." => {
                if parts.len() > 1 {
                    parts.pop();
                }
                if k == segs.len() - 1 {
                    parts.push("");
                }
            }
            s => parts.push(s),
        }
    }
    let mut p = parts.join("/");
    if !p.starts_with('/') {
        p.insert(0, '/');
    }
    format!("{}{}{}", origin, p, tail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_text_keeps_the_words_and_drops_the_markup() {
        let html = "<html><head><title>Moons</title><style>p{color:red}</style><script>var x = '<p>';</script></head>
<body><h1>The  Moon</h1><!-- a comment --><p>Earth&#39;s moon is &lt;big&gt; &amp; bright.</p>
<ul><li>Phobos</li><li>Deimos</li></ul><p>a&nbsp;b &#x2014; c</p></body></html>";
        assert_eq!(page_text(html), "Moons\n\nThe Moon\n\nEarth's moon is <big> & bright.\n- Phobos\n- Deimos\n\na b \u{2014} c");
    }

    #[test]
    fn page_text_keeps_to_the_main_part_when_there_is_one() {
        let html = "<title>A &amp; B</title><nav><a href=/>Home</a></nav><header>Site</header>
<main id=content><div class=\"lang-dropdown\"><ul><li>English</li><div><li>Sinhala</li></div></ul></div><h1>Story</h1>
<div aria-hidden=true>secret</div><p>Text.</p><aside>Ad</aside><form><button>Go</button></form><img src=a.png></main><footer>(c)</footer>";
        assert_eq!(page_text(html), "A & B

Story

Text.");
    }

    #[test]
    fn links_are_found_and_made_full() {
        let html = r##"<a href="/wiki/Moon">The <b>Moon</b></a> <a class=x href='b.html'>B</a> <A HREF="https://x.org/">X</A>
            <a href="#top">top</a> <a href="mailto:a@b.c">mail</a> <a href="/wiki/Moon">again</a> <a name="n">no link</a>"##;
        let links = page_links(html, Some("https://en.wikipedia.org/wiki/Mars?x=1"));
        assert_eq!(
            links,
            vec![
                ("The Moon".to_string(), "https://en.wikipedia.org/wiki/Moon".to_string()),
                ("B".to_string(), "https://en.wikipedia.org/wiki/b.html".to_string()),
                ("X".to_string(), "https://x.org/".to_string()),
            ]
        );
    }

    #[test]
    fn addresses_resolve_like_a_browser() {
        let base = "https://a.com/docs/guide/page.html?q=1";
        assert_eq!(resolve(base, "other.html"), "https://a.com/docs/guide/other.html");
        assert_eq!(resolve(base, "../up.html"), "https://a.com/docs/up.html");
        assert_eq!(resolve(base, "./"), "https://a.com/docs/guide/");
        assert_eq!(resolve(base, "/root"), "https://a.com/root");
        assert_eq!(resolve(base, "//cdn.com/x.js"), "https://cdn.com/x.js");
        assert_eq!(resolve(base, "?page=2"), "https://a.com/docs/guide/page.html?page=2");
        assert_eq!(resolve("https://a.com", "x"), "https://a.com/x");
        assert_eq!(resolve(base, "http://b.com/"), "http://b.com/");
    }

    #[test]
    fn url_encoding_escapes_all_but_the_safe_characters() {
        assert_eq!(url_encode("moon landing & 1969/7"), "moon%20landing%20%26%201969%2F7");
        assert_eq!(url_encode("café"), "caf%C3%A9");
    }

    #[test]
    fn shell_runs_commands_and_reports_codes() {
        let (out, _, code) = run_command("echo hello", None, None, 30.0).unwrap();
        assert_eq!(out.trim(), "hello");
        assert_eq!(code, 0);
        let (_, _, code) = run_command("exit 3", None, None, 30.0).unwrap();
        assert_eq!(code, 3);
        let cat = if cfg!(windows) { "findstr x" } else { "grep x" };
        let (out, _, _) = run_command(cat, None, Some("axe\nbee\nfox\n"), 30.0).unwrap();
        assert_eq!(out.replace('\r', "").trim(), "axe\nfox");
    }

    #[test]
    fn a_timeout_stops_what_the_command_started() {
        // a program started by the command writes a file after 3 seconds,
        // unless the timeout stops it too
        let file = std::env::temp_dir().join(format!("antares_tree_{}.txt", std::process::id()));
        let _ = std::fs::remove_file(&file);
        let f = file.to_string_lossy().to_string();
        let command = if cfg!(windows) {
            format!("cmd /C \"ping -n 4 127.0.0.1 >nul & echo late> \"{}\"\"", f)
        } else {
            format!("(sleep 3; echo late > '{}') & wait", f)
        };
        let (_, errors, code) = run_command(&command, None, None, 0.5).unwrap();
        assert_eq!(code, -1);
        assert!(errors.contains("stopped after 0.5 seconds"));
        std::thread::sleep(Duration::from_secs(5));
        assert!(!file.exists(), "a program the command started kept running after the timeout");
    }
}
