//! Talking to language models: Anthropic's Claude, OpenAI (and services
//! that copy its interface, through the `url` option), and Ollama for models
//! running on your own computer. Conversations, tools (the model calls your
//! Antares functions), answers in a shape you choose, images, streaming and
//! embeddings.
//!
//! No HTTP library is linked in: requests go through the `curl` program that
//! ships with Windows, macOS and Linux, which keeps the Antares binary tiny.

use crate::json::{self, J};
use crate::value::Value;
use std::cell::RefCell;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::rc::Rc;

/// An Antares function a model may call.
#[derive(Clone)]
pub struct Tool {
    pub name: String,
    pub about: String,
    /// Each input's name, and its kind ("number", "text", "yes/no",
    /// "list") when it was given.
    pub inputs: Vec<(String, Option<String>)>,
    pub run: Value,
    /// Whether the user is asked before each call.
    pub confirm: bool,
}

/// The built-in functions that can be tools by themselves: (name, inputs,
/// description, ask the user first?).
const BUILTIN_TOOLS: [(&str, &[&str], &str, bool); 6] = [
    ("shell", &["command"], "Runs a command in the computer's shell (cmd on Windows, sh elsewhere) and gives its output, errors and exit code", true),
    ("read_page", &["url"], "Fetches a web page and gives its readable text", false),
    ("fetch", &["url"], "Fetches a web address and gives back exactly what it sends", false),
    ("read", &["file"], "Reads a text file", false),
    ("write", &["file", "text"], "Writes text to a file, replacing anything in it", true),
    ("files", &["folder"], "The names of the files and folders in a folder", false),
];

#[derive(Clone)]
pub struct ModelSpec {
    pub provider: String,
    pub name: String,
    pub max_tokens: Option<u64>,
    pub system: Option<String>,
    pub effort: Option<String>,
    pub stream: bool,
    pub url: Option<String>,
    pub tools: Vec<Tool>,
    /// For a conversation: the messages so far, in the provider's own form.
    pub history: Option<RefCell<Vec<J>>>,
}

const PROVIDERS: &str = "anthropic, openai, ollama, echo";

fn tool_of(v: &Value) -> Result<Tool, String> {
    let kinds = ["number", "text", "yes/no", "list"];
    match v {
        Value::Fn(c) => Ok(Tool {
            name: c.def.name.to_string(),
            about: String::new(),
            inputs: c.def.params.iter().map(|p| (p.to_string(), None)).collect(),
            run: v.clone(),
            confirm: false,
        }),
        Value::Rust(r) => Ok(Tool { name: r.name.clone(), about: String::new(), inputs: r.params.iter().map(|p| (p.clone(), None)).collect(), run: v.clone(), confirm: false }),
        Value::Map(m) => {
            let get = |k: &str| m.iter().find(|(n, _)| &**n == k).map(|(_, v)| v);
            let run = get("run").ok_or("a tool described with a map needs `run`: the function to call, as in {run: weather, about: \"...\"}")?;
            let mut t = tool_of(run)?;
            if let Some(Value::Str(n)) = get("name") {
                t.name = n.to_string();
            }
            if let Some(a) = get("about") {
                t.about = a.show();
            }
            match get("confirm") {
                Some(Value::Bool(c)) => t.confirm = *c,
                Some(_) => return Err("a tool's `confirm` is yes or no".into()),
                None => {}
            }
            if let Some(Value::Map(ins)) = get("inputs") {
                let mut named = vec![];
                for (k, kind) in ins.iter() {
                    let kind = kind.show();
                    if !kinds.contains(&kind.as_str()) {
                        return Err(format!("the tool input '{}' is described as \"{}\"; inputs are \"number\", \"text\", \"yes/no\" or \"list\"", k, kind));
                    }
                    named.push((k.to_string(), Some(kind)));
                }
                if named.len() != t.inputs.len() {
                    return Err(format!("the tool '{}' takes {} input(s), but `inputs` describes {}", t.name, t.inputs.len(), named.len()));
                }
                t.inputs = named;
            }
            Ok(t)
        }
        Value::Native(n, _) => match BUILTIN_TOOLS.iter().find(|t| t.0 == *n) {
            Some((name, inputs, about, confirm)) => Ok(Tool {
                name: name.to_string(),
                about: about.to_string(),
                inputs: inputs.iter().map(|p| (p.to_string(), Some("text".to_string()))).collect(),
                run: v.clone(),
                confirm: *confirm,
            }),
            None => Err(format!(
                "the built-in function '{}' can't be a tool by itself; wrap it in a function of your own (the built-ins that can: {})",
                n,
                BUILTIN_TOOLS.iter().map(|t| t.0).collect::<Vec<_>>().join(", ")
            )),
        },
        other => Err(format!("a tool is a function, or a map like {{run: weather, about: \"...\"}}, not a {}", other.kind())),
    }
}

pub fn make_spec(spec: &Value, opts: Option<&Value>) -> Result<ModelSpec, String> {
    let s = match spec {
        Value::Str(s) => s.to_string(),
        other => return Err(format!("a model is described with text like \"anthropic:claude-opus-5-5\", not a {}", other.kind())),
    };
    let (provider, name) = match s.split_once(':') {
        Some((p, n)) => (p.trim().to_string(), n.trim().to_string()),
        None if s == "echo" => ("echo".to_string(), "echo".to_string()),
        None if s.starts_with("claude") => ("anthropic".to_string(), s.clone()),
        None => {
            return Err(format!(
                "I don't know the model \"{}\". Write it with its provider, like \"anthropic:claude-opus-5-5\", \"openai:gpt-5\" or \"ollama:llama3.2\", or \"echo\" for a pretend model that repeats the prompt",
                s
            ))
        }
    };
    if !PROVIDERS.split(", ").any(|p| p == provider) {
        return Err(format!("I don't know the model provider \"{}\". Available: {}", provider, PROVIDERS));
    }
    let mut m = ModelSpec { provider, name, max_tokens: None, system: None, effort: None, stream: false, url: None, tools: vec![], history: None };
    if let Some(o) = opts {
        let Value::Map(entries) = o else {
            return Err("the options after `with` must be a map, like {system: \"Be brief\"}".into());
        };
        for (k, v) in entries.iter() {
            match (&**k, v) {
                ("system", Value::Str(s)) => m.system = Some(s.to_string()),
                ("max_tokens", v) if v.scalar().is_some() => m.max_tokens = Some(v.scalar().unwrap().max(1.0) as u64),
                ("effort", Value::Str(e)) => {
                    if !["low", "medium", "high", "xhigh", "max"].contains(&&**e) {
                        return Err(format!("effort is \"low\", \"medium\", \"high\", \"xhigh\" or \"max\", not \"{}\"", e));
                    }
                    if m.provider != "anthropic" && m.provider != "echo" {
                        return Err("the effort option works with Anthropic models".into());
                    }
                    m.effort = Some(e.to_string());
                }
                ("stream", Value::Bool(b)) => m.stream = *b,
                ("url", Value::Str(u)) => m.url = Some(u.trim_end_matches('/').to_string()),
                ("tools", Value::List(ts)) => {
                    m.tools = ts.iter().map(tool_of).collect::<Result<_, _>>()?;
                    let mut seen = std::collections::HashSet::new();
                    for t in &m.tools {
                        let ok = !t.name.is_empty() && t.name.len() <= 64 && t.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
                        if !ok {
                            return Err(format!("the tool name '{}' must be letters, digits, _ or -; give it a name with {{run: f, name: \"...\"}}", t.name));
                        }
                        if !seen.insert(t.name.clone()) {
                            return Err(format!("two tools are called '{}'; give one another name with {{run: f, name: \"...\"}}", t.name));
                        }
                    }
                }
                (other, _) => {
                    return Err(format!(
                        "I don't know the model option '{}' (or it has the wrong kind of value). Available: system (text), max_tokens (number), effort (text), stream (yes/no), url (text), tools (a list)",
                        other
                    ))
                }
            }
        }
    }
    Ok(m)
}

/// A conversation: the same model, remembering what was said. Made from a
/// conversation, it starts with a copy of that one's history.
pub fn conversation(m: &ModelSpec) -> ModelSpec {
    let mut c = m.clone();
    let start = m.history.as_ref().map(|h| h.borrow().clone()).unwrap_or_default();
    c.history = Some(RefCell::new(start));
    c
}

/// What was said in a conversation: (role, text) for each message.
pub fn history(m: &ModelSpec) -> Option<Vec<(String, String)>> {
    let h = m.history.as_ref()?.borrow();
    let mut out = vec![];
    for msg in h.iter() {
        let role = msg.get("role").and_then(J::str).unwrap_or("").to_string();
        let text = match msg.get("content") {
            Some(J::Str(s)) => s.clone(),
            Some(J::Arr(blocks)) => blocks
                .iter()
                .filter_map(|b| match b.get("type").and_then(J::str) {
                    Some("text") => b.get("text").and_then(J::str).map(str::to_string),
                    Some("tool_use") => Some(format!("[uses the tool {}]", b.get("name").and_then(J::str).unwrap_or("?"))),
                    Some("tool_result") => Some(format!("[tool result] {}", tool_result_text(b))),
                    Some("image") | Some("image_url") => Some("[an image]".into()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        // OpenAI-style tool calls and results, written the same way as Anthropic's
        let mut text = if role == "tool" { format!("[tool result] {}", text) } else { text };
        if let Some(J::Arr(calls)) = msg.get("tool_calls") {
            for c in calls {
                let name = c.get("function").and_then(|f| f.get("name")).and_then(J::str).unwrap_or("?");
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&format!("[uses the tool {}]", name));
            }
        }
        if !text.is_empty() || role == "assistant" {
            out.push((role, text));
        }
    }
    Some(out)
}

fn tool_result_text(b: &J) -> String {
    match b.get("content") {
        Some(J::Str(s)) => s.clone(),
        _ => String::new(),
    }
}

// ---------- prompts: text and images ----------

pub enum Part {
    Text(String),
    Image { media: String, data: String },
}

/// image(file): a picture to put in a prompt, as in ask m ["What is this?", image("cat.png")].
pub fn image_value(file: &str) -> Result<Value, String> {
    let bytes = match crate::builtins::embedded_file(file) {
        Some(Some(b)) => b.to_vec(),
        Some(None) => return Err(format!("image: no file was embedded as \"{}\"", file)),
        None => std::fs::read(file).map_err(|e| format!("image couldn't read \"{}\": {}", file, e))?,
    };
    let lower = file.to_ascii_lowercase();
    let media = if lower.ends_with(".png") {
        "image/png"
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        "image/jpeg"
    } else if lower.ends_with(".gif") {
        "image/gif"
    } else if lower.ends_with(".webp") {
        "image/webp"
    } else {
        return Err(format!("image works with .png, .jpg, .gif and .webp files, not \"{}\"", file));
    };
    Ok(Value::Map(Rc::new(vec![
        ("image".into(), Value::Str(file.into())),
        ("media".into(), Value::Str(media.into())),
        ("data".into(), Value::Str(base64(&bytes).into())),
    ])))
}

fn base64(b: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(b.len().div_ceil(3) * 4);
    for c in b.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

fn image_part(v: &Value) -> Option<Part> {
    let Value::Map(m) = v else { return None };
    let get = |k: &str| m.iter().find(|(n, _)| &**n == k).map(|(_, v)| v);
    match (get("image"), get("media"), get("data")) {
        (Some(_), Some(Value::Str(media)), Some(Value::Str(data))) => Some(Part::Image { media: media.to_string(), data: data.to_string() }),
        _ => None,
    }
}

/// A prompt as parts: text, or a list of texts and images.
pub fn parts_of(prompt: &Value) -> Vec<Part> {
    match prompt {
        Value::List(items) => items
            .iter()
            .flat_map(|v| match v {
                // a list inside the prompt list, such as several images after `with`
                Value::List(inner) => inner.iter().map(|x| image_part(x).unwrap_or_else(|| Part::Text(x.show()))).collect::<Vec<_>>(),
                v => vec![image_part(v).unwrap_or_else(|| Part::Text(v.show()))],
            })
            .collect(),
        v => vec![image_part(v).unwrap_or_else(|| Part::Text(v.show()))],
    }
}

// ---------- shapes: answers as values ----------

/// The shape of an answer asked for with `as`.
#[derive(Clone, PartialEq, Debug)]
pub enum Shape {
    Text,
    Number,
    YesNo,
    List(Box<Shape>),
    Map(Vec<(String, Shape)>),
}

impl Shape {
    /// How it is written after `as` (and how compiled programs carry it).
    pub fn spec(&self) -> String {
        match self {
            Shape::Text => "text".into(),
            Shape::Number => "number".into(),
            Shape::YesNo => "yes/no".into(),
            Shape::List(s) => format!("list of {}", s.spec()),
            Shape::Map(fields) => format!("{{{}}}", fields.iter().map(|(k, s)| format!("{}: {}", k, s.spec())).collect::<Vec<_>>().join(", ")),
        }
    }

    pub fn from_spec(s: &str) -> Option<Shape> {
        let (shape, rest) = Self::read(s.trim())?;
        rest.trim().is_empty().then_some(shape)
    }

    fn read(s: &str) -> Option<(Shape, &str)> {
        let s = s.trim_start();
        if let Some(r) = s.strip_prefix("text") {
            return Some((Shape::Text, r));
        }
        if let Some(r) = s.strip_prefix("number") {
            return Some((Shape::Number, r));
        }
        if let Some(r) = s.strip_prefix("yes/no") {
            return Some((Shape::YesNo, r));
        }
        if let Some(r) = s.strip_prefix("list of ") {
            let (inner, r) = Self::read(r)?;
            return Some((Shape::List(Box::new(inner)), r));
        }
        let mut r = s.strip_prefix('{')?;
        let mut fields = vec![];
        loop {
            r = r.trim_start();
            if let Some(after) = r.strip_prefix('}') {
                return Some((Shape::Map(fields), after));
            }
            let colon = r.find(':')?;
            let key = r[..colon].trim().to_string();
            let (shape, after) = Self::read(&r[colon + 1..])?;
            fields.push((key, shape));
            r = after.trim_start();
            r = r.strip_prefix(',').unwrap_or(r);
        }
    }

    /// The JSON schema for it, for structured answers.
    pub fn schema(&self) -> J {
        let s = |t: &str| J::Obj(vec![("type".into(), J::Str(t.into()))]);
        match self {
            Shape::Text => s("string"),
            Shape::Number => s("number"),
            Shape::YesNo => s("boolean"),
            Shape::List(inner) => J::Obj(vec![("type".into(), J::Str("array".into())), ("items".into(), inner.schema())]),
            Shape::Map(fields) => object_schema(fields.iter().map(|(k, s)| (k.clone(), s.schema())).collect()),
        }
    }

    /// A made-up answer of this shape (for the pretend echo model).
    fn example(&self, prompt: &str) -> J {
        match self {
            Shape::Text => J::Str(prompt.into()),
            Shape::Number => J::Num(0.0),
            Shape::YesNo => J::Bool(false),
            Shape::List(_) => J::Arr(vec![]),
            Shape::Map(fields) => J::Obj(fields.iter().map(|(k, s)| (k.clone(), s.example(prompt))).collect()),
        }
    }

    /// Checks a structured answer against the shape.
    fn check(&self, j: &J) -> bool {
        match (self, j) {
            (Shape::Text, J::Str(_)) | (Shape::Number, J::Num(_)) | (Shape::YesNo, J::Bool(_)) => true,
            (Shape::List(inner), J::Arr(items)) => items.iter().all(|i| inner.check(i)),
            (Shape::Map(fields), J::Obj(_)) => fields.iter().all(|(k, s)| j.get(k).is_some_and(|v| s.check(v))),
            _ => false,
        }
    }
}

fn object_schema(props: Vec<(String, J)>) -> J {
    let required = J::Arr(props.iter().map(|(k, _)| J::Str(k.clone())).collect());
    J::Obj(vec![
        ("type".into(), J::Str("object".into())),
        ("properties".into(), J::Obj(props)),
        ("required".into(), required),
        ("additionalProperties".into(), J::Bool(false)),
    ])
}

// ---------- asking ----------

/// Asks a model. `want` asks for an answer of that shape (structured output);
/// `suffix` is added to the prompt; `run_tool(i, input)` calls tool `i` with
/// the model's input and gives back the result as text (or a problem).
pub fn ask(
    m: &ModelSpec,
    prompt: &[Part],
    suffix: &str,
    want: Option<&Shape>,
    run_tool: &mut dyn FnMut(usize, &J) -> Result<String, String>,
) -> Result<J, String> {
    let mut parts: Vec<&Part> = prompt.iter().collect();
    let extra = Part::Text(suffix.to_string());
    if !suffix.is_empty() {
        parts.push(&extra);
    }
    let mut messages = m.history.as_ref().map(|h| h.borrow().clone()).unwrap_or_default();
    let start = messages.len();
    let answer = match m.provider.as_str() {
        "echo" => {
            let text: String = prompt.iter().filter_map(|p| if let Part::Text(t) = p { Some(t.as_str()) } else { None }).collect::<Vec<_>>().join(" ");
            messages.push(anthropic_user(&parts));
            let reply = match want {
                Some(s) => s.example(&text),
                None => J::Str(text),
            };
            let shown = match &reply {
                J::Str(s) => s.clone(),
                other => other.dump(),
            };
            if m.stream {
                println!("{}", shown);
            }
            messages.push(J::Obj(vec![("role".into(), J::Str("assistant".into())), ("content".into(), J::Str(shown))]));
            reply
        }
        "anthropic" => {
            messages.push(anthropic_user(&parts));
            let text = anthropic_turns(m, &mut messages, want, run_tool)?;
            structured(text, want)?
        }
        _ => {
            messages.push(openai_user(&parts));
            let text = openai_turns(m, &mut messages, want, run_tool)?;
            structured(text, want)?
        }
    };
    if let Some(h) = &m.history {
        h.borrow_mut().extend(messages.drain(start..));
    }
    Ok(answer)
}

/// The answer: plain text, or for a shape, the value inside the JSON reply.
fn structured(text: String, want: Option<&Shape>) -> Result<J, String> {
    let Some(shape) = want else { return Ok(J::Str(text)) };
    let j = json::parse(text.trim()).map_err(|_| format!("I asked for {} but the model said: {}", shape.spec(), text.trim()))?;
    let answer = match j {
        J::Obj(mut fields) if fields.len() == 1 && fields[0].0 == "answer" => fields.remove(0).1,
        other => other,
    };
    if !shape.check(&answer) {
        return Err(format!("I asked for {} but the model said: {}", shape.spec(), text.trim()));
    }
    Ok(answer)
}

/// Structured answers are asked for as {"answer": ...}: the outside of an
/// answer's shape must be a map.
fn answer_schema(s: &Shape) -> J {
    object_schema(vec![("answer".into(), s.schema())])
}

fn tool_schema(t: &Tool) -> J {
    let props = t
        .inputs
        .iter()
        .map(|(name, kind)| {
            let s = match kind.as_deref() {
                Some("number") => J::Obj(vec![("type".into(), J::Str("number".into()))]),
                Some("text") => J::Obj(vec![("type".into(), J::Str("string".into()))]),
                Some("yes/no") => J::Obj(vec![("type".into(), J::Str("boolean".into()))]),
                Some("list") => J::Obj(vec![("type".into(), J::Str("array".into()))]),
                _ => J::Obj(vec![("description".into(), J::Str("a number, text, yes/no or list".into()))]),
            };
            (name.clone(), s)
        })
        .collect::<Vec<_>>();
    let required = J::Arr(t.inputs.iter().map(|(n, _)| J::Str(n.clone())).collect());
    J::Obj(vec![("type".into(), J::Str("object".into())), ("properties".into(), J::Obj(props)), ("required".into(), required)])
}

fn tool_about(t: &Tool) -> String {
    if t.about.is_empty() {
        let names: Vec<&str> = t.inputs.iter().map(|(n, _)| n.as_str()).collect();
        format!("Runs the function {}({}) and gives back its result.", t.name, names.join(", "))
    } else {
        t.about.clone()
    }
}

fn obj(fields: Vec<(&str, J)>) -> J {
    J::Obj(fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn s(x: &str) -> J {
    J::Str(x.into())
}

const MAX_TOOL_ROUNDS: usize = 25;

// ---------- Anthropic ----------

fn anthropic_user(parts: &[&Part]) -> J {
    let only_text = parts.iter().all(|p| matches!(p, Part::Text(_)));
    let content = if only_text {
        J::Str(parts.iter().filter_map(|p| if let Part::Text(t) = p { Some(t.as_str()) } else { None }).collect::<Vec<_>>().join("\n"))
    } else {
        J::Arr(
            parts
                .iter()
                .map(|p| match p {
                    Part::Text(t) => obj(vec![("type", s("text")), ("text", s(t))]),
                    Part::Image { media, data } => obj(vec![
                        ("type", s("image")),
                        ("source", obj(vec![("type", s("base64")), ("media_type", s(media)), ("data", s(data))])),
                    ]),
                })
                .collect(),
        )
    };
    obj(vec![("role", s("user")), ("content", content)])
}

/// Models that accept Anthropic's server-side refusal fallback.
fn supports_fallback(name: &str) -> bool {
    ["claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-fable-5-2"].iter().any(|p| name.starts_with(p))
}

fn anthropic_turns(m: &ModelSpec, messages: &mut Vec<J>, want: Option<&Shape>, run_tool: &mut dyn FnMut(usize, &J) -> Result<String, String>) -> Result<String, String> {
    let key = std::env::var("ANTHROPIC_API_KEY").ok().filter(|k| !k.trim().is_empty());
    if key.is_none() && m.url.is_none() {
        return Err("to ask a Claude model, set the ANTHROPIC_API_KEY environment variable to your API key first".into());
    }
    let fallback = supports_fallback(&m.name);
    let url = format!("{}/v1/messages", m.url.as_deref().unwrap_or("https://api.anthropic.com"));
    let mut headers = vec!["anthropic-version: 2026-10-21".to_string(), "content-type: application/json".to_string()];
    if let Some(k) = &key {
        headers.push(format!("x-api-key: {}", k.trim()));
    }
    if fallback {
        headers.push("anthropic-beta: server-side-fallback-2026-07-01".into());
    }
    for _ in 0..MAX_TOOL_ROUNDS {
        let mut body = vec![
            ("model", s(&m.name)),
            ("max_tokens", J::Num(m.max_tokens.unwrap_or(if m.stream { 64000 } else { 16000 }) as f64)),
            ("messages", J::Arr(messages.clone())),
        ];
        if let Some(sys) = &m.system {
            body.push(("system", s(sys)));
        }
        if !m.tools.is_empty() {
            body.push((
                "tools",
                J::Arr(m.tools.iter().map(|t| obj(vec![("name", s(&t.name)), ("description", s(&tool_about(t))), ("input_schema", tool_schema(t))])).collect()),
            ));
        }
        let mut config = vec![];
        if let Some(e) = &m.effort {
            config.push(("effort", s(e)));
        }
        if let Some(shape) = want {
            config.push(("format", obj(vec![("type", s("json_schema")), ("schema", answer_schema(shape))])));
        }
        if !config.is_empty() {
            body.push(("output_config", obj(config)));
        }
        if fallback {
            body.push(("fallbacks", s("default")));
        }
        if m.stream {
            body.push(("stream", J::Bool(true)));
        }
        // a structured answer streams too, but only plain text is shown as it arrives
        let reply = if m.stream { anthropic_stream(&url, &headers, &obj(body).dump(), want.is_none())? } else { post(&url, &headers, &obj(body).dump(), "anthropic")? };
        if let Some(e) = reply.get("error") {
            return Err(format!("the model service reported: {}", e.get("message").and_then(J::str).unwrap_or("unknown error")));
        }
        let stop = reply.get("stop_reason").and_then(J::str).unwrap_or("");
        if stop == "refusal" {
            return Err("the model declined to answer this request".into());
        }
        let content = match reply.get("content") {
            Some(J::Arr(c)) => c.clone(),
            _ => return Err(format!("the model service sent back something unexpected: {}", short(&reply.dump()))),
        };
        let text: String = content.iter().filter(|b| b.get("type").and_then(J::str) == Some("text")).filter_map(|b| b.get("text").and_then(J::str)).collect();
        messages.push(obj(vec![("role", s("assistant")), ("content", J::Arr(content.clone()))]));
        let calls: Vec<&J> = content.iter().filter(|b| b.get("type").and_then(J::str) == Some("tool_use")).collect();
        if stop == "tool_use" && !calls.is_empty() {
            let mut results = vec![];
            for c in calls {
                let id = c.get("id").and_then(J::str).unwrap_or("");
                let name = c.get("name").and_then(J::str).unwrap_or("");
                let input = c.get("input").cloned().unwrap_or(J::Obj(vec![]));
                let (out, failed) = call_tool(m, name, &input, run_tool);
                let mut r = vec![("type", s("tool_result")), ("tool_use_id", s(id)), ("content", s(&out))];
                if failed {
                    r.push(("is_error", J::Bool(true)));
                }
                results.push(obj(r));
            }
            // every result goes back together, in one message
            messages.push(obj(vec![("role", s("user")), ("content", J::Arr(results))]));
            continue;
        }
        if text.is_empty() && stop == "max_tokens" {
            return Err("the model ran out of room before answering. Give it more with: with {max_tokens: 32000}".into());
        }
        return Ok(text);
    }
    Err(format!("the model kept calling tools for {} rounds without answering", MAX_TOOL_ROUNDS))
}

/// Runs the tool the model asked for. Gives (result text, whether it failed).
fn call_tool(m: &ModelSpec, name: &str, input: &J, run_tool: &mut dyn FnMut(usize, &J) -> Result<String, String>) -> (String, bool) {
    match m.tools.iter().position(|t| t.name == name) {
        Some(i) => match run_tool(i, input) {
            Ok(out) => (out, false),
            Err(e) => (format!("Problem: {}", e), true),
        },
        None => (format!("There is no tool called {}", name), true),
    }
}

/// A streamed reply, put back together: text appears as it arrives.
fn anthropic_stream(url: &str, headers: &[String], body: &str, show: bool) -> Result<J, String> {
    let mut blocks: Vec<J> = vec![];
    let mut partial: Vec<String> = vec![];
    let mut stop = String::new();
    let mut error: Option<J> = None;
    let mut printed = false;
    stream_lines(url, headers, body, "anthropic", &mut |data| {
        let Ok(ev) = json::parse(data) else { return };
        match ev.get("type").and_then(J::str) {
            Some("content_block_start") => {
                let i = ev.get("index").and_then(num).unwrap_or(blocks.len() as f64) as usize;
                let b = ev.get("content_block").cloned().unwrap_or(J::Obj(vec![]));
                while blocks.len() <= i {
                    blocks.push(J::Null);
                    partial.push(String::new());
                }
                blocks[i] = b;
            }
            Some("content_block_delta") => {
                let i = ev.get("index").and_then(num).unwrap_or(0.0) as usize;
                let Some(d) = ev.get("delta") else { return };
                if i >= blocks.len() {
                    return;
                }
                match d.get("type").and_then(J::str) {
                    Some("text_delta") => {
                        let t = d.get("text").and_then(J::str).unwrap_or("");
                        if show {
                            print!("{}", t);
                            let _ = std::io::stdout().flush();
                            printed = true;
                        }
                        append_field(&mut blocks[i], "text", t);
                    }
                    Some("thinking_delta") => append_field(&mut blocks[i], "thinking", d.get("thinking").and_then(J::str).unwrap_or("")),
                    Some("signature_delta") => set_field(&mut blocks[i], "signature", s(d.get("signature").and_then(J::str).unwrap_or(""))),
                    Some("input_json_delta") => partial[i].push_str(d.get("partial_json").and_then(J::str).unwrap_or("")),
                    _ => {}
                }
            }
            Some("content_block_stop") => {
                let i = ev.get("index").and_then(num).unwrap_or(0.0) as usize;
                if i < blocks.len() && blocks[i].get("type").and_then(J::str) == Some("tool_use") {
                    let input = if partial[i].trim().is_empty() { J::Obj(vec![]) } else { json::parse(&partial[i]).unwrap_or(J::Obj(vec![])) };
                    set_field(&mut blocks[i], "input", input);
                }
            }
            Some("message_delta") => {
                if let Some(r) = ev.get("delta").and_then(|d| d.get("stop_reason")).and_then(J::str) {
                    stop = r.to_string();
                }
            }
            Some("error") => error = ev.get("error").cloned(),
            _ => {}
        }
    })?;
    if printed {
        println!();
    }
    let mut reply = vec![("content".to_string(), J::Arr(blocks.into_iter().filter(|b| !matches!(b, J::Null)).collect())), ("stop_reason".to_string(), J::Str(stop))];
    if let Some(e) = error {
        reply.push(("error".into(), e));
    }
    Ok(J::Obj(reply))
}

fn num(j: &J) -> Option<f64> {
    match j {
        J::Num(n) => Some(*n),
        _ => None,
    }
}

fn append_field(b: &mut J, key: &str, more: &str) {
    if let J::Obj(fields) = b {
        match fields.iter_mut().find(|(k, _)| k == key) {
            Some((_, J::Str(t))) => t.push_str(more),
            _ => fields.push((key.to_string(), J::Str(more.to_string()))),
        }
    }
}

fn set_field(b: &mut J, key: &str, v: J) {
    if let J::Obj(fields) = b {
        match fields.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = v,
            None => fields.push((key.to_string(), v)),
        }
    }
}

// ---------- OpenAI and services like it (including Ollama) ----------

fn openai_user(parts: &[&Part]) -> J {
    let only_text = parts.iter().all(|p| matches!(p, Part::Text(_)));
    let content = if only_text {
        J::Str(parts.iter().filter_map(|p| if let Part::Text(t) = p { Some(t.as_str()) } else { None }).collect::<Vec<_>>().join("\n"))
    } else {
        J::Arr(
            parts
                .iter()
                .map(|p| match p {
                    Part::Text(t) => obj(vec![("type", s("text")), ("text", s(t))]),
                    Part::Image { media, data } => obj(vec![("type", s("image_url")), ("image_url", obj(vec![("url", s(&format!("data:{};base64,{}", media, data)))]))]),
                })
                .collect(),
        )
    };
    obj(vec![("role", s("user")), ("content", content)])
}

/// The service's address and the headers it needs.
fn openai_target(m: &ModelSpec) -> Result<(String, Vec<String>), String> {
    let base = match (&m.url, m.provider.as_str()) {
        (Some(u), _) => u.clone(),
        (None, "ollama") => "http://localhost:11434/v1".to_string(),
        (None, _) => "https://api.openai.com/v1".to_string(),
    };
    let mut headers = vec!["content-type: application/json".to_string()];
    let key = std::env::var(if m.provider == "ollama" { "OLLAMA_API_KEY" } else { "OPENAI_API_KEY" }).ok().filter(|k| !k.trim().is_empty());
    match key {
        Some(k) => headers.push(format!("authorization: Bearer {}", k.trim())),
        None if m.provider == "openai" && m.url.is_none() => {
            return Err("to ask an OpenAI model, set the OPENAI_API_KEY environment variable to your API key first".into())
        }
        None => {}
    }
    Ok((base, headers))
}

fn openai_turns(m: &ModelSpec, messages: &mut Vec<J>, want: Option<&Shape>, run_tool: &mut dyn FnMut(usize, &J) -> Result<String, String>) -> Result<String, String> {
    let (base, headers) = openai_target(m)?;
    let url = format!("{}/chat/completions", base);
    for _ in 0..MAX_TOOL_ROUNDS {
        let mut all = vec![];
        if let Some(sys) = &m.system {
            all.push(obj(vec![("role", s("system")), ("content", s(sys))]));
        }
        all.extend(messages.iter().cloned());
        let mut body = vec![("model", s(&m.name)), ("messages", J::Arr(all))];
        if let Some(n) = m.max_tokens {
            body.push((if m.provider == "openai" { "max_completion_tokens" } else { "max_tokens" }, J::Num(n as f64)));
        }
        if !m.tools.is_empty() {
            body.push((
                "tools",
                J::Arr(
                    m.tools
                        .iter()
                        .map(|t| obj(vec![("type", s("function")), ("function", obj(vec![("name", s(&t.name)), ("description", s(&tool_about(t))), ("parameters", tool_schema(t))]))]))
                        .collect(),
                ),
            ));
        }
        if let Some(shape) = want {
            body.push((
                "response_format",
                obj(vec![("type", s("json_schema")), ("json_schema", obj(vec![("name", s("answer")), ("schema", answer_schema(shape)), ("strict", J::Bool(true))]))]),
            ));
        }
        if m.stream {
            body.push(("stream", J::Bool(true)));
        }
        let reply = if m.stream { openai_stream(&url, &headers, &obj(body).dump(), &m.provider, want.is_none())? } else { post(&url, &headers, &obj(body).dump(), &m.provider)? };
        if let Some(e) = reply.get("error") {
            let msg = match e {
                J::Str(t) => t.clone(),
                other => other.get("message").and_then(J::str).unwrap_or("unknown error").to_string(),
            };
            return Err(format!("the model service reported: {}", msg));
        }
        let choice = match reply.get("choices") {
            Some(J::Arr(c)) if !c.is_empty() => c[0].clone(),
            _ => return Err(format!("the model service sent back something unexpected: {}", short(&reply.dump()))),
        };
        let msg = choice.get("message").cloned().unwrap_or(J::Obj(vec![]));
        let text = msg.get("content").and_then(J::str).unwrap_or("").to_string();
        if msg.get("refusal").and_then(J::str).is_some_and(|r| !r.is_empty()) {
            return Err("the model declined to answer this request".into());
        }
        let calls = match msg.get("tool_calls") {
            Some(J::Arr(c)) if !c.is_empty() => c.clone(),
            _ => vec![],
        };
        // the reply goes into the history as the service wrote it
        let mut kept = vec![("role", s("assistant")), ("content", if text.is_empty() && !calls.is_empty() { J::Null } else { s(&text) })];
        if !calls.is_empty() {
            kept.push(("tool_calls", J::Arr(calls.clone())));
        }
        messages.push(obj(kept));
        if !calls.is_empty() {
            for c in &calls {
                let id = c.get("id").and_then(J::str).unwrap_or("");
                let f = c.get("function");
                let name = f.and_then(|f| f.get("name")).and_then(J::str).unwrap_or("");
                let args = f.and_then(|f| f.get("arguments")).and_then(J::str).unwrap_or("{}");
                let input = json::parse(if args.trim().is_empty() { "{}" } else { args }).unwrap_or(J::Obj(vec![]));
                let (out, _) = call_tool(m, name, &input, run_tool);
                messages.push(obj(vec![("role", s("tool")), ("tool_call_id", s(id)), ("content", s(&out))]));
            }
            continue;
        }
        if text.is_empty() && choice.get("finish_reason").and_then(J::str) == Some("length") {
            return Err("the model ran out of room before answering. Give it more with: with {max_tokens: 32000}".into());
        }
        return Ok(text);
    }
    Err(format!("the model kept calling tools for {} rounds without answering", MAX_TOOL_ROUNDS))
}

fn openai_stream(url: &str, headers: &[String], body: &str, provider: &str, show: bool) -> Result<J, String> {
    let mut text = String::new();
    let mut calls: Vec<(String, String, String)> = vec![]; // id, name, arguments
    let mut finish = String::new();
    let mut error: Option<J> = None;
    stream_lines(url, headers, body, provider, &mut |data| {
        if data.trim() == "[DONE]" {
            return;
        }
        let Ok(ev) = json::parse(data) else { return };
        if let Some(e) = ev.get("error") {
            error = Some(e.clone());
            return;
        }
        let Some(J::Arr(choices)) = ev.get("choices") else { return };
        let Some(c) = choices.first() else { return };
        if let Some(r) = c.get("finish_reason").and_then(J::str) {
            finish = r.to_string();
        }
        let Some(d) = c.get("delta") else { return };
        if let Some(t) = d.get("content").and_then(J::str) {
            if show {
                print!("{}", t);
                let _ = std::io::stdout().flush();
            }
            text.push_str(t);
        }
        if let Some(J::Arr(tc)) = d.get("tool_calls") {
            for call in tc {
                let i = call.get("index").and_then(num).unwrap_or(calls.len() as f64) as usize;
                while calls.len() <= i {
                    calls.push(Default::default());
                }
                if let Some(id) = call.get("id").and_then(J::str) {
                    calls[i].0 = id.to_string();
                }
                if let Some(f) = call.get("function") {
                    if let Some(n) = f.get("name").and_then(J::str) {
                        calls[i].1.push_str(n);
                    }
                    if let Some(a) = f.get("arguments").and_then(J::str) {
                        calls[i].2.push_str(a);
                    }
                }
            }
        }
    })?;
    if show && !text.is_empty() {
        println!();
    }
    if let Some(e) = error {
        return Ok(obj(vec![("error", e)]));
    }
    let mut msg = vec![("role", s("assistant")), ("content", s(&text))];
    if !calls.is_empty() {
        msg.push((
            "tool_calls",
            J::Arr(calls.iter().map(|(id, name, args)| obj(vec![("id", s(id)), ("type", s("function")), ("function", obj(vec![("name", s(name)), ("arguments", s(args))]))])).collect()),
        ));
    }
    Ok(obj(vec![("choices", J::Arr(vec![obj(vec![("message", obj(msg)), ("finish_reason", s(&finish))])]))]))
}

// ---------- embeddings ----------

/// Embeddings: one row of numbers per text, so that similar texts get
/// similar rows.
pub fn embed(m: &ModelSpec, texts: &[String]) -> Result<Vec<Vec<f64>>, String> {
    match m.provider.as_str() {
        "echo" => Ok(texts.iter().map(|t| echo_embedding(t)).collect()),
        "anthropic" => Err("Anthropic doesn't make embeddings. Use an embedding model from another provider, like model e = \"ollama:nomic-embed-text\" or \"openai:text-embedding-3-small\"".into()),
        _ => {
            let (base, headers) = openai_target(m)?;
            let body = obj(vec![("model", s(&m.name)), ("input", J::Arr(texts.iter().map(|t| s(t)).collect()))]);
            let reply = post(&format!("{}/embeddings", base), &headers, &body.dump(), &m.provider)?;
            if let Some(e) = reply.get("error") {
                let msg = match e {
                    J::Str(t) => t.clone(),
                    other => other.get("message").and_then(J::str).unwrap_or("unknown error").to_string(),
                };
                return Err(format!("the model service reported: {}", msg));
            }
            let Some(J::Arr(data)) = reply.get("data") else {
                return Err(format!("the model service sent back something unexpected: {}", short(&reply.dump())));
            };
            let mut rows: Vec<(usize, Vec<f64>)> = vec![];
            for (k, d) in data.iter().enumerate() {
                let i = d.get("index").and_then(num).map(|x| x as usize).unwrap_or(k);
                let Some(J::Arr(v)) = d.get("embedding") else { return Err("the model service sent an embedding that isn't a list of numbers".into()) };
                rows.push((i, v.iter().map(|x| num(x).unwrap_or(0.0)).collect()));
            }
            rows.sort_by_key(|r| r.0);
            if rows.len() != texts.len() {
                return Err(format!("asked for {} embeddings but got {}", texts.len(), rows.len()));
            }
            Ok(rows.into_iter().map(|r| r.1).collect())
        }
    }
}

/// The pretend echo model's embedding: counts of letter triples, hashed into
/// 64 numbers and scaled to length 1. Texts that share words come out similar.
fn echo_embedding(t: &str) -> Vec<f64> {
    let mut v = vec![0.0; 64];
    let chars: Vec<char> = format!(" {} ", t.to_lowercase()).chars().collect();
    for w in chars.windows(3) {
        let mut h: u32 = 2166136261;
        for c in w {
            h = (h ^ *c as u32).wrapping_mul(16777619);
        }
        v[(h % 64) as usize] += 1.0;
    }
    let len = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if len > 0.0 {
        v.iter_mut().for_each(|x| *x /= len);
    }
    v
}

// ---------- sending requests ----------

/// The request body goes in a temporary file and the headers (with any API
/// key) travel through curl's configuration on its standard input, so a key
/// never shows up in the list of running programs.
fn curl_config(url: &str, headers: &[String], body: &str, streaming: bool) -> Result<(String, std::path::PathBuf), String> {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let path = std::env::temp_dir().join(format!("antares_{}_{}.json", std::process::id(), nanos));
    std::fs::write(&path, body).map_err(|e| format!("couldn't prepare the request: {}", e))?;
    let quote = |x: &str| x.replace('\\', "\\\\").replace('"', "\\\"");
    let mut config = format!("url = \"{}\"\n", quote(url));
    for h in headers {
        config.push_str(&format!("header = \"{}\"\n", quote(h)));
    }
    config.push_str(&format!("data-binary = \"@{}\"\n", quote(&path.to_string_lossy().replace('\\', "/"))));
    config.push_str("silent\nshow-error\nmax-time = 600\n");
    if streaming {
        config.push_str("no-buffer\n");
    }
    Ok((config, path))
}

fn unreachable(provider: &str, detail: &str) -> String {
    let hint = if provider == "ollama" { " (is Ollama running? Start it, and get the model with: ollama pull <name>)" } else { "" };
    format!("couldn't reach the model: {}{}", detail.trim(), hint)
}

fn spawn_curl(config: &str) -> Result<std::process::Child, String> {
    let mut child = Command::new("curl")
        .args(["--config", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("couldn't start curl to reach the model: {}", e))?;
    child.stdin.take().unwrap().write_all(config.as_bytes()).map_err(|e| format!("couldn't send the request: {}", e))?;
    Ok(child)
}

fn post(url: &str, headers: &[String], body: &str, provider: &str) -> Result<J, String> {
    let (config, path) = curl_config(url, headers, body, false)?;
    let result = (|| {
        let out = spawn_curl(&config)?.wait_with_output().map_err(|e| format!("the request failed: {}", e))?;
        if !out.status.success() {
            return Err(unreachable(provider, &String::from_utf8_lossy(&out.stderr)));
        }
        Ok(String::from_utf8_lossy(&out.stdout).to_string())
    })();
    let _ = std::fs::remove_file(&path);
    let text = result?;
    json::parse(&text).map_err(|_| format!("the model service sent back something unexpected: {}", short(&text)))
}

/// Reads a streamed reply line by line and hands each `data:` line to `on`.
fn stream_lines(url: &str, headers: &[String], body: &str, provider: &str, on: &mut dyn FnMut(&str)) -> Result<(), String> {
    let (config, path) = curl_config(url, headers, body, true)?;
    let result = (|| {
        let mut child = spawn_curl(&config)?;
        let mut raw = String::new();
        let reader = BufReader::new(child.stdout.take().unwrap());
        for line in reader.lines() {
            let line = line.map_err(|e| format!("the request failed: {}", e))?;
            match line.strip_prefix("data:") {
                Some(d) => on(d.trim_start()),
                None => {
                    if !line.starts_with("event:") && !line.trim().is_empty() {
                        raw.push_str(&line);
                    }
                }
            }
        }
        let out = child.wait_with_output().map_err(|e| format!("the request failed: {}", e))?;
        if !out.status.success() {
            return Err(unreachable(provider, &String::from_utf8_lossy(&out.stderr)));
        }
        // a service that answered with an error instead of a stream
        if !raw.is_empty() {
            if let Ok(j) = json::parse(&raw) {
                if let Some(e) = j.get("error") {
                    let msg = match e {
                        J::Str(t) => t.clone(),
                        other => other.get("message").and_then(J::str).unwrap_or("unknown error").to_string(),
                    };
                    return Err(format!("the model service reported: {}", msg));
                }
            }
        }
        Ok(())
    })();
    let _ = std::fs::remove_file(&path);
    result
}

fn short(t: &str) -> String {
    t.chars().take(300).collect()
}

// ---------- the built-in functions ----------

fn model_of<'a>(v: &'a Value, name: &str) -> Result<&'a Rc<ModelSpec>, String> {
    match v {
        Value::Model(m) => Ok(m),
        other => Err(format!("{} needs a model (made with `model name = \"...\"`), not a {}", name, other.kind())),
    }
}

/// conversation(model): the model, remembering everything said to it with `ask`.
pub fn b_conversation(_: &mut crate::interp::Interp, a: Vec<Value>) -> Result<Value, String> {
    crate::builtins::want(&a, 1, "conversation")?;
    Ok(Value::Model(Rc::new(conversation(model_of(&a[0], "conversation")?))))
}

/// history(chat): what was said in a conversation, as a list of {role, text}.
pub fn b_history(_: &mut crate::interp::Interp, a: Vec<Value>) -> Result<Value, String> {
    crate::builtins::want(&a, 1, "history")?;
    let h = history(model_of(&a[0], "history")?).ok_or("history needs a conversation, made with conversation(model)")?;
    Ok(Value::List(Rc::new(
        h.into_iter()
            .map(|(role, text)| Value::Map(Rc::new(vec![("role".into(), Value::Str(role.into())), ("text".into(), Value::Str(text.into()))])))
            .collect(),
    )))
}

/// image(file): a picture to put in a prompt.
pub fn b_image(_: &mut crate::interp::Interp, a: Vec<Value>) -> Result<Value, String> {
    crate::builtins::want(&a, 1, "image")?;
    image_value(&crate::builtins::text(&a[0], "image")?)
}

/// embed(model, text): a row of numbers that stands for the meaning of the
/// text; with a list of texts, one row per text.
pub fn b_embed(_: &mut crate::interp::Interp, a: Vec<Value>) -> Result<Value, String> {
    crate::builtins::want(&a, 2, "embed")?;
    let m = model_of(&a[0], "embed")?;
    let (texts, many) = match &a[1] {
        Value::List(l) => (l.iter().map(|v| v.show()).collect::<Vec<_>>(), true),
        v => (vec![v.show()], false),
    };
    if texts.is_empty() {
        return Ok(crate::value::wrap(crate::tensor::new(vec![0, 0], vec![])));
    }
    let rows = embed(m, &texts)?;
    let width = rows[0].len();
    if rows.iter().any(|r| r.len() != width) {
        return Err("the model service sent embeddings of different lengths".into());
    }
    let shape = if many { vec![rows.len(), width] } else { vec![width] };
    Ok(crate::value::wrap(crate::tensor::new(shape, rows.concat())))
}

/// similarity(a, b): how alike two rows of numbers are (cosine similarity,
/// from -1 to 1). With a table as `b`, one score per row of the table.
pub fn b_similarity(_: &mut crate::interp::Interp, a: Vec<Value>) -> Result<Value, String> {
    crate::builtins::want(&a, 2, "similarity")?;
    let q = crate::builtins::ten(&a[0], "similarity")?;
    let t = crate::builtins::ten(&a[1], "similarity")?;
    let qd = q.vals();
    let td = t.vals();
    if q.shape.len() != 1 || qd.is_empty() {
        return Err("similarity needs a row of numbers first, like the result of embed".into());
    }
    let n = qd.len();
    if td.len() % n != 0 || t.shape.last() != Some(&n) {
        return Err(format!("similarity can't compare a row of {} numbers with shape {}", n, crate::tensor::fmt_shape(&t.shape)));
    }
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let qn = norm(&qd);
    let scores: Vec<f64> = td
        .chunks(n)
        .map(|row| {
            let d = norm(row) * qn;
            if d == 0.0 { 0.0 } else { row.iter().zip(qd.iter()).map(|(x, y)| x * y).sum::<f64>() / d }
        })
        .collect();
    if t.shape.len() == 1 {
        Ok(Value::Num(scores[0]))
    } else {
        Ok(crate::value::wrap(crate::tensor::new(vec![scores.len()], scores)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_round_trip_and_check() {
        let spec = "{name: text, age: number, tags: list of text, ok: yes/no, inner: {x: number}}";
        let s = Shape::from_spec(spec).unwrap();
        assert_eq!(s.spec(), spec);
        assert_eq!(Shape::from_spec("list of number"), Some(Shape::List(Box::new(Shape::Number))));
        assert!(Shape::from_spec("{a: banana}").is_none());
        let good = json::parse(r#"{"name":"Ada","age":36,"tags":["a"],"ok":true,"inner":{"x":1}}"#).unwrap();
        assert!(s.check(&good));
        let bad = json::parse(r#"{"name":"Ada","age":"36","tags":[],"ok":true,"inner":{"x":1}}"#).unwrap();
        assert!(!s.check(&bad));
        let schema = answer_schema(&s).dump();
        assert!(schema.contains(r#""additionalProperties":false"#) && schema.contains(r#""required":["answer"]"#));
    }

    #[test]
    fn structured_answers_are_unwrapped() {
        let s = Shape::List(Box::new(Shape::Number));
        assert!(matches!(structured(r#"{"answer":[1,2]}"#.into(), Some(&s)), Ok(J::Arr(v)) if v.len() == 2));
        assert!(structured(r#"{"answer":["x"]}"#.into(), Some(&s)).is_err());
        assert!(structured("not json".into(), Some(&s)).unwrap_err().contains("list of number"));
    }

    #[test]
    fn base64_matches_the_standard() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn echo_embeddings_are_unit_length_and_similar_for_similar_text() {
        let a = echo_embedding("the cat sat on the mat");
        let b = echo_embedding("a cat sat on a mat");
        let c = echo_embedding("quantum chromodynamics");
        let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>();
        assert!((dot(&a, &a) - 1.0).abs() < 1e-9);
        assert!(dot(&a, &b) > dot(&a, &c));
    }
}
