//! Talking to language models. No HTTP library is linked in: requests go
//! through the `curl` program that ships with Windows, macOS and Linux, which
//! keeps the Antares binary tiny.

use crate::json::{self, J};
use crate::value::Value;
use std::io::Write;
use std::process::{Command, Stdio};

pub struct ModelSpec {
    pub provider: String,
    pub name: String,
    pub max_tokens: u64,
    pub system: Option<String>,
}

pub fn make_spec(spec: &Value, opts: Option<&Value>) -> Result<ModelSpec, String> {
    let s = match spec {
        Value::Str(s) => s.to_string(),
        other => {
            return Err(format!(
                "a model is described with text like \"anthropic:claude-opus-5-5\", not a {}",
                other.kind()
            ))
        }
    };
    let (provider, name) = match s.split_once(':') {
        Some((p, n)) => (p.trim().to_string(), n.trim().to_string()),
        None if s == "echo" => ("echo".to_string(), "echo".to_string()),
        None if s.starts_with("claude") => ("anthropic".to_string(), s.clone()),
        None => {
            return Err(format!(
                "I don't know the model \"{}\". Write it as \"anthropic:claude-opus-5-5\", or \"echo\" for a pretend model that repeats the prompt",
                s
            ))
        }
    };
    if provider != "anthropic" && provider != "echo" {
        return Err(format!(
            "I don't know the model provider \"{}\". Available: anthropic, echo",
            provider
        ));
    }
    let mut m = ModelSpec { provider, name, max_tokens: 16000, system: None };
    if let Some(o) = opts {
        let entries = match o {
            Value::Map(e) => e,
            _ => return Err("the options after `with` must be a map, like {system: \"Be brief\"}".into()),
        };
        for (k, v) in entries.iter() {
            match (&**k, v) {
                ("system", Value::Str(s)) => m.system = Some(s.to_string()),
                ("max_tokens", v) if v.scalar().is_some() => m.max_tokens = v.scalar().unwrap().max(1.0) as u64,
                (other, _) => {
                    return Err(format!(
                        "I don't know the model option '{}'. Available: system (text), max_tokens (number)",
                        other
                    ))
                }
            }
        }
    }
    Ok(m)
}

pub fn ask(m: &ModelSpec, prompt: &str, suffix: &str) -> Result<String, String> {
    match m.provider.as_str() {
        "echo" => Ok(prompt.to_string()),
        _ => ask_anthropic(m, &format!("{}{}", prompt, suffix)),
    }
}

/// Models that accept Anthropic's server-side refusal fallback.
fn supports_fallback(name: &str) -> bool {
    ["claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-fable-5-2"]
        .iter()
        .any(|p| name.starts_with(p))
}

fn ask_anthropic(m: &ModelSpec, prompt: &str) -> Result<String, String> {
    let key = std::env::var("ANTHROPIC_API_KEY").ok().filter(|k| !k.trim().is_empty()).ok_or(
        "to ask a Claude model, set the ANTHROPIC_API_KEY environment variable to your API key first",
    )?;
    let fallback = supports_fallback(&m.name);

    let mut body = format!(
        "{{\"model\":{},\"max_tokens\":{},\"messages\":[{{\"role\":\"user\",\"content\":{}}}]",
        json::escape(&m.name),
        m.max_tokens,
        json::escape(prompt)
    );
    if let Some(sys) = &m.system {
        body.push_str(&format!(",\"system\":{}", json::escape(sys)));
    }
    if fallback {
        body.push_str(",\"fallbacks\":\"default\"");
    }
    body.push('}');

    // The request body goes in a temp file and the API key travels through
    // curl's stdin config, so the key never shows up in the process list.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!("antares_{}_{}.json", std::process::id(), nanos));
    std::fs::write(&path, &body).map_err(|e| format!("couldn't prepare the request: {}", e))?;
    let path_str = path.to_string_lossy().replace('\\', "/");

    let mut config = String::new();
    config.push_str("url = \"https://api.anthropic.com/v1/messages\"\n");
    config.push_str(&format!("header = \"x-api-key: {}\"\n", key.trim()));
    config.push_str("header = \"anthropic-version: 2026-10-21\"\n");
    config.push_str("header = \"content-type: application/json\"\n");
    if fallback {
        config.push_str("header = \"anthropic-beta: server-side-fallback-2026-07-01\"\n");
    }
    config.push_str(&format!("data-binary = \"@{}\"\n", path_str));
    config.push_str("silent\nshow-error\nmax-time = 600\n");

    let result = (|| {
        let mut child = Command::new("curl")
            .args(["--config", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("couldn't start curl to reach the model: {}", e))?;
        child
            .stdin
            .take()
            .unwrap()
            .write_all(config.as_bytes())
            .map_err(|e| format!("couldn't send the request: {}", e))?;
        let out = child.wait_with_output().map_err(|e| format!("the request failed: {}", e))?;
        if !out.status.success() {
            return Err(format!(
                "couldn't reach the model: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).to_string())
    })();
    let _ = std::fs::remove_file(&path);
    let text = result?;

    let j = json::parse(&text).map_err(|_| {
        format!("the model service sent back something unexpected: {}", text.chars().take(200).collect::<String>())
    })?;
    if j.get("type").and_then(J::str) == Some("error") {
        let msg = j.get("error").and_then(|e| e.get("message")).and_then(J::str).unwrap_or("unknown error");
        return Err(format!("the model service reported: {}", msg));
    }
    if j.get("stop_reason").and_then(J::str) == Some("refusal") {
        return Err("the model declined to answer this request".into());
    }
    let mut answer = String::new();
    if let Some(J::Arr(blocks)) = j.get("content") {
        for b in blocks {
            if b.get("type").and_then(J::str) == Some("text") {
                if let Some(t) = b.get("text").and_then(J::str) {
                    answer.push_str(t);
                }
            }
        }
    }
    if answer.is_empty() && j.get("stop_reason").and_then(J::str) == Some("max_tokens") {
        return Err("the model ran out of room before answering. Give it more with: with {max_tokens: 32000}".into());
    }
    Ok(answer)
}
