"""A pretend model service for testing Antares's model support offline.

It speaks just enough of the Anthropic Messages API and the OpenAI chat
completions and embeddings APIs (plain and streamed) to check what Antares
sends: tool calls and their results, structured answers, images,
conversations, effort, and that thinking blocks come back unchanged.

Run tests/check_models.py, which starts this and runs tests/models.ant.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SIGNATURE = "sig-abc123"


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def has_image(content):
    return isinstance(content, list) and any(b.get("type") in ("image", "image_url") for b in content if isinstance(b, dict))


def sample(schema):
    """A value that fits a JSON schema."""
    t = schema.get("type")
    if t == "object":
        return {k: sample(v) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        item = schema.get("items", {"type": "string"})
        return [sample(item), sample(item)]
    return {"string": "Ada", "number": 36, "integer": 36, "boolean": True}.get(t)


def decide(messages, tools, schema, system, extra):
    """What the pretend model does: ("tool", name, input) or ("text", reply)."""
    last = messages[-1]
    content = last.get("content")
    results = []
    if isinstance(content, list):
        results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
    if last.get("role") == "tool":
        results = [{"content": content}]
    if results:
        got = "; ".join(str(r.get("content")) for r in results)
        return ("text", f"The tool said: {got}")
    prompt = text_of(content)
    if schema is not None:
        return ("text", json.dumps(sample(schema)))
    if has_image(content):
        return ("text", "I see an image")
    if tools and "weather" in prompt.lower():
        t = tools[0]
        props = list((t.get("input_schema") or t.get("parameters") or {}).get("properties", {}).keys())
        return ("tool", t["name"], {p: ("Paris" if i == 0 else 2) for i, p in enumerate(props)})
    if "my name" in prompt.lower() and len(messages) > 1:
        earlier = " / ".join(text_of(m.get("content")) for m in messages[:-1] if m.get("role") == "user")
        return ("text", f"Earlier you said: {earlier}")
    if "system?" in prompt:
        return ("text", f"system={system}")
    if "effort?" in prompt:
        return ("text", f"effort={extra.get('effort')}")
    return ("text", f"heard: {prompt}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def body(self):
        return json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")

    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_events(self, events, named):
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for e in events:
            if named:
                self.wfile.write(f"event: {e['type']}\n".encode())
            self.wfile.write(f"data: {json.dumps(e) if isinstance(e, dict) else e}\n\n".encode())
            self.wfile.flush()

    def do_POST(self):
        req = self.body()
        if self.path == "/v1/messages":
            self.anthropic(req)
        elif self.path == "/v1/chat/completions":
            self.openai(req)
        elif self.path == "/v1/embeddings":
            vecs = [[float(len(t)), float(t.count("a")), 1.0] for t in req["input"]]
            self.send_json({"data": [{"index": i, "embedding": v} for i, v in reversed(list(enumerate(vecs)))]})
        else:
            self.send_json({"error": {"message": f"no such endpoint {self.path}"}}, 404)

    # ----- Anthropic -----

    def anthropic(self, req):
        if self.headers.get("anthropic-version") != "2026-10-21":
            return self.send_json({"type": "error", "error": {"message": "bad anthropic-version header"}}, 400)
        msgs = req["messages"]
        # an assistant turn that called a tool must come back with its thinking block unchanged
        for m in msgs:
            if m["role"] == "assistant" and isinstance(m["content"], list):
                kinds = [b["type"] for b in m["content"]]
                if "tool_use" in kinds and not any(b["type"] == "thinking" and b.get("signature") == SIGNATURE for b in m["content"]):
                    return self.send_json({"type": "error", "error": {"message": "thinking block missing or changed"}}, 400)
        fmt = req.get("output_config", {}).get("format")
        schema = fmt["schema"] if fmt else None
        if schema is not None and schema.get("additionalProperties") is not False:
            return self.send_json({"type": "error", "error": {"message": "schema needs additionalProperties: false"}}, 400)
        what = decide(msgs, req.get("tools", []), schema, req.get("system"), req.get("output_config", {}))
        thinking = {"type": "thinking", "thinking": "", "signature": SIGNATURE}
        if what[0] == "tool":
            blocks = [thinking, {"type": "tool_use", "id": "toolu_1", "name": what[1], "input": what[2]}]
            stop = "tool_use"
        else:
            blocks = [thinking, {"type": "text", "text": what[1]}]
            stop = "end_turn"
        if not req.get("stream"):
            return self.send_json({"type": "message", "role": "assistant", "content": blocks, "stop_reason": stop})
        events = [{"type": "message_start", "message": {"type": "message", "role": "assistant", "content": []}}]
        for i, b in enumerate(blocks):
            if b["type"] == "thinking":
                events.append({"type": "content_block_start", "index": i, "content_block": {"type": "thinking", "thinking": ""}})
                events.append({"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta", "signature": SIGNATURE}})
            elif b["type"] == "text":
                events.append({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}})
                half = len(b["text"]) // 2
                for piece in (b["text"][:half], b["text"][half:]):
                    events.append({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": piece}})
            else:
                events.append({"type": "content_block_start", "index": i, "content_block": {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}})
                raw = json.dumps(b["input"])
                for piece in (raw[:5], raw[5:]):
                    events.append({"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": piece}})
            events.append({"type": "content_block_stop", "index": i})
        events.append({"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 5}})
        events.append({"type": "message_stop"})
        self.send_events(events, True)

    # ----- OpenAI -----

    def openai(self, req):
        msgs = [m for m in req["messages"] if m["role"] != "system"]
        system = next((m["content"] for m in req["messages"] if m["role"] == "system"), None)
        rf = req.get("response_format")
        schema = rf["json_schema"]["schema"] if rf else None
        tools = [t["function"] for t in req.get("tools", [])]
        what = decide(msgs, tools, schema, system, {})
        if what[0] == "tool":
            call = {"id": "call_1", "type": "function", "function": {"name": what[1], "arguments": json.dumps(what[2])}}
            message = {"role": "assistant", "content": None, "tool_calls": [call]}
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": what[1]}
            finish = "stop"
        if not req.get("stream"):
            return self.send_json({"choices": [{"index": 0, "message": message, "finish_reason": finish}]})
        events = []
        if what[0] == "tool":
            args = message["tool_calls"][0]["function"]["arguments"]
            events.append({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": what[1], "arguments": args[:4]}}]}}]})
            events.append({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": args[4:]}}]}}]})
        else:
            half = len(what[1]) // 2
            for piece in (what[1][:half], what[1][half:]):
                events.append({"choices": [{"index": 0, "delta": {"content": piece}}]})
        events.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        events.append("[DONE]")
        self.send_events(events, False)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(server.server_address[1], flush=True)
    server.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
