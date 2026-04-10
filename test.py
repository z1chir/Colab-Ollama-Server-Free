"""MARK: Colab Ollama Test (Standard Libs)"""

# @title Config
URL = "https://professional-established-diffs-pirates.trycloudflare.com"  # @param {type:"string"}
MODEL = "qwen3.5:27b"  # @param {type:"string"}

import time
import requests
import json

# --- Helpers ---
now = lambda: time.perf_counter() * 1000

def chat(messages, stream=False, format=None, tools=None, **kwargs):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": 0, **kwargs},
    }
    if format:
        payload["format"] = format
    if tools:
        # Convert simple function list to Ollama tool schema
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": f.__name__,
                    "description": f.__doc__ or "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"}
                        },
                        "required": ["a", "b"]
                    }
                }
            } for f in tools
        ]

    response = requests.post(f"{URL}/api/chat", json=payload, stream=stream)

    if stream:
        return response
    return response.json()

def elapsed(t0):
    return f"{now() - t0:.0f}ms"

def extract_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()[1:]
        raw = "\n".join(lines[:-1] if lines[-1].startswith("```") else lines).strip()
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start:end + 1] if start != -1 and end > start else raw)

# --- Tests ---
def test_ping():
    t0 = now()
    requests.get(f"{URL}/api/tags")
    print(f"🏓 Ping: {elapsed(t0)} ✅")

def test_chat():
    print("🤖 Chat: ", end="", flush=True)
    t0, ttft = now(), None
    resp = chat([{"role": "user", "content": "Hi!"}], stream=True)

    full_content = ""
    for line in resp.iter_lines():
        if line:
            chunk = json.loads(line)
            if ttft is None:
                ttft = elapsed(t0)
            content = chunk.get("message", {}).get("content", "")
            print(content, end="", flush=True)
            full_content += content
    print(f" ({ttft} TTFT, {elapsed(t0)})")

def test_structured():
    # Simple schema definition without Pydantic
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }

    t0 = now()
    resp = chat(
        [
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": "John is 25 years old."},
        ],
        format=schema,
    )
    person = extract_json(resp["message"]["content"])
    print(f"📋 Structured: {person} ({elapsed(t0)})")

def test_tool():
    def mul(a: int, b: int) -> int:
        """Multiply."""
        return a * b

    t0 = now()
    resp = chat([{"role": "user", "content": "100*100?"}], tools=[mul])

    message = resp.get("message", {})
    if "tool_calls" in message:
        tc = message["tool_calls"][0]["function"]
        args = tc["arguments"]
        print(f"🔧 Tool: {tc['name']}(...)={mul(**args)} ({elapsed(t0)})")
    else:
        print(f"🔧 Tool: {message.get('content')} ({elapsed(t0)})")

if __name__ == "__main__":
    test_ping()
    test_chat()
    test_structured()
    test_tool()
    print("✅ Done")
