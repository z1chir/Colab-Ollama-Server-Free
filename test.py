"""MARK: Colab Ollama Test"""

# @title Config
URL = "https://url.com"  # @param {type:"string"}
MODEL = "qwen3.5:9b"  # @param {type:"string"}

import time
from ollama import Client
from pydantic import BaseModel

client = Client(host=URL)

# --- Helpers ---
now = lambda: time.perf_counter() * 1000

def chat(messages, **kwargs):
    return client.chat(model=MODEL, messages=messages, options={"temperature": 0}, think=False, **kwargs)

def elapsed(t0):
    return f"{now() - t0:.0f}ms"

def extract_json(raw: str) -> str:
    """Strip markdown fences and extract first JSON object."""
    if raw.startswith("```"):
        lines = raw.splitlines()[1:]
        raw = "\n".join(lines[:-1] if lines[-1].startswith("```") else lines).strip()
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start:end + 1] if start != -1 and end > start else raw

# --- Tests ---
def test_ping():
    t0 = now()
    client.list()
    print(f"🏓 Ping: {elapsed(t0)} ✅")

def test_chat():
    print("🤖 Chat: ", end="", flush=True)
    t0, ttft = now(), None
    for chunk in chat([{"role": "user", "content": "Hi!"}], stream=True):
        if ttft is None:
            ttft = elapsed(t0)
        print(chunk.message.content, end="", flush=True)
    print(f" ({ttft} TTFT, {elapsed(t0)})")

def test_structured():
    class Person(BaseModel):
        name: str
        age: int

    t0 = now()
    resp = chat(
        [
            {"role": "system", "content": "Return only valid JSON that matches the provided schema."},
            {"role": "user", "content": "John is 25 years old."},
        ],
        format=Person.model_json_schema(),
    )
    person = Person.model_validate_json(extract_json(resp.message.content.strip()))
    print(f"📋 Structured: {person.model_dump()} ({elapsed(t0)})")

def test_tool():
    def mul(a: int, b: int) -> int:
        """Multiply."""
        return a * b

    t0 = now()
    resp = chat([{"role": "user", "content": "100*100?"}], tools=[mul])
    if resp.message.tool_calls:
        tc = resp.message.tool_calls[0]
        print(f"🔧 Tool: {tc.function.name}(...)={mul(**tc.function.arguments)} ({elapsed(t0)})")
    else:
        print(f"🔧 Tool: {resp.message.content} ({elapsed(t0)})")

if __name__ == "__main__":
    test_ping()
    test_chat()
    test_structured()
    test_tool()
    print("✅ Done")
