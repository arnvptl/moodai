"""
MoodAI Model Health Check — pings every model in the fallback chain
and reports which ones are alive.

Usage:
    python moodai/ping.py          # requires AI_API_KEY (and optionally GROQ_API_KEY, OPENROUTER_API_KEY)
"""

import os
import sys
import time
from google import genai

# ── Configuration ──────────────────────────────────────────────────────────────
AI_API_KEY = os.getenv("AI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not AI_API_KEY:
    print("Error: AI_API_KEY environment variable must be set.")
    sys.exit(1)

# ── Clients ────────────────────────────────────────────────────────────────────
gemini_client = genai.Client(api_key=AI_API_KEY)

groq_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
    except (ImportError, Exception) as e:
        print(f"WARNING: Groq client unavailable: {e}")

openrouter_client = None
if OPENROUTER_API_KEY:
    try:
        from openai import OpenAI
        openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
    except (ImportError, Exception) as e:
        print(f"WARNING: OpenRouter client unavailable: {e}")

# ── Fallback chain (must mirror auto.py) ──────────────────────────────────────
FALLBACK_CHAIN = [
    ("gemini",     "gemini-3.8-flash"),
    ("openrouter", "poolside/laguna-s-2.1:free"),
    ("openrouter", "nvidia/nemotron-3.5-super-vl:free"),
    ("groq",       "openai/gpt-oss-120b"),
    ("groq",       "qwen/qwen3.8-27b"),
    ("openrouter", "google/gemma-4-26b-a4b-it:free"),
    ("gemini",     "gemini-3.6-flash"),
    ("openrouter", "cohere/north-mini-code:free"),
    ("groq",       "openai/gpt-oss-20b"),
    ("gemini",     "gemini-3.5-flash-lite"),
]

PING_PROMPT = "Reply with exactly one word: pong"

# ── Ping helpers ───────────────────────────────────────────────────────────────

def ping_gemini(model: str) -> tuple[bool, str, float]:
    """Returns (ok, detail, latency_sec)."""
    start = time.perf_counter()
    try:
        resp = gemini_client.models.generate_content(
            model=model,
            contents=PING_PROMPT,
            config=genai.types.GenerateContentConfig(temperature=0.0),
        )
        latency = time.perf_counter() - start
        text = (resp.text or "").strip()
        return (True, text[:80], latency)
    except Exception as e:
        latency = time.perf_counter() - start
        return (False, str(e)[:80], latency)


def ping_openai_compat(client, client_name: str, model: str) -> tuple[bool, str, float]:
    """Ping any OpenAI-compatible client (Groq, OpenRouter). Returns (ok, detail, latency_sec)."""
    if client is None:
        return (False, f"no {client_name} API key", 0.0)
    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PING_PROMPT}],
            temperature=0.0,
        )
        latency = time.perf_counter() - start
        text = (resp.choices[0].message.content or "").strip()
        return (True, text[:80], latency)
    except Exception as e:
        latency = time.perf_counter() - start
        return (False, str(e)[:80], latency)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("  MoodAI — Model Health Check")
    print("=" * 90)
    print(f"  Gemini API key     : {'set' if AI_API_KEY else 'MISSING'}")
    print(f"  Groq API key       : {'set' if GROQ_API_KEY else 'not set (Groq models will be skipped)'}")
    print(f"  OpenRouter API key : {'set' if OPENROUTER_API_KEY else 'not set (OpenRouter models will be skipped)'}")
    print(f"  Models to ping     : {len(FALLBACK_CHAIN)}")
    print("=" * 90)
    print()

    results = []
    total_ok = 0

    for idx, (provider, model) in enumerate(FALLBACK_CHAIN, 1):
        label = f"{provider}/{model}"
        print(f"  [{idx}/{len(FALLBACK_CHAIN)}] Pinging {label} ...", end=" ", flush=True)

        if provider == "gemini":
            ok, detail, latency = ping_gemini(model)
        elif provider == "groq":
            ok, detail, latency = ping_openai_compat(groq_client, "GROQ", model)
        elif provider == "openrouter":
            ok, detail, latency = ping_openai_compat(openrouter_client, "OPENROUTER", model)
        else:
            ok, detail, latency = False, "unknown provider", 0.0

        status = "✓ OK" if ok else "✗ FAIL"
        print(f"{status}  ({latency:.2f}s)")

        results.append((idx, provider, model, ok, detail, latency))
        if ok:
            total_ok += 1

    # ── Summary table ──────────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("  RESULTS")
    print("=" * 90)
    print(f"  {'#':<4} {'Provider':<12} {'Model':<45} {'Status':<8} {'Latency':<10} {'Response / Error'}")
    print(f"  {'─'*4} {'─'*12} {'─'*45} {'─'*8} {'─'*10} {'─'*30}")

    for idx, provider, model, ok, detail, latency in results:
        status_str = "✓ OK" if ok else "✗ FAIL"
        latency_str = f"{latency:.2f}s" if latency > 0 else "—"
        print(f"  {idx:<4} {provider:<12} {model:<45} {status_str:<8} {latency_str:<10} {detail}")

    print()
    print(f"  Total: {total_ok}/{len(FALLBACK_CHAIN)} models operational")

    if total_ok == len(FALLBACK_CHAIN):
        print("  🟢 All models healthy!")
    elif total_ok > 0:
        print("  🟡 Some models are down — fallback chain partially operational.")
    else:
        print("  🔴 ALL models are down! MoodAI will not be able to answer questions.")

    print("=" * 90)

    # Exit with non-zero if everything is dead (useful in CI)
    sys.exit(0 if total_ok > 0 else 1)


if __name__ == "__main__":
    main()
