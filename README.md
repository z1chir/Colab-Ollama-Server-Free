# Colab Ollama Server Free

Run Ollama with GPU on Google Colab, exposed via Cloudflare tunnel.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HamzaYslmn/Colab-Ollama-Server-Free/blob/main/server.ipynb?accelerator=GPU&gpuType=T4)

## Quick Start

1. Click badge above → opens Colab with **Tesla T4** selected
2. Run cell → wait for tunnel URL (e.g. `https://xxx.trycloudflare.com`)
3. Copy URL to `test.py` or your `.env`

## Files

- `server.ipynb` — Installs Ollama, starts GPU server, opens tunnel
- `test.py` — Client-side test script

## Local Testing

```bash
# Run tests
uv run python src/modules/llm/colab/test.py
```

## Config

Edit `MODEL` in the Colab form sidebar (default: `qwen3.5:9b`)
