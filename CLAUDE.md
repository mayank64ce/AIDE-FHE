# CLAUDE.md

## Project Overview

AIDE-FHE is an LLM-driven agent that autonomously solves FHE (Fully Homomorphic Encryption) challenges using tree-search. It generates `eval()` function bodies in C++, injects them into challenge templates, executes via Docker, and iterates based on validation metrics.

## Commands

```bash
pip install -e .
aide-fhe challenge_dir=/path/to/challenge
```

## Architecture

- **`aide/backend/`** -- LLM abstraction layer. Two backends: `backend_openai.py` (for `gpt-*`, `o*`, `codex-*`, or when `OPENAI_BASE_URL` is set) and `backend_openrouter.py` (everything else).
- **`aide/journal.py`** -- `Node` and `Journal` dataclasses forming the solution tree. Nodes store code, execution results, metrics, and FHE-specific fields.
- **`aide/fhe/agent.py`** -- FHE agent with draft/debug/improve loop. Generates only `eval()` body.
- **`aide/fhe/challenge_parser.py`** -- Parses `challenge.md` into `FHEChallengeSpec`.
- **`aide/fhe/interpreters/`** -- Docker-based executors selected by challenge type: `black_box.py`, `white_box.py`, `ml_inference.py`, `non_openfhe.py`.
- **`aide/fhe/run.py`** -- CLI entry point (`aide-fhe`).
- **`aide/fhe/config.yaml`** -- Default configuration.

## LLM Routing

In `aide/backend/__init__.py`:
- `gpt-*`, `o\d+`, `codex-*` → OpenAI
- Everything else → OpenRouter (includes Claude, Gemini, DeepSeek, etc.)
- If `OPENAI_BASE_URL` is set → OpenAI backend for all models

## Environment Variables

- `OPENAI_API_KEY` -- OpenAI models
- `OPENROUTER_API_KEY` -- OpenRouter models
- `OPENAI_BASE_URL` -- Override for local LLMs

## Output

Results saved to `logs/<exp_name>/`: `config.yaml`, `challenge_spec.json`, `journal.json`, `best_solution.txt`.
