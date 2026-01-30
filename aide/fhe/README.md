# AIDE-FHE Module

Core module for autonomous FHE challenge solving. The agent generates `eval()` function bodies, injects them into challenge templates, executes in Docker, and iterates via tree-search.

## File Structure

```
aide/fhe/
├── run.py                 # CLI entry point (aide-fhe)
├── config.py              # Configuration loading
├── config.yaml            # Default settings
├── agent.py               # FHE agent (draft/debug/improve loop)
├── challenge_parser.py    # Parses challenge.md → FHEChallengeSpec
└── interpreters/
    ├── __init__.py        # Interpreter factory (selects by challenge type)
    ├── base.py            # Base class: Docker build/run, error parsing
    ├── black_box.py       # Pre-encrypted testcases
    ├── white_box.py       # OpenFHE with fherma-validator
    ├── ml_inference.py    # ML training + encrypted inference
    └── non_openfhe.py     # HElayers, SEAL, Swift-HE, etc.
```

## How It Works

1. `challenge_parser.py` extracts scheme (CKKS/BFV/BGV), constraints (depth, batch size), available keys, and scoring from `challenge.md`
2. `interpreters/__init__.py` selects the right interpreter based on `ChallengeType`
3. `agent.py` generates only the `eval()` body (e.g. OpenFHE C++ calls), which the interpreter injects into `yourSolution.cpp`
4. The interpreter builds and runs the solution in Docker, parses validation output into accuracy metrics
5. The agent uses tree-search (draft -> debug/improve) guided by these metrics until the accuracy threshold is met or steps are exhausted

## Error Types

Detected and classified by the base interpreter (`base.py`):

- `BUILD_ERROR` -- CMake/make failure
- `COMPILATION_ERROR` -- C++ syntax or API errors
- `RUNTIME_ERROR` -- Exception during execution
- `DEPTH_EXCEEDED` -- Multiplicative depth budget exceeded
- `TIMEOUT` -- Execution exceeded time limit

## Configuration

Defaults in `config.yaml`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `agent.steps` | 50 | Max iterations |
| `agent.code.model` | `gpt-5-mini-2025-08-07` | Code generation model |
| `agent.code.reasoning_effort` | `high` | Thinking mode (low/medium/high) |
| `agent.search.num_drafts` | 10 | Initial drafts before improving |
| `agent.search.debug_prob` | 0.5 | Probability of debugging vs improving |
| `exec.build_timeout` | 600 | Docker build timeout (seconds) |
| `exec.run_timeout` | 1800 | FHE execution timeout (seconds) |
