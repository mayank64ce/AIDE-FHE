# AIDE-FHE: Autonomous FHE Challenge Solver

An LLM-driven agent that autonomously solves Fully Homomorphic Encryption (FHE) challenges using tree-search. Built on top of [AIDE ML](https://github.com/WecoAI/aideml), it generates, evaluates, and iteratively improves OpenFHE C++ solutions.

![Tree Search Visualization](https://github.com/WecoAI/aideml/assets/8918572/2401529c-b97e-4029-aed2-c3f376f54c3c)

## How It Works

The agent operates as a tree-search over the space of FHE solutions:

1. **Parse** the challenge spec (`challenge.md`) to extract scheme, constraints, and scoring
2. **Draft** initial `eval()` function implementations using an LLM
3. **Execute** solutions in Docker containers against test cases
4. **Evaluate** results by parsing validator output (accuracy, errors, score)
5. **Improve or debug** based on feedback, branching the solution tree

Supported challenge types: black-box (pre-encrypted), white-box OpenFHE, ML inference, and non-OpenFHE (HElayers, SEAL, etc.).

## Installation

```bash
git clone https://github.com/your-org/aide-fhe.git
cd aide-fhe
pip install -e .
```

Requires Python 3.10+ and Docker.

## Challenge Data

```bash
huggingface-cli download yifeiz29/fhe-challenge-data --repo-type dataset --local-dir fhe_challenge
```

## Usage

```bash
# Set your LLM API key
export OPENAI_API_KEY=<your-key>
# Or for OpenRouter models:
export OPENROUTER_API_KEY=<your-key>

# Run on an FHE challenge
aide-fhe challenge_dir=fhe_challenge/black_box/challenge_relu

# With custom model and iterations
aide-fhe challenge_dir=fhe_challenge/black_box/challenge_relu \
       agent.steps=30 \
       agent.code.model=deepseek/deepseek-v3.2 \
       agent.code.reasoning_effort=high
```

Results are saved to `logs/<run-name>/` including the best solution, journal, and parsed challenge spec.
