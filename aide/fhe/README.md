# AIDE-FHE: Autonomous FHE Challenge Solver

AIDE-FHE is an extension of AIDE ML that autonomously solves Fully Homomorphic Encryption (FHE) challenges using LLM-driven code generation and tree-search optimization.

## Quick Start

```bash
# Basic usage
aide-fhe challenge_dir=/path/to/challenge

# With custom settings
aide-fhe challenge_dir=/path/to/challenge agent.steps=30 agent.code.model=gpt-4o

# With specific testcase
aide-fhe challenge_dir=/path/to/challenge testcase_dir=/path/to/testcase
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           aide-fhe CLI                               │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        1. Challenge Parser                           │
│  challenge.md → FHEChallengeSpec (type, scheme, constraints, keys)  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     2. Interpreter Factory                           │
│  ChallengeType → BlackBox | WhiteBox | MLInference | NonOpenFHE     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          3. FHE Agent                                │
│  Tree-search: draft → debug/improve → best solution                 │
│  Generates ONLY eval() function body                                │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    4. Template Injection                             │
│  Code injected into yourSolution.cpp template                       │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    5. Docker Execution                               │
│  Build → Run → Validate → Accuracy metrics                          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      6. Feedback Loop                                │
│  Accuracy/errors → Agent decides: draft more | debug | improve      │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Challenge Parser (`challenge_parser.py`)

Parses `challenge.md` to extract:
- **Challenge Type**: `black_box`, `white_box_openfhe`, `ml_inference`, `non_openfhe`
- **Scheme**: CKKS, BFV, BGV, TFHE
- **Constraints**: Multiplicative depth, batch size, scale parameters
- **Keys**: Available rotation indices, bootstrapping support
- **Scoring**: Accuracy threshold, error tolerance

### 2. Interpreters (`interpreters/`)

Type-specific execution environments:

| Type | Interpreter | Description |
|------|-------------|-------------|
| `black_box` | `BlackBoxInterpreter` | Pre-encrypted testcases, uses challenge's Dockerfile |
| `white_box_openfhe` | `WhiteBoxInterpreter` | OpenFHE with fherma-validator image |
| `ml_inference` | `MLInferenceInterpreter` | ML model training + encrypted inference |
| `non_openfhe` | `NonOpenFHEInterpreter` | HElayers, Swift-HE, SEAL, etc. |

### 3. FHE Agent (`agent.py`)

Tree-search algorithm with three operations:

```
                    ┌─────────┐
                    │  Start  │
                    └────┬────┘
                         │
              ┌──────────▼──────────┐
              │   Draft Phase       │
              │  (num_drafts nodes) │
              └──────────┬──────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌─────▼─────┐   ┌─────▼─────┐
    │  Debug  │    │  Improve  │   │   Draft   │
    │ (buggy) │    │  (good)   │   │   (new)   │
    └────┬────┘    └─────┬─────┘   └─────┬─────┘
         │               │               │
         └───────────────┴───────────────┘
                         │
              ┌──────────▼──────────┐
              │   Best Solution     │
              └─────────────────────┘
```

**Search Policy:**
1. Create `num_drafts` initial solutions
2. After drafting: probabilistically choose between:
   - **Debug** (fix buggy nodes) - `debug_prob` chance
   - **Improve** (enhance good nodes) - `1 - debug_prob` chance
3. Stop when accuracy threshold reached or steps exhausted

### 4. Code Generation

The agent generates **only the `eval()` function body**, not complete files:

```cpp
// Template (yourSolution.cpp)
void CKKSTaskSolver::eval() {
    // <<<YOUR_CODE_HERE>>>  ← Agent generates this
}

// Agent output example:
auto y = m_cc->EvalMult(m_InputC, 2.0);
m_OutputC = m_cc->EvalAdd(y, 1.0);
```

### 5. Execution Flow

```
Agent generates code
        │
        ▼
┌───────────────────┐
│ Inject into       │
│ yourSolution.cpp  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Docker Build      │──→ BUILD_ERROR (if fails)
│ (cmake, make)     │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Docker Run        │──→ RUNTIME_ERROR / DEPTH_EXCEEDED
│ (execute app)     │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Validate Output   │──→ accuracy, error metrics
│ (decrypt & check) │
└─────────┬─────────┘
          │
          ▼
    Feedback to Agent
```

## Configuration

### Command Line Options

```bash
aide-fhe \
  challenge_dir=/path/to/challenge \
  agent.steps=30 \                    # Max iterations
  agent.search.num_drafts=5 \         # Initial drafts before debug/improve
  agent.search.debug_prob=0.5 \       # Probability of debugging vs improving
  agent.code.model=gpt-4o \           # LLM model
  agent.code.temp=0.5 \               # Temperature
  agent.code.reasoning_effort=high \  # For thinking models (low/medium/high)
  exec.build_timeout=600 \            # Docker build timeout (seconds)
  exec.run_timeout=3000               # FHE execution timeout (seconds)
```

### Default Configuration (`config.yaml`)

```yaml
agent:
  steps: 20
  early_stop_threshold: 0.99
  code:
    model: gpt-4o
    temp: 0.5
    reasoning_effort: medium
  search:
    num_drafts: 3
    debug_prob: 0.5
    max_debug_depth: 5

exec:
  build_timeout: 600
  run_timeout: 3000
```

## Output Files

After running, find results in `logs/<exp_name>/`:

```
logs/
└── 42-happy-dancing-penguin/
    ├── config.yaml           # Run configuration
    ├── challenge_spec.json   # Parsed challenge specification
    ├── journal.json          # All nodes with code, results, metrics
    └── best_solution.txt     # Best performing solution code
```

### Journal Format (v3)

```json
{
  "nodes": [
    {
      "step": 0,
      "id": "abc123...",
      "parent": null,           // null for drafts, parent ID for debug/improve
      "code": "auto y = ...",
      "plan": "Approach: ...",
      "is_buggy": false,
      "analysis": "Success: accuracy 0.95",
      "metric": {"value": 0.95, "maximize": true},
      "_term_out": ["Build output...", "Validation: accuracy=0.95"]
    },
    {
      "step": 1,
      "parent": "abc123...",    // This is a debug of node 0
      ...
    }
  ],
  "__version": "3"
}
```

## Error Detection

The interpreter detects and categorizes errors:

| Error Type | Description |
|------------|-------------|
| `BUILD_ERROR` | CMake/make compilation failure |
| `COMPILATION_ERROR` | C++ syntax or OpenFHE API errors |
| `LINKER_ERROR` | Missing symbols or libraries |
| `RUNTIME_ERROR` | Exception during execution |
| `DEPTH_EXCEEDED` | Multiplicative depth budget exceeded |
| `COMPILER_BUG` | GCC internal compiler error |
| `TIMEOUT` | Execution exceeded time limit |

## Supported Challenge Types

### Black Box Challenges
- Pre-encrypted testcases provided
- Uses challenge's own Dockerfile
- Solution only accesses ciphertexts and crypto operations

### White Box OpenFHE Challenges
- Uses `fherma-validator` Docker image
- Full access to plaintext for validation
- Standard OpenFHE template

### ML Inference Challenges
- Training data provided in plaintext
- Model must be trained and converted to FHE
- Inference performed on encrypted data

### Non-OpenFHE Challenges
- HElayers (IBM)
- swift-homomorphic-encryption (Apple)
- Other FHE libraries

## File Structure

```
aide/fhe/
├── __init__.py
├── README.md              # This file
├── run.py                 # CLI entry point
├── config.py              # Configuration dataclasses
├── agent.py               # FHE agent with tree-search
├── challenge_parser.py    # Parse challenge.md
└── interpreters/
    ├── __init__.py        # Interpreter factory
    ├── base.py            # Base class with common functionality
    ├── black_box.py       # Pre-encrypted testcase handling
    ├── white_box.py       # OpenFHE with fherma-validator
    ├── ml_inference.py    # ML model + encrypted inference
    └── non_openfhe.py     # Other FHE libraries
```

## Example Workflow

1. **Parse Challenge**
   ```
   challenge.md → FHEChallengeSpec(
     type=WHITE_BOX_OPENFHE,
     scheme=CKKS,
     constraints=Constraints(depth=10, batch_size=4096),
     task="Compute sign(x) for x in [-1, 1]"
   )
   ```

2. **Create Interpreter**
   ```
   WhiteBoxInterpreter(spec, workspace_dir, timeouts)
   ```

3. **Agent Drafts Solution**
   ```cpp
   // Draft 0: Polynomial approximation
   auto y = m_cc->EvalMult(m_InputC, 1.5);
   for (int i = 0; i < 3; i++) {
       auto y2 = m_cc->EvalMult(y, y);
       auto t = m_cc->EvalSub(3.0, y2);
       y = m_cc->EvalMult(y, t);
       y = m_cc->EvalMult(y, 0.5);
   }
   m_OutputC = y;
   ```

4. **Execute & Validate**
   ```
   Build: SUCCESS
   Run: SUCCESS
   Accuracy: 0.82
   ```

5. **Agent Improves**
   ```
   Node 0 accuracy 0.82 < 0.99 threshold
   → Create improvement node with refined polynomial
   ```

6. **Iterate Until Success**
   ```
   Node 5: accuracy 0.99 ✓
   → Save as best_solution.txt
   ```
