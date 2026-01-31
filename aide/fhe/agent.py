"""
FHE Agent with Template-Aware Code Generation.

This agent generates ONLY the eval() function body, not full files.
Template files (Dockerfile, CMakeLists.txt, main.cpp, etc.) are provided
by the challenge and handled by type-specific interpreters.

Workflow:
1. Parse challenge.md → FHEChallengeSpec with ChallengeType
2. Load template context (yourSolution.cpp, config.json, etc.)
3. Generate eval() body based on challenge type
4. Interpreter injects code into template and executes
5. Parse validation result directly (no LLM needed)
6. Agent iterates: draft → improve/debug
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from dataclasses_json import DataClassJsonMixin

from ..backend import query
from ..journal import Journal, Node
from ..utils.response import wrap_code
from ..utils.metric import MetricValue, WorstMetricValue

from .challenge_parser import (
    FHEChallengeSpec,
    ChallengeType,
    Scheme,
    Library,
)
from .interpreters.ml_inference import TrainingDataLoader

logger = logging.getLogger("aide.fhe")


@dataclass
class AgentConfig(DataClassJsonMixin):
    """Agent configuration."""
    # Code generation model
    code_model: str = "gpt-5.2-2025-12-11"
    code_temp: float = 0.5
    reasoning_effort: str = "high"  # low, medium, high (for thinking models)
    web_search: bool = True  # Enable web search for FHE docs/examples

    # Feedback model (lesson extraction)
    feedback_enabled: bool = True
    feedback_model: str = "gpt-4o-mini"
    feedback_temp: float = 0.3
    max_lessons: int = 10

    # Search settings
    num_drafts: int = 3
    debug_prob: float = 0.5
    max_debug_depth: int = 5

    # Execution
    timeout: int = 600


class FHEAgent:
    """
    FHE Agent with template-aware code generation.

    Key design:
    - Generates ONLY eval() function body
    - Interpreter handles template injection
    - Challenge type determines prompt structure
    - Uses validation metrics for improvement
    """

    def __init__(
        self,
        spec: FHEChallengeSpec,
        journal: Journal,
        config: Optional[AgentConfig] = None,
        workspace_dir: Optional[Path] = None,
    ):
        self.spec = spec
        self.journal = journal
        self.config = config or AgentConfig()
        self.workspace_dir = workspace_dir or Path(".")

        # Lessons learned from past failures (for feedback LLM)
        self.lessons_learned: list[str] = []

        # Load template context once
        self._template_context = self._load_template_context()

        # Load training data for ML inference challenges
        self._training_data = None
        if spec.challenge_type == ChallengeType.ML_INFERENCE:
            data_dir = spec.challenge_dir / "data"
            self._training_data = TrainingDataLoader.load(data_dir)
            if self._training_data:
                logger.info(f"Loaded training data: {self._training_data.X_shape}")

    def _load_template_context(self) -> dict[str, str]:
        """Load relevant template files for prompts."""
        context = {}

        if not self.spec.template_dir or not self.spec.template_dir.exists():
            return context

        # Load key template files based on challenge type
        if self.spec.challenge_type == ChallengeType.BLACK_BOX:
            # Black box: yourSolution.cpp, yourSolution.h
            for name in ["yourSolution.cpp", "yourSolution.h"]:
                path = self.spec.template_dir / name
                if path.exists():
                    context[name] = path.read_text()

        elif self.spec.challenge_type == ChallengeType.WHITE_BOX_OPENFHE:
            # White box: yourSolution.cpp, config.json
            for name in ["yourSolution.cpp", "yourSolution.h", "config.json"]:
                path = self.spec.template_dir / name
                if path.exists():
                    context[name] = path.read_text()

        elif self.spec.challenge_type == ChallengeType.ML_INFERENCE:
            # ML: app.py or yourSolution.cpp depending on Python/C++
            # Also load header files for variable extraction
            for name in ["app.py", "yourSolution.cpp", "config.json"]:
                path = self.spec.template_dir / name
                if path.exists():
                    context[name] = path.read_text()
            # Load any .h files (cifar10.h, fraud.h, yourSolution.h, etc.)
            for header in self.spec.template_dir.glob("*.h"):
                context[header.name] = header.read_text()

        elif self.spec.challenge_type == ChallengeType.NON_OPENFHE:
            # Non-OpenFHE: main.swift or app.py + config.json
            for name in ["main.swift", "app.py", "Package.swift", "config.json"]:
                path = self.spec.template_dir / name
                if path.exists():
                    context[name] = path.read_text()
            # Swift templates store main.swift in Sources/ subdirectory
            if "main.swift" not in context:
                swift_main = self.spec.template_dir / "Sources" / "main.swift"
                if swift_main.exists():
                    context["main.swift"] = swift_main.read_text()

        return context

    def _is_python_template(self) -> bool:
        """Check if this challenge uses a Python template."""
        # Check for Python files in template context
        if any(name.endswith('.py') for name in self._template_context):
            return True
        # Also check template directory
        if self.spec.template_dir and self.spec.template_dir.exists():
            return any(self.spec.template_dir.glob('*.py'))
        return False

    def _extract_template_variables(self) -> dict[str, list[str]]:
        """
        Extract variables from template files.

        Returns dict with:
        - 'context': CryptoContext/HeContext variable
        - 'inputs': Input ciphertext variables
        - 'output': Output ciphertext variable
        - 'public_key': Public key variable
        - 'is_python': True if Python template
        """
        import re

        # Check for Python templates first
        if self._is_python_template():
            return self._extract_python_template_variables()

        # Check for Swift templates
        if self.spec.library == Library.SWIFT_HE:
            return self._extract_swift_template_variables()

        # C++ template extraction (default)
        result = {
            'context': 'm_cc',
            'inputs': ['m_InputC'],
            'output': 'm_OutputC',
            'public_key': 'm_PublicKey',
            'is_python': False,
        }

        # Try to find header file content
        header_content = None
        for name in ['yourSolution.h', 'fraud.h', 'cifar10.h']:
            if name in self._template_context:
                header_content = self._template_context[name]
                break

        # Also try to load from template dir if not in context
        if not header_content and self.spec.template_dir:
            for header_name in ['yourSolution.h', 'fraud.h', 'cifar10.h']:
                header_path = self.spec.template_dir / header_name
                if header_path.exists():
                    header_content = header_path.read_text()
                    break

        if not header_content:
            return result

        # Extract Ciphertext member variables
        ciphertext_pattern = r'Ciphertext<DCRTPoly>\s+(m_\w+)'
        ciphertext_vars = re.findall(ciphertext_pattern, header_content)

        # Categorize variables
        inputs = []
        output = None
        for var in ciphertext_vars:
            if 'Output' in var:
                output = var
            else:
                inputs.append(var)

        if inputs:
            result['inputs'] = inputs
        if output:
            result['output'] = output

        # Extract CryptoContext variable
        cc_pattern = r'CryptoContext<DCRTPoly>\s+(m_\w+)'
        cc_match = re.search(cc_pattern, header_content)
        if cc_match:
            result['context'] = cc_match.group(1)

        # Extract PublicKey variable
        pk_pattern = r'PublicKey<DCRTPoly>\s+(m_\w+)'
        pk_match = re.search(pk_pattern, header_content)
        if pk_match:
            result['public_key'] = pk_match.group(1)

        return result

    def _extract_python_template_variables(self) -> dict[str, list[str]]:
        """
        Extract variables from Python template files.

        Handles:
        - OpenFHE-Python: def solve(input, context, pub_key) -> returns output
        - HElayers: def solve(input_ctxts, word_sizes, text, he_context) -> returns CTile
        """
        import re

        result = {
            'context': 'context',
            'inputs': ['input'],
            'output': 'output',
            'public_key': 'pub_key',
            'is_python': True,
            'library': 'openfhe-python',  # default
        }

        # Find Python template content
        py_content = None
        for name in ['app.py', 'main.py']:
            if name in self._template_context:
                py_content = self._template_context[name]
                break

        if not py_content and self.spec.template_dir:
            for py_name in ['app.py', 'main.py']:
                py_path = self.spec.template_dir / py_name
                if py_path.exists():
                    py_content = py_path.read_text()
                    break

        if not py_content:
            return result

        # Detect library type
        if 'pyhelayers' in py_content or 'pyhe' in py_content:
            result['library'] = 'helayers'
            result['context'] = 'he_context'

        # Extract solve() function signature
        # Pattern: def solve(arg1, arg2, ...) or def solve(arg1, arg2, ...):
        solve_pattern = r'def\s+solve\s*\(([^)]*)\)'
        match = re.search(solve_pattern, py_content)
        if match:
            params_str = match.group(1)
            params = [p.strip().split(':')[0].strip() for p in params_str.split(',') if p.strip()]

            # Parse parameters based on library
            if result['library'] == 'helayers':
                # HElayers: def solve(input_ctxts, word_sizes, text, he_context)
                inputs = []
                for p in params:
                    if 'context' in p.lower():
                        result['context'] = p
                    elif 'input' in p.lower() or 'ctxt' in p.lower():
                        inputs.append(p)
                    # word_sizes, text are plaintext - not included in crypto vars
                if inputs:
                    result['inputs'] = inputs
                # HElayers returns the result (not assigned to output variable)
                result['output'] = 'return value'
            else:
                # OpenFHE-Python: def solve(input, context, pub_key)
                inputs = []
                for p in params:
                    if 'context' in p.lower() or p == 'cc':
                        result['context'] = p
                    elif 'key' in p.lower() or 'pub' in p.lower():
                        result['public_key'] = p
                    elif 'input' in p.lower():
                        inputs.append(p)
                if inputs:
                    result['inputs'] = inputs
                # OpenFHE-Python returns output
                result['output'] = 'return value'

        return result

    def _extract_swift_template_variables(self) -> dict[str, list[str]]:
        """Extract variables from Swift template files."""
        import re

        result = {
            'context': 'context',
            'inputs': ['cipher1'],
            'output': 'result',
            'public_key': 'evaluationKey',
            'is_swift': True,
            'is_python': False,
        }

        swift_content = self._template_context.get('main.swift', '')
        if not swift_content:
            return result

        # Extract input variable names from CLI argument loading
        # Pattern: let varName = try loadCiphertext(path: argName, ...)
        load_pattern = r'let\s+(\w+)\s*=\s*try\s+loadCiphertext\('
        inputs = re.findall(load_pattern, swift_content)
        if inputs:
            result['inputs'] = inputs

        # Extract evaluation key variable
        eval_key_pattern = r'let\s+(\w+)\s*=\s*try\s+loadEvaluationKey\('
        ek_match = re.search(eval_key_pattern, swift_content)
        if ek_match:
            result['public_key'] = ek_match.group(1)

        # Extract context variable
        ctx_pattern = r'let\s+(\w+)\s*=\s*try\s+Context\('
        ctx_match = re.search(ctx_pattern, swift_content)
        if ctx_match:
            result['context'] = ctx_match.group(1)

        return result

    def _format_variable_docs(self, vars: dict) -> list[str]:
        """Format variable documentation for prompts."""
        # Check if Swift template
        if vars.get('is_swift', False):
            return self._format_swift_variable_docs(vars)
        # Check if Python template
        if vars.get('is_python', False):
            return self._format_python_variable_docs(vars)

        # C++ template formatting
        lines = [
            "IMPORTANT - Use these EXACT variable names (class members):",
            f"  {vars['context']}       - CryptoContext<DCRTPoly> (NOT 'cc')",
        ]

        # Format input variables
        if len(vars['inputs']) == 1:
            lines.append(f"  {vars['inputs'][0]}   - Input Ciphertext<DCRTPoly>")
        else:
            for inp in vars['inputs']:
                # Try to make descriptive name from variable
                desc = inp.replace('m_', '').replace('C', ' Ciphertext')
                lines.append(f"  {inp}  - {desc} (Ciphertext<DCRTPoly>)")

        lines.extend([
            f"  {vars['output']}  - Output Ciphertext (ASSIGN to this, don't return)",
            f"  {vars['public_key']} - PublicKey<DCRTPoly>",
            "",
            f"The eval() function is void - assign result to {vars['output']}:",
            f"  {vars['output']} = result;  // CORRECT",
            "  return result;       // WRONG - eval() is void!",
        ])

        return lines

    def _format_swift_variable_docs(self, vars: dict) -> list[str]:
        """Format variable documentation for Swift templates."""
        lines = [
            "IMPORTANT - Use these EXACT variable names (already loaded in template):",
            f"  {vars['context']}       - BFV Context for encryption parameters",
        ]
        for inp in vars['inputs']:
            lines.append(f"  {inp}        - Input encrypted ciphertext (CanonicalCiphertext)")
        lines.extend([
            f"  {vars['public_key']} - EvaluationKey (for rotation + relinearization)",
            "",
            "Available operations (Swift operators):",
            "  cipher1 + cipher2           // Add ciphertexts",
            "  cipher1 - cipher2           // Subtract ciphertexts",
            "  cipher1 * cipher2           // Multiply ciphertexts",
            "  cipher1 + plaintext         // Add plaintext",
            "  cipher1 * plaintext         // Multiply by plaintext",
            "  -cipher1                    // Negation",
            f"  cipher.rotateColumns(by: step, using: {vars['public_key']})  // Rotate slots",
            f"  cipher.relinearize(using: {vars['public_key']})              // After multiplication",
            "  context.encode(values: [UInt64], format: .coefficient)        // Create plaintext",
            "",
            f"Assign your result to 'result' variable (already declared as 'var result = ...').",
        ])
        return lines

    def _format_python_variable_docs(self, vars: dict) -> list[str]:
        """Format variable documentation for Python templates."""
        library = vars.get('library', 'openfhe-python')

        if library == 'helayers':
            lines = [
                "IMPORTANT - Use these EXACT parameter names (function arguments):",
                f"  {vars['context']}     - HElayers HeContext object",
            ]
            for inp in vars['inputs']:
                lines.append(f"  {inp}    - Input CTile(s) from HElayers")
            lines.extend([
                "",
                "The solve() function RETURNS the encrypted result:",
                "  return result_ctile  # CORRECT - return your CTile result",
                "",
                "HElayers API hints:",
                "  - CTile operations: add, sub, multiply, square",
                "  - Use he_context for encoding/encrypting plaintext",
                "  - Rotation: ctile.rotate(offset)",
            ])
        else:
            # OpenFHE-Python
            lines = [
                "IMPORTANT - Use these EXACT parameter names (function arguments):",
                f"  {vars['context']}     - CryptoContext (OpenFHE-Python)",
            ]
            for inp in vars['inputs']:
                lines.append(f"  {inp}       - Input Ciphertext (encrypted)")
            if vars.get('public_key'):
                lines.append(f"  {vars['public_key']}    - PublicKey for encryption")
            lines.extend([
                "",
                "The solve() function RETURNS the encrypted result:",
                "  return output  # CORRECT - return your Ciphertext result",
                "",
                "OpenFHE-Python API hints:",
                "  - context.EvalAdd(ct1, ct2) - Add ciphertexts",
                "  - context.EvalMult(ct1, ct2) - Multiply ciphertexts",
                "  - context.EvalRotate(ct, offset) - Rotate slots",
                "  - context.MakeCKKSPackedPlaintext(values) - Create plaintext",
            ])

        return lines

    # ==================== Feedback & Learning ====================

    def _extract_lesson_from_failure(self, node: Node) -> Optional[str]:
        """
        Use feedback LLM to extract actionable lesson from a failure.

        Returns a specific lesson learned, or None if feedback is disabled
        or the lesson couldn't be extracted.
        """
        if not self.config.feedback_enabled:
            return None

        # Skip if no error output
        if not node.term_out or not node.is_buggy:
            return None

        # Check if this is a repeated error pattern (don't waste LLM call)
        error_snippet = node.term_out[-500:] if len(node.term_out) > 500 else node.term_out
        for existing_lesson in self.lessons_learned:
            # Simple similarity check - if error keywords appear in existing lesson
            if any(keyword in existing_lesson.lower() for keyword in
                   ['rotation key', 'depth', 'enable', 'compilation']
                   if keyword.lower() in error_snippet.lower()):
                # Skip extracting lesson for similar errors
                logger.debug("Similar error pattern detected, skipping lesson extraction")
                return None

        try:
            # For white-box, need more context (CONFIG + CODE sections)
            code_limit = 2000 if self.spec.challenge_type == ChallengeType.WHITE_BOX_OPENFHE else 800

            prompt = {
                "Task": "You are an FHE debugging expert. Extract ONE specific, actionable lesson from this failure.",
                "Challenge Type": self.spec.task,
                "Code Snippet": node.code[:code_limit] if len(node.code) > code_limit else node.code,
                "Error Output": error_snippet,
                "Instructions": [
                    "Identify the ROOT CAUSE of the failure (not just symptoms)",
                    "Provide a SPECIFIC fix or action to avoid this mistake",
                    "Be specific about config.json changes if needed (which field, what value)",
                    "Reference exact file/line if compilation error",
                    "Format: Start with 'Lesson:' followed by the mistake and fix",
                    "Be concise (2-3 sentences maximum)",
                    "Focus on what to DO, not what went wrong",
                ],
                "Examples": [
                    "Lesson: When using EvalRotate(k), add k to indexes_for_rotation_key array in config.json",
                    "Lesson: Depth exceeded means reduce multiplications or increase mult_depth in config.json",
                    "Lesson: Use m_cc, m_InputC, m_OutputC exactly - wrong variable names cause compilation errors",
                ]
            }

            response = query(
                system_message=prompt,
                user_message=None,
                model=self.config.feedback_model,
                temperature=self.config.feedback_temp,
            )

            # Clean up response
            lesson = response.strip()

            # Validate lesson format
            if not lesson.lower().startswith('lesson:'):
                lesson = f"Lesson: {lesson}"

            # Truncate if too long
            if len(lesson) > 300:
                lesson = lesson[:297] + "..."

            logger.info(f"📝 {lesson}")
            return lesson

        except Exception as e:
            logger.warning(f"Failed to extract lesson: {e}")
            return None

    def _update_lessons(self, node: Node) -> None:
        """Update lessons learned from a buggy node."""
        if not node.is_buggy or not self.config.feedback_enabled:
            return

        lesson = self._extract_lesson_from_failure(node)
        if lesson and lesson not in self.lessons_learned:
            self.lessons_learned.append(lesson)

            # Keep only most recent N lessons
            if len(self.lessons_learned) > self.config.max_lessons:
                self.lessons_learned = self.lessons_learned[-self.config.max_lessons:]

    def _build_lessons_prompt(self) -> dict:
        """Build prompt section with past lessons learned."""
        if not self.lessons_learned:
            return {}

        return {
            "⚠️ CRITICAL LESSONS FROM PAST FAILURES": [
                "You have made these mistakes before. DO NOT repeat them!",
                "",
                *self.lessons_learned,
                "",
                "Apply these lessons to your current solution!",
            ]
        }

    # ==================== Search Policy ====================

    def search_policy(self) -> Optional[Node]:
        """Select node to work on (None = draft new)."""
        # Initial drafting phase
        if len(self.journal.draft_nodes) < self.config.num_drafts:
            logger.debug("[search] drafting (not enough drafts)")
            return None

        # Get debuggable nodes (buggy leaves within max depth)
        debuggable = [
            n for n in self.journal.buggy_nodes
            if n.is_leaf and n.debug_depth <= self.config.max_debug_depth
        ]

        # Get good nodes for improvement
        good_nodes = self.journal.good_nodes

        # If no good nodes exist, must debug (can't improve nothing)
        if not good_nodes:
            if debuggable:
                logger.debug("[search] debugging (no good nodes, must debug)")
                # Prioritize compile errors (usually easier to fix)
                compile_errors = [n for n in debuggable if self._is_compile_error(n)]
                if compile_errors:
                    return random.choice(compile_errors)
                return random.choice(debuggable)
            else:
                # No good nodes and no debuggable nodes - draft again
                logger.debug("[search] drafting (no good or debuggable nodes)")
                return None

        # Both good and buggy nodes exist - use probability to choose
        if debuggable and random.random() < self.config.debug_prob:
            logger.debug("[search] debugging (by probability)")
            compile_errors = [n for n in debuggable if self._is_compile_error(n)]
            if compile_errors:
                return random.choice(compile_errors)
            return random.choice(debuggable)

        # Improve best working node
        logger.debug("[search] improving best node")
        return self.journal.get_best_node()

    def _is_compile_error(self, node: Node) -> bool:
        """Check if node has compile error."""
        out = (node.term_out or "").lower()
        return any(kw in out for kw in ["error:", "make[", "cmake error"])

    # ==================== Prompt Building ====================

    def _build_challenge_prompt(self) -> dict:
        """Build challenge description section."""
        spec = self.spec

        prompt = {
            "Task": spec.task,
            "Description": spec.task_description,
            "Scheme": spec.scheme.value if spec.scheme else "CKKS",
            "Library": spec.library.value if spec.library else "OpenFHE",
        }

        # Add output format if available
        if spec.output_format:
            prompt["Expected Output"] = spec.output_format

        if spec.constraints:
            prompt["Constraints"] = {
                "Multiplicative Depth": spec.constraints.depth,
                "Batch Size": spec.constraints.batch_size,
                "Input Range": f"[{spec.constraints.input_range[0]}, {spec.constraints.input_range[1]}]",
            }

        if spec.keys:
            prompt["Available Keys"] = {
                "Public Key": spec.keys.public,
                "Multiplication Key": spec.keys.multiplication,
                "Rotation Indices": spec.keys.rotation_indices,
            }

        # Add training data info for ML inference challenges
        if spec.challenge_type == ChallengeType.ML_INFERENCE and self._training_data:
            prompt["Training Data"] = self._training_data.summary()
            prompt["Data Loading Code"] = self._get_data_loading_code()

        # Add reference data for white-box challenges (e.g., KNN dataset)
        if spec.challenge_type == ChallengeType.WHITE_BOX_OPENFHE:
            data_dir = spec.challenge_dir / "data"
            if data_dir.exists():
                data_info = self._load_reference_data(data_dir)
                if data_info:
                    prompt["Reference Data"] = data_info

        # Add useful links from challenge.md (e.g., Polycircuit, tutorials)
        if spec.useful_links:
            links_info = []
            for link in spec.useful_links:
                link_str = f"- {link['name']}: {link['url']}"
                if link.get('description'):
                    link_str += f" ({link['description']})"
                links_info.append(link_str)
            prompt["Useful Resources"] = links_info

        return {"Challenge Specification": prompt}

    def _get_data_loading_code(self) -> str:
        """Generate code snippet for loading training data."""
        if not self._training_data:
            return "# No training data available"

        data_dir = self.spec.challenge_dir / "data"

        # Read data_info.json if available for preprocessing notes
        data_info = None
        data_info_path = data_dir / "data_info.json"
        if data_info_path.exists():
            import json
            with open(data_info_path) as f:
                data_info = json.load(f)

        if self._training_data.data_format == "numpy":
            x_file = "X_train.npy"
            for name in ["X_train.npy", "X_train_flat.npy", "x_train.npy"]:
                if (data_dir / name).exists():
                    x_file = name
                    break
            y_file = "y_train.npy"
            for name in ["y_train.npy", "Y_train.npy"]:
                if (data_dir / name).exists():
                    y_file = name
                    break

            code = f'''import numpy as np
X_train = np.load("{data_dir}/{x_file}")
y_train = np.load("{data_dir}/{y_file}")'''

            # Add preprocessing notes if available
            if data_info and "preprocessing_notes" in data_info:
                notes = "\n# ".join(data_info["preprocessing_notes"])
                code += f"\n# Preprocessing notes:\n# {notes}"

            return code

        elif self._training_data.data_format == "csv":
            return f'''import pandas as pd
X_train = pd.read_csv("{data_dir}/X_train.csv").values
y_train = pd.read_csv("{data_dir}/y_train.csv").values.flatten()'''

        elif self._training_data.data_format == "parquet":
            parquet_files = list(data_dir.glob("*.parquet"))
            if parquet_files:
                return f'''import pandas as pd
import numpy as np
df = pd.read_parquet("{parquet_files[0]}")
X_train = np.stack(df["embedding"].values)  # embeddings
y_train = df["label"].values'''

        return "# Unknown data format"

    def _load_reference_data(self, data_dir: Path) -> dict:
        """Load reference data for white-box challenges (e.g., KNN dataset)."""
        import pandas as pd

        result = {}

        # Check for CSV files
        csv_files = list(data_dir.glob("*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Get numeric columns only
                numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()

                # Docker path where data will be mounted
                docker_path = f"/fherma/data/{csv_file.name}"

                file_info = {
                    "docker_path": docker_path,  # Path inside Docker container
                    "num_rows": len(df),
                    "columns": numeric_cols,
                    "sample": df[numeric_cols].head(5).values.tolist() if numeric_cols else [],
                }

                # Add note about reading from Docker path
                file_info["note"] = (
                    f"Dataset with {len(df)} rows. "
                    f"Read from Docker path: {docker_path}"
                )

                result[csv_file.name] = file_info
            except Exception as e:
                result[csv_file.name] = {"error": str(e)}

        return result if result else None

    def _build_template_prompt(self) -> dict:
        """Build template context section."""
        if not self._template_context:
            return {}

        # Show key template structure
        templates = {}
        for name, content in self._template_context.items():
            # Truncate large files, but keep important parts
            if len(content) > 3000:
                lines = content.split('\n')
                relevant_sections = []

                # Keywords that indicate important code sections
                important_keywords = [
                    'class ', 'struct ', 'def ', 'func ', 'void ', 'eval(',
                    'solve(', 'public:', 'private:', '#include', 'import ',
                    'Ciphertext', 'CryptoContext', 'EvalMult', 'EvalAdd',
                    'this->input', 'this->output', 'this->cc'
                ]

                i = 0
                while i < len(lines):
                    line = lines[i]
                    if any(kw in line for kw in important_keywords):
                        # Include context around important lines
                        start = max(0, i - 2)
                        end = min(len(lines), i + 20)
                        section = lines[start:end]
                        relevant_sections.append('\n'.join(section))
                        i = end  # Skip ahead
                    else:
                        i += 1

                if relevant_sections:
                    content = '\n\n// ... (truncated) ...\n\n'.join(relevant_sections)
                else:
                    # Fallback: first 1500 chars + last 500 chars
                    content = content[:1500] + "\n\n// ... (truncated) ...\n\n" + content[-500:]

            templates[name] = content

        return {"Template Files": templates}

    def _build_eval_instructions(self) -> dict:
        """Build instructions for what to implement."""
        spec = self.spec

        # Extract template variables dynamically
        template_vars = self._extract_template_variables()

        if spec.challenge_type == ChallengeType.BLACK_BOX:
            return {
                "What to Implement": [
                    "Implement ONLY the body of the eval() function.",
                    "The eval() function is in yourSolution.cpp.",
                    "Input ciphertexts are already deserialized and available.",
                    f"Output must be assigned to {template_vars['output']}.",
                    "Do NOT modify main.cpp, Dockerfile, or CMakeLists.txt.",
                ],
                "C++ Template Variables": self._format_variable_docs(template_vars),
            }

        elif spec.challenge_type == ChallengeType.WHITE_BOX_OPENFHE:
            # Get current config for display
            config_content = self._template_context.get('config.json', '')

            result = {
                "What to Implement": [
                    "Implement the body of the eval() function.",
                    "Template handles CryptoContext creation and key generation.",
                    "fherma-validator will build, run, and validate your solution.",
                    "",
                    "You MAY also modify config.json parameters if needed:",
                    "- Add rotation key indexes if your algorithm requires them",
                    "- Adjust mult_depth if you need more levels",
                    "- Enable bootstrapping for very deep computations",
                ],
                "C++ Template Variables": self._format_variable_docs(template_vars),
            }

            if config_content:
                result["Current config.json"] = config_content
                result["Config Modification (Optional)"] = [
                    "To modify config, provide the COMPLETE config.json:",
                    "",
                    "### CONFIG ###",
                    "<your complete modified config.json here>",
                    "",
                    "Common modifications:",
                    "- indexes_for_rotation_key: add rotation indices you need",
                    "- mult_depth: increase if you get 'depth exhausted' error",
                ]

            return result

        elif spec.challenge_type == ChallengeType.ML_INFERENCE:
            is_python = any(f.endswith('.py') for f in self._template_context)
            weights_dir = self.workspace_dir / "weights"

            if is_python:
                return {
                    "What to Implement": [
                        "Provide THREE sections:",
                        "",
                        "1. CONFIG.JSON - Update rotation keys and parameters as needed",
                        "",
                        "2. TRAINING CODE (runs LOCALLY, not in Docker):",
                        "   - Loads and trains on the provided data",
                        f"   - IMPORTANT: Save weights to LOCAL path: {weights_dir}/",
                        "   - DO NOT use /fherma/ paths - that only exists in Docker!",
                        f"   - Example: arr.astype(np.float64).tofile('{weights_dir}/name.bin')",
                        "",
                        "3. INFERENCE CODE (runs in Docker):",
                        "   - Loads weights from /fherma/weights/*.bin (Docker mount)",
                        "   - Implements FHE inference with OpenFHE-Python",
                        "   - Returns encrypted output",
                    ],
                }
            # C++ template
            var_docs = self._format_variable_docs(template_vars)
            # Get input range from constraints
            input_range = "[0, 255]"  # default for images
            if spec.constraints and spec.constraints.input_range:
                input_range = f"[{spec.constraints.input_range[0]}, {spec.constraints.input_range[1]}]"

            # Get max value for scaling
            max_val = 255.0
            if spec.constraints and spec.constraints.input_range:
                max_val = spec.constraints.input_range[1]

            return {
                "What to Implement": [
                    "Provide THREE sections:",
                    "",
                    "1. CONFIG - Update config.json (rotation keys, mult_depth)",
                    "",
                    "2. TRAINING CODE (Python, runs locally):",
                    "   - Use PyTorch, TensorFlow, or scikit-learn",
                    f"   - NORMALIZE input to [0,1] by dividing by {max_val}",
                    f"   - Save weights to: {weights_dir}/ using .tofile() for binary format",
                    "   - Explore different model architectures to maximize accuracy",
                    "   - Print accuracy to verify model quality",
                    "",
                    "3. INFERENCE CODE (C++ eval() body, runs in Docker):",
                    "   - Load weights from /fherma/weights/",
                    f"   - SCALE encrypted input by 1/{max_val} first (one plaintext multiply)",
                ],
                "CRITICAL - Input Scaling": [
                    f"Encrypted input is in range {input_range}, but normalized training works better.",
                    f"In C++ eval(), scale input FIRST:",
                    f"  auto input_norm = m_cc->EvalMult(m_InputC, m_cc->MakeCKKSPackedPlaintext(std::vector<double>(slot_count, 1.0/{max_val})));",
                    "Then use input_norm with your normalized-trained weights.",
                ],
                "C++ Template Variables": var_docs,
            }

        elif spec.challenge_type == ChallengeType.NON_OPENFHE:
            # Get current config for display
            config_content = self._template_context.get('config.json', '')

            if spec.library == Library.SWIFT_HE:
                result = {
                    "What to Implement": [
                        "Implement the solve() function body in Swift.",
                        "Use swift-homomorphic-encryption APIs.",
                        "",
                        "You MAY also modify config.json parameters if needed.",
                    ],
                }
            else:
                result = {
                    "What to Implement": [
                        "Implement the solve() function body.",
                        f"Use {spec.library.value} library APIs.",
                        "",
                        "You MAY also modify config.json parameters if needed.",
                    ],
                }

            if config_content:
                result["Current config.json"] = config_content
                result["Config Modification (Optional)"] = [
                    "To modify config, provide the COMPLETE config.json:",
                    "",
                    "### CONFIG ###",
                    "<your complete modified config.json here>",
                ]

            return result

        return {}

    def _build_response_format(self) -> dict:
        """Build expected response format."""
        spec = self.spec

        if spec.challenge_type == ChallengeType.BLACK_BOX:
            return {
                "Response Format": (
                    "Provide your implementation in this format:\n\n"
                    "1. Brief explanation of your approach (2-3 sentences)\n\n"
                    "2. The eval() function body:\n"
                    "```cpp\n"
                    "// Your implementation here\n"
                    "// Do NOT include function signature or braces\n"
                    "```\n\n"
                    "IMPORTANT: Only provide the code that goes INSIDE eval(), "
                    "not the function signature or surrounding code."
                )
            }

        elif spec.challenge_type == ChallengeType.WHITE_BOX_OPENFHE:
            return {
                "Response Format": (
                    "Provide BOTH sections (config.json controls crypto parameters):\n\n"
                    "### CONFIG ###\n"
                    "{\n"
                    '  "mult_depth": 20,\n'
                    '  "indexes_for_rotation_key": [1, 2, 4],\n'
                    '  "scheme": "CKKS"\n'
                    "}\n\n"
                    "### CODE ###\n"
                    "// Your eval() function body here\n\n"
                    "IMPORTANT: Both sections are required for white-box challenges."
                )
            }

        elif spec.challenge_type == ChallengeType.ML_INFERENCE:
            if any(f.endswith('.py') for f in self._template_context):
                return {
                    "Response Format": (
                        "Provide THREE code blocks with ### markers:\n\n"
                        "```json\n### CONFIG ###\n{...}\n```\n\n"
                        "```python\n### TRAINING CODE ###\n# training script\n```\n\n"
                        "```python\n### INFERENCE CODE ###\n# solve() body\n```"
                    )
                }
            return {
                "Response Format": (
                    "Provide your solution with these markers:\n\n"
                    "```\n"
                    "### CONFIG ###\n"
                    "{...your config.json updates...}\n\n"
                    "### TRAINING CODE ###\n"
                    "# Python training script\n\n"
                    "### INFERENCE CODE ###\n"
                    "// C++ eval() body\n"
                    "```"
                )
            }

        elif spec.challenge_type == ChallengeType.NON_OPENFHE:
            if spec.library == Library.SWIFT_HE:
                return {
                    "Response Format": (
                        "Provide your implementation in this format:\n\n"
                        "1. Brief approach explanation\n\n"
                        "2. (OPTIONAL) If you need to modify config.json, provide the COMPLETE file:\n"
                        "```json\n"
                        "### CONFIG ###\n"
                        "{ ... complete config.json ... }\n"
                        "```\n\n"
                        "3. The solve() function body:\n"
                        "```swift\n"
                        "### CODE ###\n"
                        "// Implementation\n"
                        "```"
                    )
                }
            return {
                "Response Format": (
                    "Provide your implementation in this format:\n\n"
                    "1. Brief approach\n\n"
                    "2. (OPTIONAL) If you need to modify config.json, provide the COMPLETE file:\n"
                    "```json\n"
                    "### CONFIG ###\n"
                    "{ ... complete config.json ... }\n"
                    "```\n\n"
                    "3. The solve() function body:\n"
                    "```python\n"
                    "### CODE ###\n"
                    "# Implementation\n"
                    "```"
                )
            }

        return {}

    # ==================== Code Generation ====================

    def _generate_code(self, prompt: dict) -> tuple[str, str]:
        """Query LLM for plan and code."""
        response = query(
            system_message=prompt,
            user_message=None,
            model=self.config.code_model,
            temperature=self.config.code_temp,
            reasoning_effort=self.config.reasoning_effort,
            web_search=self.config.web_search,
        )

        # Extract code from response
        code = self._extract_code(response)

        # Extract explanation (text before code)
        plan = response.split("```")[0].strip() if "```" in response else ""

        return plan, code

    def _extract_code(self, text: str) -> str:
        """Extract code block from response."""
        import re

        # For ML inference: preserve both training and inference code blocks
        if self.spec.challenge_type == ChallengeType.ML_INFERENCE:
            return self._extract_ml_code(text)

        # For white box and non-OpenFHE: also check for CONFIG section
        if self.spec.challenge_type in [ChallengeType.WHITE_BOX_OPENFHE, ChallengeType.NON_OPENFHE]:
            return self._extract_whitebox_code(text)

        # Find code block
        pattern = r'```(?:\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            # Return the main code block (usually the longest)
            return max(matches, key=len).strip()

        # Fallback: check for code-like content
        lines = text.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            if any(kw in line for kw in ['#include', 'def ', 'func ', 'cc->', 'EvalMult']):
                in_code = True
            if in_code:
                code_lines.append(line)

        return '\n'.join(code_lines) if code_lines else text

    def _extract_whitebox_code(self, text: str) -> str:
        """Extract code and optional CONFIG section for white box challenges."""
        import re

        # Find all code blocks (handle both complete and incomplete markdown)
        pattern = r'```(?:\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)

        # If no complete code blocks found, try to find incomplete ones (missing closing ```)
        if not matches:
            # Look for ```cpp or ``` followed by code
            incomplete_pattern = r'```(?:\w+)?\n(.*?)(?=```|\Z)'
            matches = re.findall(incomplete_pattern, text, re.DOTALL)

        config_json = None
        main_code = None

        for block in matches:
            block_lower = block.lower()
            # Check for config markers - detect by ### CONFIG ### or common config keys
            config_keys = ['"indexes_for_rotation_key"', '"mult_depth"', '"ring_dimension"',
                          '"poly_degree"', '"plaintext_modulus"', '"scheme"', '"batch_size"']
            if '### config' in block_lower or any(key in block for key in config_keys):
                config_json = block.strip()
                # Remove the ### CONFIG ### marker if present
                if config_json.lower().startswith('### config'):
                    config_json = '\n'.join(config_json.split('\n')[1:]).strip()
            # Check for explicit ### CODE ### marker
            elif '### code' in block_lower:
                main_code = block.strip()
                # Remove the ### CODE ### marker if present
                if main_code.lower().startswith('### code'):
                    main_code = '\n'.join(main_code.split('\n')[1:]).strip()
            # Check for C++ code (eval function body)
            elif any(kw in block for kw in ['EvalMult', 'EvalAdd', 'm_cc->', 'Ciphertext', 'm_OutputC']):
                main_code = block.strip()
            # Check for Swift code
            elif any(kw in block for kw in ['Bfv<', 'Rlwe<', '.decrypt(', '.encrypt(', 'HeContext', 'SerializedCiphertext', 'GaloisKey']):
                main_code = block.strip()
            # Fallback: longest block is probably the code
            elif not main_code and len(block.strip()) > 50:
                main_code = block.strip()

        # Combine with markers for interpreter
        result_parts = []
        if config_json:
            result_parts.append("### CONFIG ###")
            result_parts.append(config_json)

        if main_code:
            result_parts.append("\n### CODE ###")
            result_parts.append(main_code)
        elif matches:
            # Fallback: use longest code block
            result_parts.append("\n### CODE ###")
            result_parts.append(max(matches, key=len).strip())

        if result_parts:
            return "\n".join(result_parts)

        return text

    def _extract_ml_code(self, text: str) -> str:
        """Extract config, training, and inference code for ML challenges."""
        import re

        result_parts = []

        # Find all code blocks
        pattern = r'```(?:\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)

        config_json = None
        training_code = None
        inference_code = None

        for block in matches:
            block_lower = block.lower()
            # Check for config markers - detect by ### CONFIG ### or common config keys
            config_keys = ['"indexes_for_rotation_key"', '"mult_depth"', '"ring_dimension"',
                          '"poly_degree"', '"plaintext_modulus"', '"scheme"', '"batch_size"']
            if '### config' in block_lower or any(key in block for key in config_keys):
                config_json = block.strip()
                # Remove the ### CONFIG ### marker if present
                if config_json.lower().startswith('### config'):
                    config_json = '\n'.join(config_json.split('\n')[1:]).strip()
            # Check for training markers
            elif '### training' in block_lower or 'training code' in block_lower[:100]:
                training_code = block.strip()
            # Check for inference markers
            elif '### inference' in block_lower or 'inference code' in block_lower[:100]:
                inference_code = block.strip()
            # Check for solve() function pattern (inference)
            elif 'def solve' in block_lower or 'return ' in block_lower:
                if not inference_code:
                    inference_code = block.strip()
            # Check for training patterns (np.save, model.fit, etc.)
            elif any(kw in block_lower for kw in ['np.save', 'model.fit', 'sklearn', 'x_train']):
                if not training_code:
                    training_code = block.strip()

        # Combine with markers for interpreter
        if config_json:
            result_parts.append("### CONFIG ###")
            result_parts.append(config_json)

        if training_code:
            result_parts.append("\n### TRAINING CODE ###")
            result_parts.append(training_code)

        if inference_code:
            result_parts.append("\n### INFERENCE CODE ###")
            result_parts.append(inference_code)

        if result_parts:
            return "\n".join(result_parts)

        # Fallback: return longest code block
        if matches:
            return max(matches, key=len).strip()

        return text

    # ==================== Agent Operations ====================

    def _draft(self) -> Node:
        """Create initial solution draft."""
        prompt = {
            "Role": (
                "You are an expert in Fully Homomorphic Encryption (FHE). "
                "Implement a solution for the given challenge. "
                "Search the web for FHE examples if needed (github.com, openfhe-development.readthedocs.io)."
            ),
            "CRITICAL": [
                "NEVER decrypt inputs or use secret keys.",
                "NEVER return dummy/placeholder output - implement real FHE computation.",
                "Derive optimal parameters (degree, rotations, scaling) analytically before coding.",
            ],
        }

        # Add lessons learned (high priority - right after critical section)
        prompt |= self._build_lessons_prompt()

        # Web search guidance
        prompt["Web Search Guidance"] = [
            "You can search online for:",
            "- OpenFHE API: site:github.com/openfheorg [function_name]",
            "- Error fixes: site:github.com/openfheorg/issues '[error]'",
            "- Examples: github openfhe [operation] example",
            "DO NOT search for FHERMA challenge solutions - implement yourself",
        ]

        prompt |= self._build_challenge_prompt()
        prompt |= self._build_template_prompt()
        prompt |= self._build_eval_instructions()

        # Add memory of past attempts
        memory = self.journal.generate_summary()
        if memory:
            prompt["Previous Attempts (learn from these outcomes)"] = memory

        prompt |= self._build_response_format()

        plan, code = self._generate_code(prompt)
        return Node(plan=plan, code=code)

    def _improve(self, parent: Node) -> Node:
        """Improve working solution."""
        prompt = {
            "Role": (
                "You are improving a working FHE solution. "
                "Analyze the current results and make improvements to increase accuracy. "
                "Search the web for better algorithms if needed."
            ),
            "CRITICAL": [
                "NEVER decrypt inputs or use secret keys.",
                "NEVER return dummy/placeholder output - implement real FHE computation.",
                "Refine parameters based on validation results (increase degree, adjust scaling, etc.).",
            ],
        }

        # Add lessons learned (high priority)
        prompt |= self._build_lessons_prompt()

        # Web search guidance
        prompt["Web Search Guidance"] = [
            "You can search online for:",
            "- OpenFHE API: site:github.com/openfheorg [function_name]",
            "- Error fixes: site:github.com/openfheorg/issues '[error]'",
            "- Examples: github openfhe [operation] example",
            "DO NOT search for FHERMA challenge solutions - implement yourself",
        ]

        prompt |= self._build_challenge_prompt()
        prompt |= self._build_template_prompt()  # Template context for API understanding
        prompt |= self._build_eval_instructions()  # Available variables

        # Add memory of past attempts (like original AIDE)
        memory = self.journal.generate_summary()
        if memory:
            prompt["Previous Attempts (build on successes, learn from failures)"] = memory

        prompt["Current Solution"] = {
            "Code": wrap_code(parent.code),
            "Plan": parent.plan or "No plan provided",
            "Results": parent.analysis or "No analysis available",
            "Accuracy": str(parent.metric) if parent.metric else "Unknown",
        }

        prompt["Improvement Focus"] = [
            "Identify what limits the current accuracy.",
            "Consider better polynomial approximations or more iterations.",
            "Ensure depth budget is used efficiently.",
            "Try to reduce numerical errors in the computation.",
        ]

        prompt |= self._build_response_format()

        plan, code = self._generate_code(prompt)
        return Node(plan=plan, code=code, parent=parent)

    def _debug(self, parent: Node) -> Node:
        """Fix buggy solution - feed error + full context to LLM."""
        error_text = parent.term_out or ""

        prompt = {
            "Role": (
                "Debug this FHE solution. Carefully read the error output, "
                "understand what went wrong, and fix the code. "
                "Search OpenFHE docs for correct API usage if needed."
            ),
        }

        # Add lessons learned FIRST - most critical for debugging
        prompt |= self._build_lessons_prompt()

        # Web search guidance
        prompt["Web Search Guidance"] = [
            "You can search online for:",
            "- OpenFHE API: site:github.com/openfheorg [function_name]",
            "- Error fixes: site:github.com/openfheorg/issues '[error]'",
            "- Examples: github openfhe [operation] example",
            "DO NOT search for FHERMA challenge solutions - implement yourself",
        ]

        prompt |= self._build_challenge_prompt()
        prompt |= self._build_template_prompt()  # Template context for API understanding
        prompt |= self._build_eval_instructions()  # Available variables

        # Add memory of past attempts to avoid repeating mistakes
        memory = self.journal.generate_summary()
        if memory:
            prompt["Previous Attempts (understand why these failed)"] = memory

        prompt["Buggy Solution"] = {
            "Code": wrap_code(parent.code),
            "Plan": parent.plan or "No plan provided",
        }
        prompt["Error Output"] = wrap_code(error_text[-5000:])  # Last 5000 chars

        prompt["Debugging Instructions"] = [
            "If accuracy is low, recompute parameters; if depth exceeded, reduce complexity.",
            "Read the error message carefully and fix the issue.",
            "Common fixes: depth exceeded → reduce polynomial degree; missing rotation key [k] → add k to config.json",
            "CRITICAL: Do NOT fall back to dummy/placeholder output.",
            "If the approach is flawed, implement a working algorithm.",
            "Goal is HIGH ACCURACY, not just code that compiles.",
        ]

        prompt |= self._build_response_format()

        plan, code = self._generate_code(prompt)
        return Node(plan=plan, code=code, parent=parent)

    # ==================== Main Loop ====================

    def step(self, exec_callback: Callable) -> None:
        """Execute one agent step."""
        parent = self.search_policy()

        action = "draft" if parent is None else ("debug" if parent.is_buggy else "improve")
        logger.info(f"Agent step: {action}")

        if parent is None:
            node = self._draft()
        elif parent.is_buggy:
            node = self._debug(parent)
        else:
            node = self._improve(parent)

        # Execute via interpreter
        exec_result = exec_callback(node.code)

        # Parse results
        self._parse_result(node, exec_result)

        # Save node solution to workspace
        self._save_node_solution(node, exec_result)

        # Add to journal
        self.journal.append(node)

    def _parse_result(self, node: Node, exec_result: Any) -> None:
        """Parse execution result directly (no LLM needed)."""
        # Get build/run success directly from result
        build_success = getattr(exec_result, 'build_success', False)
        run_success = getattr(exec_result, 'run_success', False)
        output_generated = getattr(exec_result, 'output_generated', False)

        # Combine build and run output for complete error/validation info
        # This ensures debug stage sees compile errors, not just runtime errors
        # Note: _term_out must be a list (joined by term_out property)
        output_lines = []

        # Include build output on failure
        if hasattr(exec_result, 'build_output') and exec_result.build_output:
            build_out = exec_result.build_output
            if not build_success:
                output_lines.append("=== BUILD OUTPUT ===\n")
                if isinstance(build_out, list):
                    output_lines.extend(build_out)
                else:
                    output_lines.append(str(build_out))

        # Always include run output (contains validation results on success, errors on failure)
        if hasattr(exec_result, 'run_output') and exec_result.run_output:
            run_out = exec_result.run_output
            if isinstance(run_out, list):
                output_lines.extend(run_out)
            else:
                output_lines.append(str(run_out))

        # Use get_feedback() if available (provides structured error info)
        if hasattr(exec_result, 'get_feedback') and (not build_success or not run_success):
            output_lines.append("\n=== FEEDBACK ===\n")
            output_lines.append(exec_result.get_feedback())

        node._term_out = output_lines if output_lines else []

        # Get timing
        node.exec_time = getattr(exec_result, 'total_time', 0)

        # Determine if buggy
        node.is_buggy = not (build_success and run_success and output_generated)

        # Get validation metrics
        validation = getattr(exec_result, 'validation', None)

        # Set metric and build analysis summary
        analysis_parts = []

        if node.is_buggy:
            node.metric = WorstMetricValue()
            if not build_success:
                # Use specific error info if available
                error_type = getattr(exec_result, 'error_type', None)
                error_msg = getattr(exec_result, 'error_message', None)
                if error_type and error_msg:
                    analysis_parts.append(f"Build failed ({error_type}): {error_msg[:200]}")
                else:
                    analysis_parts.append("Build failed - compile errors")
            elif not run_success:
                error_type = getattr(exec_result, 'error_type', None)
                error_msg = getattr(exec_result, 'error_message', None)
                if error_type and error_msg:
                    analysis_parts.append(f"Runtime failed ({error_type}): {error_msg[:200]}")
                else:
                    analysis_parts.append("Runtime error during execution")
            elif not output_generated:
                analysis_parts.append("No output generated")
        else:
            accuracy = None
            if validation and validation.accuracy is not None:
                accuracy = validation.accuracy
                analysis_parts.append(f"Accuracy: {accuracy:.4f}")
                if hasattr(validation, 'max_error') and validation.max_error is not None:
                    analysis_parts.append(f"Max error: {validation.max_error:.6f}")
                if hasattr(validation, 'mean_error') and validation.mean_error is not None:
                    analysis_parts.append(f"Mean error: {validation.mean_error:.6f}")

            if accuracy is not None:
                node.metric = MetricValue(accuracy, maximize=True)
            else:
                node.metric = MetricValue(0.0, maximize=True)
                analysis_parts.append("Executed successfully but no accuracy metric")

        # Set analysis summary for journal memory
        node.analysis = "; ".join(analysis_parts) if analysis_parts else "No analysis available"

        logger.info(f"Node: buggy={node.is_buggy}, metric={node.metric}")

        # Extract lesson from failure (feedback LLM)
        self._update_lessons(node)

    def _save_node_solution(self, node: Node, exec_result: Any) -> None:
        """Save node solution to workspace/solutions/node_XXX/."""
        import json
        import shutil

        app_build_dir = getattr(exec_result, 'app_build_dir', None)
        if not app_build_dir or not app_build_dir.exists():
            return

        # Create solutions directory
        solutions_dir = self.workspace_dir / "solutions"
        solutions_dir.mkdir(exist_ok=True)

        # Node index
        node_idx = len(self.journal.nodes)
        node_dir = solutions_dir / f"node_{node_idx:03d}"

        # Copy app_build contents to node directory
        if node_dir.exists():
            shutil.rmtree(node_dir, ignore_errors=True)
        shutil.copytree(app_build_dir, node_dir, dirs_exist_ok=True)

        # Write metadata
        metadata = {
            "node_id": node_idx,
            "is_buggy": node.is_buggy,
            "metric": node.metric.value if node.metric else None,
            "analysis": node.analysis,
        }
        (node_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
