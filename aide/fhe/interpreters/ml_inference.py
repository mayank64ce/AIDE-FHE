"""
ML Inference FHE Interpreter - Two-Stage Approach.

Stage 1: Training (Plaintext)
- Agent generates training code (full Python script)
- System executes training → saves weights to workspace

Stage 2: Inference (Encrypted)
- Agent generates inference code using trained weights
- System runs fherma-validator for encrypted inference
"""

import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .white_box import WhiteBoxInterpreter, FHERMA_VALIDATOR_IMAGE
from .base import ExecutionResult

logger = logging.getLogger("aide.fhe")


@dataclass
class TrainingData:
    """Container for training data with metadata."""
    X_train: any  # numpy array or similar
    y_train: any
    X_shape: tuple
    y_shape: tuple
    num_classes: Optional[int] = None
    feature_names: Optional[list] = None
    data_format: str = "numpy"  # numpy, csv, parquet
    dtype: Optional[str] = None  # Data type (uint8, float32, etc.)
    value_range: Optional[tuple] = None  # (min, max) of feature values

    def summary(self) -> str:
        """Generate human-readable summary for agent prompt."""
        lines = [
            "Training Data:",
            f"  X_train shape: {self.X_shape}",
            f"  y_train shape: {self.y_shape}",
            f"  Data format: {self.data_format}",
        ]
        if self.dtype:
            lines.append(f"  Data type: {self.dtype}")
        if self.value_range:
            lines.append(f"  Value range: [{self.value_range[0]}, {self.value_range[1]}]")
        if self.num_classes:
            lines.append(f"  Number of classes: {self.num_classes}")
        if self.feature_names and len(self.feature_names) <= 20:
            lines.append(f"  Features: {self.feature_names}")
        return "\n".join(lines)


class TrainingDataLoader:
    """Load training data from various formats."""

    @staticmethod
    def load(data_dir: Path) -> Optional[TrainingData]:
        """Auto-detect and load training data."""
        if not data_dir.exists():
            return None

        # Try different formats
        loaders = [
            TrainingDataLoader._load_numpy,
            TrainingDataLoader._load_csv,
            TrainingDataLoader._load_parquet,
        ]

        for loader in loaders:
            result = loader(data_dir)
            if result:
                return result

        return None

    @staticmethod
    def _load_numpy(data_dir: Path) -> Optional[TrainingData]:
        """Load .npy files."""
        import numpy as np

        X_path = None
        y_path = None

        # Find X file (prefer flat/pre-processed versions first)
        for name in ["X_train_flat.npy", "X_train.npy", "x_train.npy"]:
            if (data_dir / name).exists():
                X_path = data_dir / name
                break

        # Find y file
        for name in ["y_train.npy", "Y_train.npy"]:
            if (data_dir / name).exists():
                y_path = data_dir / name
                break

        if not X_path or not y_path:
            return None

        X = np.load(X_path)
        y = np.load(y_path)

        num_classes = len(np.unique(y)) if y.ndim == 1 else None

        return TrainingData(
            X_train=X,
            y_train=y,
            X_shape=X.shape,
            y_shape=y.shape,
            num_classes=num_classes,
            data_format="numpy",
            dtype=str(X.dtype),
            value_range=(float(X.min()), float(X.max())),
        )

    @staticmethod
    def _load_csv(data_dir: Path) -> Optional[TrainingData]:
        """Load .csv files."""
        import pandas as pd
        import numpy as np

        X_path = data_dir / "X_train.csv"
        y_path = data_dir / "y_train.csv"

        if not X_path.exists() or not y_path.exists():
            return None

        X_df = pd.read_csv(X_path)
        y_df = pd.read_csv(y_path)

        X = X_df.values
        y = y_df.values.flatten() if y_df.shape[1] == 1 else y_df.values

        num_classes = len(np.unique(y)) if y.ndim == 1 and len(np.unique(y)) < 100 else None

        return TrainingData(
            X_train=X,
            y_train=y,
            X_shape=X.shape,
            y_shape=y.shape,
            num_classes=num_classes,
            feature_names=X_df.columns.tolist(),
            data_format="csv",
            dtype=str(X.dtype),
            value_range=(float(X.min()), float(X.max())),
        )

    @staticmethod
    def _load_parquet(data_dir: Path) -> Optional[TrainingData]:
        """Load .parquet files (e.g., sentiment analysis)."""
        import pandas as pd
        import numpy as np

        # Find parquet file
        parquet_files = list(data_dir.glob("*.parquet"))
        if not parquet_files:
            return None

        df = pd.read_parquet(parquet_files[0])

        # Handle different column names
        X = None
        y = None

        if "embedding" in df.columns:
            # Sentiment analysis format: embedding column contains vectors
            X = np.stack(df["embedding"].values)
            if "label" in df.columns:
                y = df["label"].values
        else:
            # Generic format: last column is target
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

        if X is None:
            return None

        num_classes = len(np.unique(y)) if y is not None and len(np.unique(y)) < 100 else None

        return TrainingData(
            X_train=X,
            y_train=y,
            X_shape=X.shape,
            y_shape=y.shape if y is not None else None,
            num_classes=num_classes,
            feature_names=df.columns.tolist(),
            data_format="parquet",
            dtype=str(X.dtype),
            value_range=(float(X.min()), float(X.max())),
        )


class MLInferenceInterpreter(WhiteBoxInterpreter):
    """
    Two-stage interpreter for ML inference FHE challenges.

    Stage 1: Training
    - Agent generates training code (train.py)
    - System executes training code
    - Weights saved to workspace/weights/

    Stage 2: Inference
    - Agent generates inference code (solve() body with embedded weights)
    - fherma-validator runs encrypted inference

    Code Format Expected:
    The agent provides both training and inference code in a single response:

    ```python
    ### TRAINING CODE ###
    # Full training script here...
    ```

    ```python
    ### INFERENCE CODE ###
    # solve() function body here...
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_python = self._detect_python_template()
        self.training_data: Optional[TrainingData] = None
        self.weights_dir = self.workspace_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # Track training state
        self.training_completed = False
        self.trained_weights_path: Optional[Path] = None

        # Load training data on init
        self._load_training_data()

    def _detect_python_template(self) -> bool:
        """Check if template uses Python (openfhe-python)."""
        if self.spec.template_dir:
            return "openfhe-python" in str(self.spec.template_dir)
        for f in self.spec.template_files:
            if f.endswith(".py"):
                return True
        return False

    def _load_training_data(self) -> None:
        """Load training data from challenge directory."""
        data_dir = self.spec.challenge_dir / "data"
        self.training_data = TrainingDataLoader.load(data_dir)
        if self.training_data:
            logger.info(f"Loaded training data: {self.training_data.X_shape}")

    def get_training_data_path(self) -> Path:
        """Return path to training data directory."""
        return self.spec.challenge_dir / "data"

    def get_training_data_summary(self) -> str:
        """Generate summary for agent prompt."""
        if self.training_data is None:
            return "No training data found."
        return self.training_data.summary()

    def get_data_loading_code(self) -> str:
        """Generate code snippet for loading training data."""
        data_dir = self.get_training_data_path()

        if self.training_data is None:
            return "# No training data available"

        if self.training_data.data_format == "numpy":
            # Check actual file names (prefer flat/pre-processed versions first)
            x_file = "X_train.npy"
            for name in ["X_train_flat.npy", "X_train.npy", "x_train.npy"]:
                if (data_dir / name).exists():
                    x_file = name
                    break
            y_file = "y_train.npy"
            for name in ["y_train.npy", "Y_train.npy"]:
                if (data_dir / name).exists():
                    y_file = name
                    break

            return f'''import numpy as np

# Load training data
X_train = np.load("{data_dir}/{x_file}")
y_train = np.load("{data_dir}/{y_file}")
print(f"X_train: {{X_train.shape}}, y_train: {{y_train.shape}}")
'''

        elif self.training_data.data_format == "csv":
            return f'''import pandas as pd
import numpy as np

# Load training data
X_train = pd.read_csv("{data_dir}/X_train.csv").values
y_train = pd.read_csv("{data_dir}/y_train.csv").values.flatten()
print(f"X_train: {{X_train.shape}}, y_train: {{y_train.shape}}")
'''

        elif self.training_data.data_format == "parquet":
            parquet_files = list(data_dir.glob("*.parquet"))
            if parquet_files:
                return f'''import pandas as pd
import numpy as np

# Load training data
df = pd.read_parquet("{parquet_files[0]}")
X_train = np.stack(df["embedding"].values)  # 768-dim embeddings
y_train = df["label"].values  # 0=negative, 1=neutral, 2=positive
print(f"X_train: {{X_train.shape}}, y_train: {{y_train.shape}}")
'''

        return "# Unknown data format"

    def validate(self, output_path: Path, testcase_path: Path) -> "ValidationResult":
        """
        Validate ML inference results.

        For classification tasks (num_classes > 1), uses argmax to determine
        predicted class and compares to expected class label.

        For regression tasks, falls back to parent's per-slot comparison.
        """
        from .base import ValidationResult

        result_file = self.app_build_dir / "result.json"
        if not result_file.exists():
            result = ValidationResult()
            result.details["error"] = "result.json not found"
            return result

        try:
            data = json.loads(result_file.read_text())

            # Check for compilation error
            if data.get("compilation_error"):
                result = ValidationResult()
                result.details = data
                result.details["error"] = data["compilation_error"]
                return result

            # Check if this is a classification task
            is_classification = (
                self.training_data is not None and
                self.training_data.num_classes is not None and
                self.training_data.num_classes > 1
            )

            if is_classification:
                return self._validate_classification(data)
            else:
                # Fall back to parent's per-slot validation for regression
                return super().validate(output_path, testcase_path)

        except json.JSONDecodeError as e:
            result = ValidationResult()
            result.details["error"] = f"Failed to parse result.json: {e}"
            return result
        except Exception as e:
            result = ValidationResult()
            result.details["error"] = f"Validation error: {e}"
            logger.exception("Error in validate()")
            return result

    def _validate_classification(self, data: dict) -> "ValidationResult":
        """
        Validate classification results using argmax.

        For each testcase:
        - result: list of logits/scores (one per class)
        - expected_output: list with single class index [class_idx]

        Accuracy = number of correct predictions / total predictions
        """
        from .base import ValidationResult

        result = ValidationResult()
        result.details = data

        testcases = data.get("testcases", [])
        if not testcases:
            result.details["error"] = "No testcases in result.json"
            return result

        total_predictions = 0
        correct_predictions = 0
        all_errors = []

        num_classes = self.training_data.num_classes

        for tc in testcases:
            runs = tc.get("runs", [])
            for run in runs:
                actual = run.get("result", [])
                expected = run.get("expected_output", [])

                # Check for runtime error
                if run.get("error"):
                    logger.warning(f"Runtime error in testcase: {run['error'][:200]}")
                    continue

                if not actual or not expected:
                    continue

                # Get expected class
                expected_class = expected[0] if isinstance(expected, list) else expected

                # Get predicted class via argmax of first num_classes values
                logits = actual[:num_classes] if len(actual) >= num_classes else actual
                if not logits:
                    continue

                try:
                    predicted_class = max(range(len(logits)), key=lambda i: float(logits[i]))
                    total_predictions += 1

                    if predicted_class == int(expected_class):
                        correct_predictions += 1

                    # Track error as distance from correct class logit
                    all_errors.append(0.0 if predicted_class == int(expected_class) else 1.0)

                    logger.debug(
                        f"Classification: expected={expected_class}, "
                        f"predicted={predicted_class}, logits={logits[:5]}..."
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error parsing classification result: {e}")

        # Compute accuracy
        if total_predictions > 0:
            result.accuracy = correct_predictions / total_predictions
            result.total_slots = total_predictions
            result.mean_error = 1.0 - result.accuracy  # Error rate
            result.max_error = 1.0 if correct_predictions < total_predictions else 0.0

            # Use accuracy threshold from spec
            threshold = 0.8
            if self.spec and self.spec.scoring:
                threshold = self.spec.scoring.accuracy_threshold

            result.passed = result.accuracy >= threshold

            logger.info(
                f"Classification validation: accuracy={result.accuracy:.4f} "
                f"({correct_predictions}/{total_predictions}), "
                f"threshold={threshold}"
            )
        else:
            result.details["error"] = "No valid predictions found"

        return result

    def _parse_two_stage_code(self, code: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse agent response to extract config, training and inference code.

        Expected format:
        ### CONFIG ###
        <json config>

        ### TRAINING CODE ###
        <training code>

        ### INFERENCE CODE ###
        <inference code>

        Returns:
            (config_json, training_code, inference_code) - any can be None if not found
        """
        config_json = None
        training_code = None
        inference_code = None

        # Pattern for config
        config_pattern = r'###\s*CONFIG\s*###\s*\n(.*?)(?=###\s*TRAINING|###\s*INFERENCE|$)'
        config_match = re.search(config_pattern, code, re.DOTALL | re.IGNORECASE)
        if config_match:
            config_json = config_match.group(1).strip()

        # Pattern 1: Look for explicit markers
        training_pattern = r'###\s*TRAINING\s+CODE\s*###\s*\n(.*?)(?=###\s*INFERENCE|$)'
        inference_pattern = r'###\s*INFERENCE\s+CODE\s*###\s*\n(.*?)$'

        training_match = re.search(training_pattern, code, re.DOTALL | re.IGNORECASE)
        inference_match = re.search(inference_pattern, code, re.DOTALL | re.IGNORECASE)

        if training_match:
            training_code = training_match.group(1).strip()
        if inference_match:
            inference_code = inference_match.group(1).strip()

        # Pattern 2: Look for code blocks with markers in comments
        if not training_code or not inference_code:
            # Find all code blocks
            code_blocks = re.findall(r'```(?:python)?\n(.*?)```', code, re.DOTALL)

            for block in code_blocks:
                block_lower = block.lower()
                if 'training' in block_lower[:200] or 'train' in block_lower[:100]:
                    if not training_code:
                        training_code = block.strip()
                elif 'inference' in block_lower[:200] or 'solve' in block_lower[:100]:
                    if not inference_code:
                        inference_code = block.strip()

        # Pattern 3: If only one code block and it's the inference (solve body)
        if not inference_code and not training_code:
            # Fall back to treating entire code as inference (backward compatibility)
            inference_code = code.strip()

        return config_json, training_code, inference_code

    def _log_available_weights(self) -> list:
        """
        Log available weight files for debugging.

        Weight files are now loaded by the LLM-generated code at runtime,
        not auto-injected. This function just logs what's available.

        Weights at workspace/weights/ are accessible at /fherma/weights/ inside Docker.
        """
        weight_files = []

        # Check for binary weight files (.bin) - new preferred format
        for wfile in self.weights_dir.glob("*.bin"):
            size = wfile.stat().st_size
            # Check both float32 (4 bytes) and double (8 bytes) interpretations
            count_f32 = size // 4  # 4 bytes per float32
            count_f64 = size // 8  # 8 bytes per double
            weight_files.append(f"{wfile.name}: {count_f32} floats or {count_f64} doubles ({size} bytes)")
            logger.info(f"Weight file available: {wfile.name} ({count_f32} float32 or {count_f64} double)")

        # Also check for legacy numpy files (.npy)
        for wfile in self.weights_dir.glob("*.npy"):
            try:
                import numpy as np
                arr = np.load(wfile)
                weight_files.append(f"{wfile.name}: shape={arr.shape}, size={arr.size}")
                logger.info(f"Legacy numpy weight: {wfile.name} shape={arr.shape}")
            except Exception as e:
                weight_files.append(f"{wfile.name}: error loading - {e}")
                logger.warning(f"Failed to inspect {wfile}: {e}")

        if not weight_files:
            logger.warning("No weight files found in weights directory")

        return weight_files

    def execute_training(self, training_code: str) -> Tuple[bool, str, Optional[Path]]:
        """
        Execute training code and return weights path.

        Args:
            training_code: Python code that trains a model and saves weights

        Returns:
            (success, output, weights_path)
        """
        # Write training code
        train_script = self.workspace_dir / "train.py"
        train_script.write_text(training_code)

        logger.info(f"Executing training script: {train_script}")

        # Run training
        try:
            env = os.environ.copy()
            result = subprocess.run(
                ["python3", str(train_script)],
                capture_output=True,
                text=True,
                timeout=1200,  # 20 minutes for training
                cwd=str(self.workspace_dir),
                env=env,
            )

            output = result.stdout + "\n" + result.stderr
            success = result.returncode == 0

            logger.info(f"Training {'succeeded' if success else 'failed'}")
            if not success:
                logger.debug(f"Training output:\n{output[:2000]}")

            # Check for weights file in weights_dir
            weights_path = None
            for ext in [".npy", ".npz", ".json", ".pkl", ".pt", ".pth"]:
                for wfile in self.weights_dir.glob(f"*{ext}"):
                    weights_path = wfile
                    logger.info(f"Found weights: {weights_path}")
                    break
                if weights_path:
                    break

            # Also check workspace root for weights*
            if not weights_path:
                for ext in [".npy", ".npz", ".json", ".pkl"]:
                    for wfile in self.workspace_dir.glob(f"weights*{ext}"):
                        weights_path = wfile
                        logger.info(f"Found weights in workspace: {weights_path}")
                        break
                    if weights_path:
                        break

            # Check for any .npy files in workspace (common pattern)
            if not weights_path:
                for wfile in self.workspace_dir.glob("*.npy"):
                    weights_path = wfile
                    logger.info(f"Found .npy file: {weights_path}")
                    break

            return success, output, weights_path

        except subprocess.TimeoutExpired:
            return False, "Training timeout (>20 minutes)", None
        except Exception as e:
            return False, f"Training error: {e}", None

    def execute(self, code: str, testcase_path: Optional[Path] = None) -> ExecutionResult:
        """
        Execute two-stage ML inference.

        1. Parse code to extract training and inference portions
        2. Execute training if present
        3. Execute inference via fherma-validator

        Args:
            code: Combined code from agent (training + inference)
            testcase_path: Path to testcase directory

        Returns:
            ExecutionResult with combined results
        """
        start_time = time.time()

        # Reset training state for each new execution
        # Each code submission has its own training code
        self.reset_training()

        # Parse the two-stage code (config, training, inference)
        config_json, training_code, inference_code = self._parse_two_stage_code(code)

        # Store config for later use with inference (applied by parent's _prepare_source)
        self._pending_config_json = config_json

        build_output = []
        run_output = []

        # Stage 1: Training (if training code present and not already completed)
        if training_code and not self.training_completed:
            build_output.append("=== STAGE 1: TRAINING ===\n")

            success, output, weights_path = self.execute_training(training_code)
            build_output.append(output)

            if not success:
                result = ExecutionResult(
                    build_success=False,
                    run_success=False,
                    output_generated=False,
                    build_output=build_output,
                    run_output=["Training failed - cannot proceed to inference"],
                    total_time=time.time() - start_time,
                    validation=None,
                )
                # Set error info for proper feedback
                result.error_type = "TRAINING_ERROR"
                result.error_message = self._extract_training_error(output)
                return result

            if weights_path:
                self.trained_weights_path = weights_path
                build_output.append(f"\nWeights saved to: {weights_path}")
            else:
                build_output.append("\nWarning: No weights file found after training")

            self.training_completed = True
            build_output.append("\n=== Training completed ===\n")

        # Stage 2: Inference
        if not inference_code:
            result = ExecutionResult(
                build_success=True,
                run_success=False,
                output_generated=False,
                build_output=build_output,
                run_output=["No inference code provided"],
                total_time=time.time() - start_time,
                validation=None,
            )
            result.error_type = "MISSING_CODE"
            result.error_message = "No inference code provided - include ### INFERENCE CODE ### section"
            return result

        run_output.append("=== STAGE 2: INFERENCE ===\n")

        # Log available weights (LLM code loads them at runtime from /fherma/weights/)
        weight_files = self._log_available_weights()
        if weight_files:
            run_output.append(f"Weight files available at /fherma/weights/: {weight_files}\n")

        # Use parent class (WhiteBoxInterpreter) for inference execution
        # This injects the inference code into the template and runs fherma-validator
        # Include config with inference code so parent's _prepare_source applies it
        if self._pending_config_json:
            full_inference_code = f"### CONFIG ###\n{self._pending_config_json}\n\n### CODE ###\n{inference_code}"
        else:
            full_inference_code = inference_code
        inference_result = super().execute(full_inference_code, testcase_path)

        # Combine results
        if hasattr(inference_result, 'build_output') and inference_result.build_output:
            if isinstance(inference_result.build_output, list):
                build_output.extend(inference_result.build_output)
            else:
                build_output.append(str(inference_result.build_output))

        if hasattr(inference_result, 'run_output') and inference_result.run_output:
            if isinstance(inference_result.run_output, list):
                run_output.extend(inference_result.run_output)
            else:
                run_output.append(str(inference_result.run_output))

        result = ExecutionResult(
            build_success=inference_result.build_success,
            run_success=inference_result.run_success,
            output_generated=inference_result.output_generated,
            build_output=build_output,
            run_output=run_output,
            total_time=time.time() - start_time,
            validation=inference_result.validation,
        )
        # Copy error info from inference result if present
        if hasattr(inference_result, 'error_type') and inference_result.error_type:
            result.error_type = inference_result.error_type
        if hasattr(inference_result, 'error_message') and inference_result.error_message:
            result.error_message = inference_result.error_message
        return result

    def _prepare_source(self, code: str) -> None:
        """
        Prepare source for inference.

        The code should contain trained weights embedded (or be the solve() body).
        """
        # Parse CONFIG and CODE sections (same logic as parent)
        config_json, actual_code = self._parse_code_with_config(code)

        # Clear previous build
        if self.app_build_dir.exists():
            shutil.rmtree(self.app_build_dir, ignore_errors=True)
        self.app_build_dir.mkdir(parents=True, exist_ok=True)

        # Copy template files
        if self.spec.template_dir and self.spec.template_dir.exists():
            for f in self.spec.template_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, self.app_build_dir / f.name)

        # Replace config.json if provided
        if config_json:
            self._write_config(config_json)

        # Use actual code (without config section) for injection
        code = actual_code

        # Inject code
        if self.is_python:
            solution_file = self.app_build_dir / "app.py"
            if solution_file.exists():
                self._inject_python_code(solution_file, code)
            else:
                (self.app_build_dir / "app.py").write_text(code)
        else:
            # C++ template - need to handle weights at file scope, body in eval()
            solution_file = self.app_build_dir / "yourSolution.cpp"
            if not solution_file.exists():
                for cpp_file in self.app_build_dir.glob("*.cpp"):
                    if cpp_file.name not in ["main.cpp"]:
                        solution_file = cpp_file
                        break

            if solution_file.exists():
                self._inject_cpp_code(solution_file, code)

    def _inject_python_code(self, solution_file: Path, code: str) -> None:
        """Inject code into Python solution file."""
        template = solution_file.read_text()

        # Separate weight declarations from solve body
        weight_declarations, solve_body = self._split_weights_and_body(code)

        # If code contains its own "def solve" definition, extract just the body
        solve_body = self._extract_solve_body(solve_body)

        # Insert weight declarations after imports, before class definition
        if weight_declarations:
            # Find position after imports
            insert_pos = 0
            import_pattern = r'(?:^(?:import|from)\s+.+$\n?)+'
            import_match = re.search(import_pattern, template, re.MULTILINE)
            if import_match:
                insert_pos = import_match.end()
            else:
                # Insert at beginning
                insert_pos = 0

            template = template[:insert_pos] + "\n" + weight_declarations + "\n" + template[insert_pos:]

        # Pattern 1: def solve(...): with placeholder body
        # Match the entire solve function and replace its body
        pattern = r'(def\s+solve\s*\([^)]*\)\s*:)\s*\n(\s+)(?:# put your solution here\n\s+output = Ciphertext\(\)\n\s+return output|pass|\.\.\.)'
        match = re.search(pattern, template)
        if match:
            indent = match.group(2)
            # Indent the code properly
            indented_code = "\n".join(indent + line if line.strip() else line
                                       for line in solve_body.split("\n"))
            new_content = template[:match.start()] + match.group(1) + "\n" + indented_code + template[match.end():]
            solution_file.write_text(new_content)
            return

        # Pattern 2: Just replace placeholder comment
        pattern = r'#\s*put your solution here.*?return output'
        if re.search(pattern, template, re.DOTALL):
            new_content = re.sub(pattern, solve_body, template, flags=re.DOTALL)
            solution_file.write_text(new_content)
            return

        # Fallback: append (this is for Python only since we're in _inject_python_code)
        with open(solution_file, 'a') as f:
            f.write("\n# Agent-generated code\n")
            f.write(solve_body)

    def _inject_cpp_code(self, solution_file: Path, code: str) -> None:
        """Inject code into C++ solution file.

        The LLM's code includes weight loading logic directly in the eval() body.
        We just need to inject the eval() body into the template.
        """
        template = solution_file.read_text()

        # Strip any ### markers and host #includes
        eval_body = self._strip_code_markers(code)

        # Strip void eval() { } wrapper if LLM included it (we only want the body)
        eval_body = self._extract_cpp_eval_body(eval_body)

        # Inject eval body into eval() function
        # Pattern 1: Empty eval function body
        pattern1 = r'(void\s+\w+::eval\s*\(\s*\)\s*\{)\s*(\})'
        if re.search(pattern1, template):
            new_content = re.sub(pattern1, rf'\1\n{eval_body}\n\2', template)
            solution_file.write_text(new_content)
            return

        # Pattern 2: TODO comment in eval
        pattern2 = r'(void\s+\w+::eval\s*\(\s*\)\s*\{[^}]*)(//\s*TODO[^\n]*\n)([^}]*\})'
        if re.search(pattern2, template):
            new_content = re.sub(pattern2, rf'\1{eval_body}\n\3', template)
            solution_file.write_text(new_content)
            return

        # Pattern 3: Your implementation comment
        pattern3 = r'//\s*[Yy]our\s+implementation\s+here[^\n]*\n'
        if re.search(pattern3, template):
            new_content = re.sub(pattern3, f'{eval_body}\n', template)
            solution_file.write_text(new_content)
            return

        # Fallback: just inject into empty eval
        logger.warning("Could not find C++ injection point, using fallback")
        solution_file.write_text(template)

    def _strip_code_markers(self, code: str) -> str:
        """Remove ### markers and problematic #include statements from code."""
        lines = code.split('\n')
        filtered = []
        for line in lines:
            stripped = line.strip()
            # Skip marker lines (### TRAINING/INFERENCE/CONFIG ###)
            if stripped.startswith('###') and stripped.endswith('###'):
                continue
            # Skip #include statements for host filesystem paths
            # These won't exist inside Docker container
            if stripped.startswith('#include') and ('/mnt/' in stripped or '/home/' in stripped or '/tmp/' in stripped):
                logger.warning(f"Stripping host filesystem #include: {stripped}")
                continue
            filtered.append(line)
        return '\n'.join(filtered)

    def _extract_solve_body(self, code: str) -> str:
        """Extract the body from a def solve(): definition if present."""
        # Pattern: def solve(...):
        match = re.search(r'^def\s+solve\s*\([^)]*\)\s*:\s*$', code, re.MULTILINE)
        if match:
            # Find the function body (everything after the def line, dedented)
            after_def = code[match.end():]
            lines = after_def.split('\n')

            # Find the base indentation
            body_lines = []
            base_indent = None
            for line in lines:
                if line.strip():  # Non-empty line
                    if base_indent is None:
                        # Count leading spaces
                        base_indent = len(line) - len(line.lstrip())
                    # Remove the base indentation
                    if line[:base_indent].strip() == '':
                        body_lines.append(line[base_indent:])
                    else:
                        body_lines.append(line.lstrip())
                else:
                    body_lines.append('')

            return '\n'.join(body_lines).strip()

        return code

    def _extract_cpp_eval_body(self, code: str) -> str:
        """Extract the body from a void eval() { } definition if present.

        LLMs sometimes wrap their C++ inference code in void eval() { }
        or void ClassName::eval() { } but we only want the body since it
        gets injected into the template's existing void ClassName::eval() function.
        """
        code = code.strip()

        # Pattern 1: void eval() or void ClassName::eval() anywhere in the code
        # Use re.search since LLM may include comments before the function definition
        pattern = r'void\s+(?:\w+::)?eval\s*\(\s*\)\s*\{'
        match = re.search(pattern, code)
        if match:
            # Find matching closing brace
            after_open = code[match.end():]

            # Track brace depth to find the matching close
            depth = 1
            pos = 0
            for i, ch in enumerate(after_open):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        pos = i
                        break

            if depth == 0:
                # Extract body between { and matching }
                body = after_open[:pos].strip()
                logger.debug(f"Stripped void eval() wrapper, body length: {len(body)}")
                return body

        # Pattern 2: Just { ... } wrapper (already stripped the void eval part)
        if code.startswith('{') and code.endswith('}'):
            # Check if it's a simple wrapper (only one top-level brace pair)
            depth = 0
            is_wrapper = True
            for i, ch in enumerate(code):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and i < len(code) - 1:
                        # Found a closing brace before the end - not a simple wrapper
                        is_wrapper = False
                        break

            if is_wrapper and depth == 0:
                # Strip outer braces
                body = code[1:-1].strip()
                logger.debug(f"Stripped outer braces wrapper, body length: {len(body)}")
                return body

        return code

    def build(self) -> tuple[bool, list]:
        """Validate build files."""
        if self.is_python:
            required = ["app.py", "config.json"]
        else:
            required = ["CMakeLists.txt"]

        missing = [f for f in required if not (self.app_build_dir / f).exists()]

        if missing:
            return False, [f"Missing required files: {missing}"]

        return True, ["Build files prepared"]

    def reset_training(self) -> None:
        """Reset training state for a fresh attempt."""
        self.training_completed = False
        self.trained_weights_path = None

        # Clear weights directory
        if self.weights_dir.exists():
            shutil.rmtree(self.weights_dir, ignore_errors=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)

    def _extract_training_error(self, output: str) -> str:
        """Extract meaningful error message from training output."""
        # Look for common Python error patterns
        patterns = [
            r'(\w+Error): (.+?)(?:\n|$)',  # NameError: name 'x' is not defined
            r'(\w+Exception): (.+?)(?:\n|$)',  # ValueError: ...
            r'Traceback.*?\n(\w+Error: .+?)(?:\n|$)',  # Last line of traceback
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.DOTALL)
            if match:
                if match.lastindex == 2:
                    return f"{match.group(1)}: {match.group(2)}"[:200]
                else:
                    return match.group(1)[:200]

        # Look for "Error:" or "error:" lines
        for line in output.split('\n'):
            if 'error' in line.lower() and len(line.strip()) > 10:
                return line.strip()[:200]

        # Fallback: last non-empty line
        lines = [l.strip() for l in output.split('\n') if l.strip()]
        if lines:
            return lines[-1][:200]

        return "Training failed - check output for details"
