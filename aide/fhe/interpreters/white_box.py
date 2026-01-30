"""
White Box OpenFHE Interpreter.

Handles OpenFHE challenges validated with fherma-validator:
- Uses yashalabinc/fherma-validator Docker image
- Validator generates keys, encrypts inputs, runs solution, validates output
- Parses result.json for accuracy metrics
"""

import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Optional

from .base import BaseInterpreter, ExecutionResult, ValidationResult

logger = logging.getLogger("aide.fhe")

FHERMA_VALIDATOR_IMAGE = "yashalabinc/fherma-validator"


class WhiteBoxInterpreter(BaseInterpreter):
    """
    Interpreter for white box OpenFHE challenges.

    Uses fherma-validator which:
    1. Reads config.json for crypto parameters
    2. Generates CryptoContext and keys
    3. Encrypts test inputs
    4. Builds and runs the solution
    5. Decrypts output and validates accuracy
    6. Writes result.json

    Workflow:
    1. Copy template files to app_build/
    2. Inject eval() code into yourSolution.cpp
    3. Run fherma-validator with mounted directories
    4. Parse result.json for metrics
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app_build_dir = self.workspace_dir / "app_build"
        self.app_build_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_source(self, code: str) -> None:
        """
        Prepare app_build/ directory with template + injected code.

        If code contains ### CONFIG ### section, replace config.json entirely.
        """
        # Parse CONFIG and CODE sections
        config_json, actual_code = self._parse_code_with_config(code)

        # Clear previous build (ignore errors from Docker-created files)
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

        # Find solution file - look for yourSolution.cpp first, then any .cpp with eval()
        solution_file = self.app_build_dir / "yourSolution.cpp"
        if not solution_file.exists():
            # Look for any .cpp file containing eval() function
            for cpp_file in self.app_build_dir.glob("*.cpp"):
                if cpp_file.name == "main.cpp":
                    continue
                content = cpp_file.read_text()
                if "void" in content and "eval()" in content:
                    solution_file = cpp_file
                    break

        if solution_file.exists():
            self._inject_code(solution_file, actual_code)
        else:
            # No template - create solution.cpp
            (self.app_build_dir / "solution.cpp").write_text(actual_code)

    def _parse_code_with_config(self, code: str) -> tuple[str | None, str]:
        """
        Parse code that may contain CONFIG and CODE sections.

        Returns:
            (config_json, actual_code) - config_json is the raw JSON string, None if no CONFIG section
        """
        config_json = None
        actual_code = code

        # Check for ### CONFIG ### section
        if '### CONFIG ###' in code:
            parts = code.split('### CONFIG ###', 1)
            if len(parts) > 1:
                rest = parts[1]
                # Find where code section starts
                if '### CODE ###' in rest:
                    config_part, code_part = rest.split('### CODE ###', 1)
                    actual_code = code_part.strip()
                    config_json = config_part.strip()
                else:
                    # No explicit CODE marker - extract JSON block, rest is code
                    lines = rest.strip().split('\n')
                    config_lines = []
                    code_start = 0
                    in_json = False
                    brace_count = 0

                    for i, line in enumerate(lines):
                        if '{' in line:
                            in_json = True
                        if in_json:
                            config_lines.append(line)
                            brace_count += line.count('{') - line.count('}')
                            if brace_count <= 0:
                                code_start = i + 1
                                break

                    config_json = '\n'.join(config_lines)
                    actual_code = '\n'.join(lines[code_start:]).strip()

                # Validate JSON
                if config_json:
                    try:
                        parsed = json.loads(config_json)
                        logger.info(f"Parsed config with keys: {list(parsed.keys())}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid config JSON: {e}")
                        config_json = None

        elif '### CODE ###' in code:
            # Just CODE section, no config
            actual_code = code.split('### CODE ###', 1)[1].strip()

        return config_json, actual_code

    def _write_config(self, config_json: str) -> None:
        """
        Write config.json file (complete replacement).
        """
        config_file = self.app_build_dir / "config.json"
        config_file.write_text(config_json)
        logger.info("Replaced config.json with LLM-provided config")

    def build(self) -> tuple[bool, list]:
        """
        Build is handled by fherma-validator.

        This method just validates that files are in place.
        """
        required_files = ["yourSolution.cpp", "yourSolution.h", "main.cpp", "CMakeLists.txt"]
        missing = []

        for f in required_files:
            if not (self.app_build_dir / f).exists():
                missing.append(f)

        if missing:
            # Check for alternative file names
            if "yourSolution.cpp" in missing and (self.app_build_dir / "solution.cpp").exists():
                missing.remove("yourSolution.cpp")

        if missing:
            return False, [f"Missing required files: {missing}"]

        return True, ["Build files prepared"]

    def run(self, testcase_path: Optional[Path] = None) -> tuple[bool, list]:
        """
        Run solution using fherma-validator.

        fherma-validator expects:
        - --project-folder: Directory with solution source
        - --testcase: Path to test_case.json
        """
        # Find test_case.json
        if testcase_path and testcase_path.is_file():
            test_case_json = testcase_path
        elif testcase_path and testcase_path.is_dir():
            test_case_json = testcase_path / "test_case.json"
        elif self.spec.challenge_dir:
            test_case_json = self.spec.challenge_dir / "tests" / "test_case.json"
        else:
            return False, ["ERROR: No test_case.json found"]

        if not test_case_json.exists():
            return False, [f"ERROR: test_case.json not found at {test_case_json}"]

        # Mount paths
        # fherma-validator expects:
        #   /fherma/app_build/ - solution source
        #   /fherma/tests/test_case.json - test case

        # We mount the workspace directory at /fherma
        # Copy test_case.json to workspace/tests/
        tests_dir = self.workspace_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        shutil.copy2(test_case_json, tests_dir / "test_case.json")

        # Build volumes dict
        volumes = {str(self.workspace_dir): "/fherma"}

        # Also mount data directory if it exists (for KNN, etc.)
        if self.spec.challenge_dir:
            data_dir = self.spec.challenge_dir / "data"
            if data_dir.exists() and data_dir.is_dir():
                volumes[str(data_dir)] = "/fherma/data"
                logger.info(f"Mounting data directory: {data_dir} -> /fherma/data")

        return self.run_docker(
            image=FHERMA_VALIDATOR_IMAGE,
            command=[
                "--project-folder=/fherma/app_build",
                "--testcase=/fherma/tests/test_case.json",
            ],
            volumes=volumes,
            timeout=self.run_timeout,
        )

    def validate(self, output_path: Path, testcase_path: Path) -> ValidationResult:
        """
        Parse result.json written by fherma-validator.

        fherma-validator outputs:
        {
            "compilation_error": null,
            "testcases": [{
                "scheme": "CKKS",
                "runs": [{
                    "result": [...],
                    "expected_output": [...],
                    "time": 1.23
                }]
            }]
        }

        We compute accuracy as: % of slots where |result - expected| < threshold
        Threshold comes from spec.scoring.error_threshold (default 0.001).
        """
        result = ValidationResult()

        result_file = self.app_build_dir / "result.json"
        if not result_file.exists():
            result.details["error"] = "result.json not found"
            return result

        try:
            data = json.loads(result_file.read_text())
            result.details = data

            # Check for compilation error
            if data.get("compilation_error"):
                result.details["error"] = data["compilation_error"]
                return result

            # Parse fherma-validator testcases structure
            testcases = data.get("testcases", [])
            if not testcases:
                result.details["error"] = "No testcases in result.json"
                return result

            # Aggregate results across all testcases and runs
            all_errors = []
            total_slots = 0
            total_correct = 0

            # Get threshold from challenge spec
            threshold = 0.001  # Default
            if self.spec and self.spec.scoring:
                threshold = self.spec.scoring.error_threshold

            for tc in testcases:
                runs = tc.get("runs", [])
                for run in runs:
                    actual = run.get("result", [])
                    expected = run.get("expected_output", [])

                    if not actual or not expected:
                        continue

                    # Compute per-slot errors
                    for a, e in zip(actual, expected):
                        try:
                            err = abs(float(a) - float(e))
                            all_errors.append(err)
                            total_slots += 1
                            if err < threshold:
                                total_correct += 1
                        except (ValueError, TypeError):
                            # Skip non-numeric values
                            pass

            # Compute metrics
            if total_slots > 0:
                result.accuracy = total_correct / total_slots
                result.total_slots = total_slots
                result.mean_error = sum(all_errors) / len(all_errors)
                result.max_error = max(all_errors)
                result.passed = result.accuracy >= (self.spec.scoring.accuracy_threshold if self.spec and self.spec.scoring else 0.8)

                logger.info(
                    f"Validation: accuracy={result.accuracy:.4f} "
                    f"({total_correct}/{total_slots}), "
                    f"threshold={threshold}, "
                    f"max_error={result.max_error:.6f}"
                )
            else:
                result.details["error"] = "No valid result/expected pairs found"

        except json.JSONDecodeError as e:
            result.details["error"] = f"Failed to parse result.json: {e}"
        except Exception as e:
            result.details["error"] = f"Validation error: {e}"
            logger.exception("Error in validate()")

        return result

    def execute(self, code: str, testcase_path: Optional[Path] = None) -> ExecutionResult:
        """
        Execute solution with fherma-validator.

        Override to combine build+run into single fherma-validator call.
        """
        result = ExecutionResult()
        start_time = time.time()

        # Validate format: white-box requires config.json
        if "### CONFIG ###" not in code:
            result.build_success = False
            result.error_type = "FORMAT_ERROR"
            result.error_message = (
                "Missing ### CONFIG ### section. "
                "White-box challenges require both config.json and code."
            )
            result.total_time = time.time() - start_time
            return result

        try:
            # Prepare source
            self._prepare_source(code)

            # Validate files
            build_ok, build_out = self.build()
            result.build_output = build_out

            if not build_ok:
                result.build_success = False
                self._analyze_build_error(result)
                result.total_time = time.time() - start_time
                return result

            result.build_success = True

            # Run fherma-validator (does build + run + validation)
            run_start = time.time()
            run_ok, run_out = self.run(testcase_path)
            result.run_time = time.time() - run_start
            result.run_output = run_out

            # Parse output for build/run status
            output_text = "\n".join(run_out)

            # Detect error type from fherma-validator output
            is_build_error = self._is_build_error(output_text)

            # fherma-validator returns 0 even on runtime errors, so check output
            has_runtime_error = self._has_runtime_error(output_text)

            if not run_ok or is_build_error or has_runtime_error:
                if is_build_error:
                    # Build/compilation error - use run_out as build_output for analysis
                    result.build_success = False
                    result.run_success = False
                    result.build_output = run_out  # Use actual fherma output, not "Build files prepared"
                    self._analyze_build_error(result)
                else:
                    # Runtime error
                    result.run_success = False
                    self._analyze_runtime_error(result)
                result.total_time = time.time() - start_time
                return result

            result.run_success = True

            # Check for output
            result_file = self.app_build_dir / "result.json"
            result.output_generated = result_file.exists()

            if result.output_generated:
                result.output_path = result_file
                result.validation = self.validate(result_file, testcase_path)

        except Exception as e:
            result.error_type = type(e).__name__
            result.error_message = str(e)
            logger.exception("WhiteBox interpreter error")

        result.total_time = time.time() - start_time
        result.app_build_dir = self.app_build_dir
        return result

    def _has_runtime_error(self, output: str) -> bool:
        """
        Detect if fherma-validator output indicates a runtime error.

        fherma-validator returns 0 even when the app crashes, so we need to
        detect runtime errors from the output text.
        """
        output_lower = output.lower()

        # fherma-validator specific patterns
        if "run error:" in output_lower:
            return True
        if "return code: -" in output_lower:  # Negative return code = signal
            return True

        # OpenFHE exception patterns
        runtime_patterns = [
            "openfheexception",
            "lbcrypto::",
            "what():",
            "terminate called",
            "segmentation fault",
            "ciphertext passed to decrypt is empty",
        ]
        for pattern in runtime_patterns:
            if pattern in output_lower:
                return True

        return False

    def _is_build_error(self, output: str) -> bool:
        """
        Detect if error is from build/compilation phase vs runtime.

        Build errors include:
        - CMake errors
        - Compilation errors (g++ errors with file:line format)
        - Linker errors

        Runtime errors include:
        - OpenFHE exceptions (lbcrypto::)
        - Segfaults during execution
        - FHE-specific errors (depth exceeded, etc.)
        """
        output_lower = output.lower()

        # Runtime error indicators (check first - more specific)
        runtime_indicators = [
            "lbcrypto::",  # OpenFHE exception
            "openfheexception",
            "what():",  # C++ exception
            "segmentation fault",
            "removing last element",  # Depth exceeded
            "multiplicative depth",
            "cannot decrypt",
            "ciphertext passed to decrypt is empty",
        ]
        for indicator in runtime_indicators:
            if indicator in output_lower:
                return False

        # Build error indicators
        build_indicators = [
            "cmake error",
            "cmake warning",
            "make[",  # Make output like "make[2]: ***"
            "error: ",  # Compilation error
            "undefined reference",  # Linker error
            "fatal error:",  # Fatal compilation error
            "collect2: error",  # Linker error
            "cannot find -l",  # Missing library
            "no such file or directory",  # Missing header/file (during build)
        ]
        for indicator in build_indicators:
            if indicator in output_lower:
                return True

        # Check for file:line:col: error pattern (compilation error)
        if re.search(r'\w+\.cpp:\d+:\d+: error:', output):
            return True

        # Default: assume runtime error (safer for FHE challenges)
        return False

    def cleanup(self) -> None:
        """Cleanup app_build directory."""
        if self.app_build_dir.exists():
            shutil.rmtree(self.app_build_dir, ignore_errors=True)
