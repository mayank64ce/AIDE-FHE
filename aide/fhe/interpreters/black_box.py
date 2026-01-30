"""
Black Box FHE Interpreter.

Handles challenges with pre-encrypted testcases:
- Uses challenge's own Dockerfile
- Mounts pre-encrypted testcase directory
- Uses challenge's verifier.cpp for validation
"""

import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from .base import BaseInterpreter, ExecutionResult, ValidationResult

logger = logging.getLogger("aide.fhe")


class BlackBoxInterpreter(BaseInterpreter):
    """
    Interpreter for black box FHE challenges.

    Black box challenges have:
    - Pre-encrypted testcases (tests/testcase1/, tests/testcase2/, etc.)
    - Their own Dockerfile that builds solution + verifier
    - verifier.cpp that decrypts and validates output

    Workflow:
    1. Copy template files + inject eval() code
    2. Build using challenge's Dockerfile
    3. Run with mounted testcase directory
    4. Parse verifier output for accuracy
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = f"fhe-black-box-{self.spec.task}"
        self.current_testcase: Optional[Path] = None

    def _prepare_source(self, code: str) -> None:
        """
        Prepare source by copying challenge files + injecting code.

        Black box challenges need:
        - Dockerfile, CMakeLists.txt, verifier.cpp from challenge dir
        - templates/openfhe/* with eval() injected
        """
        challenge_dir = self.spec.challenge_dir

        # Copy challenge-level files
        for filename in ["Dockerfile", "CMakeLists.txt", "verifier.cpp"]:
            src = challenge_dir / filename
            if src.exists():
                shutil.copy2(src, self.workspace_dir / filename)

        # Copy template files
        if self.spec.template_dir and self.spec.template_dir.exists():
            template_dest = self.workspace_dir / "templates" / "openfhe"
            template_dest.mkdir(parents=True, exist_ok=True)

            for f in self.spec.template_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, template_dest / f.name)

            # Inject code into yourSolution.cpp
            solution_file = template_dest / "yourSolution.cpp"
            if solution_file.exists():
                self._inject_code(solution_file, code)

    def build(self) -> tuple[bool, list]:
        """Build using challenge's Dockerfile."""
        dockerfile = self.workspace_dir / "Dockerfile"

        if not dockerfile.exists():
            return False, ["ERROR: Dockerfile not found in challenge directory"]

        return self.docker_build(
            dockerfile=dockerfile,
            context=self.workspace_dir,
            tag=self.image_name,
            timeout=self.build_timeout,
        )

    def run(self, testcase_path: Optional[Path] = None) -> tuple[bool, list]:
        """
        Run solution with mounted testcase.

        The Dockerfile's CMD expects /data to contain:
        - cc.txt (CryptoContext)
        - key_pub.txt (Public key)
        - key_mult.txt (Multiplication key)
        - key_rot.txt (Rotation keys)
        - input.txt (Input ciphertext)
        - key_secret.txt (Secret key - for verification)
        - expected_output.txt (Expected values - for verification)
        """
        if testcase_path is None:
            # Use first available testcase
            if self.spec.testcase_dirs:
                testcase_path = self.spec.testcase_dirs[0]
            else:
                return False, ["ERROR: No testcase available"]

        self.current_testcase = testcase_path

        # Clean up any stale output.txt that might have been created by previous
        # runs (could be owned by root from Docker, preventing overwrites)
        output_file = testcase_path / "output.txt"
        if output_file.exists():
            try:
                output_file.unlink()
            except PermissionError:
                # If we can't delete it, try via sudo (won't work in most cases)
                subprocess.run(["sudo", "rm", "-f", str(output_file)],
                             capture_output=True, timeout=5)

        # Run with testcase mounted at /data
        return self.run_docker(
            image=self.image_name,
            command=[],  # Use Dockerfile's CMD
            volumes={str(testcase_path): "/data"},
            timeout=self.run_timeout,
        )

    def execute(self, code: str, testcase_path: Optional[Path] = None) -> ExecutionResult:
        """
        Execute solution for black box challenge.

        Overrides base to add runtime error detection from output.
        """
        result = ExecutionResult()
        start_time = time.time()

        try:
            # Prepare source
            self._prepare_source(code)

            # Build
            build_start = time.time()
            result.build_success, result.build_output = self.build()
            result.build_time = time.time() - build_start

            if not result.build_success:
                self._analyze_build_error(result)
                result.total_time = time.time() - start_time
                return result

            # Run
            run_start = time.time()
            run_ok, run_out = self.run(testcase_path)
            result.run_time = time.time() - run_start
            result.run_output = run_out

            # Check for runtime errors in output (docker may return 0 even on crash)
            output_text = "\n".join(run_out) if isinstance(run_out, list) else str(run_out)
            has_runtime_error = self._has_runtime_error(output_text)

            logger.info(f"BlackBox run_ok={run_ok}, has_runtime_error={has_runtime_error}")

            if not run_ok or has_runtime_error:
                result.run_success = False
                self._analyze_runtime_error(result)
                result.total_time = time.time() - start_time
                return result

            result.run_success = True

            # Check for output and validate
            if testcase_path is None and self.spec.testcase_dirs:
                testcase_path = self.spec.testcase_dirs[0]

            output_file = testcase_path / "output.txt" if testcase_path else None
            result.output_generated = output_file and output_file.exists()

            if result.output_generated and testcase_path:
                result.output_path = output_file
                # Pass run_out to validate() to avoid running solution twice
                result.validation = self.validate(output_file, testcase_path, run_out)

        except Exception as e:
            result.error_type = type(e).__name__
            result.error_message = str(e)
            logger.exception("BlackBox interpreter error")

        result.total_time = time.time() - start_time
        return result

    def validate(self, output_path: Path, testcase_path: Path, run_output: list = None) -> ValidationResult:
        """
        Parse validation from verifier output.

        The black box verifier prints results in format:
        - Accuracy: 0.95
        - Slots passed: 3900/4096
        - Fatal errors: 5

        Args:
            output_path: Path to output file
            testcase_path: Path to testcase directory
            run_output: Output from run() - if provided, parse this instead of re-running
        """
        result = ValidationResult()

        # Use provided run output if available, otherwise would need to re-run
        # (but we should always have run_output from execute())
        if run_output:
            output_text = "\n".join(run_output) if isinstance(run_output, list) else str(run_output)
        else:
            # Fallback: re-run (shouldn't happen in normal flow)
            logger.warning("validate() called without run_output - re-running solution")
            success, output = self.run_docker(
                image=self.image_name,
                command=[],
                volumes={str(testcase_path): "/data"},
                timeout=self.run_timeout,
            )
            output_text = "\n".join(output)

        # Parse accuracy
        acc_match = re.search(r'[Aa]ccuracy[:\s]+(\d+\.?\d*)', output_text)
        if acc_match:
            result.accuracy = float(acc_match.group(1))
            if result.accuracy > 1:
                result.accuracy /= 100  # Handle percentage

        # Parse slots
        slots_match = re.search(r'[Ss]lots\s+passed[:\s]+(\d+)/(\d+)', output_text)
        if slots_match:
            passed = int(slots_match.group(1))
            total = int(slots_match.group(2))
            result.total_slots = total
            if result.accuracy is None:
                result.accuracy = passed / total

        # Parse fatal errors
        fatal_match = re.search(r'[Ff]atal\s+errors?[:\s]+(\d+)', output_text)
        if fatal_match:
            result.fatal_error_count = int(fatal_match.group(1))

        # Parse mean/max error
        mean_match = re.search(r'[Mm]ean\s+error[:\s]+(\d+\.?\d*)', output_text)
        if mean_match:
            result.mean_error = float(mean_match.group(1))

        max_match = re.search(r'[Mm]ax\s+error[:\s]+(\d+\.?\d*)', output_text)
        if max_match:
            result.max_error = float(max_match.group(1))

        # Determine if passed
        threshold = self.spec.scoring.accuracy_threshold
        max_fatal = self.spec.scoring.max_fatal_errors

        if result.accuracy is not None:
            result.passed = (
                result.accuracy >= threshold and
                result.fatal_error_count <= max_fatal
            )

        return result

    def _has_runtime_error(self, output: str) -> bool:
        """
        Detect if output indicates a runtime error.

        Docker may return 0 even when the solution or verifier crashes.
        """
        output_lower = output.lower()

        # Common runtime error patterns
        runtime_patterns = [
            "openfheexception",
            "lbcrypto::",
            "config_error",
            "what():",
            "terminate called",
            "segmentation fault",
            "aborted",
            "core dumped",
            "ciphertext passed to decrypt is empty",
        ]
        for pattern in runtime_patterns:
            if pattern in output_lower:
                return True

        return False

    def cleanup(self) -> None:
        """Remove Docker image."""
        try:
            subprocess.run(
                ["docker", "rmi", "-f", self.image_name],
                capture_output=True,
                timeout=30,
            )
        except Exception:
            pass
