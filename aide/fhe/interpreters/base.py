"""
Base interpreter for FHE challenges.

Provides common functionality for all challenge types:
- Template loading and code injection
- Docker execution
- Result parsing
"""

import logging
import os
import re
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dataclasses_json import DataClassJsonMixin

from ..challenge_parser import FHEChallengeSpec

logger = logging.getLogger("aide.fhe")


@dataclass
class ValidationResult(DataClassJsonMixin):
    """Result from validating a solution."""
    passed: bool = False
    accuracy: Optional[float] = None
    error_count: int = 0
    fatal_error_count: int = 0
    total_slots: int = 0
    mean_error: float = 0.0
    max_error: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """Result of executing an FHE solution."""

    # Build phase
    build_success: bool = False
    build_output: list = field(default_factory=list)
    build_time: float = 0.0

    # Run phase
    run_success: bool = False
    run_output: list = field(default_factory=list)
    run_time: float = 0.0

    # Output
    output_generated: bool = False
    output_path: Optional[Path] = None

    # Validation
    validation: Optional[ValidationResult] = None

    # Error info
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Timing
    total_time: float = 0.0

    # Build directory (for saving solutions)
    app_build_dir: Optional[Path] = None

    @property
    def success(self) -> bool:
        """True if build and run succeeded."""
        return self.build_success and self.run_success

    @property
    def accuracy(self) -> Optional[float]:
        """Accuracy from validation, if available."""
        if self.validation:
            return self.validation.accuracy
        return None

    def _filter_docker_noise(self, output: list) -> list:
        """Filter out Docker build step noise, keep actual error output."""
        filtered = []
        skip_patterns = [
            "Step ",  # Docker build steps like "Step 1/17"
            " ---> ",  # Docker layer hashes
            "Using cache",
            "Sending build context",
        ]

        for line in output:
            # Skip Docker build step noise
            if any(p in line for p in skip_patterns):
                continue
            # Keep everything else (errors, warnings, make output, etc.)
            if line.strip():
                filtered.append(line)

        return filtered if filtered else output[-50:]  # Fallback to last 50 lines

    def get_feedback(self) -> str:
        """Generate feedback string for the agent."""
        lines = []

        if not self.build_success:
            lines.append("BUILD FAILED")
            lines.append(f"Error: {self.error_type}")
            lines.append(f"Message: {self.error_message}")
            lines.append("")
            lines.append("Build output:")
            filtered_output = self._filter_docker_noise(self.build_output)
            lines.extend(filtered_output[-50:])  # Last 50 relevant lines
            return "\n".join(lines)

        if not self.run_success:
            lines.append("RUNTIME FAILED")
            lines.append(f"Error: {self.error_type}")
            lines.append(f"Message: {self.error_message}")
            lines.append("")
            lines.append("Runtime output (last 50 lines):")
            lines.extend(self.run_output[-50:])
            return "\n".join(lines)

        if not self.output_generated:
            lines.append("NO OUTPUT GENERATED")
            lines.append("The solution ran but did not produce an output file.")
            lines.append("Check that output is written to the correct path.")
            return "\n".join(lines)

        if self.validation:
            lines.append(f"VALIDATION: {'PASSED' if self.validation.passed else 'FAILED'}")
            lines.append(f"Accuracy: {self.validation.accuracy:.4f}" if self.validation.accuracy else "Accuracy: N/A")
            lines.append(f"Mean error: {self.validation.mean_error:.6f}")
            lines.append(f"Max error: {self.validation.max_error:.6f}")
            lines.append(f"Slots: {self.validation.total_slots}")
            return "\n".join(lines)

        lines.append("EXECUTION COMPLETE")
        lines.append(f"Build time: {self.build_time:.2f}s")
        lines.append(f"Run time: {self.run_time:.2f}s")
        return "\n".join(lines)


class BaseInterpreter(ABC):
    """
    Base class for FHE interpreters.

    Handles:
    - Loading template files
    - Injecting generated code into templates
    - Docker execution
    - Result parsing

    Subclasses implement:
    - build(): Build the solution
    - run(): Run the solution
    - validate(): Validate the output
    """

    def __init__(
        self,
        spec: FHEChallengeSpec,
        workspace_dir: Path | str,
        build_timeout: int = 600,
        run_timeout: int = 6000,
    ):
        self.spec = spec
        self.workspace_dir = Path(workspace_dir).resolve()
        self.build_timeout = build_timeout
        self.run_timeout = run_timeout

        # Create workspace directories
        self.src_dir = self.workspace_dir / "src"
        self.build_dir = self.workspace_dir / "build"
        self.output_dir = self.workspace_dir / "output"

        self.src_dir.mkdir(parents=True, exist_ok=True)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, code: str, testcase_path: Optional[Path] = None) -> ExecutionResult:
        """
        Execute a solution.

        Args:
            code: Generated code (just the eval() function body or full file)
            testcase_path: Path to testcase directory (optional)

        Returns:
            ExecutionResult with build/run/validation status
        """
        result = ExecutionResult()
        start_time = time.time()

        try:
            # 1. Prepare source files
            self._prepare_source(code)

            # 2. Build
            build_start = time.time()
            result.build_success, result.build_output = self.build()
            result.build_time = time.time() - build_start

            if not result.build_success:
                self._analyze_build_error(result)
                result.total_time = time.time() - start_time
                return result

            # 3. Run
            run_start = time.time()
            result.run_success, result.run_output = self.run(testcase_path)
            result.run_time = time.time() - run_start

            if not result.run_success:
                self._analyze_runtime_error(result)
                result.total_time = time.time() - start_time
                return result

            # 4. Check output
            result.output_generated, result.output_path = self._check_output()

            # 5. Validate
            if result.output_generated and testcase_path:
                result.validation = self.validate(result.output_path, testcase_path)

        except Exception as e:
            result.error_type = type(e).__name__
            result.error_message = str(e)
            logger.exception("Interpreter error")

        result.total_time = time.time() - start_time
        return result

    def _prepare_source(self, code: str) -> None:
        """
        Prepare source files from template + generated code.

        Default behavior: copy template files and inject code into yourSolution.cpp
        Subclasses can override for different behavior.
        """
        # Copy template files if available
        if self.spec.template_dir and self.spec.template_dir.exists():
            for f in self.spec.template_dir.iterdir():
                if f.is_file():
                    dest = self.src_dir / f.name
                    shutil.copy2(f, dest)
                    logger.debug(f"Copied template: {f.name}")

        # Inject code into yourSolution.cpp
        solution_file = self.src_dir / "yourSolution.cpp"
        if solution_file.exists():
            self._inject_code(solution_file, code)
        else:
            # No template - write code as solution.cpp
            (self.src_dir / "solution.cpp").write_text(code)

    def _inject_code(self, solution_file: Path, code: str) -> None:
        """
        Inject generated code into the eval() function.

        Looks for placeholder comments like:
        - // TODO: Implement eval()
        - // Your implementation here
        - void CKKSTaskSolver::eval() { }
        - void ClassName::eval() { // Agent implements... }
        """
        template = solution_file.read_text()

        # Strip any void eval() { } wrapper if LLM included it
        code = self._strip_eval_wrapper(code)

        # Pattern 1: Empty eval function body (just whitespace)
        pattern1 = r'(void\s+\w+::eval\s*\(\s*\)\s*\{)\s*(\})'
        if re.search(pattern1, template):
            new_content = re.sub(pattern1, rf'\1\n{code}\n\2', template)
            solution_file.write_text(new_content)
            return

        # Pattern 2: eval() with only comment lines inside (common template pattern)
        # Matches: void ClassName::eval() { // comments only \n // more comments \n }
        pattern2 = r'(void\s+\w+::eval\s*\(\s*\)\s*\{)\s*((?:\s*//[^\n]*\n)+)\s*(\})'
        if re.search(pattern2, template):
            new_content = re.sub(pattern2, rf'\1\n{code}\n\3', template)
            solution_file.write_text(new_content)
            return

        # Pattern 3: TODO comment anywhere in eval
        pattern3 = r'(void\s+\w+::eval\s*\(\s*\)\s*\{[^}]*)(//\s*TODO[^\n]*\n)([^}]*\})'
        if re.search(pattern3, template):
            new_content = re.sub(pattern3, rf'\1{code}\n\3', template)
            solution_file.write_text(new_content)
            return

        # Pattern 4: Your implementation comment
        pattern4 = r'//\s*[Yy]our\s+implementation\s+here[^\n]*\n'
        if re.search(pattern4, template):
            new_content = re.sub(pattern4, f'{code}\n', template)
            solution_file.write_text(new_content)
            return

        # Pattern 5: Find eval() and replace body using brace matching
        match = re.search(r'void\s+\w+::eval\s*\(\s*\)\s*\{', template)
        if match:
            start = match.end()
            # Find matching closing brace
            depth = 1
            pos = start
            while pos < len(template) and depth > 0:
                if template[pos] == '{':
                    depth += 1
                elif template[pos] == '}':
                    depth -= 1
                pos += 1
            if depth == 0:
                # Found matching brace - replace body
                new_content = template[:start] + '\n' + code + '\n' + template[pos-1:]
                solution_file.write_text(new_content)
                return

        # Fallback: Replace entire file
        logger.warning("Could not find injection point, replacing entire file")
        solution_file.write_text(code)

    def _strip_eval_wrapper(self, code: str) -> str:
        """Strip void eval() { } wrapper if LLM included it."""
        code = code.strip()

        # Pattern: void eval() { ... } or void ClassName::eval() { ... }
        # Use re.search since LLM may include comments before the function definition
        match = re.search(r'void\s+(?:\w+::)?eval\s*\(\s*\)\s*\{', code)
        if match:
            # Find matching closing brace
            after_open = code[match.end():]
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
                return after_open[:pos].strip()

        return code

    @abstractmethod
    def build(self) -> tuple[bool, list]:
        """
        Build the solution.

        Returns:
            (success, output_lines)
        """
        pass

    @abstractmethod
    def run(self, testcase_path: Optional[Path] = None) -> tuple[bool, list]:
        """
        Run the solution.

        Args:
            testcase_path: Path to testcase directory

        Returns:
            (success, output_lines)
        """
        pass

    @abstractmethod
    def validate(self, output_path: Path, testcase_path: Path) -> ValidationResult:
        """
        Validate the solution output.

        Args:
            output_path: Path to output ciphertext
            testcase_path: Path to testcase directory

        Returns:
            ValidationResult with accuracy metrics
        """
        pass

    def _check_output(self) -> tuple[bool, Optional[Path]]:
        """Check if output was generated."""
        output_names = ["output.bin", "output.txt", "result.bin", "result.txt"]

        for name in output_names:
            path = self.output_dir / name
            if path.exists() and path.stat().st_size > 0:
                return True, path

        return False, None

    def _analyze_build_error(self, result: ExecutionResult) -> None:
        """Analyze build error and extract the actual error message."""
        output = "\n".join(result.build_output)
        result.error_type = "BUILD_ERROR"
        result.error_message = self._extract_error_message(output)

    def _analyze_runtime_error(self, result: ExecutionResult) -> None:
        """Analyze runtime error and extract the actual error message."""
        output = "\n".join(result.run_output)
        result.error_type = "RUNTIME_ERROR"
        result.error_message = self._extract_error_message(output)

    def _extract_error_message(self, output: str) -> str:
        """
        Extract clean, actionable error message from output.

        For FHE challenges, extracts just the essential info:
        - Missing key index → "EvalKey for index [X] not found"
        - Depth exceeded → "Depth exhausted" or "levels exceeded"
        - Feature not enabled → "Enable(X) must be called"
        - Empty ciphertext → "ciphertext passed to Decrypt is empty"
        """
        # 1. Try to find C++ exception what() message
        what_match = re.search(r"what\(\):\s*(.+?)(?:\n|$)", output)
        if what_match:
            raw_msg = what_match.group(1).strip()
            # Extract just the error message, removing file paths
            # Format: /path/to/file.cpp:123:FunctionName(): actual message
            # We want just "actual message"
            clean_msg = self._clean_fhe_error(raw_msg)
            return clean_msg

        # 2. Look for specific FHE error patterns
        fhe_patterns = [
            (r"EvalKey for index \[(\d+)\] is not found", lambda m: f"EvalKey for index [{m.group(1)}] not found"),
            (r"EvalAtIndex.*index\s*\[?(\d+)\]?.*not found", lambda m: f"Rotation key for index {m.group(1)} not found"),
            (r"ciphertext passed to (\w+) is empty", lambda m: f"Ciphertext passed to {m.group(1)} is empty"),
            (r"Removing last element.*renders it invalid", lambda _: "Depth exhausted - too many operations"),
            (r"number of levels.*is less than (\d+)", lambda m: f"Need {m.group(1)} levels but depth exhausted"),
            (r"Enable\((\w+)\) must be called", lambda m: f"Enable({m.group(1)}) must be called first"),
            (r"(\w+) operation has not been enabled", lambda m: f"{m.group(1)} operation not enabled"),
        ]

        for pattern, formatter in fhe_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return formatter(match)

        # 3. Look for compile errors
        compile_match = re.search(r"error:\s*(.+?)(?:\n|$)", output)
        if compile_match:
            return compile_match.group(1).strip()[:200]

        # 4. Look for generic error lines
        error_lines = []
        for line in output.split('\n'):
            line_lower = line.lower()
            # Skip build noise
            if any(skip in line_lower for skip in ['make[', 'cmake', 'building', '===', '---']):
                continue
            # Skip JSON lines (result.json output)
            if line.strip().startswith('{') and line.strip().endswith('}'):
                continue
            if any(kw in line_lower for kw in ['error', 'exception', 'failed', 'abort']):
                cleaned = line.strip()
                if cleaned and len(cleaned) > 5:
                    error_lines.append(cleaned)

        if error_lines:
            return error_lines[-1][:200]

        # 5. Fallback
        lines = [l.strip() for l in output.split('\n') if l.strip() and not l.startswith('===')]
        if lines:
            return lines[-1][:200]

        return "Unknown error"

    def _clean_fhe_error(self, raw_msg: str) -> str:
        """Clean FHE error message by removing file paths and extracting key info."""
        # Remove file path prefix: /path/file.cpp:123:FunctionName():
        # Pattern: starts with /, contains .cpp or .h, followed by :line: or just :
        cleaned = re.sub(r'^/[^:]+\.(cpp|h|hpp):\d+:?', '', raw_msg).strip()
        cleaned = re.sub(r'^\w+\(\):\s*', '', cleaned).strip()  # Remove FunctionName():

        # Extract key info from common patterns
        if "EvalKey for index" in cleaned:
            match = re.search(r"EvalKey for index \[(\d+)\]", cleaned)
            if match:
                return f"EvalKey for index [{match.group(1)}] not found"

        if "ciphertext passed to" in cleaned.lower():
            match = re.search(r"ciphertext passed to (\w+)", cleaned, re.IGNORECASE)
            if match:
                return f"Ciphertext passed to {match.group(1)} is empty"

        if "Enable(" in cleaned:
            match = re.search(r"Enable\((\w+)\)", cleaned)
            if match:
                return f"Enable({match.group(1)}) must be called first"

        # Depth exhausted patterns
        if "Removing last element" in cleaned or "DCRTPoly" in cleaned:
            return "Depth exhausted - too many multiplicative operations"

        if "number of levels" in cleaned.lower():
            match = re.search(r"less than (\d+)", cleaned)
            if match:
                return f"Depth exhausted - need {match.group(1)} more levels"
            return "Depth exhausted - insufficient levels"

        # Remove [called from: ...] suffix
        cleaned = re.sub(r'\s*\[called from:.*\]$', '', cleaned)

        return cleaned[:200] if len(cleaned) > 200 else cleaned

    def run_docker(
        self,
        image: str,
        command: list,
        volumes: dict = None,
        workdir: str = None,
        timeout: int = None,
    ) -> tuple[bool, list]:
        """
        Run a command in Docker.

        Args:
            image: Docker image name
            command: Command to run
            volumes: Dict of host_path: container_path
            workdir: Working directory in container
            timeout: Timeout in seconds

        Returns:
            (success, output_lines)
        """
        timeout = timeout or self.run_timeout

        docker_cmd = ["docker", "run", "--rm"]

        # Run as current user to avoid permission issues with created files
        docker_cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])

        if volumes:
            for host, container in volumes.items():
                docker_cmd.extend(["-v", f"{host}:{container}"])

        if workdir:
            docker_cmd.extend(["-w", workdir])

        docker_cmd.append(image)
        docker_cmd.extend(command)

        try:
            env = os.environ.copy()
            env["DOCKER_BUILDKIT"] = "0"

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

            output = (result.stdout + "\n" + result.stderr).strip().split("\n")
            return result.returncode == 0, output

        except subprocess.TimeoutExpired:
            return False, [f"TIMEOUT: Execution exceeded {timeout}s"]
        except Exception as e:
            return False, [f"ERROR: {e}"]

    def docker_build(
        self,
        dockerfile: Path,
        context: Path,
        tag: str,
        timeout: int = 600,
    ) -> tuple[bool, list]:
        """
        Build a Docker image.

        Args:
            dockerfile: Path to Dockerfile
            context: Build context directory
            tag: Image tag
            timeout: Build timeout

        Returns:
            (success, output_lines)
        """
        try:
            env = os.environ.copy()
            env["DOCKER_BUILDKIT"] = "0"

            result = subprocess.run(
                [
                    "docker", "build",
                    "-f", str(dockerfile),
                    "-t", tag,
                    str(context),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

            output = (result.stdout + "\n" + result.stderr).strip().split("\n")
            return result.returncode == 0, output

        except subprocess.TimeoutExpired:
            return False, [f"TIMEOUT: Build exceeded {timeout}s"]
        except Exception as e:
            return False, [f"ERROR: {e}"]

    def cleanup(self) -> None:
        """Cleanup workspace and Docker resources."""
        # Subclasses can override for specific cleanup
        pass
