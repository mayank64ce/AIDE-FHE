"""
Non-OpenFHE Interpreter.

Handles challenges using alternative FHE libraries:
- IBM HElayers (Python)
- Apple swift-homomorphic-encryption (Swift)
"""

import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from .base import BaseInterpreter, ExecutionResult, ValidationResult
from ..challenge_parser import Library

logger = logging.getLogger("aide.fhe")


class NonOpenFHEInterpreter(BaseInterpreter):
    """
    Interpreter for non-OpenFHE FHE challenges.

    Supports:
    - HElayers (Python): Uses ibm helayers Docker image
    - Swift HE: Uses swift Docker image

    These challenges have their own:
    - Dockerfile for building
    - Validator logic
    - Library-specific templates
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = f"fhe-{self.spec.library.value.lower()}-{self.spec.task}"
        self.app_build_dir = self.workspace_dir / "app_build"
        self.app_build_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_source(self, code: str) -> None:
        """Prepare source files for the specific library.

        If code contains ### CONFIG ### section, replace config.json entirely.
        """
        # Parse CONFIG and CODE sections
        config_json, actual_code = self._parse_code_with_config(code)

        # Store code for later injection (used by verify.sh flow)
        self._pending_code = actual_code

        # Clear previous build - use Docker to handle root-owned artifacts
        if self.app_build_dir.exists():
            # First try normal cleanup
            shutil.rmtree(self.app_build_dir, ignore_errors=True)
            # If .build or other root-owned dirs still exist, use Docker to clean
            if self.app_build_dir.exists():
                try:
                    subprocess.run(
                        ["docker", "run", "--rm",
                         "-v", f"{self.app_build_dir}:/cleanup",
                         "alpine", "rm", "-rf", "/cleanup/.build", "/cleanup/.swiftpm"],
                        capture_output=True, timeout=60)
                except Exception:
                    pass
                shutil.rmtree(self.app_build_dir, ignore_errors=True)
        self.app_build_dir.mkdir(parents=True, exist_ok=True)

        # Copy template files
        if self.spec.template_dir and self.spec.template_dir.exists():
            for f in self.spec.template_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, self.app_build_dir / f.name)
                elif f.is_dir():
                    shutil.copytree(f, self.app_build_dir / f.name)

        # Add .dockerignore to exclude build artifacts
        dockerignore = self.app_build_dir / ".dockerignore"
        dockerignore.write_text(".build\n.swiftpm\n*.o\n*.d\n")

        # Replace config.json if provided
        if config_json:
            self._write_config(config_json)

        # Inject code based on library
        if self.spec.library == Library.HELAYERS:
            self._inject_helayers_code(actual_code)
        elif self.spec.library == Library.SWIFT_HE:
            self._inject_swift_code(actual_code)

    def _write_config(self, config_json: str) -> None:
        """Write config.json file (complete replacement)."""
        config_file = self.app_build_dir / "config.json"
        config_file.write_text(config_json)
        logger.info("Replaced config.json with LLM-provided config")

    def _inject_helayers_code(self, code: str) -> None:
        """Inject code into HElayers Python template."""
        solution_file = self.app_build_dir / "app.py"
        if not solution_file.exists():
            solution_file.write_text(code)
            return

        template = solution_file.read_text()

        # Strip def solve() wrapper if LLM included it
        code = self._strip_python_func_wrapper(code, "solve")

        # Pattern 1: def solve(...): with simple placeholder (pass, return [], ...)
        pattern = r'(def\s+solve\s*\([^)]*\)\s*:)\s*\n\s*(?:pass|return\s+\[\]|\.\.\.)'
        if re.search(pattern, template):
            new_content = re.sub(pattern, rf'\1\n{code}', template)
            solution_file.write_text(new_content)
            return

        # Pattern 2: Replace from "# TODO:" comment to placeholder return
        # This handles templates with docstrings and TODO comments
        pattern = r'(def\s+solve\s*\([^)]*\)\s*:.*?)(#\s*TODO:.*?)((?:if\s+input_ctxts:|return\s+input_ctxts|return\s+None|return\s+result).*?)(\n\ndef\s|\n\nclass\s|\Z)'
        match = re.search(pattern, template, re.DOTALL)
        if match:
            # Replace TODO section and placeholder with new code
            new_content = template[:match.start(2)] + code + "\n" + match.group(4)
            solution_file.write_text(new_content)
            return

        # Pattern 3: Replace # TODO comment and everything after until next function
        pattern = r'#\s*TODO:\s*[Ii]mplement.*?(?=\n\ndef\s|\n\nclass\s|\Z)'
        if re.search(pattern, template, re.DOTALL):
            new_content = re.sub(pattern, code, template, flags=re.DOTALL)
            solution_file.write_text(new_content)
            return

        # Pattern 4: # Your implementation comment
        pattern = r'#\s*[Yy]our\s+implementation[^\n]*\n'
        if re.search(pattern, template):
            new_content = re.sub(pattern, f'{code}\n', template)
            solution_file.write_text(new_content)
            return

        # Fallback: replace file
        logger.warning("HElayers: Could not find injection point, replacing file")
        solution_file.write_text(code)

    def _strip_python_func_wrapper(self, code: str, func_name: str) -> str:
        """Strip def func_name(): wrapper if present."""
        code = code.strip()
        pattern = rf'^def\s+{func_name}\s*\([^)]*\)\s*:'
        match = re.match(pattern, code)
        if match:
            # Extract body (everything after the def line, dedented)
            after_def = code[match.end():]
            lines = after_def.split('\n')
            body_lines = []
            base_indent = None
            for line in lines:
                if line.strip():
                    if base_indent is None:
                        base_indent = len(line) - len(line.lstrip())
                    if len(line) >= base_indent and line[:base_indent].strip() == '':
                        body_lines.append(line[base_indent:])
                    else:
                        body_lines.append(line.lstrip())
                else:
                    body_lines.append('')
            return '\n'.join(body_lines).strip()
        return code

    def _inject_swift_code(self, code: str) -> None:
        """Inject code into Swift template."""
        # Swift template is in Sources/main.swift
        sources_dir = self.app_build_dir / "Sources"
        if not sources_dir.exists():
            sources_dir.mkdir(parents=True)

        solution_file = sources_dir / "main.swift"
        if not solution_file.exists():
            # Check if it's in app_build directly
            alt_file = self.app_build_dir / "main.swift"
            if alt_file.exists():
                solution_file = alt_file
            else:
                solution_file.write_text(code)
                return

        template = solution_file.read_text()

        # Pattern 1: func solve(...) { placeholder }
        pattern = r'(func\s+solve\s*\([^)]*\)\s*(?:throws\s*)?(?:->\s*\[[^\]]+\]\s*)?\{)\s*\n\s*(?:return\s+\[\]|fatalError\([^)]*\))'
        if re.search(pattern, template):
            new_content = re.sub(pattern, rf'\1\n{code}', template)
            solution_file.write_text(new_content)
            return

        # Pattern 2: Replace // TODO block and placeholder (var result = ...)
        # This handles templates without func solve() but with TODO comments
        pattern = r'(//\s*=+\s*\n\s*//\s*TODO:.*?//\s*=+\s*\n)(.*?)(var\s+result\s*=.*?\n)'
        match = re.search(pattern, template, re.DOTALL)
        if match:
            # Replace the placeholder section with code
            new_content = template[:match.start(1)] + code + "\n" + template[match.end():]
            solution_file.write_text(new_content)
            return

        # Pattern 3: Replace // TODO: Implement comment block
        pattern = r'//\s*TODO:\s*[Ii]mplement.*?(?=\n\s*//\s*Save output|\n\s*let\s+serialized|\Z)'
        if re.search(pattern, template, re.DOTALL):
            new_content = re.sub(pattern, code + "\n", template, flags=re.DOTALL)
            solution_file.write_text(new_content)
            return

        # Pattern 4: // Your implementation comment
        pattern = r'//\s*[Yy]our\s+implementation[^\n]*\n'
        if re.search(pattern, template):
            new_content = re.sub(pattern, f'{code}\n', template)
            solution_file.write_text(new_content)
            return

        # Pattern 5: Replace "// Placeholder:" or similar
        pattern = r'//\s*[Pp]laceholder:.*?\n.*?var\s+result\s*=.*?\n'
        if re.search(pattern, template, re.DOTALL):
            new_content = re.sub(pattern, code + "\n", template, flags=re.DOTALL)
            solution_file.write_text(new_content)
            return

        # Fallback
        logger.warning("Swift: Could not find injection point, replacing file")
        solution_file.write_text(code)

    def build(self) -> tuple[bool, list]:
        """Build using challenge's Dockerfile."""
        # Copy Dockerfile and other build files
        challenge_dir = self.spec.challenge_dir

        for filename in ["Dockerfile", "Package.swift", "validator.py"]:
            src = challenge_dir / filename
            if src.exists():
                shutil.copy2(src, self.workspace_dir / filename)

        # Also copy validator directory if exists
        validator_dir = challenge_dir / "validator"
        if validator_dir.exists():
            dest = self.workspace_dir / "validator"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(validator_dir, dest)

        dockerfile = self.workspace_dir / "Dockerfile"
        if not dockerfile.exists():
            return False, ["ERROR: Dockerfile not found"]

        return self.docker_build(
            dockerfile=dockerfile,
            context=self.workspace_dir,
            tag=self.image_name,
            timeout=self.build_timeout,
        )

    def run(self, testcase_path: Optional[Path] = None) -> tuple[bool, list]:
        """Run solution using challenge's verify.sh or Docker."""
        # Check for verify.sh in challenge directory
        verify_sh = self.spec.challenge_dir / "verify.sh"

        if verify_sh.exists():
            # Copy verify.sh to workspace
            shutil.copy2(verify_sh, self.workspace_dir / "verify.sh")

            # Copy templates directory if exists (needed by verify.sh)
            # IMPORTANT: verify.sh wipes app_build/ and copies from templates/,
            # so we must propagate injected code from app_build/ back to templates/
            templates_src = self.spec.challenge_dir / "templates"
            if templates_src.exists():
                templates_dst = self.workspace_dir / "templates"
                if templates_dst.exists():
                    shutil.rmtree(templates_dst, ignore_errors=True)
                shutil.copytree(templates_src, templates_dst)

                # Propagate injected code + config from app_build/ to templates/
                # (_prepare_source() already injected code into app_build/)
                if self.spec.library == Library.SWIFT_HE:
                    # Copy injected main.swift from app_build to templates
                    for src_rel in ["Sources/main.swift", "config.json"]:
                        src = self.app_build_dir / src_rel
                        dst = templates_dst / "swift" / src_rel
                        if src.exists():
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src, dst)
                            logger.info(f"Propagated {src_rel} to templates/swift/")

                    # Add .dockerignore to templates to prevent build artifacts
                    dockerignore = templates_dst / "swift" / ".dockerignore"
                    dockerignore.write_text(".build\n.swiftpm\n*.o\n*.d\n")

                elif self.spec.library == Library.HELAYERS:
                    # Copy injected app.py and config from app_build to templates
                    for filename in ["app.py", "config.json"]:
                        src = self.app_build_dir / filename
                        dst = templates_dst / "helayers" / filename
                        if src.exists():
                            shutil.copy2(src, dst)
                            logger.info(f"Propagated {filename} to templates/helayers/")

            # Copy tests directory if exists (needed by verify.sh)
            tests_src = self.spec.challenge_dir / "tests"
            if tests_src.exists():
                tests_dst = self.workspace_dir / "tests"
                if tests_dst.exists():
                    shutil.rmtree(tests_dst, ignore_errors=True)
                shutil.copytree(tests_src, tests_dst)

            # Copy validator directory if exists (needed by verify.sh)
            validator_src = self.spec.challenge_dir / "validator"
            if validator_src.exists():
                validator_dst = self.workspace_dir / "validator"
                if validator_dst.exists():
                    shutil.rmtree(validator_dst, ignore_errors=True)
                shutil.copytree(validator_src, validator_dst)

            # Copy build/validation files from challenge dir to workspace
            for filename in ["Dockerfile", "validator.py"]:
                src = self.spec.challenge_dir / filename
                if src.exists():
                    shutil.copy2(src, self.workspace_dir / filename)

            # Patch verify.sh to ensure Docker-created files are writable
            # verify.sh recreates app_build/ from templates, but Docker
            # containers may run as different users and need write access
            verify_content = (self.workspace_dir / "verify.sh").read_text()
            if "chmod -R a+rw" not in verify_content:
                # Add chmod after app_build preparation (before Docker run)
                verify_content = verify_content.replace(
                    'echo "=== Running',
                    'chmod -R a+rw "$SCRIPT_DIR" 2>/dev/null || true\necho "=== Running'
                )
                (self.workspace_dir / "verify.sh").write_text(verify_content)

            # Run verify.sh
            try:
                env = os.environ.copy()
                result = subprocess.run(
                    ["bash", str(self.workspace_dir / "verify.sh")],
                    capture_output=True,
                    text=True,
                    timeout=self.run_timeout,
                    cwd=str(self.workspace_dir),
                    env=env,
                )
                output = (result.stdout + "\n" + result.stderr).strip().split("\n")
                return result.returncode == 0, output
            except subprocess.TimeoutExpired:
                return False, [f"TIMEOUT: Execution exceeded {self.run_timeout}s"]
            except Exception as e:
                return False, [f"ERROR: {e}"]

        # Fallback: run Docker image directly
        volumes = {str(self.app_build_dir): "/app/solution"}

        if testcase_path:
            volumes[str(testcase_path)] = "/app/testcase"

        return self.run_docker(
            image=self.image_name,
            command=[],
            volumes=volumes,
            timeout=self.run_timeout,
        )

    def validate(self, output_path: Path, testcase_path: Path) -> ValidationResult:
        """Parse validation results."""
        result = ValidationResult()

        # Check for result.json in various locations
        result_locations = [
            self.app_build_dir / "result.json",
            self.workspace_dir / "result.json",
            self.workspace_dir / "app_build" / "result.json",
        ]

        result_file = None
        for loc in result_locations:
            if loc.exists():
                result_file = loc
                break

        if result_file is None:
            result.details["error"] = "result.json not found"
            return result

        try:
            data = json.loads(result_file.read_text())

            result.accuracy = data.get("accuracy", 0.0)
            result.passed = data.get("passed", False)
            result.total_slots = data.get("total_tests", data.get("total_slots", 0))
            result.error_count = data.get("failed_tests", data.get("errors", 0))
            result.details = data

            # Calculate accuracy if not provided
            if result.accuracy == 0.0 and result.total_slots > 0:
                passed = result.total_slots - result.error_count
                result.accuracy = passed / result.total_slots

        except json.JSONDecodeError as e:
            result.details["error"] = f"Failed to parse result.json: {e}"

        return result

    def execute(self, code: str, testcase_path: Optional[Path] = None) -> ExecutionResult:
        """Execute solution for non-OpenFHE challenge."""
        result = ExecutionResult()
        start_time = time.time()

        try:
            # Prepare source
            self._prepare_source(code)

            # Check if verify.sh handles build+run (skip separate build step)
            verify_sh = self.spec.challenge_dir / "verify.sh"
            has_verify_sh = verify_sh.exists()

            if not has_verify_sh:
                # Build separately only when no verify.sh
                build_start = time.time()
                build_ok, build_out = self.build()
                result.build_time = time.time() - build_start
                result.build_success = build_ok
                result.build_output = build_out

                if not build_ok:
                    self._analyze_build_error(result)
                    result.total_time = time.time() - start_time
                    return result
            else:
                # verify.sh handles build internally
                result.build_success = True
                result.build_output = ["Build handled by verify.sh"]

            # Run (uses verify.sh if available, otherwise Docker directly)
            run_start = time.time()
            run_ok, run_out = self.run(testcase_path)
            result.run_time = time.time() - run_start
            result.run_output = run_out

            # Check for errors in output
            output_text = "\n".join(run_out) if isinstance(run_out, list) else str(run_out)

            if not run_ok or self._has_runtime_error(output_text):
                # When verify.sh handles build+run, detect if failure is build or runtime
                if has_verify_sh and self._is_build_error(output_text):
                    result.build_success = False
                    result.build_output = run_out
                    self._analyze_build_error(result)
                else:
                    result.run_success = False
                    self._analyze_runtime_error(result)

                # Check for partial result.json (validator may have written it)
                result_file = self.app_build_dir / "result.json"
                if not result_file.exists():
                    result_file = self.workspace_dir / "result.json"
                if result_file.exists():
                    result.output_generated = True
                    result.validation = self.validate(result_file, testcase_path)

                result.total_time = time.time() - start_time
                return result

            result.run_success = True

            # Check for output/result
            result.output_generated = any(
                (self.app_build_dir / f).exists()
                for f in ["result.json", "output.bin", "output.txt"]
            )

            if result.output_generated:
                result.validation = self.validate(
                    self.app_build_dir / "result.json",
                    testcase_path
                )

        except Exception as e:
            result.error_type = type(e).__name__
            result.error_message = str(e)
            logger.exception("NonOpenFHE interpreter error")

        result.total_time = time.time() - start_time
        return result

    def _has_runtime_error(self, output: str) -> bool:
        """Detect if output indicates a runtime error."""
        output_lower = output.lower()

        # Common runtime error patterns for HElayers and Swift
        # NOTE: Patterns must be specific enough to avoid false positives
        # on normal output (e.g., "Error rate: 0.01" or "Errors: 0")
        runtime_patterns = [
            "traceback (most recent call last)",
            "fatal error",
            "segmentation fault",
            "aborted",
            "core dumped",
            "panic:",  # Swift
            "fatalerror",  # Swift
            "assertion failed",
            "exception:",
            "terminate called",
        ]

        for pattern in runtime_patterns:
            if pattern in output_lower:
                return True

        # Check for Python exception patterns (NameError:, ValueError:, etc.)
        # but NOT "error:" alone (too broad - matches "Error rate: 0.0")
        if re.search(r'\w+Error:', output) or re.search(r'\w+Exception:', output):
            return True

        return False

    def _is_build_error(self, output: str) -> bool:
        """Detect if verify.sh output indicates a build/compilation error."""
        output_lower = output.lower()

        # Swift build error patterns
        swift_build_patterns = [
            "swift build",  # Swift build command failed
            "compiling ",  # Swift compilation output
            "error: no such module",
            "cannot find type",
            "cannot find '",
            "use of unresolved identifier",
            "value of type",
            "expected declaration",
            "linker command failed",
        ]
        for pattern in swift_build_patterns:
            if pattern in output_lower:
                # Verify it's actually a build failure (not just mentioning "swift build")
                if "error:" in output_lower or "fatal:" in output_lower:
                    return True

        # Docker build error patterns
        docker_build_patterns = [
            "error building image",
            "dockerfile parse error",
            "error checking context",
        ]
        for pattern in docker_build_patterns:
            if pattern in output_lower:
                return True

        # CMake/make patterns (for C++ challenges)
        cmake_patterns = [
            "cmake error",
            "make[",
            "undefined reference",
            "fatal error:",
            "collect2: error",
        ]
        for pattern in cmake_patterns:
            if pattern in output_lower:
                return True

        return False

    def cleanup(self) -> None:
        """Cleanup Docker image and build artifacts."""
        try:
            subprocess.run(
                ["docker", "rmi", "-f", self.image_name],
                capture_output=True,
                timeout=30,
            )
        except Exception:
            pass

        if self.app_build_dir.exists():
            shutil.rmtree(self.app_build_dir, ignore_errors=True)
