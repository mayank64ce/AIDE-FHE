"""
Challenge parser for FHE challenges.

Parses challenge.md to extract:
- Challenge type (black_box, white_box_openfhe, ml_inference, non_openfhe)
- Encryption scheme (CKKS, BFV, BGV)
- Constraints (depth, batch size, input range)
- Available keys
- Task specification
- Scoring parameters
"""

import json
import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger("aide.fhe")


class ChallengeType(str, Enum):
    """Types of FHE challenges."""
    BLACK_BOX = "black_box"           # Pre-encrypted testcases
    WHITE_BOX_OPENFHE = "white_box_openfhe"  # OpenFHE with fherma-validator
    ML_INFERENCE = "ml_inference"      # ML model + encrypted inference
    NON_OPENFHE = "non_openfhe"       # HElayers, Swift, etc.


class Scheme(str, Enum):
    """FHE encryption schemes."""
    CKKS = "CKKS"
    BFV = "BFV"
    BGV = "BGV"
    TFHE = "TFHE"


class Library(str, Enum):
    """FHE libraries."""
    OPENFHE = "OpenFHE"
    HELAYERS = "HElayers"
    SWIFT_HE = "swift-homomorphic-encryption"
    SEAL = "SEAL"


@dataclass
class Constraints(DataClassJsonMixin):
    """FHE constraints from challenge specification."""
    depth: int = 10
    batch_size: int = 4096
    scale_mod_size: int = 50
    first_mod_size: int = 60
    ring_dimension: Optional[int] = None
    input_range: tuple = (-1.0, 1.0)
    plaintext_modulus: Optional[int] = None  # For BFV/BGV
    security_level: str = "HEStd_128_classic"


@dataclass
class Keys(DataClassJsonMixin):
    """Available keys from challenge specification."""
    public: bool = True
    secret: bool = False  # Only for validation
    multiplication: bool = True
    rotation_indices: list = field(default_factory=list)
    bootstrapping: bool = False


@dataclass
class Scoring(DataClassJsonMixin):
    """Scoring specification."""
    metric_type: str = "accuracy"  # accuracy, rmse, mae
    error_threshold: float = 0.001
    accuracy_threshold: float = 0.8
    max_fatal_errors: int = 40
    score_per_slot: float = 10.0


@dataclass
class FHEChallengeSpec(DataClassJsonMixin):
    """Complete FHE challenge specification."""

    # Challenge identity
    challenge_id: Optional[str] = None
    challenge_name: Optional[str] = None
    challenge_dir: Optional[Path] = None

    # Challenge type - critical for execution flow
    challenge_type: ChallengeType = ChallengeType.WHITE_BOX_OPENFHE

    # Encryption
    scheme: Scheme = Scheme.CKKS
    library: Library = Library.OPENFHE
    library_version: Optional[str] = None

    # Constraints
    constraints: Constraints = field(default_factory=Constraints)

    # Keys
    keys: Keys = field(default_factory=Keys)

    # Task
    task: str = "unknown"
    task_description: str = ""
    function_signature: str = ""  # e.g., "sign(x)", "max(0, x)"

    # Input/Output
    input_format: str = ""
    output_format: str = ""
    num_inputs: int = 1
    num_outputs: int = 1

    # Scoring
    scoring: Scoring = field(default_factory=Scoring)

    # Template info (populated from challenge directory)
    template_dir: Optional[Path] = None
    template_files: list = field(default_factory=list)
    has_dockerfile: bool = False
    has_verifier: bool = False

    # Testcase info
    testcase_dirs: list = field(default_factory=list)
    has_test_case_json: bool = False

    # Raw text
    raw_text: str = ""

    # Useful links from challenge.md (e.g., Polycircuit, tutorials)
    useful_links: list = field(default_factory=list)

    def save(self, path: Path) -> None:
        """Save spec to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "FHEChallengeSpec":
        """Load spec from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


def parse_challenge(challenge_path: Path | str) -> FHEChallengeSpec:
    """
    Parse challenge.md and directory structure to create FHEChallengeSpec.

    Challenge type is detected from challenge.md content:
    - "black box" / "pre-encrypted" → BLACK_BOX
    - "white box" + "openfhe" → WHITE_BOX_OPENFHE
    - "machine learning" / "inference" / "training" → ML_INFERENCE
    - "helayers" / "swift" / non-openfhe lib → NON_OPENFHE
    """
    challenge_path = Path(challenge_path)

    if challenge_path.is_file():
        challenge_dir = challenge_path.parent
        challenge_file = challenge_path
    else:
        challenge_dir = challenge_path
        challenge_file = challenge_path / "challenge.md"

    if not challenge_file.exists():
        raise FileNotFoundError(f"challenge.md not found: {challenge_file}")

    text = challenge_file.read_text()
    spec = FHEChallengeSpec(raw_text=text, challenge_dir=challenge_dir)

    # Parse all components
    _parse_metadata(text, spec)
    _parse_challenge_type(text, spec)
    _parse_scheme(text, spec)
    _parse_library(text, spec)
    _parse_constraints(text, spec)
    _parse_keys(text, spec)
    _parse_task(text, spec)
    _parse_scoring(text, spec)
    _parse_useful_links(text, spec)
    _parse_directory_structure(challenge_dir, spec)

    logger.info(f"Parsed challenge: {spec.challenge_name}")
    logger.info(f"  Type: {spec.challenge_type.value}")
    logger.info(f"  Scheme: {spec.scheme.value}, Library: {spec.library.value}")
    logger.info(f"  Task: {spec.task}")

    return spec


def _parse_metadata(text: str, spec: FHEChallengeSpec) -> None:
    """Extract challenge name and ID."""
    # Title from first heading
    title_match = re.search(r'^#\s+(.+?)(?:\n|$)', text, re.MULTILINE)
    if title_match:
        spec.challenge_name = title_match.group(1).strip()

    # Challenge ID if present
    id_match = re.search(r'challenge_id[:\s]+([a-f0-9]+)', text, re.IGNORECASE)
    if id_match:
        spec.challenge_id = id_match.group(1)


def _parse_challenge_type(text: str, spec: FHEChallengeSpec) -> None:
    """
    Detect challenge type from challenge.md content.

    Priority:
    1. Non-OpenFHE library detection (HElayers, Swift) - always takes precedence
    2. Explicit "Challenge type: Black Box" → BLACK_BOX
    3. ML-specific challenge names (cifar, house_prediction, etc.) → ML_INFERENCE
    4. Explicit "Challenge type: White Box" → WHITE_BOX_OPENFHE
    5. Other black box indicators → BLACK_BOX
    6. Default to WHITE_BOX_OPENFHE
    """
    text_lower = text.lower()

    # FIRST: Check for non-OpenFHE libraries - these always override
    # even if the challenge says "White Box"
    if re.search(r'helayers|ibm\s+fhe|pyhelayers', text_lower):
        spec.challenge_type = ChallengeType.NON_OPENFHE
        return

    if re.search(r'swift-homomorphic-encryption|apple.*swift.*homomorphic|swift\s+he', text_lower):
        spec.challenge_type = ChallengeType.NON_OPENFHE
        return

    # Check for explicit black box declaration
    explicit_type_match = re.search(
        r'challenge\s+type[:\s]+([^\n]+)',
        text_lower
    )
    if explicit_type_match:
        type_text = explicit_type_match.group(1).strip()
        if 'black' in type_text and 'box' in type_text:
            spec.challenge_type = ChallengeType.BLACK_BOX
            return

    # Check for ML-specific challenge names - these are very specific
    # (General "machine learning" mentions don't count)
    ml_challenge_patterns = [
        r'cifar[-\s]?10',
        r'mnist',
        r'sentiment\s+(analysis|classification)',
        r'fraud\s+detection',
        r'house\s+(price\s+)?prediction',
        r'svm\s+(model|classification|fraud)',
        r'training\s+data\s+(is\s+)?provided',
        r'neural\s+network\s+(model|inference|evaluation)',
        r'cnn\s+(inference|classification)',
    ]
    for pattern in ml_challenge_patterns:
        if re.search(pattern, text_lower):
            spec.challenge_type = ChallengeType.ML_INFERENCE
            return

    # Check for explicit white box declaration (after ML check)
    if explicit_type_match:
        type_text = explicit_type_match.group(1).strip()
        if 'white' in type_text and 'box' in type_text:
            spec.challenge_type = ChallengeType.WHITE_BOX_OPENFHE
            return

    # Check for other black box indicators
    black_box_patterns = [
        r'black\s*box',
        r'pre-encrypted\s+test',
        r'testcase.*pre-encrypted',
        r'encrypted\s+input\s+is\s+provided',
        r'ciphertext\s+is\s+already\s+provided',
    ]
    for pattern in black_box_patterns:
        if re.search(pattern, text_lower):
            spec.challenge_type = ChallengeType.BLACK_BOX
            return

    # Default to white box OpenFHE
    spec.challenge_type = ChallengeType.WHITE_BOX_OPENFHE


def _parse_scheme(text: str, spec: FHEChallengeSpec) -> None:
    """Extract encryption scheme."""
    text_upper = text.upper()

    if "CKKS" in text_upper:
        spec.scheme = Scheme.CKKS
    elif "BFV" in text_upper:
        spec.scheme = Scheme.BFV
    elif "BGV" in text_upper:
        spec.scheme = Scheme.BGV
    elif "TFHE" in text_upper or "BOOLEAN" in text_upper:
        spec.scheme = Scheme.TFHE


def _parse_library(text: str, spec: FHEChallengeSpec) -> None:
    """Extract FHE library and version."""
    text_lower = text.lower()

    # Detect library
    if re.search(r'helayers|pyhelayers', text_lower):
        spec.library = Library.HELAYERS
    elif re.search(r'swift-homomorphic-encryption|apple.*swift', text_lower):
        spec.library = Library.SWIFT_HE
    elif re.search(r'\bseal\b', text_lower):
        spec.library = Library.SEAL
    else:
        spec.library = Library.OPENFHE

    # Extract version for OpenFHE
    if spec.library == Library.OPENFHE:
        version_match = re.search(r'openfhe[:\s]*v?(\d+\.\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if version_match:
            spec.library_version = version_match.group(1)


def _parse_constraints(text: str, spec: FHEChallengeSpec) -> None:
    """Extract crypto constraints."""
    constraints = spec.constraints

    # Multiplicative depth
    for pattern in [
        r'[Mm]ultiplicative\s+depth[:\s]+(\d+)',
        r'depth[:\s]+(\d+)',
        r'depth\s*(?:budget|limit)?[:\s]*(\d+)',
    ]:
        match = re.search(pattern, text)
        if match:
            constraints.depth = int(match.group(1))
            break

    # Batch size - look for JSON format first, then explicit declarations
    for pattern in [
        r'"batch_size":\s*(\d+)',           # JSON format: "batch_size": 65536
        r'batch_size\s*[=:]\s*(\d+)',       # batch_size = X or batch_size: X
        r'[Bb]atch\s+[Ss]ize:\s*(\d+)',     # Batch Size: X (direct, no greedy match)
        r'[Vv]ector\s+length[:\s]+(\d+)',
    ]:
        match = re.search(pattern, text)
        if match:
            val = int(match.group(1))
            # Sanity check: batch_size should be at least 1024 for most FHE tasks
            if val >= 1024:
                constraints.batch_size = val
                break

    # Scale mod size
    match = re.search(r'[Ss]cale[Mm]od[Ss]ize[:\s]+(\d+)', text)
    if match:
        constraints.scale_mod_size = int(match.group(1))

    # Ring dimension
    match = re.search(r'[Rr]ing\s+dimension[:\s]+(\d+)', text)
    if match:
        constraints.ring_dimension = int(match.group(1))

    # Input range
    match = re.search(r'[Rr]ange[:\s]*\[([^\]]+)\]', text)
    if match:
        try:
            parts = [float(x.strip()) for x in match.group(1).split(',')]
            if len(parts) == 2:
                constraints.input_range = tuple(parts)
        except ValueError:
            pass


def _parse_keys(text: str, spec: FHEChallengeSpec) -> None:
    """Extract available keys."""
    keys = spec.keys

    keys.public = bool(re.search(r'public\s+key', text, re.IGNORECASE))
    keys.multiplication = bool(re.search(r'multiplication|relinearization|key_mult', text, re.IGNORECASE))
    keys.bootstrapping = bool(re.search(r'bootstrap', text, re.IGNORECASE))

    # Rotation indices
    rot_match = re.search(r'rotation\s+key[^[]*\[([^\]]+)\]', text, re.IGNORECASE)
    if rot_match:
        try:
            indices = [int(x.strip()) for x in rot_match.group(1).split(',')]
            keys.rotation_indices = indices
        except ValueError:
            pass


def _parse_task(text: str, spec: FHEChallengeSpec) -> None:
    """Identify the computational task."""
    text_lower = text.lower()

    # For ML inference challenges, use challenge name or folder name as task
    # This avoids generic patterns like "max" matching in ML challenge descriptions
    if spec.challenge_type == ChallengeType.ML_INFERENCE:
        if spec.challenge_dir:
            # Use folder name: challenge_cifar10 -> cifar10
            folder_name = spec.challenge_dir.name
            spec.task = folder_name.replace('challenge_', '')
        elif spec.challenge_name:
            # Use challenge name: "CIFAR-10 Image Classification" -> "cifar10_image_classification"
            spec.task = re.sub(r'[^a-z0-9]+', '_', spec.challenge_name.lower()).strip('_')[:50]
        else:
            spec.task = "ml_inference"
        spec.function_signature = "model(encrypted_input)"

        # Still extract task description
        intro_match = re.search(
            r'##\s*Introduction\s*\n(.*?)(?=\n##|\Z)',
            text, re.DOTALL | re.IGNORECASE
        )
        if intro_match:
            spec.task_description = intro_match.group(1).strip()[:1000]
        return

    # STEP 1: Try matching the challenge TITLE (first heading) — most reliable
    # Title patterns are checked against the first heading only
    title = (spec.challenge_name or "").lower()

    title_task_map = [
        (r'\bgelu\b', "gelu", "gelu(x)"),
        (r'\brelu\b', "relu", "max(0, x)"),
        (r'\bsoftmax\b', "softmax", "softmax(x)"),
        (r'\bsigmoid\b|logistic', "sigmoid", "1/(1+exp(-x))"),
        (r'\bsign\b', "sign", "sign(x)"),
        (r'\btanh\b', "tanh", "tanh(x)"),
        (r'singular\s+value|svd', "svd", "svd(A)"),
        (r'invertible\s+matrix', "invertible_matrix", "det(A) != 0"),
        (r'matrix\s+mult', "matrix_multiplication", "A @ B"),
        (r'array\s+sort', "array_sorting", "sort(x)"),
        (r'max\s+element', "max", "max(x)"),
        (r'k-?nearest|knn', "knn", "knn(x, k)"),
        (r'lookup\s+table', "lookup_table", "table[idx]"),
        (r'\bshl\b|shift\s+left', "shl", "x << n"),
        (r'\bparity\b', "parity", "parity(x)"),
        (r'string\s+search', "string_search", "find(str, text)"),
        (r'set\s+membership', "set_membership", "x in S"),
    ]

    for pattern, task, signature in title_task_map:
        if re.search(pattern, title):
            spec.task = task
            spec.function_signature = signature
            break
    else:
        # STEP 2: Fall back to content patterns (more specific first, broad last)
        task_patterns = [
            # Very specific patterns first
            (r'\bgelu\b', "gelu", "gelu(x)"),
            (r'\bsoftmax\b', "softmax", "softmax(x)"),
            (r'singular\s+value\s+decomposition|\bsvd\b', "svd", "svd(A)"),
            (r'\bshl\b|shift\s+left', "shl", "x << n"),
            (r'\bparity\b', "parity", "parity(x)"),
            (r'invertible\s+matrix|matrix\s+inver', "invertible_matrix", "det(A) != 0"),
            # Then standard patterns with word boundaries
            (r'\brelu\b|max\s*\(\s*0', "relu", "max(0, x)"),
            (r'\bsigmoid\b|logistic\s+function', "sigmoid", "1/(1+exp(-x))"),
            (r'sign\s*\(|sign\s+function|sign\s+evaluation', "sign", "sign(x)"),
            (r'\btanh\b', "tanh", "tanh(x)"),
            (r'matrix\s+mult|matmul', "matrix_multiplication", "A @ B"),
            (r'array\s+sort|sorting\s+algorithm', "array_sorting", "sort(x)"),
            (r'max\s+element|maximum\s+element', "max", "max(x)"),
            (r'knn|k-?nearest\s+neighbor', "knn", "knn(x, k)"),
            (r'lookup\s+table', "lookup_table", "table[idx]"),
            (r'string\s+search|substring', "string_search", "find(str, text)"),
            (r'set\s+membership', "set_membership", "x in S"),
        ]

        for pattern, task, signature in task_patterns:
            if re.search(pattern, text_lower):
                spec.task = task
                spec.function_signature = signature
                break
        else:
            spec.task = "custom"
            spec.function_signature = "f(x)"

    # Extract task description from Introduction section
    intro_match = re.search(
        r'##\s*Introduction\s*\n(.*?)(?=\n##|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    if intro_match:
        spec.task_description = intro_match.group(1).strip()[:1000]

    # Extract output format - look for "Output" section or description
    output_patterns = [
        r'\*\*Output\*\*[:\s]*(.*?)(?:\n\n|\n\*\*|\n##|\Z)',
        r'Output[:\s]+(.*?)(?:\n\n|\n##|\Z)',
        r'output\s+(?:should|must|will)\s+(.*?)(?:\n\n|\n##|\Z)',
    ]
    for pattern in output_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            spec.output_format = match.group(1).strip()[:500]
            break


def _parse_scoring(text: str, spec: FHEChallengeSpec) -> None:
    """Extract scoring parameters."""
    scoring = spec.scoring

    # Error threshold
    match = re.search(r'(?:error|threshold)[:\s<]+(\d+\.?\d*)', text, re.IGNORECASE)
    if match:
        scoring.error_threshold = float(match.group(1))

    # Accuracy threshold
    match = re.search(r'(?:min|minimum)\s*(?:slot\s*)?accuracy[:\s]+(\d+\.?\d*)', text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        scoring.accuracy_threshold = val / 100 if val > 1 else val

    # Fatal errors
    match = re.search(r'(?:fatal|allowed)\s*(?:_)?error[s]?[:\s]+(\d+)', text, re.IGNORECASE)
    if match:
        scoring.max_fatal_errors = int(match.group(1))


def _parse_useful_links(text: str, spec: FHEChallengeSpec) -> None:
    """Extract useful links section from challenge.md."""
    # Find the "Useful links" section
    links_match = re.search(
        r'##\s*[Uu]seful\s+[Ll]inks?\s*\n(.*?)(?=\n##|\Z)',
        text, re.DOTALL
    )
    if not links_match:
        return

    links_section = links_match.group(1)

    # Extract markdown links: match each line starting with -
    # Pattern captures: [name](url) and rest of line as description
    link_pattern = r'-\s*\[([^\]]+)\]\(([^)]+)\)([^\n]*)'
    for match in re.finditer(link_pattern, links_section):
        link_name = match.group(1).strip().rstrip(':')  # Remove trailing colon
        link_url = match.group(2).strip()
        desc_raw = match.group(3).strip()

        # Clean description: remove nested markdown links, leading dashes
        link_desc = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', desc_raw)  # [text](url) -> text
        link_desc = re.sub(r'^[\s—–-]+', '', link_desc).strip()

        # Filter out generic FHERMA links, keep challenge-specific resources
        if any(skip in link_url.lower() for skip in ['fherma.io/how_it_works', 'fhe.org/resources']):
            continue

        spec.useful_links.append({
            "name": link_name,
            "url": link_url,
            "description": link_desc
        })


def _parse_directory_structure(challenge_dir: Path, spec: FHEChallengeSpec) -> None:
    """Analyze challenge directory structure."""

    # Find template directory
    template_candidates = [
        challenge_dir / "templates" / "openfhe",
        challenge_dir / "templates" / "helayers",
        challenge_dir / "templates" / "swift",
        challenge_dir / "templates" / "openfhe-python",
    ]

    for template_dir in template_candidates:
        if template_dir.exists():
            spec.template_dir = template_dir
            spec.template_files = [f.name for f in template_dir.iterdir() if f.is_file()]
            break

    # Check for Dockerfile
    dockerfile = challenge_dir / "Dockerfile"
    spec.has_dockerfile = dockerfile.exists()

    # Check for verifier
    verifier = challenge_dir / "verifier.cpp"
    spec.has_verifier = verifier.exists()

    # Find testcases
    tests_dir = challenge_dir / "tests"
    if tests_dir.exists():
        # Black box style: tests/testcase1, tests/testcase2
        for d in tests_dir.iterdir():
            if d.is_dir() and d.name.startswith("testcase"):
                spec.testcase_dirs.append(d)

        # White box style: tests/test_case.json
        test_case_json = tests_dir / "test_case.json"
        spec.has_test_case_json = test_case_json.exists()
