"""
FHE Challenge solving module for AIDE.

This module extends AIDE to solve Fully Homomorphic Encryption (FHE) challenges.

Architecture:
- challenge_parser: Parse challenge.md to extract spec and detect challenge type
- interpreters: Type-specific execution (black_box, white_box, ml_inference, non_openfhe)
- agent: Template-aware code generation (only eval() body)
- run: Entry point with Rich UI
"""

from .challenge_parser import (
    FHEChallengeSpec,
    ChallengeType,
    Scheme,
    Library,
    parse_challenge,
)
from .agent import FHEAgent, AgentConfig
from .interpreters import create_interpreter
from .config import FHEConfig, load_fhe_config

__all__ = [
    # Challenge parsing
    "FHEChallengeSpec",
    "ChallengeType",
    "Scheme",
    "Library",
    "parse_challenge",
    # Agent
    "FHEAgent",
    "AgentConfig",
    # Interpreters
    "create_interpreter",
    # Config
    "FHEConfig",
    "load_fhe_config",
]
