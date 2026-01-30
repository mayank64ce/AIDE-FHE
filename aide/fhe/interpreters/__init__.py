"""
Type-specific FHE interpreters.

Each challenge type has its own interpreter:
- BlackBoxInterpreter: Pre-encrypted testcases, uses challenge's Dockerfile
- WhiteBoxInterpreter: Uses fherma-validator Docker image
- MLInferenceInterpreter: Training + inference with fherma-validator
- NonOpenFHEInterpreter: HElayers, Swift, etc.
"""

from .base import BaseInterpreter, ExecutionResult, ValidationResult
from .black_box import BlackBoxInterpreter
from .white_box import WhiteBoxInterpreter
from .ml_inference import MLInferenceInterpreter
from .non_openfhe import NonOpenFHEInterpreter

from ..challenge_parser import ChallengeType, FHEChallengeSpec


def create_interpreter(
    spec: FHEChallengeSpec,
    workspace_dir,
    build_timeout: int = 600,
    run_timeout: int = 6000,
) -> BaseInterpreter:
    """
    Factory function to create the appropriate interpreter for a challenge.

    Args:
        spec: Parsed challenge specification
        workspace_dir: Directory for build artifacts
        build_timeout: Timeout for Docker build (seconds)
        run_timeout: Timeout for FHE computation (seconds)

    Returns:
        Appropriate interpreter instance for the challenge type
    """
    interpreters = {
        ChallengeType.BLACK_BOX: BlackBoxInterpreter,
        ChallengeType.WHITE_BOX_OPENFHE: WhiteBoxInterpreter,
        ChallengeType.ML_INFERENCE: MLInferenceInterpreter,
        ChallengeType.NON_OPENFHE: NonOpenFHEInterpreter,
    }

    interpreter_class = interpreters.get(spec.challenge_type, WhiteBoxInterpreter)
    return interpreter_class(
        spec=spec,
        workspace_dir=workspace_dir,
        build_timeout=build_timeout,
        run_timeout=run_timeout,
    )


__all__ = [
    "BaseInterpreter",
    "ExecutionResult",
    "ValidationResult",
    "BlackBoxInterpreter",
    "WhiteBoxInterpreter",
    "MLInferenceInterpreter",
    "NonOpenFHEInterpreter",
    "create_interpreter",
]
