"""
Configuration for FHE challenges.
"""

from dataclasses import dataclass, field
from pathlib import Path

import coolname
from omegaconf import OmegaConf
import logging

logger = logging.getLogger("aide.fhe")


@dataclass
class StageConfig:
    """LLM stage configuration."""
    model: str
    temp: float
    reasoning_effort: str = "high"  # low, medium, high (for thinking models)
    web_search: bool = True  # Enable web search for FHE docs/examples


@dataclass
class FeedbackConfig:
    """Feedback LLM configuration for lesson extraction."""
    enabled: bool = True
    model: str = "gpt-4o-mini"
    temp: float = 0.3
    max_lessons: int = 10


@dataclass
class SearchConfig:
    """Tree search parameters."""
    num_drafts: int
    debug_prob: float
    max_debug_depth: int


@dataclass
class AgentConfig:
    """Agent configuration."""
    steps: int
    early_stop_threshold: float  # Stop when accuracy reaches this (0.0 to disable)
    code: StageConfig
    feedback: FeedbackConfig
    search: SearchConfig


@dataclass
class ExecConfig:
    """Execution configuration."""
    build_timeout: int  # Timeout for Docker build (seconds)
    run_timeout: int    # Timeout for FHE computation (seconds)


@dataclass
class FHEConfig:
    """Configuration for FHE challenge solving."""

    challenge_dir: Path

    log_dir: Path
    workspace_dir: Path

    exp_name: str

    exec: ExecConfig
    agent: AgentConfig


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    if not dir.exists():
        return 0
    max_index = -1
    for p in dir.iterdir():
        try:
            current_index = int(p.name.split("-")[0])
            if current_index > max_index:
                max_index = current_index
        except ValueError:
            pass
    return max_index + 1


def load_fhe_config(
    config_path: Path | None = None,
    use_cli_args: bool = True,
) -> FHEConfig:
    """Load FHE configuration from YAML file and CLI arguments."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    cfg = OmegaConf.load(config_path)

    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

    return prep_fhe_config(cfg)


def prep_fhe_config(cfg) -> FHEConfig:
    """Prepare and validate FHE configuration."""
    # Validate required fields
    if cfg.challenge_dir is None:
        raise ValueError("`challenge_dir` must be provided")

    cfg.challenge_dir = Path(cfg.challenge_dir).resolve()

    if not cfg.challenge_dir.exists():
        raise ValueError(f"Challenge directory not found: {cfg.challenge_dir}")

    # Check for challenge.md
    challenge_file = cfg.challenge_dir / "challenge.md"
    if not challenge_file.exists():
        if cfg.challenge_dir.is_file() and cfg.challenge_dir.name == "challenge.md":
            cfg.challenge_dir = cfg.challenge_dir.parent
        else:
            raise ValueError(f"challenge.md not found in {cfg.challenge_dir}")

    # Setup output directories
    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)

    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    top_workspace_dir.mkdir(parents=True, exist_ok=True)

    # Generate experiment name
    ind = max(_get_next_logindex(top_log_dir), _get_next_logindex(top_workspace_dir))
    cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)
    cfg.exp_name = f"{ind}-{cfg.exp_name}"

    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # Validate against schema
    cfg_schema = OmegaConf.structured(FHEConfig)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    logger.info(f"FHE Config: {cfg.exp_name}")
    logger.info(f"  Challenge: {cfg.challenge_dir}")
    logger.info(f"  Workspace: {cfg.workspace_dir}")

    return cfg
