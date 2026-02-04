"""
Entry point for FHE challenge solving mode.

Usage:
    aide-fhe challenge_dir=/path/to/challenge

    # With custom testcase directory
    aide-fhe challenge_dir=/path/to/challenge testcase_dir=/path/to/testcase

    # With custom settings
    aide-fhe challenge_dir=/path/to/challenge agent.steps=20 agent.code.model=gpt-4o

Architecture:
    1. Parse challenge.md → FHEChallengeSpec with ChallengeType
    2. Create type-specific interpreter (black_box, white_box, ml_inference, non_openfhe)
    3. Agent generates eval() function body only
    4. Interpreter injects code into templates and executes
    5. Validator returns accuracy metrics
    6. Agent iterates based on feedback
"""

from dotenv import load_dotenv
load_dotenv()

import atexit
import logging
import shutil

from omegaconf import OmegaConf
from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.status import Status
from rich.text import Text
from rich.tree import Tree

from ..journal import Journal, Node
from ..utils.serialize import dump_json

from .challenge_parser import parse_challenge
from .config import FHEConfig, load_fhe_config
from .agent import FHEAgent, AgentConfig
from .interpreters import create_interpreter

logger = logging.getLogger("aide.fhe")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def _aggregate_results(results: list) -> object:
    """Aggregate execution results from multiple testcases.

    Strategy: Use minimum accuracy (worst case) to ensure robustness.
    A solution must work well on ALL testcases.
    """
    from dataclasses import dataclass, field

    @dataclass
    class AggregatedResult:
        build_success: bool
        run_success: bool
        output_generated: bool
        validation: object
        run_output: list
        build_output: list
        total_time: float
        error_type: str = None
        error_message: str = None

        def get_feedback(self) -> str:
            """Generate feedback string for the agent."""
            lines = []
            if not self.build_success:
                lines.append("BUILD FAILED")
                lines.append(f"Error: {self.error_type}")
                lines.append(f"Message: {self.error_message}")
                lines.append("")
                lines.append("Build output:")
                lines.extend(self.build_output[-50:])
            elif not self.run_success:
                lines.append("RUNTIME FAILED")
                lines.append(f"Error: {self.error_type}")
                lines.append(f"Message: {self.error_message}")
                lines.append("")
                lines.append("Runtime output (last 50 lines):")
                lines.extend(self.run_output[-50:])
            return "\n".join(lines)

    # All must build successfully
    build_success = all(getattr(r, 'build_success', False) for r in results)
    run_success = all(getattr(r, 'run_success', False) for r in results)
    output_generated = all(getattr(r, 'output_generated', False) for r in results)

    # Get error info from first failed result
    error_type = None
    error_message = None
    build_output = []
    for r in results:
        if not getattr(r, 'build_success', True):
            error_type = getattr(r, 'error_type', 'BUILD_ERROR')
            error_message = getattr(r, 'error_message', 'Build failed')
            build_output = getattr(r, 'build_output', [])
            break
        if not getattr(r, 'run_success', True):
            error_type = getattr(r, 'error_type', 'RUNTIME_ERROR')
            error_message = getattr(r, 'error_message', 'Runtime error')
            break

    # Aggregate validation: use minimum accuracy (worst case)
    validations = [getattr(r, 'validation', None) for r in results]
    valid_validations = [v for v in validations if v and v.accuracy is not None]

    @dataclass
    class AggregatedValidation:
        accuracy: float | None
        max_error: float | None
        mean_error: float | None

    if valid_validations:
        # Minimum accuracy across all testcases
        min_accuracy = min(v.accuracy for v in valid_validations)
        max_error = max((v.max_error for v in valid_validations if v.max_error is not None), default=None)
        mean_errors = [v.mean_error for v in valid_validations if v.mean_error is not None]
        mean_error = sum(mean_errors) / len(mean_errors) if mean_errors else None
        validation = AggregatedValidation(min_accuracy, max_error, mean_error)
    else:
        validation = AggregatedValidation(None, None, None)

    # Combine outputs
    run_output = []
    for i, r in enumerate(results):
        if hasattr(r, 'run_output'):
            run_output.append(f"=== Testcase {i+1} ===")
            run_output.extend(r.run_output if isinstance(r.run_output, list) else [r.run_output])

    total_time = sum(getattr(r, 'total_time', 0) for r in results)

    return AggregatedResult(
        build_success=build_success,
        run_success=run_success,
        output_generated=output_generated,
        validation=validation,
        run_output=run_output,
        build_output=build_output,
        total_time=total_time,
        error_type=error_type,
        error_message=error_message,
    )


def journal_to_rich_tree(journal: Journal) -> Tree:
    """Convert journal to Rich tree for visualization."""
    best_node = journal.get_best_node()

    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""
            if node is best_node:
                s = f"[{style}green]● {node.metric.value:.4f} (best)"
            else:
                s = f"[{style}green]● {node.metric.value:.4f}"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution Tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree


def run_fhe():
    """Main entry point for FHE challenge solving."""
    # Load configuration
    cfg = load_fhe_config()

    logger.info(f'Starting FHE run "{cfg.exp_name}"')

    # Parse challenge
    with Status("[cyan]Parsing challenge specification..."):
        spec = parse_challenge(cfg.challenge_dir)

    logger.info(f"Challenge parsed: {spec.challenge_name or spec.task}")
    logger.info(f"  Type: {spec.challenge_type.value}")
    logger.info(f"  Scheme: {spec.scheme.value if spec.scheme else 'Unknown'}")
    logger.info(f"  Library: {spec.library.value if spec.library else 'Unknown'}")

    # Setup workspace
    with Status("[cyan]Preparing workspace..."):
        cfg.workspace_dir.mkdir(parents=True, exist_ok=True)
        cfg.log_dir.mkdir(parents=True, exist_ok=True)

        # Copy challenge.md to workspace for reference
        challenge_file = cfg.challenge_dir / "challenge.md"
        if challenge_file.exists():
            shutil.copy2(challenge_file, cfg.workspace_dir / "challenge.md")

    # Create interpreter based on challenge type
    with Status("[cyan]Creating interpreter..."):
        interpreter = create_interpreter(
            spec=spec,
            workspace_dir=cfg.workspace_dir,
            build_timeout=cfg.exec.build_timeout,
            run_timeout=cfg.exec.run_timeout,
        )

    logger.info(f"Interpreter: {interpreter.__class__.__name__}")

    # Create journal and agent
    journal = Journal()

    agent_config = AgentConfig(
        code_model=cfg.agent.code.model,
        code_temp=cfg.agent.code.temp,
        reasoning_effort=cfg.agent.code.reasoning_effort,
        web_search=cfg.agent.code.web_search,
        feedback_enabled=cfg.agent.feedback.enabled,
        feedback_model=cfg.agent.feedback.model,
        feedback_temp=cfg.agent.feedback.temp,
        max_lessons=cfg.agent.feedback.max_lessons,
        num_drafts=cfg.agent.search.num_drafts,
        debug_prob=cfg.agent.search.debug_prob,
        max_debug_depth=cfg.agent.search.max_debug_depth,
        timeout=cfg.exec.run_timeout,  # Agent uses run timeout for tracking
    )

    agent = FHEAgent(
        spec=spec,
        journal=journal,
        config=agent_config,
        workspace_dir=cfg.workspace_dir,
    )

    # Cleanup on exit if no work done
    global_step = 0

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir, ignore_errors=True)

    atexit.register(cleanup)

    # Build challenge summary for display
    challenge_summary = _build_challenge_summary(spec)

    # Progress display
    prog = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )
    status = Status("[green]Generating code...")
    prog.add_task("Progress:", total=cfg.agent.steps, completed=global_step)

    def generate_live():
        """Generate live display content."""
        tree = journal_to_rich_tree(journal)
        prog.update(prog.task_ids[0], completed=global_step)

        # Get best accuracy
        best_node = journal.get_best_node()
        best_accuracy = best_node.metric.value if best_node else 0.0

        file_paths = [
            f"Challenge: [yellow]{cfg.challenge_dir}",
            f"Type: [cyan]{spec.challenge_type.value}",
            f"Workspace: [yellow]{cfg.workspace_dir}",
            f"Best accuracy: [green]{best_accuracy:.4f}" if best_node else "Best accuracy: [red]N/A",
        ]

        left = Group(
            Panel(
                Text(challenge_summary[:500] + "..." if len(challenge_summary) > 500 else challenge_summary),
                title="Challenge"
            ),
            prog,
            status,
        )
        right = tree
        wide = Group(*[Text(p) for p in file_paths])

        return Panel(
            Group(
                Padding(wide, (1, 1, 1, 1)),
                Columns(
                    [Padding(left, (1, 2, 1, 1)), Padding(right, (1, 1, 1, 2))],
                    equal=True,
                ),
            ),
            title=f'[b]AIDE-FHE: [bold green]"{cfg.exp_name}[/b]"',
            subtitle="Press [b]Ctrl+C[/b] to stop",
        )

    # Define execution callback that uses interpreter
    def exec_callback(code: str):
        """Execute code using the type-specific interpreter.

        Runs on all available testcases and aggregates results.
        """
        if not spec.testcase_dirs:
            return interpreter.execute(code, None)

        # Run on all testcases
        results = []
        for testcase_path in spec.testcase_dirs:
            result = interpreter.execute(code, testcase_path)
            results.append(result)

        # If only one testcase, return it directly
        if len(results) == 1:
            return results[0]

        # Aggregate results across testcases
        return _aggregate_results(results)

    # Main loop
    consecutive_failures = 0
    max_consecutive_failures = 5
    with Live(generate_live(), refresh_per_second=4, screen=True) as live:
        while global_step < cfg.agent.steps:
            try:
                status.update("[magenta]Building & executing...")
                agent.step(exec_callback)
                status.update("[green]Generating code...")

                save_run(cfg, journal, spec)
                global_step = len(journal)
                live.update(generate_live())

                # Reset failure counter on successful step
                consecutive_failures = 0

                # Check if we've achieved target accuracy (configurable)
                best = journal.get_best_node()
                threshold = cfg.agent.early_stop_threshold
                if threshold > 0 and best and best.metric.value and best.metric.value >= threshold:
                    logger.info(f"Achieved high accuracy: {best.metric.value}")
                    break

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                consecutive_failures += 1
                logger.exception(f"Error in step {global_step}: {e}")

                # Stop only if we have too many consecutive failures
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"Stopping after {consecutive_failures} consecutive failures")
                    break
                else:
                    logger.info(f"Continuing after failure ({consecutive_failures}/{max_consecutive_failures})...")
                    global_step += 1
                    continue

    # Final summary
    print_summary(cfg, journal, spec)

    # Cleanup interpreter
    interpreter.cleanup()


def _build_challenge_summary(spec) -> str:
    """Build a summary string for display."""
    lines = [
        f"Task: {spec.task}",
        f"Type: {spec.challenge_type.value}",
    ]

    if spec.scheme:
        lines.append(f"Scheme: {spec.scheme.value}")

    if spec.library:
        lines.append(f"Library: {spec.library.value}")

    if spec.constraints:
        if spec.constraints.depth:
            lines.append(f"Depth: {spec.constraints.depth}")
        if spec.constraints.batch_size:
            lines.append(f"Batch: {spec.constraints.batch_size}")

    if spec.task_description:
        lines.append(f"\n{spec.task_description[:300]}")

    return "\n".join(lines)


def save_run(cfg: FHEConfig, journal: Journal, spec):
    """Save current run state."""
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # Save journal
    dump_json(journal, cfg.log_dir / "journal.json")

    # Save config
    OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")

    # Save challenge spec
    dump_json(spec, cfg.log_dir / "challenge_spec.json")

    # Save best solution
    best_node = journal.get_best_node(only_good=False)
    if best_node:
        (cfg.log_dir / "best_solution.txt").write_text(best_node.code)


def print_summary(cfg: FHEConfig, journal: Journal, spec):
    """Print final summary."""
    print("\n" + "=" * 60)
    print("FHE Challenge Run Complete")
    print("=" * 60)

    print(f"\nChallenge: {spec.challenge_name or spec.task}")
    print(f"Type: {spec.challenge_type.value}")
    print(f"Scheme: {spec.scheme.value if spec.scheme else 'Unknown'}")

    print(f"\nTotal iterations: {len(journal)}")
    print(f"Good solutions: {len(journal.good_nodes)}")
    print(f"Buggy solutions: {len(journal.buggy_nodes)}")

    best_node = journal.get_best_node()
    if best_node:
        print(f"\nBest solution:")
        print(f"  Accuracy: {best_node.metric.value:.4f}")
        plan_text = best_node.plan or ""
        print(f"  Plan: {plan_text[:200]}..." if len(plan_text) > 200 else f"  Plan: {plan_text}")

    print(f"\nOutput files:")
    print(f"  Journal: {cfg.log_dir / 'journal.json'}")
    print(f"  Best solution: {cfg.log_dir / 'best_solution.txt'}")
    print(f"  Workspace: {cfg.workspace_dir}")


def main():
    """CLI entry point."""
    run_fhe()


if __name__ == "__main__":
    main()
