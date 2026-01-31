"""Backend for local vLLM server.

vLLM serves an OpenAI-compatible API, so we use the openai client
pointed at the local server. Models are specified as "vllm/<model-name>"
(e.g., "vllm/Qwen/Qwen3-4B-Instruct-2507") and the "vllm/" prefix is
stripped before sending to the server.

The server is auto-launched on first query and shut down on process exit.

Environment variables:
    VLLM_BASE_URL:       Override server URL (default: http://localhost:8000/v1)
    VLLM_GPU_COUNT:      Number of GPUs for tensor parallelism (default: 1)
    VLLM_MAX_MODEL_LEN:  Max sequence length (default: let vLLM decide)
    VLLM_EXTRA_ARGS:     Extra CLI args for vllm serve (space-separated)
"""

import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import time
from urllib.parse import urlparse

import requests
from funcy import notnone, once, select_values
import openai

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore
_server_process: subprocess.Popen = None  # type: ignore
_current_model: str = None  # type: ignore

TIMEOUT_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

VLLM_DEFAULT_HOST = "localhost"
VLLM_DEFAULT_PORT = 8000


def _get_base_url() -> str:
    return os.getenv("VLLM_BASE_URL", f"http://{VLLM_DEFAULT_HOST}:{VLLM_DEFAULT_PORT}/v1")


def _get_health_url() -> str:
    """Derive health endpoint from base URL (strip /v1)."""
    base = _get_base_url()
    if base.endswith("/v1"):
        return base[:-3] + "/health"
    parsed = urlparse(base)
    return f"{parsed.scheme}://{parsed.netloc}/health"


def _server_is_running() -> bool:
    """Check if a vLLM server is already responding."""
    try:
        resp = requests.get(_get_health_url(), timeout=2)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def _stop_server():
    """Shut down the vLLM server we started."""
    global _server_process
    if _server_process is not None and _server_process.poll() is None:
        logger.info("Shutting down vLLM server...")
        _server_process.terminate()
        try:
            _server_process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _server_process.kill()
            _server_process.wait(timeout=5)
        logger.info("vLLM server stopped.")
    _server_process = None


def _start_server(model: str) -> None:
    """
    Launch `vllm serve <model>` as a background process.

    Waits until the health endpoint responds (up to 10 minutes).
    """
    global _server_process, _current_model

    # Already running with correct model
    if _server_process is not None and _server_process.poll() is None and _current_model == model:
        return

    # Server from a previous model — stop it first
    if _server_process is not None:
        _stop_server()

    # If an external server is already running, just use it
    if _server_is_running():
        logger.info("External vLLM server detected, using it directly.")
        _current_model = model
        return

    # Parse host/port from base URL
    base = _get_base_url()
    parsed = urlparse(base)
    host = parsed.hostname or VLLM_DEFAULT_HOST
    port = parsed.port or VLLM_DEFAULT_PORT

    # Build command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", os.getenv("VLLM_GPU_COUNT", "1"),
    ]

    max_model_len = os.getenv("VLLM_MAX_MODEL_LEN")
    if max_model_len:
        cmd += ["--max-model-len", max_model_len]

    extra_args = os.getenv("VLLM_EXTRA_ARGS", "").split()
    if extra_args:
        cmd += extra_args

    logger.info(f"Starting vLLM server: {' '.join(cmd)}")

    log_path = os.path.join(os.getcwd(), "vllm_server.log")
    log_file = open(log_path, "w")
    _server_process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    # Register cleanup
    atexit.register(_stop_server)

    # Wait for server to be ready (up to 10 minutes for large models)
    timeout = 600
    start = time.time()
    logger.info(f"Waiting for vLLM server to be ready (log: {log_path})...")
    while time.time() - start < timeout:
        # Check if process died
        if _server_process.poll() is not None:
            log_file.close()
            with open(log_path) as f:
                tail = f.read()[-2000:]
            raise RuntimeError(
                f"vLLM server exited with code {_server_process.returncode}.\n"
                f"Last output:\n{tail}"
            )
        if _server_is_running():
            elapsed = time.time() - start
            logger.info(f"vLLM server ready in {elapsed:.1f}s")
            _current_model = model
            return
        time.sleep(2)

    # Timed out
    _stop_server()
    raise RuntimeError(
        f"vLLM server did not start within {timeout}s. Check {log_path} for details."
    )


def _ensure_server(model: str):
    """Make sure vLLM is serving the requested model."""
    _start_server(model)


@once
def _setup_vllm_client():
    global _client
    base_url = _get_base_url()
    _client = openai.OpenAI(
        base_url=base_url,
        api_key="EMPTY",  # vLLM doesn't require a real key
        max_retries=0,
    )
    logger.info(f"vLLM client initialized: {base_url}")


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query a local vLLM server via OpenAI-compatible chat completions API.

    Auto-launches the vLLM server if not already running.
    """
    filtered_kwargs: dict = select_values(notnone, model_kwargs)

    # Strip "vllm/" prefix from model name
    model = filtered_kwargs.get("model", "")
    if model.startswith("vllm/"):
        model = model[len("vllm/"):]
        filtered_kwargs["model"] = model

    # Start server if needed, then init client
    _ensure_server(model)
    _setup_vllm_client()

    # vLLM uses max_tokens (not max_output_tokens)
    # Keep as-is since chat completions API uses max_tokens

    # Build messages
    messages = opt_messages_to_list(system_message, user_message)

    # Handle function calling if supported
    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    # Remove params not supported by vLLM
    filtered_kwargs.pop("reasoning_effort", None)
    filtered_kwargs.pop("web_search", None)

    logger.info(f"vLLM request: model={model}")

    t0 = time.time()
    try:
        response = backoff_create(
            _client.chat.completions.create,
            TIMEOUT_EXCEPTIONS,
            messages=messages,
            **filtered_kwargs,
        )
    except openai.BadRequestError as e:
        if func_spec is not None and (
            "tool" in str(e).lower() or "function" in str(e).lower()
        ):
            logger.warning(
                "Function calling not supported by this vLLM model. "
                "Falling back to plain text generation."
            )
            filtered_kwargs.pop("tools", None)
            filtered_kwargs.pop("tool_choice", None)
            response = backoff_create(
                _client.chat.completions.create,
                TIMEOUT_EXCEPTIONS,
                messages=messages,
                **filtered_kwargs,
            )
        else:
            raise
    req_time = time.time() - t0

    # Parse response
    message = response.choices[0].message

    if (
        hasattr(message, "tool_calls")
        and message.tool_calls
        and func_spec is not None
    ):
        tool_call = message.tool_calls[0]
        if tool_call.function.name == func_spec.name:
            try:
                output = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as ex:
                logger.error(f"Error decoding function arguments: {tool_call.function.arguments}")
                raise ex
        else:
            logger.warning(
                f"Function name mismatch: expected {func_spec.name}, "
                f"got {tool_call.function.name}. Fallback to text."
            )
            output = message.content
    else:
        output = message.content

    in_tokens = response.usage.prompt_tokens if response.usage else 0
    out_tokens = response.usage.completion_tokens if response.usage else 0

    info = {
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "model": response.model,
        "created": getattr(response, "created", None),
    }

    logger.info(
        f"vLLM call completed - {response.model} - {req_time:.2f}s - "
        f"{in_tokens + out_tokens} tokens (in: {in_tokens}, out: {out_tokens})"
    )

    return output, req_time, in_tokens, out_tokens, info
