"""vLLM serving script (with fallback)."""
import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)

HAS_VLLM = False
try:
    from vllm import LLM, SamplingParams  # noqa: F401
    HAS_VLLM = True
except ImportError:
    logger.info("vLLM not installed — this script requires vllm package")


def serve_with_vllm(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
) -> None:
    """Serve a model using vLLM.

    Args:
        model_path: Path to the model.
        host: Server host.
        port: Server port.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        max_model_len: Maximum model context length.
    """
    if not HAS_VLLM:
        logger.error("vLLM not installed. Install with: pip install vllm")
        sys.exit(1)

    try:
        from vllm.entrypoints.openai.api_server import run_server
        import argparse

        # Build CLI args
        sys.argv = ["vllm_serve", model_path, "--host", host, "--port", str(port)]
        if tensor_parallel_size > 1:
            sys.argv.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
        if max_model_len:
            sys.argv.extend(["--max-model-len", str(max_model_len)])

        run_server()
    except Exception as exc:
        logger.error("vLLM serve failed: %s", exc)
        logger.info("Falling back to uvicorn FastAPI server")
        from src.deploy.api_server import app
        import uvicorn
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Serve LoRA model")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    args = parser.parse_args()
    serve_with_vllm(args.model, args.host, args.port, args.tp)
