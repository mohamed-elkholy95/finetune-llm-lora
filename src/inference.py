"""Model inference."""
import logging
from typing import Dict, List, Optional

import numpy as np

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)

HAS_TRANSFORMERS = False
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    logger.info("transformers not installed")


def generate_text(
    prompt: str,
    model=None,
    tokenizer=None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate text from a prompt.

    Args:
        prompt: Input prompt.
        model: Model (or None for mock).
        tokenizer: Tokenizer.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        do_sample: Whether to sample.

    Returns:
        Generated text.
    """
    if model is None or tokenizer is None or not HAS_TRANSFORMERS:
        return mock_generate(prompt)

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as exc:
        logger.warning("Generation failed: %s", exc)
        return mock_generate(prompt)


def chat_completion(
    messages: List[Dict[str, str]],
    model=None,
    tokenizer=None,
    max_new_tokens: int = 256,
) -> str:
    """Generate chat completion.

    Args:
        messages: List of {"role": "...", "content": "..."} dicts.
        model: Model (or None for mock).
        tokenizer: Tokenizer.
        max_new_tokens: Max tokens to generate.

    Returns:
        Assistant response string.
    """
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    prompt += "\nassistant:"
    return generate_text(prompt, model, tokenizer, max_new_tokens)


def batch_inference(
    prompts: List[str],
    model=None,
    tokenizer=None,
    max_new_tokens: int = 128,
) -> List[str]:
    """Run inference on multiple prompts.

    Args:
        prompts: List of input prompts.
        model: Model.
        tokenizer: Tokenizer.
        max_new_tokens: Max tokens to generate.

    Returns:
        List of generated texts.
    """
    return [generate_text(p, model, tokenizer, max_new_tokens) for p in prompts]


def mock_generate(prompt: str) -> str:
    """Generate mock response."""
    responses = {
        "explain": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "what": "This is a complex topic with many dimensions to explore.",
        "how": "The process involves several steps that build upon each other systematically.",
        "default": "Based on the context provided, here is a comprehensive response to your question.",
    }
    prompt_lower = prompt.lower()
    for key, response in responses.items():
        if key in prompt_lower:
            return response
    return responses["default"]


def merge_and_save(
    model, tokenizer, output_path: str, safe_serialization: bool = True,
) -> Dict[str, str]:
    """Merge LoRA weights into base model and save.

    Args:
        model: PEFT model.
        tokenizer: Tokenizer.
        output_path: Directory to save.
        safe_serialization: Use safetensors.

    Returns:
        Dict with paths.
    """
    if model is None:
        logger.warning("No model to merge")
        return {"error": "No model provided"}

    try:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_path, safe_serialization=safe_serialization)
        tokenizer.save_pretrained(output_path)
        logger.info("Merged model saved to %s", output_path)
        return {"model_path": str(output_path), "tokenizer_path": str(output_path)}
    except Exception as exc:
        logger.warning("Merge failed: %s", exc)
        return {"error": str(exc)}
