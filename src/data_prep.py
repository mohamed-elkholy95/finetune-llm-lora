"""Data loading and preparation."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, RAW_DIR

logger = logging.getLogger(__name__)

HAS_DATASETS = False
try:
    from datasets import load_dataset as hf_load_dataset, Dataset
    HAS_DATASETS = True
except ImportError:
    logger.info("datasets library not available")


def load_dataset_from_hub(
    dataset_name: str = "imdb", split: str = "train", sample_size: Optional[int] = None,
) -> List[Dict]:
    """Load dataset from HuggingFace Hub.

    Args:
        dataset_name: HF dataset identifier.
        split: Dataset split.
        sample_size: Optional limit on samples.

    Returns:
        List of dicts with text/label.
    """
    if not HAS_DATASETS:
        logger.warning("datasets not available — returning synthetic data")
        return generate_synthetic_instruct(sample_size or 500)

    try:
        ds = hf_load_dataset(dataset_name, split=split)
        data = []
        for item in ds:
            text = item.get("text", item.get("prompt", ""))
            label = item.get("label", None)
            data.append({"text": text, "label": label})
        if sample_size:
            rng = np.random.default_rng(RANDOM_SEED)
            indices = rng.choice(len(data), min(sample_size, len(data)), replace=False)
            data = [data[i] for i in indices]
        logger.info("Loaded %d samples from %s/%s", len(data), dataset_name, split)
        return data
    except Exception as exc:
        logger.warning("Failed to load %s: %s — using synthetic", dataset_name, exc)
        return generate_synthetic_instruct(sample_size or 500)


def format_for_chat(data: List[Dict], system_prompt: str = "You are a helpful assistant.") -> List[Dict]:
    """Format data for chat/instruction fine-tuning.

    Args:
        data: Raw data dicts.
        system_prompt: System message.

    Returns:
        List of chat-formatted messages.
    """
    formatted = []
    for item in data:
        messages = [{"role": "system", "content": system_prompt}]
        if "instruction" in item:
            messages.append({"role": "user", "content": item["instruction"]})
            messages.append({"role": "assistant", "content": item.get("output", item.get("response", ""))})
        else:
            text = item.get("text", "")
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": item.get("label", "positive")})
        formatted.append({"messages": messages})
    return formatted


def format_for_instruct(data: List[Dict]) -> List[str]:
    """Format data for instruction tuning (text pairs).

    Args:
        data: Raw data dicts.

    Returns:
        List of formatted instruction strings.
    """
    formatted = []
    for item in data:
        if "instruction" in item:
            text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item.get('output', '')}"
        else:
            text = item.get("text", "")
        formatted.append(text)
    return formatted


def quality_filter(data: List[Dict], min_length: int = 10, max_length: int = 10000) -> List[Dict]:
    """Filter data by quality criteria.

    Args:
        data: Raw data list.
        min_length: Minimum text length.
        max_length: Maximum text length.

    Returns:
        Filtered data.
    """
    filtered = [d for d in data if min_length <= len(d.get("text", d.get("instruction", ""))) <= max_length]
    logger.info("Quality filter: %d → %d samples", len(data), len(filtered))
    return filtered


def split_data(
    data: List, test_ratio: float = 0.2, seed: int = RANDOM_SEED,
) -> Tuple[List, List]:
    """Split data into train/test.

    Args:
        data: Data list.
        test_ratio: Fraction for test.
        seed: Random seed.

    Returns:
        (train, test) tuple.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data))
    n_test = int(len(data) * test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    train = [data[i] for i in train_idx]
    test = [data[i] for i in test_idx]
    logger.info("Split: train=%d, test=%d", len(train), len(test))
    return train, test


def get_dataset_stats(data: List[Dict]) -> Dict:
    """Compute dataset statistics."""
    texts = [d.get("text", d.get("instruction", "")) for d in data]
    lengths = [len(t) for t in texts]
    return {
        "n_samples": len(data),
        "avg_length": round(float(np.mean(lengths)), 1),
        "min_length": int(min(lengths)) if lengths else 0,
        "max_length": int(max(lengths)) if lengths else 0,
        "has_labels": any("label" in d for d in data),
    }


def generate_synthetic_instruct(n_samples: int = 500, seed: int = RANDOM_SEED) -> List[Dict]:
    """Generate synthetic instruction data."""
    rng = np.random.default_rng(seed)
    instructions = [
        {"instruction": "Explain machine learning", "output": "Machine learning is a field of AI that enables computers to learn from data."},
        {"instruction": "What is deep learning?", "output": "Deep learning is a subset of ML using neural networks with many layers."},
        {"instruction": "Summarize this text", "output": "This is a summary of the provided text."},
        {"instruction": "Translate to French", "output": "Bonjour le monde"},
        {"instruction": "Write a Python function", "output": "def hello(): return 'world'"},
    ]
    data = []
    for _ in range(n_samples):
        item = instructions[rng.integers(len(instructions))].copy()
        item["text"] = f"Instruction: {item['instruction']} Response: {item['output']}"
        data.append(item)
    return data
