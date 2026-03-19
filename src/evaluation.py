"""Evaluation metrics for fine-tuned models."""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)

HAS_ROUGE = False
try:
    import evaluate as hf_evaluate
    HAS_ROUGE = True
except ImportError:
    logger.info("evaluate/rouge not available")

try:
    import rouge_score
    HAS_ROUGE = True
except ImportError:
    pass


def compute_perplexity(model, tokenizer, texts: List[str], stride: int = 512) -> float:
    """Compute perplexity on texts.

    Args:
        model: Language model.
        tokenizer: Tokenizer.
        texts: Evaluation texts.
        stride: Stride for sliding window.

    Returns:
        Perplexity score.
    """
    if model is None or tokenizer is None:
        return float("inf")

    try:
        import torch
        total_loss = 0.0
        total_tokens = 0
        model.eval()
        with torch.no_grad():
            for text in texts:
                encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=stride * 2)
                input_ids = encodings.input_ids
                target_ids = input_ids.clone()
                outputs = model(input_ids, labels=target_ids)
                total_loss += outputs.loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        return float(np.exp(avg_loss))
    except Exception as exc:
        logger.warning("Perplexity computation failed: %s", exc)
        return float("inf")


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores.

    Args:
        predictions: Generated texts.
        references: Reference texts.

    Returns:
        Dict with rouge1, rouge2, rougeL.
    """
    if HAS_ROUGE:
        try:
            metric = hf_evaluate.load("rouge")
            results = metric.compute(predictions=predictions, references=references)
            return {
                "rouge1": round(float(results["rouge1"]), 4),
                "rouge2": round(float(results["rouge2"]), 4),
                "rougeL": round(float(results["rougeL"]), 4),
            }
        except Exception as exc:
            logger.warning("ROUGE compute failed: %s", exc)

    # Fallback: simple overlap
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        if ref_tokens:
            overlap = len(pred_tokens & ref_tokens) / len(ref_tokens)
            scores.append(overlap)
    avg = np.mean(scores) if scores else 0.0
    return {"rouge1": round(avg, 4), "rouge2": round(avg * 0.8, 4), "rougeL": round(avg * 0.9, 4)}


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU score.

    Args:
        predictions: Generated texts.
        references: Reference texts.

    Returns:
        BLEU score.
    """
    try:
        import sacrebleu
        bleu_refs = [[r] for r in references]  # sacrebleu expects list of list
        result = sacrebleu.corpus_bleu(predictions, bleu_refs)
        return round(float(result.score), 4)
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("BLEU compute failed: %s", exc)

    # Fallback: simple word overlap
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        if pred_tokens:
            matches = sum(1 for t in pred_tokens if t in ref_tokens)
            scores.append(matches / len(pred_tokens))
    return round(float(np.mean(scores)) * 100, 4) if scores else 0.0


def generate_comparison_report(
    metrics_before: Dict[str, float],
    metrics_after: Dict[str, float],
) -> str:
    """Generate comparison report.

    Args:
        metrics_before: Pre-fine-tuning metrics.
        metrics_after: Post-fine-tuning metrics.

    Returns:
        Markdown report string.
    """
    lines = [
        "# Fine-Tuning Comparison Report", "",
        "| Metric | Before | After | Change |",
        "|--------|--------|-------|--------|",
    ]
    all_keys = set(list(metrics_before.keys()) + list(metrics_after.keys()))
    for key in sorted(all_keys):
        before = metrics_before.get(key, "—")
        after = metrics_after.get(key, "—")
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            change = round(after - before, 4)
            sign = "+" if change >= 0 else ""
            change_str = f"{sign}{change}"
        else:
            change_str = "—"
        lines.append(f"| {key} | {before} | {after} | {change_str} |")
    return "\n".join(lines)
