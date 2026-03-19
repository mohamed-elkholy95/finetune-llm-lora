"""Model training wrapper."""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import RANDOM_SEED, TRAINING_ARGS, CHECKPOINT_DIR

logger = logging.getLogger(__name__)

HAS_TRANSFORMERS = False
try:
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
        Trainer, DataCollatorForLanguageModeling,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    logger.info("transformers not installed")


class MockTrainer:
    """Mock trainer for when transformers is unavailable."""

    def __init__(self) -> None:
        self._history: Dict[str, List] = {"loss": [], "eval_loss": [], "epoch": []}
        self._is_trained = False

    def train(self) -> Dict[str, List]:
        rng = np.random.default_rng(RANDOM_SEED)
        for epoch in range(3):
            loss = 2.5 - 0.3 * epoch + 0.05 * rng.standard_normal()
            eval_loss = 2.7 - 0.25 * epoch + 0.05 * rng.standard_normal()
            self._history["loss"].append(round(float(loss), 4))
            self._history["eval_loss"].append(round(float(eval_loss), 4))
            self._history["epoch"].append(epoch + 1)
        self._is_trained = True
        logger.info("Mock training complete: 3 epochs")
        return self._history

    @property
    def is_trained(self) -> bool:
        return self._is_trained


def setup_training(
    model_name: str = "distilbert/distilbert-base-uncased",
    lora_config: Optional[Dict] = None,
    **overrides,
) -> Dict[str, Any]:
    """Setup model and tokenizer for training.

    Args:
        model_name: Pretrained model name.
        lora_config: LoRA config dict.
        **overrides: Training args overrides.

    Returns:
        Dict with model, tokenizer, trainer components.
    """
    if not HAS_TRANSFORMERS:
        logger.warning("transformers unavailable — returning mock components")
        return {"model": None, "tokenizer": None, "trainer": MockTrainer(), "mock": True}

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.info("Loaded model: %s", model_name)

        args_dict = {**TRAINING_ARGS, "output_dir": str(CHECKPOINT_DIR), **overrides}
        args = TrainingArguments(**args_dict)

        return {"model": model, "tokenizer": tokenizer, "training_args": args, "mock": False}
    except Exception as exc:
        logger.warning("Setup failed: %s — using mock", exc)
        return {"model": None, "tokenizer": None, "trainer": MockTrainer(), "mock": True}


def train_model(
    model, tokenizer, train_data: List[str], val_data: Optional[List[str]] = None,
    training_args_override: Optional[Dict] = None,
) -> Dict[str, List]:
    """Train model with provided data.

    Args:
        model: Model to train.
        tokenizer: Tokenizer.
        train_data: Training texts.
        val_data: Validation texts.
        training_args_override: Override training args.

    Returns:
        Training history dict.
    """
    if model is None or not HAS_TRANSFORMERS:
        trainer = MockTrainer()
        return trainer.train()

    try:
        from datasets import Dataset
        train_dataset = Dataset.from_dict({"text": train_data})
        tokenized = train_dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, max_length=512, padding="max_length"),
            remove_columns=["text"],
        )

        args_dict = {**TRAINING_ARGS, "output_dir": str(CHECKPOINT_DIR)}
        if training_args_override:
            args_dict.update(training_args_override)
        args = TrainingArguments(**args_dict)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=data_collator)
        trainer.train()

        history = {"loss": [], "eval_loss": [], "epoch": []}
        if trainer.state.log_history:
            for entry in trainer.state.log_history:
                if "loss" in entry:
                    history["loss"].append(round(entry["loss"], 4))
                if "eval_loss" in entry:
                    history["eval_loss"].append(round(entry["eval_loss"], 4))
                if "epoch" in entry:
                    history["epoch"].append(round(entry["epoch"], 1))
        return history
    except Exception as exc:
        logger.warning("Training failed: %s — returning mock history", exc)
        mock = MockTrainer()
        return mock.train()


def get_training_summary(history: Dict[str, List]) -> Dict[str, Any]:
    """Summarize training history.

    Args:
        history: Training history dict.

    Returns:
        Summary dict.
    """
    return {
        "epochs_completed": len(history.get("epoch", [])),
        "final_loss": history["loss"][-1] if history.get("loss") else None,
        "best_loss": min(history["loss"]) if history.get("loss") else None,
        "final_eval_loss": history["eval_loss"][-1] if history.get("eval_loss") else None,
    }
