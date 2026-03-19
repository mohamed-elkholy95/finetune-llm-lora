"""Fine-Tune LLM with LoRA — Configuration."""
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
REPORT_DIR = BASE_DIR / "reports"
CONFIG_DIR = BASE_DIR / "configs"

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODEL_DIR, CHECKPOINT_DIR, REPORT_DIR, CONFIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

SUPPORTED_MODELS = {
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "phi3": "microsoft/phi-3-mini-4k-instruct",
    "distilbert": "distilbert/distilbert-base-uncased",
}

DEFAULT_MODEL = "distilbert"

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

TRAINING_ARGS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "bf16": False,
    "fp16": False,
    "gradient_checkpointing": True,
    "report_to": "none",
}

API_HOST = "0.0.0.0"
API_PORT = 8003

STREAMLIT_THEME = {
    "primaryColor": "#1f77b4",
    "backgroundColor": "#0e1117",
    "secondaryBackgroundColor": "#262730",
    "textColor": "#ffffff",
}
