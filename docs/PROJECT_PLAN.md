# Project Plan — Fine-Tune LLM with LoRA

> **Duration:** 8 days  
> **Goal:** Fine-tune a 7–8B parameter language model (Llama 3, Mistral, or Phi-3) on a custom task using LoRA with 4-bit quantization, then evaluate and deploy the result.

---

## Table of Contents

1. [Phase 1: Environment & Model Setup (Day 1)](#phase-1-environment--model-setup)
2. [Phase 2: Dataset Preparation (Days 1–2)](#phase-2-dataset-preparation)
3. [Phase 3: LoRA Configuration (Days 2–3)](#phase-3-lora-configuration)
4. [Phase 4: Training Pipeline (Days 3–5)](#phase-4-training-pipeline)
5. [Phase 5: Inference & Chat (Days 5–6)](#phase-5-inference--chat)
6. [Phase 6: Evaluation (Days 6–7)](#phase-6-evaluation)
7. [Phase 7: Deployment (Days 7–8)](#phase-7-deployment)
8. [File Structure](#file-structure)
9. [Dependencies](#dependencies)
10. [Success Criteria](#success-criteria)

---

## Phase 1: Environment & Model Setup (Day 1)

### Objective
Configure the GPU environment, install all dependencies, and verify model loading before touching any data or training code.

### Hardware Requirements

| Tier | GPU | VRAM | Models Supported |
|---|---|---|---|
| Recommended | A100 80GB | 80GB | All models, full fine-tune |
| Good | A100 40GB / H100 | 40GB | 8B models with 4-bit quant |
| Minimum | RTX 3090/4090 | 24GB | 7–8B with 4-bit quant + LoRA |
| Budget | Google Colab T4 | 16GB | 7B with 4-bit quant + LoRA |
| Ultra-budget | Google Colab free | 15GB | Phi-3-mini 4-bit only |

### Setup Steps

```bash
# Create conda environment
conda create -n lora-finetune python=3.11
conda activate lora-finetune

# Install PyTorch with CUDA 12.1
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install training stack
pip install transformers>=4.36.0 peft>=0.7.0 trl>=0.7.0
pip install bitsandbytes>=0.41.0 accelerate>=0.25.0
pip install datasets>=2.16.0 evaluate>=0.4.0

# Install evaluation
pip install rouge_score sacrebleu scipy

# Verify CUDA and bitsandbytes
python -c "import torch; print(torch.cuda.is_available()); import bitsandbytes; print(bitsandbytes.__version__)"
```

### Model Access

```bash
# Authenticate with HuggingFace (required for Llama 3)
huggingface-cli login

# Verify model download (dry run)
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B', use_fast=True)
print('Tokenizer loaded:', tok.vocab_size)
"
```

### Model Selection Guide

| Model | Params | Strong At | License |
|---|---|---|---|
| `meta-llama/Meta-Llama-3-8B` | 8B | Instruction following, reasoning | Llama 3 Community |
| `mistralai/Mistral-7B-v0.1` | 7B | Code, structured output | Apache 2.0 |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | Low-resource, efficiency | MIT |

### Deliverables
- [ ] CUDA visible, `torch.cuda.is_available()` returns `True`
- [ ] bitsandbytes installed and functional
- [ ] Base model loads successfully with 4-bit quant (no errors)
- [ ] Environment snapshot saved to `environment.yml`

---

## Phase 2: Dataset Preparation (Days 1–2)

### Objective
Select a task, download and format training data in chat template format, apply quality filters, and prepare DataLoader-ready splits.

### Supported Tasks & Recommended Datasets

| Task | Dataset | HuggingFace ID | Format |
|---|---|---|---|
| Instruction following | Alpaca | `tatsu-lab/alpaca` | Alpaca |
| Domain QA | TriviaQA | `trivia_qa` | QA |
| Summarization | CNN/DailyMail | `cnn_dailymail` | Article/Summary |
| Chatbot | ShareGPT | `anon8231489123/ShareGPT_Vicuna_unfiltered` | ShareGPT |
| Code generation | Code Alpaca | `sahil2801/CodeAlpaca-20k` | Alpaca |

### Files

#### `src/data_prep.py`

```python
import json
import hashlib
from typing import Optional, Literal
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer

DatasetFormat = Literal["alpaca", "sharegpt", "qa", "summarization"]


def load_dataset_from_hub(
    dataset_name: str,
    split: str = "train",
    cache_dir: Optional[str] = None,
    streaming: bool = False,
) -> Dataset:
    """
    Load a dataset from HuggingFace Hub.

    Args:
        dataset_name: HuggingFace dataset identifier.
        split:        Dataset split ('train', 'test', 'validation').
        cache_dir:    Local cache directory. Defaults to ~/.cache/huggingface.
        streaming:    Use streaming for very large datasets.

    Returns:
        HuggingFace Dataset object.

    Examples:
        >>> ds = load_dataset_from_hub("tatsu-lab/alpaca", split="train")
        >>> len(ds)
        52002
    """
    ...


def convert_to_instruction_format(
    data: Dataset,
    source_format: DatasetFormat,
    system_prompt: str = "You are a helpful assistant.",
    max_turns: int = 10,
) -> list[list[dict]]:
    """
    Convert dataset to OpenAI-style chat message format.

    Each example is converted to a conversation:
    [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": instruction + input},
        {"role": "assistant", "content": output},
    ]

    For ShareGPT (multi-turn), preserves all turns up to max_turns.
    For Alpaca (single-turn), wraps instruction+input into user message.

    Args:
        data:          Source dataset.
        source_format: Dataset schema format.
        system_prompt: System message prepended to every conversation.
        max_turns:     Maximum number of conversation turns to keep.

    Returns:
        List of conversations (each a list of message dicts).

    Raises:
        ValueError: If source_format is not recognized.
        KeyError:   If required columns are missing from dataset.
    """
    ...


def format_for_training(
    examples: list[list[dict]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    add_eos_token: bool = True,
) -> dict[str, list]:
    """
    Apply chat template and tokenize conversations for SFTTrainer.

    Uses tokenizer.apply_chat_template() to format messages, then
    tokenizes with truncation to max_length.

    Args:
        examples:      List of conversation dicts from convert_to_instruction_format.
        tokenizer:     The model's tokenizer (must have chat_template set).
        max_length:    Maximum sequence length. Examples longer than this are dropped.
        add_eos_token: Append EOS token after assistant turn.

    Returns:
        Dict with 'input_ids', 'attention_mask', 'labels' keys.
        Labels are -100 for system+user tokens (ignore in loss), 
        actual token IDs for assistant tokens only.
    """
    ...


def split_dataset(
    dataset: Dataset,
    test_size: float = 0.1,
    val_size: float = 0.05,
    seed: int = 42,
) -> DatasetDict:
    """
    Split dataset into train/validation/test splits.

    Args:
        dataset:   Full dataset.
        test_size: Fraction for test set.
        val_size:  Fraction for validation set (applied to remaining data).
        seed:      Random seed for reproducibility.

    Returns:
        DatasetDict with 'train', 'validation', 'test' keys.
    """
    ...


def deduplicate(
    dataset: Dataset,
    text_col: str = "text",
) -> Dataset:
    """
    Remove duplicate examples using MD5 hash comparison.
    Logs number of duplicates removed.
    """
    ...


def filter_by_length(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    min_length: int = 20,
    max_length: int = 2048,
) -> Dataset:
    """
    Filter examples by tokenized length to remove too-short or too-long examples.
    Logs filtering statistics.
    """
    ...


def quality_score(
    text: str,
    min_words: int = 10,
    max_repetition_ratio: float = 0.3,
) -> float:
    """
    Heuristic quality score for a text sample. Returns 0.0–1.0.

    Penalizes:
    - Very short texts (< min_words)
    - Repetitive n-grams (ratio > max_repetition_ratio)
    - All-caps text
    - Non-ASCII heavy content

    Args:
        text:                 Input text.
        min_words:            Minimum word count for full score.
        max_repetition_ratio: Maximum ratio of repeated 3-grams.

    Returns:
        Float quality score between 0.0 and 1.0.
    """
    ...
```

### Data Quality Pipeline

```
Raw Dataset
    │
    ├─→ deduplicate()          # Remove exact/near-duplicates
    ├─→ filter_by_length()     # Remove too-short / too-long
    ├─→ quality_score() < 0.5  # Drop low-quality samples
    └─→ convert_to_instruction_format()
            │
            └─→ format_for_training() → tokenized Dataset
```

### Deliverables
- [ ] All 5 dataset formats load without error
- [ ] `convert_to_instruction_format()` produces valid chat dicts for all formats
- [ ] Quality filtering reduces dataset noise (logged stats)
- [ ] Final dataset saved to `data/processed/` as `.arrow` files

---

## Phase 3: LoRA Configuration (Days 2–3)

### Objective
Configure LoRA hyperparameters and 4-bit quantization for memory-efficient fine-tuning.

### Background: How LoRA Works

For a pre-trained weight matrix W ∈ ℝ^(d×k), LoRA adds:
```
W' = W + BA
where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), rank r << min(d, k)
```

Only A and B are trained. During inference, BA is merged into W at zero cost.

### Files

#### `src/lora_config.py`

```python
from peft import LoraConfig, TaskType, get_peft_model
from transformers import BitsAndBytesConfig
import torch
from typing import Optional


def get_lora_config(
    task_type: str = "CAUSAL_LM",
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[list[str]] = None,
    lora_dropout: float = 0.05,
    bias: str = "none",
    modules_to_save: Optional[list[str]] = None,
) -> LoraConfig:
    """
    Build a LoRA configuration for PEFT.

    Args:
        task_type:       PEFT task type. Use "CAUSAL_LM" for generative models.
        r:               LoRA rank. Higher r = more capacity but more params.
                         Typical values: 4 (minimal), 8–16 (balanced), 32–64 (high).
        lora_alpha:      Scaling factor. Effective scale = lora_alpha / r.
                         Convention: set lora_alpha = 2*r.
        target_modules:  Attention projection layers to adapt.
                         Llama 3 default: ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
                         Minimal (attention only): ["q_proj","v_proj"]
        lora_dropout:    Dropout on LoRA layers. 0.05 is typical.
        bias:            How to handle bias params. "none" trains no biases.
        modules_to_save: Modules outside LoRA to train fully (e.g., "embed_tokens").

    Returns:
        Configured LoraConfig object.

    Examples:
        >>> config = get_lora_config(r=16, lora_alpha=32)
        >>> model = get_peft_model(base_model, config)
        >>> model.print_trainable_parameters()
        trainable params: 8,388,608 || all params: 8,038,785,024 || trainable%: 0.1044
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    return LoraConfig(
        task_type=TaskType[task_type],
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        modules_to_save=modules_to_save,
    )


def get_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> BitsAndBytesConfig:
    """
    Build a BitsAndBytes 4-bit quantization configuration.

    4-bit NF4 quantization reduces a 7B model from ~14GB to ~4GB VRAM.
    Double quantization adds an additional ~0.5GB savings.

    Args:
        load_in_4bit:              Enable 4-bit loading.
        bnb_4bit_compute_dtype:    Dtype for computation. bfloat16 is recommended
                                   (requires Ampere GPU or newer).
        bnb_4bit_quant_type:       Quantization type. "nf4" outperforms "fp4".
        bnb_4bit_use_double_quant: Quantize quantization constants as well.

    Returns:
        Configured BitsAndBytesConfig.

    Memory footprint at inference:
        - 7B float32: ~28GB
        - 7B float16: ~14GB
        - 7B int8:    ~7GB
        - 7B nf4:     ~4GB  ← this config
    """
    ...


def count_trainable_parameters(model) -> tuple[int, int, float]:
    """
    Count trainable vs total parameters.

    Returns:
        (trainable_params, total_params, trainable_percentage)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, 100 * trainable / total


def get_model_memory_footprint(model) -> float:
    """Return model memory in GB (parameters + buffers)."""
    ...
```

### LoRA Rank Impact Analysis

```
                Trainable Params (Llama-3-8B)
    r=4:    ████░░░░░░   4.2M  (0.052%)
    r=8:    █████░░░░░   8.4M  (0.105%)
    r=16:   ██████░░░░  16.8M  (0.209%)   ← recommended
    r=32:   ████████░░  33.6M  (0.418%)
    r=64:   ██████████  67.1M  (0.836%)
```

### Deliverables
- [ ] 4-bit quantized model loads in <6GB VRAM
- [ ] `count_trainable_parameters()` shows <1% trainable params
- [ ] LoRA config saved to `configs/lora_config.json`

---

## Phase 4: Training Pipeline (Days 3–5)

### Objective
Train the LoRA-adapted model using the TRL SFTTrainer with gradient checkpointing, mixed precision, and proper checkpointing.

### Files

#### `src/trainer.py`

```python
import os
import json
import logging
from pathlib import Path
from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset, DatasetDict

from src.lora_config import get_lora_config, get_quantization_config

logger = logging.getLogger(__name__)


class FineTuner:
    """
    End-to-end fine-tuning pipeline for LLMs using LoRA + QLoRA.

    Handles:
    - 4-bit quantized model loading
    - LoRA adapter injection
    - SFTTrainer-based training with gradient checkpointing
    - Checkpoint saving and resumption
    - LoRA weight merging for deployment

    Usage:
        tuner = FineTuner(
            model_name="meta-llama/Meta-Llama-3-8B",
            lora_config=get_lora_config(r=16),
            quantization_config=get_quantization_config(),
        )
        tuner.prepare_dataset(dataset)
        tuner.train("outputs/run1", num_epochs=3)
        tuner.merge_and_save("models/llama3-finetuned")
    """

    def __init__(
        self,
        model_name: str,
        lora_config: LoraConfig,
        quantization_config,
        device_map: str = "auto",
        gradient_checkpointing: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Load base model with quantization and inject LoRA adapters.

        Args:
            model_name:              HuggingFace model ID or local path.
            lora_config:             PEFT LoraConfig.
            quantization_config:     BitsAndBytesConfig for 4-bit loading.
            device_map:              Device placement. 'auto' for multi-GPU.
            gradient_checkpointing:  Trade compute for memory.
            torch_dtype:             Computation dtype (bfloat16 recommended).
            cache_dir:               HuggingFace cache directory.

        After initialization:
            self.model      — LoRA-adapted model
            self.tokenizer  — model tokenizer with padding token set
        """
        ...

    def prepare_dataset(
        self,
        dataset: DatasetDict,
        text_field: str = "text",
        max_seq_length: int = 2048,
    ) -> None:
        """
        Tokenize and set dataset for training.

        Stores train/validation splits as self.train_dataset / self.eval_dataset.
        Computes and logs dataset statistics (sizes, avg length, length histogram).

        Args:
            dataset:        DatasetDict with 'train' and 'validation' keys.
            text_field:     Column name containing formatted conversation text.
            max_seq_length: Maximum tokenized length (longer examples are dropped).
        """
        ...

    def train(
        self,
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 4,
        lr: float = 2e-4,
        gradient_accumulation_steps: int = 4,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.001,
        lr_scheduler_type: str = "cosine",
        save_steps: int = 100,
        eval_steps: int = 100,
        logging_steps: int = 10,
        max_grad_norm: float = 0.3,
        resume_from_checkpoint: Optional[str] = None,
        report_to: str = "tensorboard",
    ) -> dict:
        """
        Run SFT training loop.

        Effective batch size = batch_size * gradient_accumulation_steps.
        For batch_size=4, grad_accum=4: effective batch = 16.

        Training logs:
        - Loss curve (train + eval)
        - Learning rate schedule
        - Gradient norm
        - Tokens/second throughput

        Args:
            output_dir:                   Directory for checkpoints and logs.
            num_epochs:                   Training epochs.
            batch_size:                   Per-device batch size.
            lr:                           Peak learning rate.
            gradient_accumulation_steps:  Steps to accumulate gradients.
            warmup_ratio:                 LR warmup fraction.
            weight_decay:                 AdamW weight decay.
            lr_scheduler_type:            'cosine', 'linear', or 'constant'.
            save_steps:                   Save checkpoint every N steps.
            eval_steps:                   Evaluate every N steps.
            logging_steps:                Log metrics every N steps.
            max_grad_norm:                Gradient clipping threshold.
            resume_from_checkpoint:       Path to checkpoint to resume from.
            report_to:                    Logging backend: 'tensorboard', 'wandb', 'none'.

        Returns:
            Training metrics dict:
            {
              'final_train_loss': float,
              'final_eval_loss': float,
              'best_eval_loss': float,
              'best_checkpoint': str,
              'total_steps': int,
              'total_time_seconds': float,
              'tokens_per_second': float,
            }
        """
        ...

    def evaluate(
        self,
        test_dataset: Dataset,
        batch_size: int = 8,
    ) -> dict:
        """
        Run evaluation on test set.

        Returns:
            {
              'eval_loss': float,
              'eval_perplexity': float,
              'eval_runtime': float,
              'eval_samples_per_second': float,
            }
        """
        ...

    def save_model(
        self,
        output_dir: str,
        save_tokenizer: bool = True,
    ) -> None:
        """
        Save LoRA adapter weights only (not full model).

        LoRA adapters are ~50–500MB, compared to 15–30GB for full model.
        Load with: PeftModel.from_pretrained(base_model, output_dir).
        """
        ...

    def merge_and_save(
        self,
        output_dir: str,
        safe_serialization: bool = True,
    ) -> None:
        """
        Merge LoRA weights into base model and save as a standalone model.

        Merged model can be used without PEFT library — works with vanilla
        transformers, vLLM, Ollama, and llama.cpp.

        Steps:
        1. Reload base model in float16 (not quantized)
        2. Load LoRA adapters
        3. Call model.merge_and_unload()
        4. Save merged model + tokenizer

        Args:
            output_dir:          Directory to save merged model.
            safe_serialization:  Use safetensors format (recommended).
        """
        ...

    def load_model(
        self,
        model_path: str,
        is_merged: bool = False,
    ) -> None:
        """
        Load a previously saved model.

        Args:
            model_path: Path to LoRA adapters or merged model.
            is_merged:  If True, load as standalone model.
                        If False, load as PEFT model on top of base model.
        """
        ...
```

### Training Configuration Example

```yaml
# configs/llama3_summarization.yaml
model:
  name: meta-llama/Meta-Llama-3-8B
  torch_dtype: bfloat16

lora:
  r: 16
  lora_alpha: 32
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  lora_dropout: 0.05

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true

training:
  dataset: cnn_dailymail
  task: summarization
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  lr: 2e-4
  warmup_ratio: 0.03
  max_seq_length: 2048
  output_dir: outputs/llama3-cnn
```

### Deliverables
- [ ] Training runs without OOM on target GPU
- [ ] Training loss decreases monotonically for first 500 steps
- [ ] Checkpoints save correctly and can be resumed
- [ ] `merge_and_save()` produces a standalone model that loads without PEFT

---

## Phase 5: Inference & Chat (Days 5–6)

### Objective
Implement fast text generation with proper sampling, streaming output, and a multi-turn chat interface with conversation history.

### Files

#### `src/inference.py`

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from threading import Thread
from typing import Optional, Generator
import logging

logger = logging.getLogger(__name__)


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    stream: bool = False,
) -> str:
    """
    Generate a response for a single prompt.

    Args:
        model:              Fine-tuned or base language model.
        tokenizer:          Corresponding tokenizer.
        prompt:             Input prompt string.
        max_new_tokens:     Maximum tokens to generate.
        temperature:        Sampling temperature. Higher = more random.
                            Use 0.0 for deterministic (greedy) decoding.
        top_p:              Nucleus sampling threshold. 0.9 keeps top 90%.
        top_k:              Top-k sampling. Limits vocabulary at each step.
        repetition_penalty: Penalize repeating tokens. 1.0 = no penalty.
        do_sample:          True for sampling, False for greedy decoding.
        stream:             If True, prints tokens as they are generated.

    Returns:
        Generated text (excluding the input prompt).

    Examples:
        >>> response = generate_response(model, tokenizer, "What is RAG?")
        >>> print(response)
        "Retrieval-Augmented Generation (RAG) is a technique..."
    """
    ...


def generate_stream(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> Generator[str, None, None]:
    """
    Stream generated tokens one at a time using TextIteratorStreamer.

    Runs generation in a background thread and yields tokens as
    they are produced. Suitable for real-time chat UI streaming.

    Yields:
        Token strings as they are generated.

    Usage:
        for token in generate_stream(model, tokenizer, prompt):
            print(token, end="", flush=True)
    """
    ...


def chat(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict],
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    max_history_turns: int = 10,
) -> str:
    """
    Generate a response in a multi-turn conversation.

    Applies the tokenizer's chat template to format the full
    conversation history before generation.

    Args:
        model:            Language model.
        tokenizer:        Model tokenizer (must have chat_template).
        messages:         Conversation history as list of role/content dicts.
                          [{"role": "user", "content": "Hello"},
                           {"role": "assistant", "content": "Hi there!"},
                           {"role": "user", "content": "How are you?"}]
        system_prompt:    If provided, prepended as a system message.
        max_new_tokens:   Maximum tokens for the response.
        temperature:      Sampling temperature.
        max_history_turns: Truncate history to last N turns if too long.

    Returns:
        Assistant's response text (not including history).
    """
    ...


def batch_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    batch_size: int = 8,
) -> list[str]:
    """
    Generate responses for multiple prompts in batches.

    Pads inputs to the same length within each batch.
    Significantly faster than sequential single-prompt generation.

    Args:
        prompts:    List of input prompts.
        batch_size: Number of prompts to process simultaneously.

    Returns:
        List of generated responses (same order as prompts).
    """
    ...


class InteractiveChat:
    """
    Command-line interactive chat session with the fine-tuned model.

    Maintains conversation history and supports commands:
        /clear   — Clear conversation history
        /history — Print conversation history
        /quit    — Exit the chat
        /system <prompt>  — Update system prompt

    Usage:
        chat_session = InteractiveChat(model, tokenizer, system_prompt="You are a Python expert.")
        chat_session.run()
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> None: ...

    def run(self) -> None:
        """Start the interactive chat loop."""
        ...
```

### Deliverables
- [ ] `generate_response()` returns coherent output
- [ ] `generate_stream()` streams tokens in real-time
- [ ] `chat()` maintains multi-turn context correctly
- [ ] `batch_generate()` is ≥3× faster than sequential for 16 prompts

---

## Phase 6: Evaluation (Days 6–7)

### Objective
Quantitatively compare the fine-tuned model against the base model using automatic metrics and a Streamlit human evaluation interface.

### Files

#### `src/evaluation.py`

```python
import json
import math
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import load as load_metric
import streamlit as st
import pandas as pd
from typing import Optional


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_data: Dataset,
    text_column: str = "text",
    batch_size: int = 8,
    stride: int = 512,
    max_length: int = 2048,
) -> float:
    """
    Compute perplexity on a test set using sliding window approach.

    Perplexity = exp(-1/N * Σ log P(w_i | w_1...w_{i-1}))
    Lower perplexity = better model fit to test distribution.

    Uses sliding window to handle sequences longer than model's context.

    Args:
        model:        Language model to evaluate.
        tokenizer:    Model tokenizer.
        test_data:    Dataset with text column.
        text_column:  Column containing text.
        batch_size:   Evaluation batch size.
        stride:       Window stride for long sequences.
        max_length:   Maximum sequence length.

    Returns:
        Perplexity score (float). Typical values:
        - GPT-2 on WikiText-103: ~16.0
        - Fine-tuned model on domain: 5–10
    """
    ...


def compute_bleu(
    predictions: list[str],
    references: list[str],
    max_order: int = 4,
    smooth: bool = True,
) -> dict:
    """
    Compute BLEU score for text generation tasks.

    Args:
        predictions: Model-generated texts.
        references:  Ground truth texts (one reference per prediction).
        max_order:   Maximum n-gram order (4 for BLEU-4).
        smooth:      Apply smoothing for short sentences.

    Returns:
        {
          'bleu': float,           # Overall BLEU score (0–100)
          'bleu_1': float,
          'bleu_2': float,
          'bleu_3': float,
          'bleu_4': float,
          'brevity_penalty': float,
        }
    """
    ...


def compute_rouge(
    predictions: list[str],
    references: list[str],
    rouge_types: list[str] = ["rouge1", "rouge2", "rougeL"],
) -> dict:
    """
    Compute ROUGE scores for summarization tasks.

    Args:
        predictions:  Model summaries.
        references:   Reference summaries.
        rouge_types:  Which ROUGE variants to compute.

    Returns:
        {
          'rouge1': {'precision': float, 'recall': float, 'fmeasure': float},
          'rouge2': {'precision': float, 'recall': float, 'fmeasure': float},
          'rougeL': {'precision': float, 'recall': float, 'fmeasure': float},
        }
    """
    ...


def compare_models(
    base_model: AutoModelForCausalLM,
    finetuned_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_prompts: list[str],
    references: Optional[list[str]] = None,
    metrics: list[str] = ["perplexity", "bleu", "rouge"],
) -> pd.DataFrame:
    """
    Side-by-side comparison of base model vs fine-tuned model.

    Args:
        base_model:       Original pre-trained model.
        finetuned_model:  LoRA fine-tuned model.
        tokenizer:        Shared tokenizer.
        test_prompts:     Prompts to test both models on.
        references:       Ground truth outputs (for BLEU/ROUGE).
        metrics:          Which metrics to compute.

    Returns:
        DataFrame with columns:
        ['prompt', 'base_output', 'finetuned_output', 'metric_name', 'base_score', 'ft_score', 'improvement']
    """
    ...


def human_eval_interface(
    predictions: list[dict],
    output_path: str = "data/human_eval_results.json",
) -> None:
    """
    Launch a Streamlit interface for human evaluation.

    Displays each prediction alongside the prompt and reference,
    asks raters to score on: coherence (1–5), relevance (1–5), quality (1–5).
    Supports multiple annotators with inter-annotator agreement computation.

    Args:
        predictions:  List of {"prompt": str, "base": str, "finetuned": str, "reference": str}.
        output_path:  Where to save human ratings as JSON.

    Streamlit UI:
        - Show prompt, base response, and fine-tuned response (anonymized A/B)
        - Slider ratings for coherence, relevance, overall quality
        - "Next" button to advance to next example
        - Progress bar and summary stats
        - Export ratings to JSON
    """
    ...
```

### Evaluation Targets

| Metric | Base Model | Fine-tuned Target | Notes |
|---|---|---|---|
| Perplexity (domain) | ~12 | <8 | Lower = better |
| ROUGE-L (summarization) | ~28 | >40 | Higher = better |
| BLEU-4 (QA) | ~15 | >25 | Higher = better |
| Human preference | 40% | >65% | % preferred over base |

### Deliverables
- [ ] Perplexity computed for both base and fine-tuned models
- [ ] BLEU/ROUGE scores computed and compared
- [ ] Human eval UI loads and saves ratings correctly
- [ ] Comparison report saved to `reports/model_comparison.html`

---

## Phase 7: Deployment (Days 7–8)

### Objective
Serve the merged fine-tuned model for production inference using vLLM for high throughput or FastAPI for simple use cases.

### Files

#### `src/deploy/serve_vllm.py`

```python
"""
vLLM inference server for the fine-tuned model.

vLLM provides:
- PagedAttention for memory-efficient KV cache
- Continuous batching for high throughput
- OpenAI-compatible API

Usage:
    python src/deploy/serve_vllm.py --model models/llama3-merged --port 8000
"""
import argparse
from vllm import LLM, SamplingParams
from vllm.entrypoints.openai import api_server


def load_vllm_model(
    model_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    dtype: str = "auto",
) -> LLM:
    """
    Load model with vLLM for optimized inference.

    Args:
        model_path:              Path to merged model.
        tensor_parallel_size:    Number of GPUs for tensor parallelism.
        gpu_memory_utilization:  Fraction of GPU memory to use for KV cache.
        max_model_len:           Maximum sequence length.
        dtype:                   Model dtype ('auto', 'float16', 'bfloat16').

    Returns:
        vLLM LLM instance ready for inference.
    """
    ...


def generate_vllm(
    llm: LLM,
    prompts: list[str],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
    stop_sequences: Optional[list[str]] = None,
) -> list[str]:
    """
    Batch generate with vLLM.

    vLLM automatically batches requests for maximum GPU utilization.

    Returns:
        List of generated texts (same order as prompts).
    """
    ...
```

#### `src/deploy/api_server.py`

```python
"""
FastAPI server wrapping either vLLM or transformers inference.
Provides an OpenAI-compatible chat completions endpoint.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Fine-tuned LLM API", version="1.0.0")


class ChatMessage(BaseModel):
    role: str    # "system", "user", or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


class ChatResponse(BaseModel):
    id: str
    model: str
    created: int
    choices: list[dict]
    usage: dict


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest) -> ChatResponse:
    """
    OpenAI-compatible chat completions endpoint.
    Compatible with OpenAI Python SDK by changing base_url.
    """
    ...


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}
```

### Deployment Options Comparison

| Option | Throughput | Latency | Setup | Use Case |
|---|---|---|---|---|
| vLLM | ★★★★★ | ★★★★☆ | Medium | Production, high traffic |
| Transformers | ★★☆☆☆ | ★★★☆☆ | Easy | Dev, low traffic |
| Ollama | ★★★☆☆ | ★★★★☆ | Easy | Local deployment |
| TGI | ★★★★☆ | ★★★★★ | Hard | Cloud production |

### Deliverables
- [ ] vLLM server starts and serves requests correctly
- [ ] FastAPI `/v1/chat/completions` returns valid responses
- [ ] API is compatible with OpenAI Python SDK
- [ ] Throughput ≥50 tokens/second on A100

---

## File Structure

```
07-finetune-llm-lora/
├── src/
│   ├── __init__.py
│   ├── data_prep.py
│   ├── lora_config.py
│   ├── trainer.py
│   ├── inference.py
│   ├── evaluation.py
│   └── deploy/
│       ├── __init__.py
│       ├── serve_vllm.py
│       └── api_server.py
├── configs/
│   ├── llama3_summarization.yaml
│   ├── mistral_qa.yaml
│   └── phi3_instruct.yaml
├── scripts/
│   ├── train.sh
│   ├── train_multi_gpu.sh
│   └── merge_weights.sh
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_lora_training.ipynb
│   └── 03_evaluation.ipynb
├── data/                     # gitignored
├── models/                   # gitignored
├── outputs/                  # Training checkpoints (gitignored)
├── reports/
├── requirements.txt
└── docs/
    └── PROJECT_PLAN.md
```

---

## Dependencies

```
transformers>=4.36.0
peft>=0.7.0
trl>=0.7.0
bitsandbytes>=0.41.0
accelerate>=0.25.0
datasets>=2.16.0
torch>=2.1.0
scipy>=1.11.0
evaluate>=0.4.0
rouge_score>=0.1.2
sacrebleu>=2.3.0
streamlit>=1.30.0
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
vllm>=0.3.0
pydantic>=2.5.0
```

---

## Success Criteria

1. **Environment**: Model loads in 4-bit with <6GB VRAM
2. **Dataset**: 10,000+ quality-filtered training examples prepared
3. **LoRA**: <1% trainable parameters (r=16 on 8B model)
4. **Training**: Loss converges; final eval loss < base model
5. **Perplexity**: Fine-tuned model perplexity ≥20% lower than base on domain test set
6. **ROUGE-L**: +10 points improvement on summarization vs base
7. **Merge**: Standalone merged model loads without PEFT
8. **API**: Chat completions endpoint responds in <2s at temperature=0.7
9. **vLLM**: Throughput ≥50 tokens/second
10. **Human eval**: Fine-tuned responses preferred ≥65% of time
