<div align="center">

# 🧠 LoRA Fine-Tuning LLM

**Efficient LLM fine-tuning** with LoRA/PEFT adapters and configurable training pipeline

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![Transformers](https://img.shields.io/badge/Transformers-5.3-FF6F00?style=flat-square)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-0.18-8B5CF6?style=flat-square)](https://huggingface.co/docs/peft)
[![Tests](https://img.shields.io/badge/Tests-32%20passed-success?style=flat-square)](#)

</div>

## Overview

An **LLM fine-tuning framework** using Low-Rank Adaptation (LoRA) via the PEFT library. Configurable hyperparameters, data preparation pipeline, training loop with loss tracking, inference engine, and evaluation metrics. Gracefully handles missing torch with mock fallbacks.

## Features

- 🔧 **Configurable LoRA** — Rank, alpha, dropout, and target module selection
- 📝 **Data Preparation** — Synthetic dataset generation with train/val/test splits
- 🏋️ **Training Pipeline** — Epoch-level loss tracking with eval metrics
- 🎯 **Inference Engine** — Text generation with temperature and max-length control
- 📊 **Evaluation** — Perplexity, BLEU, and accuracy metrics
- 🚀 **REST API** — Model info and generation endpoints
- 📈 **5-Page Dashboard** — Training, evaluation, chat, and configuration views

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/finetune-llm-lora.git
cd finetune-llm-lora
pip install -r requirements.txt
python -m pytest tests/ -v
streamlit run streamlit_app/app.py
```

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
