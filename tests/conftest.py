"""Shared test fixtures."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture
def sample_instruct_data():
    return [
        {"instruction": "Explain ML", "output": "ML is AI", "text": "Instruction: Explain ML Response: ML is AI"},
        {"instruction": "What is AI?", "output": "AI is artificial intelligence", "text": "Instruction: What is AI? Response: AI is artificial intelligence"},
        {"instruction": "Write code", "output": "print('hello')", "text": "Instruction: Write code Response: print('hello')"},
    ]


@pytest.fixture
def sample_texts():
    return ["Machine learning is great", "Deep learning uses neural networks", "AI is the future"]


@pytest.fixture
def sample_references():
    return ["ML is a subset of AI", "Neural networks have many layers", "Artificial intelligence will transform society"]
