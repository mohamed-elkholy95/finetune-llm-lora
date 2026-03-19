"""Tests for data preparation."""
import pytest
from src.data_prep import (
    generate_synthetic_instruct, format_for_chat, format_for_instruct,
    quality_filter, split_data, get_dataset_stats, load_dataset_from_hub,
)


class TestGenerateSynthetic:
    def test_returns_list(self):
        data = generate_synthetic_instruct(n_samples=50)
        assert isinstance(data, list)
        assert len(data) == 50

    def test_has_required_keys(self):
        data = generate_synthetic_instruct(n_samples=10)
        assert "instruction" in data[0]
        assert "output" in data[0]
        assert "text" in data[0]

    def test_reproducible(self):
        d1 = generate_synthetic_instruct(50, seed=42)
        d2 = generate_synthetic_instruct(50, seed=42)
        assert [d["instruction"] for d in d1] == [d["instruction"] for d in d2]


class TestFormatForChat:
    def test_returns_chat_format(self, sample_instruct_data):
        formatted = format_for_chat(sample_instruct_data)
        assert len(formatted) == 3
        assert "messages" in formatted[0]
        assert formatted[0]["messages"][0]["role"] == "system"

    def test_custom_system_prompt(self, sample_instruct_data):
        formatted = format_for_chat(sample_instruct_data, system_prompt="Custom bot")
        assert formatted[0]["messages"][0]["content"] == "Custom bot"


class TestFormatForInstruct:
    def test_instruct_format(self, sample_instruct_data):
        formatted = format_for_instruct(sample_instruct_data)
        assert all("### Instruction:" in t for t in formatted)
        assert all("### Response:" in t for t in formatted)


class TestQualityFilter:
    def test_filters_short(self):
        data = [{"text": "hi"}, {"text": "This is a valid text sample"}, {"text": "ok"}]
        filtered = quality_filter(data, min_length=5)
        assert len(filtered) == 1

    def test_filters_long(self):
        data = [{"text": "a" * 20000}, {"text": "normal text"}]
        filtered = quality_filter(data, max_length=1000)
        assert len(filtered) == 1


class TestSplitData:
    def test_split_ratio(self):
        data = list(range(100))
        train, test = split_data(data, test_ratio=0.2, seed=42)
        assert len(train) == 80
        assert len(test) == 20

    def test_no_overlap(self):
        data = list(range(100))
        train, test = split_data(data, test_ratio=0.2, seed=42)
        assert set(train).isdisjoint(set(test))


class TestGetDatasetStats:
    def test_stats(self, sample_instruct_data):
        stats = get_dataset_stats(sample_instruct_data)
        assert stats["n_samples"] == 3
        assert stats["avg_length"] > 0
        assert stats["has_labels"] is False


class TestLoadFromHub:
    def test_synthetic_fallback(self):
        data = load_dataset_from_hub("nonexistent_dataset_xyz", sample_size=20)
        assert len(data) == 20
