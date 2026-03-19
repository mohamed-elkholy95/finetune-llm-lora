"""Tests for evaluation metrics."""
import pytest
from src.evaluation import compute_rouge, compute_bleu, generate_comparison_report


class TestComputeRouge:
    def test_exact_match(self):
        preds = ["hello world", "foo bar"]
        refs = ["hello world", "foo bar"]
        result = compute_rouge(preds, refs)
        assert "rouge1" in result
        assert result["rouge1"] >= 0.5

    def test_no_match(self):
        preds = ["abc def"]
        refs = ["xyz qrs"]
        result = compute_rouge(preds, refs)
        assert result["rouge1"] < 0.5

    def test_returns_three_keys(self):
        result = compute_rouge(["test"], ["test"])
        assert "rouge1" in result
        assert "rouge2" in result
        assert "rougeL" in result


class TestComputeBleu:
    def test_exact_match(self):
        score = compute_bleu(
            ["This is a complete sentence with several words"],
            ["This is a complete sentence with several words"],
        )
        assert score > 50.0

    def test_no_overlap(self):
        score = compute_bleu(["abc def"], ["xyz qrs"])
        assert score == 0.0


class TestComparisonReport:
    def test_generates_markdown(self):
        before = {"rouge1": 0.3, "bleu": 15.0}
        after = {"rouge1": 0.5, "bleu": 28.0}
        report = generate_comparison_report(before, after)
        assert "# Fine-Tuning" in report
        assert "| rouge1 |" in report
        assert "+" in report

    def test_empty_metrics(self):
        report = generate_comparison_report({}, {"rouge1": 0.5})
        assert "rouge1" in report
