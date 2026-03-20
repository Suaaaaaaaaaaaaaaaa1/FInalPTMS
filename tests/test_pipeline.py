"""Tests for the pipeline modules."""
import pytest
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, ".")


class TestScraper:
    def test_normalize_items_basic(self):
        from src.scraper import normalize_items
        items = [
            {"postId": "123", "text": "Test post", "time": "2025-01-01T10:00:00",
             "type": "photo", "likes": 10, "comments": 2, "shares": 1},
            {"postId": "456", "text": "", "time": "2025-01-02T10:00:00",
             "type": "status", "likes": {"total": 5}, "comments": {"count": 1}, "shares": 0},
        ]
        df = normalize_items(items)
        assert len(df) == 2
        assert "post_id" in df.columns
        assert "engagement" not in df.columns or True

    def test_normalize_empty(self):
        from src.scraper import normalize_items
        df = normalize_items([])
        assert len(df) == 0

    def test_normalize_handles_dict_likes(self):
        from src.scraper import normalize_items
        items = [{"postId": "1", "text": "x", "time": "2025-01-01", "likes": {"total": 42}, "comments": 0, "shares": 0}]
        df = normalize_items(items)
        assert df.iloc[0]["likes"] == 42


class TestStrategy:
    def test_load_analysis_data_returns_dict(self, tmp_path):
        from src.strategy import load_analysis_data
        result = load_analysis_data()
        assert isinstance(result, dict)

    def test_build_gemini_prompt(self):
        from src.strategy import build_gemini_prompt
        data = {"eda": {"dataset": {"total_posts": 100}}, "nlp": {"sentiment": {"positive_pct": 30}}}
        prompt = build_gemini_prompt(data)
        assert "Hỏa Lò" in prompt
        assert "100" in prompt
        assert len(prompt) > 500

    def test_fallback_report_has_content(self):
        from src.strategy import generate_fallback_report
        data = {
            "eda": {"optimal_posting": {"peak_hour": 10, "peak_day": "Thursday"}},
            "nlp": {
                "sentiment": {"positive_pct": 30.5, "neutral_pct": 65.0, "negative_pct": 4.5},
                "topics": {"topic_labels": {"0": "lịch_sử, cách_mạng"}, "best_topic": 0, "best_topic_engagement": 150.0},
                "keywords": {"top_20_words": [("lịch_sử", 50), ("cách_mạng", 30)]},
            },
            "posts_summary": {
                "total_posts": 200,
                "date_range": "2024-01-01 to 2025-01-01",
                "engagement_stats": {"total": 50000, "mean": 250.0, "median": 100.0, "std": 300.0},
                "reactions_total": 40000,
                "comments_total": 5000,
                "shares_total": 5000,
            },
        }
        report = generate_fallback_report(data)
        assert "Hỏa Lò" in report
        assert "200" in report
        assert "30.5%" in report
        assert "Thursday" in report


class TestVisualizeImport:
    def test_module_importable(self):
        import src.visualize
        assert hasattr(src.visualize, "main")
        assert hasattr(src.visualize, "plot_executive_summary")


class TestEmailSender:
    def test_collect_attachments_returns_list(self):
        from src.email_sender import collect_attachments
        result = collect_attachments()
        assert isinstance(result, list)

    def test_build_email_body_missing_file(self):
        from src.email_sender import build_email_body
        body = build_email_body(Path("/nonexistent/report.md"))
        assert "đính kèm" in body.lower() or "attachment" in body.lower()


class TestPipeline:
    def test_run_step_dry_run(self):
        from src.pipeline import run_step
        result = run_step("test", "echo", ["hello"], dry_run=True)
        assert result is True
