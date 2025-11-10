import json
from datetime import datetime
from issues_agent.models import (
    Issue,
    ClassificationResult,
    ScoredIssue,
    Report,
    load_categories,
    DEFAULT_CATEGORIES,
)


def test_issue_from_raw_constructs_issue():
    raw = {
        "number": 42,
        "title": "Sample issue",
        "body": "Details here",
        "labels": [{"name": "bug"}, {"name": "urgent"}],
        "comments": 3,
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-02T00:00:00Z",
        "state": "open",
        "html_url": "https://example.com/issue/42",
        "reactions": {
            "total_count": 7,
            "+1": 3,
            "heart": 2,
            "hooray": 1,
            "rocket": 1,
            "eyes": 0,
        },
    }
    issue = Issue.from_raw(raw, repo="owner/repo")
    assert issue.number == 42
    assert issue.repo == "owner/repo"
    assert issue.title == "Sample issue"
    assert issue.labels == ["bug", "urgent"]
    assert issue.reactions_total == 7
    assert issue.reactions_positive == 3 + 2 + 1 + 1 + 0


def test_load_default_categories():
    cats = load_categories(None)
    assert cats == DEFAULT_CATEGORIES


def test_load_custom_categories_yaml(tmp_path):
    p = tmp_path / "cats.yaml"
    p.write_text("- bug\n- feature\n- custom\n")
    cats = load_categories(str(p))
    assert cats == ["bug", "feature", "custom"]


def test_load_custom_categories_json(tmp_path):
    p = tmp_path / "cats.json"
    json.dump(["bug", "feature", "enhancement"], p.open("w"))
    cats = load_categories(str(p))
    assert cats == ["bug", "feature", "enhancement"]


def test_load_categories_deduplicates_and_preserves_order(tmp_path):
    p = tmp_path / "dups.yaml"
    p.write_text("- a\n- b\n- a\n- c\n- b\n")
    cats = load_categories(str(p))
    assert cats == ["a", "b", "c"]


def test_load_categories_invalid_extension_raises(tmp_path):
    p = tmp_path / "cats.txt"
    p.write_text("bug")
    try:
        load_categories(str(p))
    except ValueError as e:
        assert "Unsupported" in str(e)
    else:
        assert False, "Expected ValueError"


def test_load_categories_empty_list_raises(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("[]")
    try:
        load_categories(str(p))
    except ValueError as e:
        assert "No categories" in str(e)
    else:
        assert False, "Expected ValueError"


def test_report_compute_metrics_counts_and_top_priority_ordering():
    issues = [
        ScoredIssue(
            number=i,
            repo="r/x",
            title=f"Issue {i}",
            category="bug" if i % 2 == 0 else "feature",
            priority_level="P1",
            rationale="test",
            score=float(i % 5),
            labels=["l"],
            html_url=f"http://example.com/{i}",
        )
        for i in range(1, 15)
    ]
    report = Report.compute_metrics(issues, ["r/x"])
    assert report.repos == ["r/x"]
    assert report.category_counts["bug"] == len([i for i in issues if i.category == "bug"])
    assert report.category_counts["feature"] == len([i for i in issues if i.category == "feature"])
    assert len(report.top_priority) == 10
    # Ensure ordering by score desc then number asc
    ordered = sorted(issues, key=lambda i: (-i.score, i.number))[:10]
    assert report.top_priority == ordered
    assert report.generated_at <= datetime.now(report.generated_at.tzinfo)
