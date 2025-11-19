import os
import io
import runpy
import pytest

from issues_agent.cli import main
from issues_agent.models import Issue, ClassificationResult, ScoredIssue

# Helpers / fakes

def make_issue(number: int) -> Issue:
    return Issue(
        number=number,
        repo="owner/name",
        title=f"Issue {number}",
        body="Body",
        labels=["bug"],
        comments=2,
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
        state="open",
        html_url=f"https://example.com/{number}",
        reactions_total=3,
        reactions_positive=2,
        raw={"number": number},
    )


def fake_classification(issue: Issue) -> ClassificationResult:
    return ClassificationResult(
        number=issue.number,
        repo=issue.repo,
        category="bug",
        priority_level="P1",
        rationale="rationale",
    )


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "endpoint")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "version")


def test_cli_missing_repos_argument_errors(capsys):
    with pytest.raises(SystemExit):
        main([])  # no args
    captured = capsys.readouterr()
    assert "--repos argument is required" in captured.err or captured.out


def test_cli_empty_issues_exits_gracefully(monkeypatch, tmp_path, capsys):
    def fake_fetch(self, repos, limit):  # noqa: D401
        return []
    monkeypatch.setattr("issues_agent.github_client.GitHubClient.fetch_issues", fake_fetch)
    exit_code = main(["--repos", "owner/name", "--output", str(tmp_path / "out.md")])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "No issues found" in captured.out
    assert not (tmp_path / "out.md").exists()


def test_cli_end_to_end_with_mocks(monkeypatch, tmp_path):
    issues = [
        {
            "number": 1,
            "title": "Issue 1",
            "body": "b",
            "labels": ["bug"],
            "comments": 1,
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "state": "open",
            "html_url": "https://example.com/1",
            "repo": "owner/name",
            "reactions_total": 1,
            "reactions_positive": 1,
            "raw": {"number": 1},
        }
    ]

    def fake_fetch(self, repos, limit):
        return issues

    def fake_classify(self, issues_list, categories):
        return [fake_classification(issues_list[0])]

    def fake_score_and_rank(issue_list, classifications):
        return [
            ScoredIssue(
                number=1,
                repo="owner/name",
                title="Issue 1",
                category="bug",
                priority_level="P1",
                rationale="rationale",
                score=0.5,
                labels=["bug"],
                html_url="https://example.com/1",
            )
        ]

    monkeypatch.setattr("issues_agent.github_client.GitHubClient.fetch_issues", fake_fetch)
    monkeypatch.setattr("issues_agent.classifier.AzureOpenAIClassifier.classify", fake_classify)
    monkeypatch.setattr("issues_agent.scoring.score_and_rank", fake_score_and_rank)

    out_path = tmp_path / "report.md"
    exit_code = main(["--repos", "owner/name", "--output", str(out_path)])
    assert exit_code == 0
    assert out_path.exists()
    contents = out_path.read_text(encoding="utf-8")
    assert "# GitHub Issues Prioritization Report" in contents
    assert "Issue 1" in contents


def test_cli_dry_run_prints_preview(monkeypatch, capsys):
    issues = [
        {
            "number": 1,
            "title": "Issue 1",
            "body": "b",
            "labels": ["bug"],
            "comments": 1,
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "state": "open",
            "html_url": "https://example.com/1",
            "repo": "owner/name",
            "reactions_total": 1,
            "reactions_positive": 1,
            "raw": {"number": 1},
        }
    ]

    def fake_fetch(self, repos, limit):
        return issues

    def fake_classify(self, issues_list, categories):
        return [fake_classification(issues_list[0])]

    def fake_score_and_rank(issue_list, classifications):
        return [
            ScoredIssue(
                number=1,
                repo="owner/name",
                title="Issue 1",
                category="bug",
                priority_level="P1",
                rationale="rationale",
                score=0.5,
                labels=["bug"],
                html_url="https://example.com/1",
            )
        ]

    monkeypatch.setattr("issues_agent.github_client.GitHubClient.fetch_issues", fake_fetch)
    monkeypatch.setattr("issues_agent.classifier.AzureOpenAIClassifier.classify", fake_classify)
    monkeypatch.setattr("issues_agent.scoring.score_and_rank", fake_score_and_rank)

    exit_code = main(["--repos", "owner/name", "--dry-run"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Dry run: showing preview" in captured.out
    assert "Issue 1" in captured.out


def test_cli_check_duplicates_flag(monkeypatch, tmp_path):
    """Test that --check-duplicates flag triggers duplicate detection."""
    issues = [
        {
            "number": 1,
            "title": "Memory leak in parser",
            "body": "Parser has memory leak",
            "labels": ["bug"],
            "comments": 1,
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "state": "open",
            "html_url": "https://example.com/1",
            "repo": "test/repo",
            "reactions_total": 1,
            "reactions_positive": 1,
            "raw": {"number": 1},
        },
        {
            "number": 2,
            "title": "Parser memory leak fix needed",
            "body": "Memory leak in parser",
            "labels": ["bug"],
            "comments": 1,
            "created_at": "2025-01-02T00:00:00Z",
            "updated_at": "2025-01-02T00:00:00Z",
            "state": "open",
            "html_url": "https://example.com/2",
            "repo": "test/repo",
            "reactions_total": 1,
            "reactions_positive": 1,
            "raw": {"number": 2},
        },
    ]

    def fake_fetch(self, repos, limit):
        return issues

    def fake_classify(self, issues_list, categories):
        return [fake_classification(Issue.from_raw(i, i["repo"])) for i in issues]

    def fake_score_and_rank(issue_list, classifications):
        return [
            ScoredIssue(
                number=i["number"],
                repo=i["repo"],
                title=i["title"],
                category="bug",
                priority_level="P1",
                rationale="rationale",
                score=0.5,
                labels=["bug"],
                html_url=i["html_url"],
            )
            for i in issues
        ]

    monkeypatch.setattr("issues_agent.github_client.GitHubClient.fetch_issues", fake_fetch)
    monkeypatch.setattr("issues_agent.classifier.AzureOpenAIClassifier.classify", fake_classify)
    monkeypatch.setattr("issues_agent.scoring.score_and_rank", fake_score_and_rank)

    out_path = tmp_path / "report.md"
    exit_code = main(["--repos", "test/repo", "--output", str(out_path), "--check-duplicates", "--duplicate-threshold", "0.4"])
    assert exit_code == 0
    assert out_path.exists()
    
    contents = out_path.read_text(encoding="utf-8")
    assert "# GitHub Issues Prioritization Report" in contents
    assert "## Potential Duplicate Issues" in contents
    assert "Memory leak in parser" in contents
    assert "Parser memory leak fix needed" in contents


def test_cli_skip_classifier_with_duplicates(monkeypatch, tmp_path):
    """Test that --skip-classifier with --check-duplicates generates duplicate-only report."""
    issues = [
        {
            "number": 1,
            "title": "Memory leak in parser",
            "body": "Parser has memory leak",
            "labels": ["bug"],
            "comments": 1,
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "state": "open",
            "html_url": "https://example.com/1",
            "repo": "test/repo",
            "reactions_total": 1,
            "reactions_positive": 1,
            "raw": {"number": 1},
        },
        {
            "number": 2,
            "title": "Parser memory leak fix needed",
            "body": "Memory leak in parser",
            "labels": ["bug"],
            "comments": 1,
            "created_at": "2025-01-02T00:00:00Z",
            "updated_at": "2025-01-02T00:00:00Z",
            "state": "open",
            "html_url": "https://example.com/2",
            "repo": "test/repo",
            "reactions_total": 1,
            "reactions_positive": 1,
            "raw": {"number": 2},
        },
    ]

    def fake_fetch(self, repos, limit):
        return issues

    monkeypatch.setattr("issues_agent.github_client.GitHubClient.fetch_issues", fake_fetch)

    out_path = tmp_path / "report.md"
    exit_code = main([
        "--repos", "test/repo", 
        "--output", str(out_path), 
        "--check-duplicates",
        "--skip-classifier",
        "--duplicate-threshold", "0.4"
    ])
    assert exit_code == 0
    assert out_path.exists()
    
    contents = out_path.read_text(encoding="utf-8")
    assert "# GitHub Issues Prioritization Report" in contents
    assert "## Potential Duplicate Issues" in contents
    assert "Memory leak in parser" in contents
    # Should not have classification sections
    assert "## Top Priority Issues" not in contents
    assert "## All Issues" not in contents
