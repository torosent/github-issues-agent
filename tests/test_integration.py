import os
from typing import List

from issues_agent.cli import main
from issues_agent import cli as cli_mod
from issues_agent.models import ClassificationResult


def _fake_issue(number: int, repo: str):
    return {
        "number": number,
        "title": f"Issue {number}",
        "body": f"Body for issue {number}",
        "labels": [{"name": "bug" if number % 2 == 0 else "feature"}],
        "comments": 3 * number,
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-02T00:00:00Z",
        "state": "open",
        "html_url": f"https://example.com/{repo}/issues/{number}",
        "repo": repo,
        "reactions_total": number,
        "reactions_positive": number // 2,
    }


def test_end_to_end_report_generation(monkeypatch, tmp_path):
    # Monkeypatch GitHubClient.fetch_issues
    from issues_agent import github_client
    from issues_agent import config as cfg

    def fake_load_config():  # noqa: D401
        return cfg.Config(
            github_token="gh_token",
            azure_openai_endpoint="endpoint",
            azure_openai_api_key="key",
            azure_openai_deployment="deploy",
            azure_openai_api_version="2025-01-01",
        )

    monkeypatch.setattr(cfg, "load_config", fake_load_config)
    monkeypatch.setattr(cli_mod, "load_config", fake_load_config)

    def fake_fetch(self, repos: List[str], limit: int = 300):  # noqa: D401
        data = []
        for r in repos:
            data.append(_fake_issue(1, r))
            data.append(_fake_issue(2, r))
        return data

    monkeypatch.setattr(github_client.GitHubClient, "fetch_issues", fake_fetch)

    # Monkeypatch classifier classify
    from issues_agent import classifier
    from issues_agent.models import Issue

    def fake_classify(self, issues: List[Issue], categories: List[str]):  # noqa: D401
        results = []
        for iss in issues:
            # Alternate categories from provided list
            cat = categories[0] if iss.number % 2 == 0 else categories[1]
            results.append(
                ClassificationResult(
                    number=iss.number,
                    repo=iss.repo,
                    category=cat,
                    priority_level="P1" if iss.number % 2 == 0 else "P2",
                    rationale=f"Rationale for {iss.number}",
                )
            )
        return results

    monkeypatch.setattr(classifier.AzureOpenAIClassifier, "classify", fake_classify)

    # Monkeypatch categories loader to simple list
    from issues_agent import models

    def fake_load_categories(path):  # noqa: D401
        return ["bug", "feature", "other"]

    monkeypatch.setattr(models, "load_categories", fake_load_categories)

    output_path = tmp_path / "report.md"
    code = main([
        "--repos",
        "owner/repo1",
        "--output",
        str(output_path),
        "--limit",
        "10",
        "--batch-size",
        "2",
    ])
    assert code == 0
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "# GitHub Issues Prioritization Report" in content
    assert "## Category:" in content  # category sections present


def test_dry_run_preview(monkeypatch, capsys, tmp_path):
    from issues_agent import github_client, classifier, models
    from issues_agent.models import Issue
    from issues_agent import config as cfg

    def fake_load_config():  # noqa: D401
        return cfg.Config(
            github_token="gh_token",
            azure_openai_endpoint="endpoint",
            azure_openai_api_key="key",
            azure_openai_deployment="deploy",
            azure_openai_api_version="2025-01-01",
        )

    monkeypatch.setattr(cfg, "load_config", fake_load_config)
    monkeypatch.setattr(cli_mod, "load_config", fake_load_config)

    def fake_fetch(self, repos: List[str], limit: int = 300):  # noqa: D401
        return [_fake_issue(1, repos[0])]

    monkeypatch.setattr(github_client.GitHubClient, "fetch_issues", fake_fetch)

    def fake_classify(self, issues: List[Issue], categories: List[str]):  # noqa: D401
        return [
            ClassificationResult(
                number=issues[0].number,
                repo=issues[0].repo,
                category=categories[0],
                priority_level="P2",
                rationale="Rationale single",
            )
        ]

    monkeypatch.setattr(classifier.AzureOpenAIClassifier, "classify", fake_classify)

    def fake_load_categories(path):  # noqa: D401
        return ["bug", "other"]

    monkeypatch.setattr(models, "load_categories", fake_load_categories)

    output_path = tmp_path / "report_dry.md"
    code = main([
        "--repos",
        "owner/repo2",
        "--output",
        str(output_path),
        "--limit",
        "5",
        "--batch-size",
        "2",
        "--dry-run",
    ])
    assert code == 0
    assert not output_path.exists()  # dry run should not write file
    captured = capsys.readouterr()
    assert "# GitHub Issues Prioritization Report" in captured.out
    assert "Dry run complete" in captured.out
