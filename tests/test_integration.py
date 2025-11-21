import pytest
from unittest.mock import MagicMock, patch
from issues_agent.cli import main

@patch("issues_agent.cli.GitHubClient")
@patch("issues_agent.cli.AzureOpenAIClient")
@patch("issues_agent.cli.analyze_issues")
def test_integration_flow(mock_analyze, mock_llm_cls, mock_gh_cls, tmp_path):
    # Setup mocks
    mock_gh = mock_gh_cls.return_value
    mock_gh.fetch_issues.return_value = [
        {"number": 1, "title": "Issue 1", "html_url": "http://github.com/owner/repo/issues/1", "body": "body1", "repo": "owner/repo"},
        {"number": 2, "title": "Issue 2", "html_url": "http://github.com/owner/repo/issues/2", "body": "body2", "repo": "owner/repo"},
    ]
    
    mock_analyze.return_value = {
        "priorities": [
            {"issue_id": 1, "priority": "P0", "reasoning": "Critical bug"},
            {"issue_id": 2, "priority": "P2", "reasoning": "Minor tweak"}
        ],
        "duplicates": [[1, 2]],
        "top_urgent_issues": [1]
    }
    
    output_file = tmp_path / "report.md"
    
    # Run CLI
    argv = ["--repos", "owner/repo", "--output", str(output_file), "--limit", "10"]
    exit_code = main(argv)
    
    # Assertions
    assert exit_code == 0
    assert output_file.exists()
    
    content = output_file.read_text()
    assert "# GitHub Issues Report" in content
    assert "Top 20 Urgent Issues" in content
    assert "All Issues (Sorted by Priority)" in content
    assert "Issue 1" in content
    assert "P0" in content
    assert "Critical bug" in content
    assert "Potential Duplicate Issues" in content
