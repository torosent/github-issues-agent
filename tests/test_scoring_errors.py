import pytest

from issues_agent.scoring import score_and_rank
from issues_agent.models import Issue, ClassificationResult


def test_score_and_rank_mismatch_numbers_raises():
    issue = Issue(
        number=1,
        repo="owner/repo",
        title="Title",
        body="Body",
        labels=[],
        comments=0,
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
        state="open",
        html_url="http://example.com",
        reactions_total=0,
        reactions_positive=0,
        raw={},
    )
    classification = ClassificationResult(
        number=2,
        repo="owner/repo",
        category="bug",
        priority_level="P2",
        rationale="Rationale",
    )
    with pytest.raises(ValueError):
        score_and_rank([issue], [classification])
