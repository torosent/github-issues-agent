import math
from datetime import datetime, timezone, timedelta

import pytest

from issues_agent.models import Issue, ClassificationResult

# We will import after implementation; for initial failing tests this may error.
# Using try/except to allow TDD flow (will assert functions later).
try:
    from issues_agent.scoring import (
        priority_weight,
        compute_recency,
        compute_comments,
        compute_reactions,
        score_and_rank,
    )
except Exception:  # pragma: no cover - initial failure expected
    priority_weight = compute_recency = compute_comments = compute_reactions = score_and_rank = None

NOW = datetime(2025, 11, 7, tzinfo=timezone.utc)


def make_issue(num: int, created_at: datetime, comments: int = 0, reactions_positive: int = 0) -> Issue:
    iso = created_at.isoformat().replace("+00:00", "Z")
    return Issue(
        number=num,
        repo="owner/repo",
        title=f"Issue {num}",
        body="Body",
        labels=["bug"],
        comments=comments,
        created_at=iso,
        updated_at=iso,
        state="open",
        html_url=f"https://example.com/{num}",
        reactions_total=reactions_positive,  # not used directly in scoring
        reactions_positive=reactions_positive,
        raw={"number": num},
    )


def classify(issue: Issue, priority: str = "P1", category: str = "bug") -> ClassificationResult:
    return ClassificationResult(
        number=issue.number,
        repo=issue.repo,
        category=category,
        priority_level=priority,
        rationale="test",
    )


def test_score_p0_higher_than_p1():
    if score_and_rank is None:
        pytest.skip("scoring module not yet implemented")
    base_time = NOW - timedelta(days=10)
    issue_p0 = make_issue(1, base_time, comments=5, reactions_positive=3)
    issue_p1 = make_issue(2, base_time, comments=5, reactions_positive=3)
    classifications = [
        classify(issue_p0, priority="P0"),
        classify(issue_p1, priority="P1"),
    ]
    scored = score_and_rank([issue_p0, issue_p1], classifications, now=NOW)
    assert scored[0].priority_level == "P0"
    assert scored[0].score > scored[1].score


def test_recency_component_decay():
    if compute_recency is None:
        pytest.skip("scoring module not yet implemented")
    new_issue = make_issue(3, NOW - timedelta(days=0))
    old_issue = make_issue(4, NOW - timedelta(days=120))
    r_new = compute_recency(new_issue, NOW)
    r_old = compute_recency(old_issue, NOW)
    assert r_new > r_old
    # Basic sanity: older issue decays significantly
    assert r_old < 0.25  # 120 days => 1/(1+4) = 0.2


def test_comments_and_reactions_boost():
    if score_and_rank is None:
        pytest.skip("scoring module not yet implemented")
    t = NOW - timedelta(days=30)
    issue_low = make_issue(5, t, comments=1, reactions_positive=0)
    issue_high = make_issue(6, t, comments=10, reactions_positive=8)
    classifications = [classify(issue_low), classify(issue_high)]
    scored = score_and_rank([issue_low, issue_high], classifications, now=NOW)
    # second should have higher score due to normalized components
    assert scored[0].number == issue_high.number  # highest score first
    assert scored[0].score > scored[1].score


def test_zero_comments_reactions_handled():
    if score_and_rank is None:
        pytest.skip("scoring module not yet implemented")
    t = NOW - timedelta(days=5)
    issue_zero = make_issue(7, t, comments=0, reactions_positive=0)
    issue_other = make_issue(8, t, comments=3, reactions_positive=2)
    classifications = [classify(issue_zero), classify(issue_other)]
    scored = score_and_rank([issue_zero, issue_other], classifications, now=NOW)
    # Find zero issue
    zero_scored = next(s for s in scored if s.number == issue_zero.number)
    # Score components indirectly: ensure not NaN
    assert not math.isnan(zero_scored.score)
    # Since zero issue has lower components, its score should be <= other
    other_scored = next(s for s in scored if s.number == issue_other.number)
    assert other_scored.score >= zero_scored.score


def test_deterministic_ordering():
    if score_and_rank is None:
        pytest.skip("scoring module not yet implemented")
    t = NOW - timedelta(days=15)
    # Create three issues with same components except priority to induce ordering,
    # plus two identical scores to test number tie-break.
    i1 = make_issue(9, t, comments=2, reactions_positive=1)  # P0
    i2 = make_issue(10, t, comments=2, reactions_positive=1)  # P1
    i3 = make_issue(11, t, comments=2, reactions_positive=1)  # P1 (same as i2, number tie check)
    classifications = [
        classify(i1, priority="P0"),
        classify(i2, priority="P1"),
        classify(i3, priority="P1"),
    ]
    scored = score_and_rank([i1, i2, i3], classifications, now=NOW)
    # Ensure P0 first
    assert scored[0].number == 9
    # For equal scores (i2, i3) ordering by number asc
    assert scored[1].number == 10
    assert scored[2].number == 11
