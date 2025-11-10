"""Scoring algorithm for GitHub Issues AI Agent (Phase 5).

Computes composite priority scores using classification results and issue
metadata. Produces a sorted list of ScoredIssue instances for downstream
reporting (Phase 6+).
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
import math
from typing import List, Dict, Optional

from .models import Issue, ClassificationResult, ScoredIssue

logger = logging.getLogger(__name__)

_PRIORITY_BASE = {"P0": 3, "P1": 2, "P2": 1}


def priority_weight(level: str) -> int:
    """Return base weight for priority level (P0=3, P1=2, P2=1).
    Defaults to 1 for unknown values (defensive)."""
    return _PRIORITY_BASE.get(level, 1)


def _parse_datetime(value: str) -> Optional[datetime]:
    """Parse an ISO 8601 timestamp used by GitHub (ending with 'Z').

    Returns a timezone-aware datetime or None on failure.
    """
    if not isinstance(value, str):
        return None
    try:
        # GitHub timestamps end with 'Z' for UTC; replace for fromisoformat
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except Exception:
        return None


def compute_recency(issue: Issue, now: datetime) -> float:
    """Compute recency component: 1 / (1 + age_days/30) in [0,1].

    On parse failure returns neutral 0.5 and logs a warning.
    """
    created_dt = _parse_datetime(issue.created_at)
    if created_dt is None:
        logger.warning("Failed to parse created_at for issue %s", issue.number)
        return 0.5
    # Ensure timezone-aware comparison
    if created_dt.tzinfo is None:
        created_dt = created_dt.replace(tzinfo=timezone.utc)
    age_days = (now - created_dt).days
    if age_days < 0:
        age_days = 0  # future timestamps treated as fresh
    return 1.0 / (1.0 + (age_days / 30.0))


def compute_comments(issue: Issue) -> float:
    """Raw comments component prior to normalization: log(1 + comments)."""
    return math.log(1 + int(issue.comments))


def compute_reactions(issue: Issue) -> float:
    """Raw positive reactions component prior to normalization: log(1 + reactions_positive)."""
    return math.log(1 + int(issue.reactions_positive))


def _normalize(values: Dict[int, float]) -> Dict[int, float]:
    """Normalize a mapping of raw values to [0,1] by dividing by max.

    Returns zeroes if all values are non-positive.
    """
    max_val = max(values.values()) if values else 0.0
    if max_val <= 0.0:
        return {k: 0.0 for k in values}
    return {k: (v / max_val) for k, v in values.items()}


def score_and_rank(
    issues: List[Issue],
    classifications: List[ClassificationResult],
    *,
    now: Optional[datetime] = None,
) -> List[ScoredIssue]:
    """Score issues and return list of ScoredIssue sorted by (-score, number).

    Validates matching issue numbers; raises ValueError on mismatch.
    Allows injection of `now` for deterministic testing.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    # Build maps for fast lookup
    issue_map: Dict[int, Issue] = {i.number: i for i in issues}
    class_map: Dict[int, ClassificationResult] = {c.number: c for c in classifications}
    if len(issue_map) != len(issues):  # duplicate numbers
        raise ValueError("Duplicate issue numbers in issues list")
    if len(class_map) != len(classifications):
        raise ValueError("Duplicate issue numbers in classifications list")
    if set(issue_map.keys()) != set(class_map.keys()):
        raise ValueError("Issue numbers mismatch between issues and classifications")

    # Precompute raw components for batch normalization
    raw_comments: Dict[int, float] = {}
    raw_reactions: Dict[int, float] = {}
    recency_components: Dict[int, float] = {}
    severity_components: Dict[int, float] = {}

    for num, issue in issue_map.items():
        cls = class_map[num]
        severity_components[num] = priority_weight(cls.priority_level) / 3.0
        recency_components[num] = compute_recency(issue, now)
        raw_comments[num] = compute_comments(issue)
        raw_reactions[num] = compute_reactions(issue)

    comments_norm = _normalize(raw_comments)
    reactions_norm = _normalize(raw_reactions)

    scored: List[ScoredIssue] = []
    for num, issue in issue_map.items():
        cls = class_map[num]
        severity = severity_components[num]
        recency = recency_components[num]
        comments_c = comments_norm[num]
        reactions_c = reactions_norm[num]
        score = (
            0.4 * severity
            + 0.2 * recency
            + 0.2 * comments_c
            + 0.2 * reactions_c
        )
        scored.append(
            ScoredIssue(
                number=issue.number,
                repo=issue.repo,
                title=issue.title,
                category=cls.category,
                priority_level=cls.priority_level,
                rationale=cls.rationale,
                score=score,
                labels=issue.labels,
                html_url=issue.html_url,
            )
        )
    if scored:
        scores = [s.score for s in scored]
        min_score = min(scores)
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        logger.info(
            "Scoring summary count=%d min=%.4f max=%.4f avg=%.4f", len(scored), min_score, max_score, avg_score
        )
    # Deterministic ordering
    scored.sort(key=lambda s: (-s.score, s.number))
    return scored


__all__ = [
    "priority_weight",
    "compute_recency",
    "compute_comments",
    "compute_reactions",
    "score_and_rank",
]
