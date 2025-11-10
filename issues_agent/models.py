"""Data models for GitHub Issues AI Agent.

Phase 3: Immutable dataclasses for issues, classification, scoring, reports,
and category loading utility supporting default, YAML, and JSON sources.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Optional
import json
import os

import yaml  # Provided via requirements (pyyaml)


DEFAULT_CATEGORIES: List[str] = [
	"bug",
	"feature",
	"documentation",
	"refactor",
	"security",
	"performance",
	"test",
	"other",
]


@dataclass(frozen=True)
class Issue:
	number: int
	repo: str
	title: str
	body: Optional[str]
	labels: List[str]
	comments: int
	created_at: str
	updated_at: str
	state: str
	html_url: str
	reactions_total: int
	reactions_positive: int
	raw: Dict

	"""Immutable representation of a GitHub issue used by the agent.

	Fields mirror a subset of the GitHub Issues API plus reaction summary
	and a retained `raw` dictionary for any future, unmodeled data needs.
	"""

	@classmethod
	def from_raw(cls, raw: Dict, repo: str) -> "Issue":
		"""Construct Issue from raw GitHub API issue dict plus explicit repo.

		The raw dict may come directly from the API (without augmentation) or
		from the GitHubClient augmented result. We defensively extract fields.
		"""
		labels = []
		raw_labels = raw.get("labels", [])
		if isinstance(raw_labels, list):
			for l in raw_labels:
				if isinstance(l, dict):
					name = l.get("name")
					if isinstance(name, str):
						labels.append(name)
				elif isinstance(l, str):
					labels.append(l)
		reactions = raw.get("reactions") or {}
		# GitHub API's reactions summary includes total_count and individual keys
		reactions_total = raw.get("reactions_total") or reactions.get("total_count") or 0
		positive_keys = {"+1", "heart", "hooray", "rocket", "eyes"}
		reactions_positive = raw.get("reactions_positive") or sum(
			int(reactions.get(k, 0)) for k in positive_keys
		)
		return cls(
			number=int(raw.get("number", 0)),
			repo=repo,
			title=str(raw.get("title")),
			body=raw.get("body"),
			labels=labels,
			comments=int(raw.get("comments", 0)),
			created_at=str(raw.get("created_at")),
			updated_at=str(raw.get("updated_at")),
			state=str(raw.get("state")),
			html_url=str(raw.get("html_url")),
			reactions_total=int(reactions_total),
			reactions_positive=int(reactions_positive),
			raw=raw,
		)


@dataclass(frozen=True)
class ClassificationResult:
	number: int
	repo: str
	category: str
	priority_level: str  # P0 | P1 | P2
	rationale: str

	"""Model-derived classification for a single issue.

	Includes assigned category, priority band, and a short rationale string.
	Validation ensures priority_level and category are present and valid.
	"""

	def __post_init__(self) -> None:  # type: ignore[override]
		if self.priority_level not in {"P0", "P1", "P2"}:
			raise ValueError(f"Invalid priority_level: {self.priority_level}")
		if not self.category:
			raise ValueError("category must be non-empty")


@dataclass(frozen=True)
class ScoredIssue:
	number: int
	repo: str
	title: str
	category: str
	priority_level: str  # P0 | P1 | P2
	rationale: str
	score: float
	labels: List[str]
	html_url: str

	"""A classified issue augmented with composite score for ranking."""

	def __post_init__(self) -> None:  # type: ignore[override]
		if self.priority_level not in {"P0", "P1", "P2"}:
			raise ValueError(f"Invalid priority_level: {self.priority_level}")
		if not self.category:
			raise ValueError("category must be non-empty")
		if not isinstance(self.score, (int, float)):
			raise ValueError("score must be numeric")


@dataclass(frozen=True)
class Report:
	generated_at: datetime
	repos: List[str]
	issues: List[ScoredIssue]
	category_counts: Dict[str, int]
	top_priority: List[ScoredIssue]

	"""Aggregate metrics and ranked subsets used for rendering markdown reports."""

	@classmethod
	def compute_metrics(cls, issues: List[ScoredIssue], repos: List[str]) -> "Report":
		"""Compute aggregate counts and top priority slice from scored issues.

		Top priority list is limited to 10 highest scoring issues (deterministic).
		"""
		if not isinstance(issues, list):
			raise ValueError("issues must be a list")
		counts: Dict[str, int] = {}
		for issue in issues:
			counts[issue.category] = counts.get(issue.category, 0) + 1
		# Sort for top priority: score desc, then number asc
		top = sorted(issues, key=lambda i: (-i.score, i.number))[:10]
		return cls(
			generated_at=datetime.now(timezone.utc),
			repos=repos,
			issues=issues,
			category_counts=counts,
			top_priority=top,
		)


def load_categories(path: Optional[str]) -> List[str]:
	"""Load categories from YAML or JSON file or return defaults.

	- If path is None: return DEFAULT_CATEGORIES.
	- Supports .yaml/.yml and .json files containing a list of strings.
	- Deduplicates while preserving first-seen order.
	- Raises ValueError on invalid extension, invalid data, or empty result.
	"""
	if path is None:
		return list(DEFAULT_CATEGORIES)
	if not isinstance(path, str) or not path:
		raise ValueError("path must be a non-empty string or None")
	ext = os.path.splitext(path)[1].lower()
	if ext not in {".yaml", ".yml", ".json"}:
		raise ValueError(f"Unsupported category file extension: {ext}")
	if not os.path.exists(path):
		raise ValueError(f"Category file does not exist: {path}")
	with open(path, "r", encoding="utf-8") as f:
		if ext in {".yaml", ".yml"}:
			data = yaml.safe_load(f)
		else:
			data = json.load(f)
	if not isinstance(data, list):
		raise ValueError("Category file must contain a list")
	seen = set()
	result: List[str] = []
	for item in data:
		if not isinstance(item, str) or not item.strip():
			raise ValueError("Each category must be a non-empty string")
		if item not in seen:
			seen.add(item)
			result.append(item)
	if not result:
		raise ValueError("No categories loaded")
	return result


__all__ = [
	"Issue",
	"ClassificationResult",
	"ScoredIssue",
	"Report",
	"load_categories",
	"DEFAULT_CATEGORIES",
]

