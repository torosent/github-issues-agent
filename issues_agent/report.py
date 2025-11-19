"""Markdown report generation utilities (Phase 6).

Generates a deterministic Markdown document from scored issues and
aggregated metrics provided by the `Report` dataclass.
"""
from __future__ import annotations

from typing import List, Optional

from .models import Report, ScoredIssue
from .duplicates import DuplicateGroup


class ReportGenerator:
    def __init__(
        self, 
        scored: List[ScoredIssue], 
        repos: List[str],
        duplicate_groups: Optional[List[DuplicateGroup]] = None
    ):
        """Initialize a report generator.

        Args:
            scored: List of scored issues (already classified and ranked).
            repos: List of repository identifiers included in the report.
            duplicate_groups: Optional list of duplicate issue groups to include in report.

        The constructor computes aggregate metrics up front using `Report.compute_metrics`.
        """
        self._scored = list(scored)
        self._repos = list(repos)
        self._duplicate_groups = duplicate_groups if duplicate_groups else []
        self._report = Report.compute_metrics(self._scored, self._repos)

    def generate(self) -> str:
        """Generate the full markdown report.

        Returns:
            A markdown string containing header, summary metrics, category counts,
            top priority issues table, full issues table, and per-category sections.

        The output is deterministic given the scored issues list.
        """
        if not self._scored and not self._duplicate_groups:
            # Minimal document for empty reports
            parts = [self._render_header(), "", "No issues to report."]
            return "\n".join(parts) + "\n"
        
        if not self._scored and self._duplicate_groups:
            # Duplicate-only report (no classification/scoring)
            parts = [
                self._render_header(),
                "",
                "## Potential Duplicate Issues",
                self._render_duplicates(),
            ]
            doc = "\n".join(p for p in parts if p is not None and p != "")
            return doc + "\n"
        parts = [
            self._render_header(),
            "",
            self._render_summary(),
            "",
            self._render_category_counts(),
        ]
        
        # Add duplicate section if duplicates exist
        if self._duplicate_groups:
            parts.extend([
                "",
                "## Potential Duplicate Issues",
                self._render_duplicates(),
            ])
        
        parts.extend([
            "",
            "## Top Priority Issues",
            self._render_top_priority(),
            "",
            "## All Issues",
            self._render_all_issues(),
            "",
            self._render_category_sections(),
        ])
        # Ensure single trailing newline, no extra spaces
        doc = "\n".join(p for p in parts if p is not None and p != "")
        return doc + "\n"

    def _render_header(self) -> str:
        """Render the header section with generation timestamp and repository list."""
        dt = self._report.generated_at.isoformat().replace("+00:00", "Z")
        repos_str = ", ".join(self._repos) if self._repos else "(none)"
        return "\n".join([
            "# GitHub Issues Prioritization Report",
            f"Generated: {dt}",
            f"Repositories: {repos_str}",
        ])

    def _render_summary(self) -> str:
        """Render summary metrics (counts, averages, priority statistics)."""
        issues = self._scored
        total = len(issues)
        repo_count = len(self._repos)
        categories_count = len(self._report.category_counts)
        top_p0 = sum(1 for i in issues if i.priority_level == "P0")
        avg_score = sum(i.score for i in issues) / total if total else 0.0
        return "\n".join(
            [
                f"- Total Issues: {total}",
                f"- Repos Count: {repo_count}",
                f"- Categories Count: {categories_count}",
                f"- Top Priority (P0) Count: {top_p0}",
                f"- Average Score: {avg_score:.3f}\n",
            ]
        )

    def _render_category_counts(self) -> str:
        """Render table listing each category and its issue count."""
        rows = ["Category | Count", "--- | ---"]
        for cat in sorted(self._report.category_counts.keys()):
            rows.append(f"{cat} | {self._report.category_counts[cat]}")
        return "\n".join(rows)

    def _render_top_priority(self) -> str:
        """Render table of highest priority (top-ranked) issues for quick scanning."""
        rows = [
            "Number | Title | Category | Priority | Score | Labels | Rationale | URL",
            "--- | --- | --- | --- | --- | --- | --- | ---",
        ]
        for issue in self._report.top_priority:
            rows.append(self._issue_row(issue))
        return "\n".join(rows)

    def _render_all_issues(self) -> str:
        """Render the full issues table including every scored issue."""
        rows = [
            "Number | Title | Category | Priority | Score | Labels | Rationale | URL",
            "--- | --- | --- | --- | --- | --- | --- | ---",
        ]
        for issue in self._scored:
            rows.append(self._issue_row(issue))
        return "\n".join(rows)

    def _render_category_sections(self) -> str:
        """Render per-category sections with concise issue rationales.

        Each issue rationale is truncated for readability and pipes are escaped for
        markdown table safety.
        """
        parts: List[str] = []
        by_cat = {}
        for issue in self._scored:
            by_cat.setdefault(issue.category, []).append(issue)
        for cat in sorted(by_cat.keys()):
            parts.append(f"## Category: {cat}")
            for issue in by_cat[cat]:
                truncated = self._truncate(issue.rationale.replace("|", "\\|"), 120)
                parts.append(
                    f"- #{issue.number} [{issue.title.replace('|', '\\|')}]({issue.html_url}) "
                    f"({issue.priority_level}, score {issue.score:.3f}) - {truncated}"
                )
        return "\n".join(parts)

    def _render_duplicates(self) -> str:
        """Render duplicate issue groups showing potential duplicates.
        
        Each group shows the similarity percentage and lists all issues
        in that group with their titles and URLs.
        """
        if not self._duplicate_groups:
            return ""
        
        parts: List[str] = []
        parts.append("The following issues may be duplicates:")
        parts.append("")
        
        for i, group in enumerate(self._duplicate_groups, 1):
            similarity_pct = group.max_similarity * 100
            parts.append(f"### Duplicate Group {i} (Similarity: {similarity_pct:.1f}%)")
            parts.append("")
            
            for issue in group.issues:
                title = issue.title.replace("|", "\\|")
                
                parts.append(
                    f"- **#{issue.number}** [{title}]({issue.html_url})"
                )
            
            parts.append("")
        
        return "\n".join(parts)

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        """Truncate a string to a maximum length preserving original content start."""
        if len(text) <= limit:
            return text
        return text[:limit]

    @staticmethod
    def _issue_row(issue: ScoredIssue) -> str:
        """Format a single issue as a markdown table row."""
        title = issue.title.replace("|", "\\|")
        labels = ", ".join(issue.labels).replace("|", "\\|") if issue.labels else ""
        rationale = issue.rationale.replace("|", "\\|").replace("\n", " ")[:200]
        if len(issue.rationale) > 200:
            rationale += "..."
        return (
            f"{issue.number} | {title} | {issue.category} | {issue.priority_level} | "
            f"{issue.score:.3f} | {labels} | {rationale} | {issue.html_url}"
        )

__all__ = ["ReportGenerator"]
