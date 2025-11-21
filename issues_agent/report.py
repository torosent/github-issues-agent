"""Markdown report generation utilities.

Generates a simple Markdown report from analyzed issues.
"""
from typing import List, Dict, Any
from datetime import datetime

class ReportGenerator:
    def __init__(
        self, 
        issues: List[Dict[str, Any]], 
        analysis_result: Dict[str, Any],
        repos: List[str]
    ):
        """Initialize a report generator.

        Args:
            issues: List of raw issue dictionaries (must contain number, title, html_url).
            analysis_result: Dictionary containing 'priorities' and 'duplicates' from analysis.
            repos: List of repository identifiers included in the report.
        """
        self.issues = {i['number']: i for i in issues}
        self.analysis_result = analysis_result
        self.repos = repos

    def generate(self) -> str:
        """Generate the full markdown report."""
        parts = [
            self._render_header(),
            "",
            self._render_summary(),
            "",
            "## Top 20 Urgent Issues",
            self._render_top_urgent_issues(limit=20),
            "",
            "## All Issues (Sorted by Priority)",
            self._render_priorities_table(),
            "",
            "## Potential Duplicate Issues",
            self._render_duplicates(),
        ]
        return "\n".join(parts) + "\n"

    def _render_header(self) -> str:
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        repos_str = ", ".join(self.repos)
        return f"# GitHub Issues Report\nGenerated: {dt}\nRepositories: {repos_str}"

    def _render_summary(self) -> str:
        total_issues = len(self.issues)
        priorities = self.analysis_result.get("priorities", [])
        p0_count = sum(1 for p in priorities if p.get("priority") == "P0")
        p1_count = sum(1 for p in priorities if p.get("priority") == "P1")
        p2_count = sum(1 for p in priorities if p.get("priority") == "P2")
        duplicates_count = len(self.analysis_result.get("duplicates", []))

        return (
            f"- Total Issues Analyzed: {total_issues}\n"
            f"- P0 Issues: {p0_count}\n"
            f"- P1 Issues: {p1_count}\n"
            f"- P2 Issues: {p2_count}\n"
            f"- Duplicate Groups: {duplicates_count}"
        )

    def _render_issues_table(self, priorities_list: List[Dict[str, Any]]) -> str:
        rows = [
            "| ID | Title | Priority | Reasoning |",
            "| --- | --- | --- | --- |"
        ]
        
        for p in priorities_list:
            issue_id = p.get("issue_id")
            issue = self.issues.get(issue_id)
            if not issue:
                continue
            
            title = issue.get("title", "").replace("|", "\\|")
            url = issue.get("html_url", "")
            priority = p.get("priority", "Unknown")
            reasoning = p.get("reasoning", "").replace("|", "\\|")
            
            rows.append(f"| [#{issue_id}]({url}) | {title} | {priority} | {reasoning} |")
            
        if len(rows) == 2:
            return "No issues found."
            
        return "\n".join(rows)

    def _get_sorted_priorities(self) -> List[Dict[str, Any]]:
        priorities = self.analysis_result.get("priorities", [])
        priority_order = {"P0": 0, "P1": 1, "P2": 2}
        return sorted(
            priorities, 
            key=lambda x: priority_order.get(x.get("priority", "P2"), 3)
        )

    def _render_top_urgent_issues(self, limit: int = 20) -> str:
        # Use LLM provided top urgent issues if available
        top_ids = self.analysis_result.get("top_urgent_issues", [])
        
        if top_ids:
            priorities_map = {p["issue_id"]: p for p in self.analysis_result.get("priorities", [])}
            top_issues_data = []
            for issue_id in top_ids[:limit]:
                if issue_id in priorities_map:
                    top_issues_data.append(priorities_map[issue_id])
            return self._render_issues_table(top_issues_data)
        
        # Fallback to sorting if LLM didn't return it
        sorted_priorities = self._get_sorted_priorities()
        top_n = sorted_priorities[:limit]
        return self._render_issues_table(top_n)

    def _render_priorities_table(self) -> str:
        sorted_priorities = self._get_sorted_priorities()
        return self._render_issues_table(sorted_priorities)

    def _render_duplicates(self) -> str:
        duplicates = self.analysis_result.get("duplicates", [])
        if not duplicates:
            return "No duplicates found."

        lines = []
        for group in duplicates:
            group_links = []
            for issue_id in group:
                issue = self.issues.get(issue_id)
                if issue:
                    title = issue.get("title", "").replace("|", "\\|")
                    url = issue.get("html_url", "")
                    group_links.append(f"[#{issue_id}]({url}) - {title}")
                else:
                    group_links.append(f"#{issue_id}")
            
            lines.append("- " + ", ".join(group_links))
            
        return "\n".join(lines)
