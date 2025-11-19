import re
from issues_agent.models import ScoredIssue, Issue
from issues_agent.report import ReportGenerator
from issues_agent.duplicates import DuplicateGroup


def make_issue(number: int, score: float, category: str, priority: str, title: str = None, rationale: str = None):
    return ScoredIssue(
        number=number,
        repo="example/repo",
        title=title or f"Issue {number}",
        category=category,
        priority_level=priority,
        rationale=rationale or f"Rationale for {number}",
        score=score,
        labels=["label1", "label2"],
        html_url=f"https://example.com/{number}",
    )


def test_empty_report_generates_minimal_markdown():
    gen = ReportGenerator([], ["example/repo"])
    md = gen.generate()
    assert "# GitHub Issues Prioritization Report" in md
    assert "No issues to report." in md
    # Should not include tables
    assert "Category | Count" not in md
    assert md.endswith("\n")


def test_header_and_summary_sections_present():
    issues = [
        make_issue(1, 10.0, "bug", "P0"),
        make_issue(2, 5.0, "feature", "P1"),
    ]
    gen = ReportGenerator(issues, ["example/repo", "another/repo"])
    md = gen.generate()
    assert "# GitHub Issues Prioritization Report" in md
    assert "- Total Issues: 2" in md
    assert "- Repos Count: 2" in md
    assert "- Categories Count: 2" in md
    assert "- Top Priority (P0) Count: 1" in md
    assert "- Average Score: 7.500" in md


def test_category_counts_table_accuracy():
    issues = [
        make_issue(1, 3.0, "bug", "P1"),
        make_issue(2, 4.0, "bug", "P2"),
        make_issue(3, 5.0, "feature", "P2"),
    ]
    gen = ReportGenerator(issues, [])
    md = gen.generate()
    # Extract category counts table section
    table_match = re.search(r"Category \| Count\n--- \| ---\n([^-#].+?)\nNumber \| Title", md, re.DOTALL)
    assert table_match, "Category counts table not found"
    rows = [r.strip() for r in table_match.group(1).strip().split("\n")]
    # Expect alphabetical ordering: bug then feature
    assert rows[0] == "bug | 2"
    assert rows[1] == "feature | 1"


def test_top_priority_table_sorting():
    issues = [
        make_issue(10, 1.0, "bug", "P0"),
        make_issue(2, 9.0, "feature", "P0"),
        make_issue(3, 9.0, "feature", "P0"),  # same score, higher number should come later
        make_issue(4, 8.5, "bug", "P1"),
    ]
    gen = ReportGenerator(issues, [])
    md = gen.generate()
    # Find top priority table lines (after header lines)
    lines = md.splitlines()
    start = lines.index("Number | Title | Category | Priority | Score | Labels | Rationale | URL")
    # top priority is first such table (after category counts table)
    top_lines = []
    for i in range(start + 2, len(lines)):
        line = lines[i]
        if line.startswith("Number | Title | Category"):
            break
        if line.startswith("##"):
            break
        top_lines.append(line)
    # Expect ordering by -score then number asc: scores 9.0 (#2 then #3), 8.5, 1.0
    nums = [int(l.split(" | ")[0]) for l in top_lines]
    assert nums[:4] == [2, 3, 4, 10]


def test_all_issues_table_scores_rounded():
    issues = [
        make_issue(1, 1.23456, "bug", "P0"),
        make_issue(2, 9.2, "feature", "P1"),
    ]
    gen = ReportGenerator(issues, [])
    md = gen.generate()
    # Locate all issues table (second table occurrence)
    occurrences = [m.start() for m in re.finditer(r"Number \| Title \| Category \| Priority \| Score \| Labels \| Rationale \| URL", md)]
    assert len(occurrences) >= 2, "Expected at least two tables"
    # Extract second table block
    after = md[occurrences[1]:]
    assert "1.235" in after  # rounded
    assert "9.200" in after  # formatted to 3 decimals


def test_category_sections_rationale_truncation():
    long_rationale = "x" * 300
    issues = [make_issue(1, 5.0, "bug", "P0", rationale=long_rationale)]
    gen = ReportGenerator(issues, [])
    md = gen.generate()
    # Find category section line for issue
    line = next(l for l in md.splitlines() if l.startswith("- #1 "))
    assert len(line) < 200  # truncated
    assert "x" * 150 not in line  # ensure not full rationale


def test_report_with_duplicates_section():
    """Test that duplicate groups are rendered in the report."""
    issues = [make_issue(i, 5.0, "bug", "P1") for i in range(1, 5)]
    
    # Create duplicate groups
    dup_issues = [
        Issue.from_raw({
            "number": 1,
            "title": "Memory leak in parser",
            "body": "Parser has memory issues",
            "labels": [],
            "comments": 0,
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "state": "open",
            "html_url": "https://github.com/test/repo/issues/1",
            "reactions": {"total_count": 0}
        }, repo="test/repo"),
        Issue.from_raw({
            "number": 2,
            "title": "Parser memory leak fix needed",
            "body": "Memory leak in the parser",
            "labels": [],
            "comments": 0,
            "created_at": "2025-01-02T00:00:00Z",
            "updated_at": "2025-01-02T00:00:00Z",
            "state": "open",
            "html_url": "https://github.com/test/repo/issues/2",
            "reactions": {"total_count": 0}
        }, repo="test/repo"),
    ]
    
    duplicate_groups = [DuplicateGroup(issues=dup_issues, max_similarity=0.85)]
    
    gen = ReportGenerator(issues, ["test/repo"], duplicate_groups=duplicate_groups)
    md = gen.generate()
    
    assert "## Potential Duplicate Issues" in md
    assert "Memory leak in parser" in md
    assert "Parser memory leak fix needed" in md
    assert "85.0%" in md  # Similarity score as percentage


def test_report_without_duplicates_no_section():
    """Test that no duplicate section is shown when no duplicates exist."""
    issues = [make_issue(1, 5.0, "bug", "P1")]
    gen = ReportGenerator(issues, ["test/repo"], duplicate_groups=[])
    md = gen.generate()
    
    assert "## Potential Duplicate Issues" not in md


def test_report_duplicates_none_parameter():
    """Test that None duplicate_groups works (backward compatibility)."""
    issues = [make_issue(1, 5.0, "bug", "P1")]
    gen = ReportGenerator(issues, ["test/repo"], duplicate_groups=None)
    md = gen.generate()
    
    assert "## Potential Duplicate Issues" not in md
