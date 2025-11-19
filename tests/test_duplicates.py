"""Tests for duplicate issue detection.

Phase 1: Test-driven development for DuplicateDetector class.
"""
from issues_agent.models import Issue
from issues_agent.duplicates import DuplicateDetector, DuplicateGroup


def test_duplicate_detector_finds_exact_duplicates():
    """Test that exact or near-exact duplicate issues are detected."""
    issues = [
        Issue.from_raw({
            "number": 1,
            "title": "Fix memory leak in parser",
            "body": "The parser has a memory leak when processing large files",
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
            "title": "Memory leak in parser needs fixing",
            "body": "Parser leaks memory when it processes large files",
            "labels": [],
            "comments": 0,
            "created_at": "2025-01-02T00:00:00Z",
            "updated_at": "2025-01-02T00:00:00Z",
            "state": "open",
            "html_url": "https://github.com/test/repo/issues/2",
            "reactions": {"total_count": 0}
        }, repo="test/repo"),
        Issue.from_raw({
            "number": 3,
            "title": "Add dark mode support",
            "body": "Users want dark mode for the UI",
            "labels": [],
            "comments": 0,
            "created_at": "2025-01-03T00:00:00Z",
            "updated_at": "2025-01-03T00:00:00Z",
            "state": "open",
            "html_url": "https://github.com/test/repo/issues/3",
            "reactions": {"total_count": 0}
        }, repo="test/repo"),
    ]
    
    detector = DuplicateDetector()
    groups = detector.find_duplicates(issues, threshold=0.5)
    
    # Should find one group with issues 1 and 2
    assert len(groups) == 1
    assert len(groups[0].issues) == 2
    issue_numbers = {issue.number for issue in groups[0].issues}
    assert issue_numbers == {1, 2}


def test_duplicate_detector_no_duplicates():
    """Test that dissimilar issues are not marked as duplicates."""
    issues = [
        Issue.from_raw({
            "number": 1,
            "title": "Fix authentication bug",
            "body": "Users cannot log in with OAuth",
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
            "title": "Add export functionality",
            "body": "Need to export data to CSV format",
            "labels": [],
            "comments": 0,
            "created_at": "2025-01-02T00:00:00Z",
            "updated_at": "2025-01-02T00:00:00Z",
            "state": "open",
            "html_url": "https://github.com/test/repo/issues/2",
            "reactions": {"total_count": 0}
        }, repo="test/repo"),
    ]
    
    detector = DuplicateDetector()
    groups = detector.find_duplicates(issues, threshold=0.8)
    
    # Should find no duplicate groups
    assert len(groups) == 0


def test_duplicate_detector_transitive_grouping():
    """Test that transitive duplicates are grouped together (A~B, B~C => {A,B,C})."""
    issues = [
        Issue.from_raw({
            "number": 1,
            "title": "Performance issue with database queries",
            "body": "Database queries are slow",
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
            "title": "Slow database query performance",
            "body": "Queries to database are performing slowly",
            "labels": [],
            "comments": 0,
            "created_at": "2025-01-02T00:00:00Z",
            "updated_at": "2025-01-02T00:00:00Z",
            "state": "open",
            "html_url": "https://github.com/test/repo/issues/2",
            "reactions": {"total_count": 0}
        }, repo="test/repo"),
        Issue.from_raw({
            "number": 3,
            "title": "Database performance needs improvement",
            "body": "The database queries need optimization",
            "labels": [],
            "comments": 0,
            "created_at": "2025-01-03T00:00:00Z",
            "updated_at": "2025-01-03T00:00:00Z",
            "state": "open",
            "html_url": "https://github.com/test/repo/issues/3",
            "reactions": {"total_count": 0}
        }, repo="test/repo"),
    ]
    
    detector = DuplicateDetector()
    groups = detector.find_duplicates(issues, threshold=0.15)
    
    # Should find one group with all three issues
    assert len(groups) == 1
    assert len(groups[0].issues) == 3
    issue_numbers = {issue.number for issue in groups[0].issues}
    assert issue_numbers == {1, 2, 3}


def test_duplicate_detector_empty_issues():
    """Test that empty issue list returns empty groups."""
    detector = DuplicateDetector()
    groups = detector.find_duplicates([], threshold=0.8)
    assert len(groups) == 0


def test_duplicate_detector_single_issue():
    """Test that a single issue returns no duplicates."""
    issues = [
        Issue.from_raw({
            "number": 1,
            "title": "Test issue",
            "body": "This is a test",
            "labels": [],
            "comments": 0,
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "state": "open",
            "html_url": "https://github.com/test/repo/issues/1",
            "reactions": {"total_count": 0}
        }, repo="test/repo"),
    ]
    
    detector = DuplicateDetector()
    groups = detector.find_duplicates(issues, threshold=0.8)
    assert len(groups) == 0


def test_duplicate_detector_handles_none_body():
    """Test that issues with None body are handled gracefully."""
    issues = [
        Issue.from_raw({
            "number": 1,
            "title": "Fix authentication",
            "body": None,
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
            "title": "Authentication fix needed",
            "body": None,
            "labels": [],
            "comments": 0,
            "created_at": "2025-01-02T00:00:00Z",
            "updated_at": "2025-01-02T00:00:00Z",
            "state": "open",
            "html_url": "https://github.com/test/repo/issues/2",
            "reactions": {"total_count": 0}
        }, repo="test/repo"),
    ]
    
    detector = DuplicateDetector()
    groups = detector.find_duplicates(issues, threshold=0.4)
    
    # Should still find duplicates based on title alone
    assert len(groups) == 1
    assert len(groups[0].issues) == 2


def test_duplicate_group_similarity_score():
    """Test that DuplicateGroup stores similarity score."""
    issues = [
        Issue.from_raw({
            "number": 1,
            "title": "Test",
            "body": "Test body",
            "labels": [],
            "comments": 0,
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "state": "open",
            "html_url": "https://github.com/test/repo/issues/1",
            "reactions": {"total_count": 0}
        }, repo="test/repo"),
    ]
    
    group = DuplicateGroup(issues=issues, max_similarity=0.95)
    assert group.max_similarity == 0.95
    assert len(group.issues) == 1
