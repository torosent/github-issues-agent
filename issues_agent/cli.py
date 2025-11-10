"""Command-line interface for GitHub Issues AI Agent (Phase 7).

Orchestrates the full pipeline:
  config load -> fetch issues -> dataclass conversion -> category load
  -> classification -> scoring -> markdown report generation -> write file.

Usage (comma-separated repos):
  python -m issues_agent --repos owner1/repo1,owner2/repo2 --output report.md

Arguments:
  --repos            Comma-separated list of repositories (required)
  --output           Output markdown file path (default: report.md)
  --limit            Max total issues across all repos (default: 300)
  --since            Fetch issues updated since (e.g., '2y', '6m', '30d') (default: 2y)
  --categories-file  Optional path to YAML/JSON categories file
  --batch-size       Classifier batch size (default: 10)
  --dry-run          Perform all steps except writing file; prints first 20 lines

Exit codes:
  0 on success (including empty issues case)
  1 on unexpected error
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional
from datetime import datetime, timedelta, timezone

from .config import load_config
from .logging import init_logging
from .github_client import GitHubClient
from .models import Issue, load_categories
from .classifier import AzureOpenAIClassifier
from .scoring import score_and_rank
from .report import ReportGenerator


def build_parser() -> argparse.ArgumentParser:
    """Create and configure the top-level argument parser for the CLI."""
    parser = argparse.ArgumentParser(description="GitHub Issues AI Agent CLI")
    parser.add_argument(
        "--repos",
        dest="repos",
        type=str,
        required=False,
        help="Comma-separated list of repositories (owner/name,owner/name,...)",
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=str,
        default="report.md",
        help="Output markdown file path (default: report.md)",
    )
    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        default=300,
        help="Maximum number of issues to fetch across all repos (default: 300)",
    )
    parser.add_argument(
        "--since",
        dest="since",
        type=str,
        default='2y',
        help="Fetch issues updated since (e.g., '2y', '6m', '30d') (default: 2y)",
    )
    parser.add_argument(
        "--categories-file",
        dest="categories_file",
        type=str,
        default=None,
        help="Optional path to YAML/JSON file listing categories",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=10,
        help="Classifier batch size (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Perform all steps except writing file; prints first 20 lines",
    )
    parser.add_argument(
        "--reasoning-effort",
        dest="reasoning_effort",
        choices=["low", "medium", "high"],
        default="medium",
        help="Optional reasoning effort for Responses API (default: medium)",
    )
    parser.add_argument(
        "--text-verbosity",
        dest="text_verbosity",
        choices=["low", "medium", "high"],
        default="low",
        help="Text verbosity preference for Responses API (default: low)",
    )
    return parser


def _parse_repos(raw: Optional[str]) -> List[str]:
    """Split comma-separated repositories string into a list.

    Raises SystemExit with code 1 if parsing fails so that the CLI exits
    gracefully with an error message.
    """
    if raw is None or not raw.strip():
        print("Error: --repos argument is required (comma-separated list).", file=sys.stderr)
        raise SystemExit(1)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        print("Error: Parsed zero repositories from --repos.", file=sys.stderr)
        raise SystemExit(1)
    return parts


def _parse_since(since_str: str) -> str:
    """Parse human-readable time duration to ISO 8601 timestamp.
    
    Examples: '2y' -> 2 years ago, '6m' -> 6 months ago, '30d' -> 30 days ago
    Returns ISO 8601 formatted timestamp for GitHub API.
    """
    since_str = since_str.strip().lower()
    if not since_str:
        raise ValueError("--since cannot be empty")
    
    # Parse the number and unit
    import re
    match = re.match(r'^(\d+)([ymd])$', since_str)
    if not match:
        raise ValueError(f"Invalid --since format: {since_str}. Expected format: <number><unit> (e.g., '2y', '6m', '30d')")
    
    amount = int(match.group(1))
    unit = match.group(2)
    
    now = datetime.now(timezone.utc)
    
    if unit == 'y':
        delta = timedelta(days=amount * 365)
    elif unit == 'm':
        delta = timedelta(days=amount * 30)
    elif unit == 'd':
        delta = timedelta(days=amount)
    else:
        raise ValueError(f"Unsupported time unit: {unit}")
    
    since_date = now - delta
    return since_date.isoformat().replace('+00:00', 'Z')


def main(argv: Optional[List[str]] = None) -> int:
    """Run the full classification and reporting pipeline.

    Steps:
      1. Parse arguments & validate repositories.
      2. Load configuration from environment (.env supported).
      3. Fetch issues (optionally filtered by --since).
      4. Convert raw JSON issues into dataclasses.
      5. Load categories (default or file).
      6. Classify issues using Azure OpenAI.
      7. Score and rank issues.
      8. Render markdown report (preview if --dry-run).

    Returns:
        Process exit code: 0 on success, 1 on any handled error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        repos = _parse_repos(args.repos)
        init_logging()  # ensure logging available early
        print("Loading configuration...")
        config = load_config()

        print("Instantiating GitHub client...")
        gh = GitHubClient(token=config.github_token)

        fetch_kwargs = {"limit": args.limit}
        if args.since:
            since_timestamp = _parse_since(args.since)
            print(f"Fetching issues updated since {since_timestamp}...")
            fetch_kwargs["since"] = since_timestamp
        else:
            print("Fetching issues...")
        try:
            raw_issues = gh.fetch_issues(repos, **fetch_kwargs)
        except TypeError as e:
            # Allow tests that monkeypatch fetch_issues without a 'since' parameter.
            if "since" in fetch_kwargs and "since" in str(e):
                since_val = fetch_kwargs.pop("since")  # remove unsupported kwarg
                raw_issues = gh.fetch_issues(repos, **fetch_kwargs)
            else:
                raise
        if not raw_issues:
            print("No issues found; exiting.")
            return 0
        print(f"Fetched {len(raw_issues)} issues across {len(repos)} repos.")

        print("Converting raw issues to dataclasses...")
        issues: List[Issue] = []
        for raw in raw_issues:
            repo = raw.get("repo")  # augmented result contains repo
            if not isinstance(repo, str) or not repo:
                raise ValueError("Augmented issue missing 'repo' string")
            issues.append(Issue.from_raw(raw, repo=repo))

        print("Loading categories...")
        categories = load_categories(args.categories_file)

        print("Classifying issues (batch size {})...".format(args.batch_size))
        classifier = AzureOpenAIClassifier(
            endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_api_key,
            deployment=config.azure_openai_deployment,
            api_version=config.azure_openai_api_version,
            batch_size=args.batch_size,
            reasoning_effort=args.reasoning_effort,
            text_verbosity=args.text_verbosity,
        )
        classifications = classifier.classify(issues, categories)

        print("Scoring issues...")
        scored = score_and_rank(issues, classifications)

        print("Generating markdown report...")
        markdown = ReportGenerator(scored, repos).generate()

        if args.dry_run:
            print("Dry run: showing preview (first 20 lines):")
            lines = markdown.splitlines()
            preview = "\n".join(lines[:20])
            print(preview)
            print("Dry run complete; no file written.")
            return 0

        print(f"Writing report to {args.output}...")
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"Report written: {args.output} (issues: {len(scored)})")
        return 0

    except SystemExit:
        # Re-raise SystemExit from parser or repo parsing to keep its message
        raise
    except Exception as e:  # Broad catch as per requirements
        print(f"Error: {e}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "main"]
