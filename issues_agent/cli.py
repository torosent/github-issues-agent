"""Command-line interface for GitHub Issues AI Agent.

Orchestrates the pipeline: Fetch -> Analyze -> Report -> Write.
"""
import argparse
import logging
import sys
from typing import List, Optional
from datetime import datetime, timedelta, timezone
import re

from .config import load_config
from .logging import init_logging
from .github_client import GitHubClient
from .llm_client import AzureOpenAIClient
from .analysis import analyze_issues
from .report import ReportGenerator

logger = logging.getLogger(__name__)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GitHub Issues AI Agent CLI")
    parser.add_argument(
        "--repos", 
        required=True, 
        help="Comma-separated list of repositories (owner/repo)"
    )
    parser.add_argument(
        "--output", 
        default="report.md", 
        help="Output markdown file path"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=100, 
        help="Max issues to fetch per repo"
    )
    parser.add_argument(
        "--since", 
        default="2y", 
        help="Fetch issues updated since (e.g. 2y, 6m, 30d)"
    )
    parser.add_argument(
        "--model", 
        default="gpt-5.1", 
        help="LLM model to use (default: gpt-5.1)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Print report to stdout instead of writing to file"
    )
    return parser

def _parse_since(since_str: str) -> str:
    """Parse human-readable time duration to ISO 8601 timestamp."""
    since_str = since_str.strip().lower()
    match = re.match(r'^(\d+)([ymd])$', since_str)
    if not match:
        # Default to 2 years if invalid
        logger.warning(
            f"Invalid --since format '{since_str}'. Expected format: <number><unit> (e.g., 2y, 6m, 30d). Defaulting to 2y."
        )
        return (datetime.now(timezone.utc) - timedelta(days=365*2)).isoformat()
    
    amount = int(match.group(1))
    unit = match.group(2)
    now = datetime.now(timezone.utc)
    
    if unit == 'y':
        delta = timedelta(days=amount * 365)
    elif unit == 'm':
        delta = timedelta(days=amount * 30)
    else:
        # unit == 'd'
        delta = timedelta(days=amount)
    
    return (now - delta).isoformat()

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    init_logging()
    config = load_config()
    
    repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    if not repos:
        logger.error("No repositories specified.")
        return 1

    try:
        # 1. Fetch
        logger.info(f"Fetching issues from {repos}...")
        gh_client = GitHubClient(token=config.github_token)
        since_iso = _parse_since(args.since)
        
        all_issues = gh_client.fetch_issues(repos, limit=args.limit, since=since_iso)
        logger.info(f"Fetched {len(all_issues)} issues.")
        
        if not all_issues:
            logger.info("No issues found.")
            return 0

        # 2. Analyze
        logger.info("Analyzing issues with LLM...")
        llm_client = AzureOpenAIClient(
            endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_api_key,
            deployment=config.azure_openai_deployment,
            api_version=config.azure_openai_api_version
        )
        
        analysis_result = analyze_issues(llm_client, all_issues, args.model)
        
        # 3. Report
        logger.info("Generating report...")
        generator = ReportGenerator(all_issues, analysis_result, repos)
        report_content = generator.generate()
        
        # 4. Write
        if args.dry_run:
            print(report_content)
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report_content)
            logger.info(f"Report written to {args.output}")
            
        return 0
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())

