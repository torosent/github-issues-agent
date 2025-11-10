"""GitHub API client for fetching issues across multiple repositories.

Phase 2 implementation: multi-repo issue fetching with pagination, PR exclusion,
reactions aggregation, global limit enforcement, and basic retry/backoff.
"""
from __future__ import annotations

import time
import logging
import re
from typing import List, Dict, Optional
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)


_POSITIVE_REACTIONS = {"+1", "heart", "hooray", "rocket", "eyes"}
_TRANSIENT_STATUS = {429, 502, 503}


class GitHubClient:
    """Thin synchronous GitHub Issues client.

    Responsibilities:
        * Fetch issues across one or more repositories with pagination.
        * Exclude pull requests (GitHub returns them in the issues endpoint).
        * Enforce a global issue fetch limit across repositories.
        * Optionally filter by `updated_at >= since` (ISO 8601 timestamp string).
        * Augment each issue with reaction counts (total & positive subset).
        * Apply basic retry/backoff for transient or network failures.

    Notes:
        - Only the specific fields required downstream are retained; the full
          raw JSON object is stored under the `raw` key for future use.
        - Positive reactions include: +1, heart, hooray, rocket, eyes.
        - Caller is responsible for providing a valid personal access token.
    """
    def __init__(
        self,
        token: str,
        base_url: str = "https://api.github.com",
        timeout: float = 10.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        if not token:
            raise ValueError("GitHub token must be provided")
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json, application/vnd.github.squirrel-girl-preview+json",
                "User-Agent": "issues-agent",
            },
            transport=transport,
        )

    def fetch_issues(self, repos: List[str], limit: int = 300, since: Optional[str] = None) -> List[Dict]:
        """Fetch issues from multiple repositories.

        Args:
            repos: List of repository identifiers in the form "owner/name".
            limit: Maximum total number of issues to collect across all repos.
            since: Optional ISO 8601 timestamp (UTC) to filter by last update.

        Returns:
            A list of augmented issue dictionaries.

        Raises:
            ValueError: If `repos` is empty.
        """
        if not repos:
            raise ValueError("At least one repository must be provided")
        if limit <= 0:
            return []
        collected: List[Dict] = []
        remaining = limit
        for repo in repos:
            if remaining <= 0:
                break
            logger.info("Starting fetch for repo %s remaining_limit=%d since=%s", repo, remaining, since or "all time")
            repo_issues = self._fetch_repo_issues(repo, remaining, since)
            collected.extend(repo_issues)
            remaining = limit - len(collected)
            logger.info(
                "Completed fetch for repo %s count=%d total_collected=%d remaining_limit=%d",
                repo,
                len(repo_issues),
                len(collected),
                remaining,
            )
        return collected

    def _fetch_repo_issues(self, repo: str, remaining: int, since: Optional[str] = None) -> List[Dict]:
        """Fetch issues for a single repository up to `remaining` items.

        Handles pagination and PR exclusion. Stops early if the global
        remaining allowance is exhausted.
        """
        owner, name = self._split_repo(repo)
        results: List[Dict] = []
        page = 1
        next_url = f"/repos/{owner}/{name}/issues"
        params = {"per_page": 100, "page": page, "state": "open"}
        if since:
            params["since"] = since
        while next_url and remaining > 0:
            logger.info("Fetching page %s for %s", page, repo)
            resp = self._get(next_url, params=params)
            resp.raise_for_status()
            issues = resp.json()
            for issue in issues:
                if "pull_request" in issue:
                    continue  # exclude PRs
                augmented = self._augment_issue_details(issue, owner, name)
                results.append(augmented)
                remaining -= 1
                if remaining <= 0:
                    break
            # Parse Link header for next page
            link = resp.headers.get("Link", "")
            next_page_path = self._parse_next_link(link)
            if remaining > 0 and next_page_path:
                # Reset for next iteration
                page += 1
                next_url = next_page_path.replace(self.base_url, "")
                params = None  # next_url already includes query params
            else:
                break
        return results

    def _get(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        retry: int = 3,
    ) -> httpx.Response:
        """Perform a GET request with retry logic.

        Retries on network exceptions and selected transient HTTP status codes.
        Also performs naive rate-limit waiting when remaining quota is zero.
        """
        attempt = 0
        backoff = 1.0
        while True:
            try:
                resp = self._client.get(url, params=params, headers=headers)
            except httpx.HTTPError as e:
                if attempt < retry:
                    logger.warning("Network error %s on %s, retrying in %.1fs", e, url, backoff)
                    time.sleep(backoff)
                    attempt += 1
                    backoff *= 2
                    continue
                logger.error("Unrecoverable network error %s on %s", e, url)
                raise

            # Rate limit handling
            if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
                reset = resp.headers.get("X-RateLimit-Reset")
                if reset and reset.isdigit():
                    wait = max(0, int(reset) - int(time.time()))
                    logger.warning("Rate limit reached, sleeping for %ss", wait)
                    time.sleep(wait)
                    attempt += 1
                    continue

            if resp.status_code in _TRANSIENT_STATUS:
                if attempt < retry:
                    logger.warning(
                        "Transient status %s on %s, retrying in %.1fs", resp.status_code, url, backoff
                    )
                    time.sleep(backoff)
                    attempt += 1
                    backoff *= 2
                    continue
            return resp

    def _augment_issue_details(self, issue: Dict, owner: str, name: str) -> Dict:
        """Add additional derived fields to a raw issue.

        Fetches reactions (best effort) and extracts a trimmed set of fields
        needed downstream. Original JSON preserved under `raw`.
        """
        number = issue.get("number")
        reactions_url = f"/repos/{owner}/{name}/issues/{number}/reactions"
        total = 0
        positive = 0
        try:
            reactions_resp = self._get(reactions_url)
            if reactions_resp.status_code >= 200 and reactions_resp.status_code < 300:
                reactions = reactions_resp.json()
                total = len(reactions)
                positive = sum(1 for r in reactions if r.get("content") in _POSITIVE_REACTIONS)
            else:
                logger.warning(
                    "Failed reactions fetch for issue %s status=%d", number, reactions_resp.status_code
                )
        except Exception as e:  # pragma: no cover (network failures rare in tests)
            logger.warning("Exception during reactions fetch for issue %s: %s", number, e)
        labels = [l.get("name") for l in issue.get("labels", []) if isinstance(l, dict)]
        return {
            "number": number,
            "title": issue.get("title"),
            "body": issue.get("body"),
            "labels": labels,
            "comments": issue.get("comments", 0),
            "created_at": issue.get("created_at"),
            "updated_at": issue.get("updated_at"),
            "state": issue.get("state"),
            "html_url": issue.get("html_url"),
            "repo": f"{owner}/{name}",
            "reactions_total": total,
            "reactions_positive": positive,
            "raw": issue,
        }

    @staticmethod
    def _split_repo(repo: str) -> tuple[str, str]:
        """Split an "owner/name" string into a tuple.

        Raises:
            ValueError: if the format is invalid.
        """
        try:
            owner, name = repo.split("/", 1)
        except ValueError:
            raise ValueError(f"Invalid repo format: {repo}. Expected 'owner/name'.")
        return owner, name

    @staticmethod
    def _parse_next_link(link_header: str) -> Optional[str]:
        """Parse the HTTP Link header for a `rel="next"` URL.

        Returns the first matching URL or None if no next page exists.
        """
        if not link_header:
            return None
        # Pattern: <URL>; rel="next"
        matches = re.findall(r'<([^>]+)>; rel="next"', link_header)
        return matches[0] if matches else None
