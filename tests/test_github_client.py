import httpx
import pytest

from issues_agent.github_client import GitHubClient


def build_issue(number: int, title: str = "Issue", repo: str = "owner/repo", **extra):
    base = {
        "number": number,
        "title": f"{title} {number}",
        "body": f"Body {number}",
        "labels": [{"name": "bug"}],
        "comments": 3,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T01:00:00Z",
        "state": "open",
        "html_url": f"https://github.com/{repo}/issues/{number}",
    }
    base.update(extra)
    return base


def make_listing_response(issues):
    return issues


def test_fetch_issues_single_repo():
    issues_page = [build_issue(1), build_issue(2)]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/owner/repo/issues":
            return httpx.Response(200, json=issues_page)
        if request.url.path.startswith("/repos/owner/repo/issues/") and request.url.path.endswith("/reactions"):
            # return empty reactions
            return httpx.Response(200, json=[])
        raise AssertionError(f"Unexpected URL {request.url}")

    transport = httpx.MockTransport(handler)
    client = GitHubClient(token="TEST", transport=transport)
    results = client.fetch_issues(["owner/repo"], limit=10)
    assert len(results) == 2
    numbers = [r["number"] for r in results]
    assert numbers == [1, 2]
    assert all(r["repo"] == "owner/repo" for r in results)


def test_fetch_issues_multiple_repos():
    repo1 = [build_issue(1, repo="owner/repo1"), build_issue(2, repo="owner/repo1")]
    repo2 = [build_issue(10, repo="owner/repo2")]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/owner/repo1/issues":
            return httpx.Response(200, json=repo1)
        if request.url.path == "/repos/owner/repo2/issues":
            return httpx.Response(200, json=repo2)
        if request.url.path.startswith("/repos/owner/repo1/issues/") and request.url.path.endswith("/reactions"):
            return httpx.Response(200, json=[])
        if request.url.path.startswith("/repos/owner/repo2/issues/") and request.url.path.endswith("/reactions"):
            return httpx.Response(200, json=[])
        raise AssertionError(f"Unexpected URL {request.url}")

    transport = httpx.MockTransport(handler)
    client = GitHubClient(token="TEST", transport=transport)
    results = client.fetch_issues(["owner/repo1", "owner/repo2"], limit=10)
    assert [r["repo"] for r in results] == ["owner/repo1", "owner/repo1", "owner/repo2"]


def test_pagination_with_link_header():
    page1 = [build_issue(1), build_issue(2)]
    page2 = [build_issue(3), build_issue(4)]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/owner/repo/issues":
            # Determine page from query
            page = request.url.params.get("page", "1")
            if page == "1":
                link = '<https://api.github.com/repos/owner/repo/issues?page=2&per_page=100>; rel="next"'
                return httpx.Response(200, json=page1, headers={"Link": link})
            elif page == "2":
                return httpx.Response(200, json=page2)
        if request.url.path.startswith("/repos/owner/repo/issues/") and request.url.path.endswith("/reactions"):
            return httpx.Response(200, json=[])
        raise AssertionError(f"Unexpected URL {request.url}")

    transport = httpx.MockTransport(handler)
    client = GitHubClient(token="TEST", transport=transport)
    results = client.fetch_issues(["owner/repo"], limit=10)
    assert [r["number"] for r in results] == [1, 2, 3, 4]


def test_exclude_pull_requests():
    issues = [build_issue(1), build_issue(2, pull_request={"url": "x"})]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/owner/repo/issues":
            return httpx.Response(200, json=issues)
        if request.url.path.startswith("/repos/owner/repo/issues/") and request.url.path.endswith("/reactions"):
            return httpx.Response(200, json=[])
        raise AssertionError(f"Unexpected URL {request.url}")

    transport = httpx.MockTransport(handler)
    client = GitHubClient(token="TEST", transport=transport)
    results = client.fetch_issues(["owner/repo"], limit=10)
    assert len(results) == 1
    assert results[0]["number"] == 1


def test_limit_enforcement():
    issues = [build_issue(i) for i in range(1, 6)]  # 5 issues

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/owner/repo/issues":
            return httpx.Response(200, json=issues)
        if request.url.path.startswith("/repos/owner/repo/issues/") and request.url.path.endswith("/reactions"):
            return httpx.Response(200, json=[])
        raise AssertionError(f"Unexpected URL {request.url}")

    transport = httpx.MockTransport(handler)
    client = GitHubClient(token="TEST", transport=transport)
    results = client.fetch_issues(["owner/repo"], limit=3)
    assert len(results) == 3
    assert [r["number"] for r in results] == [1, 2, 3]


def test_fetch_reactions_and_comments():
    issues = [build_issue(1), build_issue(2)]

    reactions_payload = {
        1: [
            {"content": "+1"},
            {"content": "heart"},
            {"content": "confused"},
        ],
        2: [
            {"content": "rocket"},
            {"content": "eyes"},
            {"content": "-1"},
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/owner/repo/issues":
            return httpx.Response(200, json=issues)
        if request.url.path.startswith("/repos/owner/repo/issues/") and request.url.path.endswith("/reactions"):
            number = int(request.url.path.split("/")[-2])
            return httpx.Response(200, json=reactions_payload[number])
        raise AssertionError(f"Unexpected URL {request.url}")

    transport = httpx.MockTransport(handler)
    client = GitHubClient(token="TEST", transport=transport)
    results = client.fetch_issues(["owner/repo"], limit=10)
    one = next(r for r in results if r["number"] == 1)
    two = next(r for r in results if r["number"] == 2)
    assert one["reactions_total"] == 3
    assert one["reactions_positive"] == 2  # +1, heart
    assert two["reactions_total"] == 3
    assert two["reactions_positive"] == 2  # rocket, eyes
