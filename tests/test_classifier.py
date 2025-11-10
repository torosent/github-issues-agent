import json
import re
import pytest

from issues_agent.models import Issue, ClassificationResult

class FakeClient:
    def __init__(self, responses):
        self._responses = responses
        self.calls = 0
        # Provide both responses.create and legacy chat.completions.create for fallback path
        self.responses = type("Responses", (), {"create": self._responses_create})()
        self.chat = type("Chat", (), {"completions": type("Completions", (), {"create": self._legacy_create})()})()

    def _responses_create(self, model, input):  # mimic responses.create
        content = self._responses[self.calls]
        self.calls += 1
        # Minimal object with output_text attribute
        return type("Resp", (), {"output_text": content})()

    def _legacy_create(self, model, messages):  # mimic chat.completions.create fallback
        content = self._responses[self.calls]
        self.calls += 1
        class Choice:  # minimal structure
            def __init__(self, content):
                self.message = type("Msg", (), {"content": content})
        return type("Resp", (), {"choices": [Choice(content)]})


def make_issue(num: int) -> Issue:
    long_body = "Body text" + " x" * 400  # long body triggers truncation
    return Issue(
        number=num,
        repo="owner/name",
        title=f"Issue {num} title",
        body=long_body,
        labels=["bug"],
        comments=2,
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
        state="open",
        html_url=f"https://example.com/{num}",
        reactions_total=5,
        reactions_positive=3,
        raw={"number": num, "title": f"Issue {num} title"},
    )


def test_classify_single_batch_returns_results():
    from issues_agent.classifier import AzureOpenAIClassifier
    issues = [make_issue(1), make_issue(2)]
    categories = ["bug", "feature", "other"]
    response_json = json.dumps([
        {"number": 1, "repo": "owner/name", "category": "bug", "priority_level": "P1", "rationale": "Looks important"},
        {"number": 2, "repo": "owner/name", "category": "feature", "priority_level": "P2", "rationale": "Minor"},
    ])
    client = FakeClient([response_json])
    clf = AzureOpenAIClassifier(endpoint="e", api_key="k", deployment="d", api_version="v", client=client, batch_size=10)

    results = clf.classify(issues, categories)
    assert len(results) == 2
    assert all(isinstance(r, ClassificationResult) for r in results)
    assert results[0].category == "bug"
    assert results[1].priority_level == "P2"


def test_classify_multiple_batches():
    from issues_agent.classifier import AzureOpenAIClassifier
    categories = ["bug", "feature", "other"]
    issues = [make_issue(i) for i in range(1, 13)]  # 12 issues, batch_size=5 => 3 calls
    # Build three response batches
    responses = []
    for batch_start in range(0, 12, 5):
        batch = issues[batch_start: batch_start + 5]
        arr = []
        for issue in batch:
            arr.append({
                "number": issue.number,
                "repo": issue.repo,
                "category": categories[issue.number % len(categories)],
                "priority_level": "P0" if issue.number % 2 == 0 else "P2",
                "rationale": "r",
            })
        responses.append(json.dumps(arr))
    client = FakeClient(responses)
    clf = AzureOpenAIClassifier(endpoint="e", api_key="k", deployment="d", api_version="v", client=client, batch_size=5)
    results = clf.classify(issues, categories)
    assert len(results) == 12
    assert client.calls == 3


def test_parse_repair_malformed_json():
    from issues_agent.classifier import AzureOpenAIClassifier
    issues = [make_issue(1)]
    categories = ["bug", "feature", "other"]
    bad_response = "Here are the results:\n" + json.dumps([
        {"number": 1, "repo": "owner/name", "category": "bug", "priority_level": "P1", "rationale": "Looks important"}
    ])
    client = FakeClient([bad_response])
    clf = AzureOpenAIClassifier(endpoint="e", api_key="k", deployment="d", api_version="v", client=client, batch_size=10)
    results = clf.classify(issues, categories)
    assert len(results) == 1
    assert results[0].category == "bug"


def test_invalid_category_fallback_to_other():
    from issues_agent.classifier import AzureOpenAIClassifier
    issues = [make_issue(1)]
    categories = ["bug", "feature", "other"]
    response_json = json.dumps([
        {"number": 1, "repo": "owner/name", "category": "unknown", "priority_level": "P1", "rationale": "Unmapped"}
    ])
    client = FakeClient([response_json])
    clf = AzureOpenAIClassifier(endpoint="e", api_key="k", deployment="d", api_version="v", client=client, batch_size=10)
    results = clf.classify(issues, categories)
    assert results[0].category == "other"


def test_invalid_priority_level_defaults_to_p2():
    from issues_agent.classifier import AzureOpenAIClassifier
    issues = [make_issue(1)]
    categories = ["bug", "feature", "other"]
    response_json = json.dumps([
        {"number": 1, "repo": "owner/name", "category": "bug", "priority_level": "HIGH", "rationale": "Bad priority"}
    ])
    client = FakeClient([response_json])
    clf = AzureOpenAIClassifier(endpoint="e", api_key="k", deployment="d", api_version="v", client=client, batch_size=10)
    results = clf.classify(issues, categories)
    assert results[0].priority_level == "P2"
    assert "adjusted" in results[0].rationale.lower()
