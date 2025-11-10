import json
import logging
import re
from typing import List, Optional, Any, Dict

from .models import Issue, ClassificationResult

logger = logging.getLogger(__name__)


class AzureOpenAIClassifier:
    """Issue classifier using Azure OpenAI GPT deployment.

    Converts batches of issue metadata into a single prompt, requests a JSON-only
    response describing category, priority band (P0/P1/P2), and a brief rationale
    per issue. Supports both the newer Responses API and a fallback to the legacy
    chat completions API for compatibility.
    """
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str,
        batch_size: int = 10,
        timeout: float = 30.0,
        client: Optional[object] = None,
        reasoning_effort: Optional[str] = None,
        text_verbosity: Optional[str] = None,
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment = deployment
        self.api_version = api_version
        self.batch_size = batch_size
        self.timeout = timeout
        self._client = client
        self.reasoning_effort = reasoning_effort
        self.text_verbosity = text_verbosity

    def _get_client(self):
        """Instantiate (or return injected) Azure OpenAI client lazily."""
        if self._client is not None:
            return self._client
        try:
            from openai import AzureOpenAI  # lazy import
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("AzureOpenAI SDK not installed") from e
        return AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
        )

    def classify(self, issues: List[Issue], categories: List[str]) -> List[ClassificationResult]:
        """Classify a list of issues into categories and priorities.

        Processes issues in batches to reduce token usage. Returns a list of
        `ClassificationResult` with one element per input issue.
        """
        results: List[ClassificationResult] = []
        total_batches = (len(issues) + self.batch_size - 1) // self.batch_size if issues else 0
        for batch_index, i in enumerate(range(0, len(issues), self.batch_size), start=1):
            batch = issues[i : i + self.batch_size]
            logger.debug(
                "Classifying batch %d/%d size=%d", batch_index, total_batches, len(batch)
            )
            results.extend(self._classify_batch(batch, categories))
        return results

    def _classify_batch(self, batch: List[Issue], categories: List[str]) -> List[ClassificationResult]:
        """Classify a single batch of issues using the model deployment."""
        messages = self._build_messages(batch, categories)
        client = self._get_client()
        logger.info("Classifying batch size %d", len(batch))
        # Use Responses API (unified) instead of chat.completions
        try:
            kwargs: Dict[str, Any] = {"model": self.deployment, "input": messages}
            # Add optional reasoning/text parameters if configured
            if self.reasoning_effort:
                kwargs["reasoning"] = {"effort": self.reasoning_effort}
            if self.text_verbosity:
                kwargs["text"] = {"verbosity": self.text_verbosity}
            try:
                resp = client.responses.create(**kwargs)  # type: ignore[attr-defined]
            except TypeError as e:
                # Remove advanced kwargs and retry for incompatible fake/SDK
                removed = False
                for k in ("reasoning", "text"):
                    if k in kwargs:
                        kwargs.pop(k)
                        removed = True
                if removed:
                    resp = client.responses.create(**kwargs)  # type: ignore[attr-defined]
                else:
                    raise
        except AttributeError:  # fallback if older client
            resp = client.chat.completions.create(model=self.deployment, messages=messages)  # type: ignore[attr-defined]
            text = resp.choices[0].message.content or ""
            return self._parse_response(text, batch, categories)
        text = self._extract_text_from_responses(resp)
        return self._parse_response(text, batch, categories)

    @staticmethod
    def _extract_text_from_responses(resp: Any) -> str:
        """Extract textual content from a Responses API result.
        Supports both `output_text` shortcut and iterating over `output` parts.
        """
        if hasattr(resp, "output_text") and isinstance(getattr(resp, "output_text"), str):
            return resp.output_text
        # Fallback: iterate over structured output
        pieces: List[str] = []
        output = getattr(resp, "output", [])
        for item in output:
            # message items may contain content list
            content_list = getattr(item, "content", [])
            for content in content_list:
                if getattr(content, "type", None) == "output_text":
                    pieces.append(getattr(content, "text", ""))
        return "".join(pieces)

    def _build_messages(self, batch: List[Issue], categories: List[str]) -> List[dict]:
        """Build system/user messages for model input.

        The user message contains a block per issue with truncated body content
        to control prompt size.
        """
        system = {
            "role": "system",
            "content": f"""You are acting as a senior product and engineering analyst. You will be given the title, description, labels, and comments from one or more GitHub issues in a software project.
                    Your task is to:
                    Evaluate each issue`s impact on the product or users (for example: number of users affected, effect on key features, risk of data loss or security concerns).
                    Evaluate its urgency (for example: occurs frequently, blocks ongoing work, relates to an imminent release).
                    Assess its potential to affect the project timeline (for example: slows down dependent tasks, delays delivery milestones, causes rework).
                    Rank the issues from highest to lowest priority.
                    For each issue, provide a clear reasoning explaining why it is placed at that priority, citing evidence from the issue content. Output ONLY a JSON array where each element has keys: number (int), repo (string), category (one of: {", ".join(categories)}), priority_level (P0|P1|P2), rationale (short explanation of why you chose this priority).""",
        }
        issue_blobs = []
        for issue in batch:
            body = issue.body or ""
            blob = (
                f"===ISSUE===\nnumber: {issue.number}\nrepo: {issue.repo}\ntitle: {issue.title}\nlabels: {', '.join(issue.labels)}\ncomments: {issue.comments}\nreactions_positive: {issue.reactions_positive}\nreactions_total: {issue.reactions_total}\nbody: {body}"
            )
            issue_blobs.append(blob)
        user = {
            "role": "user",
            "content": "\n\n".join(issue_blobs),
        }
        return [system, user]

    def _parse_response(self, text: str, batch: List[Issue], categories: List[str]) -> List[ClassificationResult]:
        """Parse and validate JSON array returned by the model.

        Attempts light repair if the model wraps the JSON in explanation text
        or introduces trailing commas.
        """
        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Initial JSON parse failed; attempting repair")
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                logger.error("No JSON array found in response")
                raise ValueError("Invalid JSON response: no array found")
            candidate = match.group(0)
            # remove trailing commas before closing ] or }
            candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError as e:
                logger.error("Repair parse failed: %s", e)
                raise ValueError("Invalid JSON after repair") from e
        if not isinstance(data, list):
            raise ValueError("JSON root is not a list")
        if len(data) != len(batch):
            raise ValueError("JSON array length mismatch batch size")

        normalized: List[ClassificationResult] = []
        other_category = None
        if "other" in categories:
            other_category = "other"
        else:
            other_category = categories[-1] if categories else "other"
        valid_priorities = {"P0", "P1", "P2"}

        issue_map = {iss.number: iss for iss in batch}
        for entry in data:
            if not isinstance(entry, dict):
                raise ValueError("Entry not an object")
            missing = {"number", "repo", "category", "priority_level", "rationale"} - set(entry.keys())
            if missing:
                raise ValueError(f"Missing keys: {missing}")
            number = entry.get("number")
            if number not in issue_map:
                raise ValueError("Number not in batch")
            category = entry.get("category")
            if category not in categories:
                category = other_category
            priority = entry.get("priority_level")
            rationale = entry.get("rationale") or ""
            if priority not in valid_priorities:
                rationale += " (priority adjusted to P2)"
                priority = "P2"
            repo_val = entry.get("repo")
            if not isinstance(repo_val, str):
                raise ValueError("repo must be a string")
            normalized.append(
                ClassificationResult(
                    number=number,
                    repo=repo_val,
                    category=category,
                    priority_level=priority,
                    rationale=rationale,
                )
            )
        return normalized
