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
            resp = client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                response_format={"type": "json_object"}
            )  # type: ignore[attr-defined]
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
                    For each issue, provide a clear reasoning explaining why it is placed at that priority, citing evidence from the issue content. 
                    
                    CRITICAL: Your response must be ONLY a valid JSON array with NO markdown code blocks, NO explanatory text before or after, and NO trailing commas. Return raw JSON only.
                    Each element must have exactly these keys: number (int), repo (string), category (one of: {", ".join(categories)}), priority_level (P0|P1|P2), rationale (short explanation of why you chose this priority).
                    Return exactly {len(batch)} items in the array, one for each issue provided.""",
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

        Attempts multiple repair strategies if the model wraps the JSON in markdown,
        explanation text, or introduces syntax errors like trailing commas.
        """
        data = None
        original_text = text
        
        # Step 1: Remove markdown code blocks (```json ... ``` or ``` ... ```)
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"```\s*$", "", text.strip(), flags=re.MULTILINE)
        text = text.strip()
        
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("Initial JSON parse failed: %s; attempting repair", e)
            
            # Step 2: Extract JSON array from surrounding text
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                logger.error("No JSON array found in response. Original text: %s", original_text[:500])
                raise ValueError("Invalid JSON response: no array found")
            
            candidate = match.group(0)
            
            # Step 3: Apply multiple repair strategies
            # Remove trailing commas before closing ] or }
            candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
            # Remove comments (// or /* */)
            candidate = re.sub(r"//.*?$", "", candidate, flags=re.MULTILINE)
            candidate = re.sub(r"/\*.*?\*/", "", candidate, flags=re.DOTALL)
            # Fix common quote issues
            candidate = candidate.replace("'", '"')
            
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError as repair_error:
                logger.error("Repair parse failed: %s. Candidate: %s", repair_error, candidate[:500])
                raise ValueError(f"Invalid JSON after repair: {repair_error}") from repair_error
        if not isinstance(data, list):
            raise ValueError("JSON root is not a list")
        if len(data) != len(batch):
            logger.error(
                "JSON array length mismatch: expected %d items for batch, got %d items. "
                "Batch issue numbers: %s, Response issue numbers: %s",
                len(batch),
                len(data),
                [issue.number for issue in batch],
                [entry.get("number") if isinstance(entry, dict) else None for entry in data]
            )
            raise ValueError(
                f"JSON array length mismatch: expected {len(batch)} items, got {len(data)} items"
            )

        normalized: List[ClassificationResult] = []
        other_category = None
        if "other" in categories:
            other_category = "other"
        else:
            other_category = categories[-1] if categories else "other"
        valid_priorities = {"P0", "P1", "P2"}

        issue_map = {iss.number: iss for iss in batch}
        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                logger.error("Entry %d is not a dict: %s", idx, type(entry))
                raise ValueError(f"Entry {idx} is not an object, got {type(entry).__name__}")
            
            missing = {"number", "repo", "category", "priority_level", "rationale"} - set(entry.keys())
            if missing:
                logger.error("Entry %d missing keys: %s. Available keys: %s", idx, missing, list(entry.keys()))
                raise ValueError(f"Entry {idx} missing required keys: {missing}")
            
            number = entry.get("number")
            if not isinstance(number, int):
                logger.error("Entry %d has invalid number type: %s", idx, type(number))
                raise ValueError(f"Entry {idx}: 'number' must be an integer, got {type(number).__name__}")
            
            if number not in issue_map:
                logger.error("Entry %d number %d not in batch. Expected: %s", idx, number, list(issue_map.keys()))
                raise ValueError(f"Entry {idx}: issue number {number} not found in batch")
            
            category = entry.get("category")
            if category not in categories:
                logger.warning("Entry %d has invalid category '%s', using '%s'", idx, category, other_category)
                category = other_category
            
            priority = entry.get("priority_level")
            rationale = entry.get("rationale") or ""
            if priority not in valid_priorities:
                logger.warning("Entry %d has invalid priority '%s', adjusting to P2", idx, priority)
                rationale += f" (priority adjusted from {priority} to P2)"
                priority = "P2"
            
            repo_val = entry.get("repo")
            if not isinstance(repo_val, str):
                logger.error("Entry %d has invalid repo type: %s", idx, type(repo_val))
                raise ValueError(f"Entry {idx}: 'repo' must be a string, got {type(repo_val).__name__}")
            
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
