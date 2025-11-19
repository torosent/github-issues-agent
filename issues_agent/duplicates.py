"""Duplicate issue detection using multiple similarity techniques.

Combines TF-IDF, semantic similarity, and LLM-based detection with issue comments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Dict, Optional
import re
import json
import logging
import os
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from issues_agent.models import Issue
from issues_agent.llm_client import AzureOpenAIClient

# Suppress tokenizers warning to avoid deadlocks in forked processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DuplicateGroup:
    """A group of issues that are likely duplicates of each other."""
    issues: List[Issue]
    max_similarity: float
    
    def __post_init__(self) -> None:  # type: ignore[override]
        if not self.issues:
            raise ValueError("DuplicateGroup must contain at least one issue")
        if not isinstance(self.max_similarity, (int, float)):
            raise ValueError("max_similarity must be numeric")


class DuplicateDetector:
    """Detects duplicate issues using multiple similarity techniques to reduce false positives."""
    
    def __init__(
        self,
        use_llm: bool = False,
        llm_client: Optional[AzureOpenAIClient] = None,
        github_client: Optional[object] = None,
        fetch_comments: bool = True
    ):
        """Initialize duplicate detector.
        
        Args:
            use_llm: Whether to use LLM for semantic similarity (more accurate but slower)
            llm_client: AzureOpenAIClient instance for LLM-based detection
            github_client: GitHub client instance for fetching issue comments
            fetch_comments: Whether to fetch and include issue comments in analysis
        """
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.github_client = github_client
        self.fetch_comments = fetch_comments
        self._comments_cache: Dict[int, List[str]] = {}
    
    def find_duplicates(
        self, 
        issues: List[Issue], 
        threshold: float = 0.8
    ) -> List[DuplicateGroup]:
        """Find groups of duplicate issues using multi-signal similarity detection.
        
        Uses either LLM-based detection (more accurate) or multi-signal TF-IDF approach:
        - LLM mode: Uses GPT to understand semantic similarity with comments
        - TF-IDF mode: Combines title, body, and structural similarity
        
        Args:
            issues: List of Issue objects to analyze
            threshold: Minimum combined similarity score (0-1) to consider duplicates
            
        Returns:
            List of DuplicateGroup objects, each containing similar issues
        """
        if not issues:
            return []
        
        if len(issues) == 1:
            return []
        
        # Fetch comments if enabled and GitHub client available
        if self.fetch_comments and self.github_client:
            logger.info("Fetching comments for %d issues...", len(issues))
            self._fetch_all_comments(issues)
        
        n = len(issues)
        combined_matrix = None
        
        if self.use_llm and self.llm_client:
            # Use LLM-based detection
            logger.info("Using LLM-based duplicate detection...")
            pairs = self._find_duplicates_with_llm(issues, threshold)
        else:
            # Use TF-IDF multi-signal approach
            logger.info("Using TF-IDF multi-signal duplicate detection...")
            title_sim_matrix = self._compute_title_similarity(issues)
            body_sim_matrix = self._compute_body_similarity(issues)
            structural_sim_matrix = self._compute_structural_similarity(issues)
            
            # Combine signals with weights
            # Title is most important (50%), body content (35%), structure (15%)
            combined_matrix = (
                0.5 * title_sim_matrix + 
                0.35 * body_sim_matrix + 
                0.15 * structural_sim_matrix
            )
            
            # Find pairs above threshold and group them
            pairs: List[tuple[int, int, float]] = []
            for i in range(n):
                for j in range(i + 1, n):
                    sim = combined_matrix[i][j]
                    # Additional filter: title similarity must be at least 0.4 to avoid false positives
                    if sim >= threshold and title_sim_matrix[i][j] >= 0.4:
                        pairs.append((i, j, sim))
        
        if not pairs:
            return []
        
        # Group connected components using union-find approach
        groups_dict: Dict[int, Set[int]] = {}
        
        for i, j, sim in pairs:
            # Find which group(s) i and j belong to
            i_group = None
            j_group = None
            
            for group_id, group_members in groups_dict.items():
                if i in group_members:
                    i_group = group_id
                if j in group_members:
                    j_group = group_id
            
            if i_group is None and j_group is None:
                # Create new group
                new_id = len(groups_dict)
                groups_dict[new_id] = {i, j}
            elif i_group is not None and j_group is None:
                # Add j to i's group
                groups_dict[i_group].add(j)
            elif i_group is None and j_group is not None:
                # Add i to j's group
                groups_dict[j_group].add(i)
            elif i_group is not None and j_group is not None and i_group != j_group:
                # Merge two groups
                groups_dict[i_group].update(groups_dict[j_group])
                del groups_dict[j_group]
        
        # Convert to DuplicateGroup objects
        result: List[DuplicateGroup] = []
        for group_indices in groups_dict.values():
            group_issues = [issues[idx] for idx in sorted(group_indices)]
            
            # Calculate max similarity within this group
            max_sim = 0.0
            if combined_matrix is not None:
                indices_list = list(group_indices)
                for i in range(len(indices_list)):
                    for j in range(i + 1, len(indices_list)):
                        sim = combined_matrix[indices_list[i]][indices_list[j]]
                        max_sim = max(max_sim, sim)
            else:
                # For LLM mode, use the similarity from pairs
                max_sim = max((s for i, j, s in pairs if i in group_indices and j in group_indices), default=0.8)
            
            result.append(DuplicateGroup(
                issues=group_issues,
                max_similarity=float(max_sim)
            ))
        
        # Sort by max similarity descending
        result.sort(key=lambda g: -g.max_similarity)
        
        return result
    
    def _compute_title_similarity(self, issues: List[Issue]):
        """Compute pairwise title similarity using sequence matching.
        
        Uses a combination of exact matching and fuzzy string matching
        to handle slight variations in phrasing.
        
        Returns:
            numpy.ndarray: n×n similarity matrix
        """
        import numpy as np
        n = len(issues)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    title_i = self._normalize_text(issues[i].title)
                    title_j = self._normalize_text(issues[j].title)
                    
                    # Use SequenceMatcher for fuzzy string matching
                    ratio = SequenceMatcher(None, title_i, title_j).ratio()
                    
                    # Boost score if key terms overlap
                    words_i = set(title_i.split())
                    words_j = set(title_j.split())
                    if words_i and words_j:
                        word_overlap = len(words_i & words_j) / max(len(words_i), len(words_j))
                        ratio = max(ratio, word_overlap)
                    
                    matrix[i][j] = ratio
        
        return matrix
    
    def _compute_body_similarity(self, issues: List[Issue]):
        """Compute pairwise body similarity using TF-IDF.
        
        Focuses on meaningful content words while filtering out
        common technical noise.
        
        Returns:
            numpy.ndarray: n×n similarity matrix
        """
        import numpy as np
        
        texts = []
        for issue in issues:
            body_text = issue.body if issue.body else ""
            # Clean and normalize body text
            cleaned = self._clean_body_text(body_text)
            texts.append(cleaned)
        
        # Check if we have any meaningful content
        if not any(texts) or all(len(t.strip()) == 0 for t in texts):
            n = len(issues)
            return np.zeros((n, n))
        
        # Vectorize using TF-IDF with stricter parameters
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=500,
            min_df=1,
            max_df=0.8,  # Ignore terms that appear in >80% of docs
            stop_words='english'
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            return cosine_similarity(tfidf_matrix)
        except ValueError:
            # Handle case where vectorization fails
            n = len(issues)
            return np.zeros((n, n))
    
    def _compute_structural_similarity(self, issues: List[Issue]):
        """Compute structural similarity based on issue metadata.
        
        Considers:
        - Label overlap
        - Text length similarity
        - Creation time proximity
        
        Returns:
            numpy.ndarray: n×n similarity matrix
        """
        import numpy as np
        n = len(issues)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                    continue
                
                score = 0.0
                
                # Label overlap (weight: 0.5)
                labels_i = set(issues[i].labels)
                labels_j = set(issues[j].labels)
                if labels_i and labels_j:
                    label_overlap = len(labels_i & labels_j) / len(labels_i | labels_j)
                    score += 0.5 * label_overlap
                elif not labels_i and not labels_j:
                    score += 0.25  # Both have no labels - weak signal
                
                # Length similarity (weight: 0.3)
                len_i = len(issues[i].title) + len(issues[i].body or "")
                len_j = len(issues[j].title) + len(issues[j].body or "")
                if len_i > 0 and len_j > 0:
                    length_ratio = min(len_i, len_j) / max(len_i, len_j)
                    score += 0.3 * length_ratio
                
                # Time proximity (weight: 0.2)
                # Issues created close together are more likely to be duplicates
                try:
                    from datetime import datetime
                    time_i = datetime.fromisoformat(issues[i].created_at.replace('Z', '+00:00'))
                    time_j = datetime.fromisoformat(issues[j].created_at.replace('Z', '+00:00'))
                    time_diff_days = abs((time_i - time_j).days)
                    # Decay over 30 days
                    time_score = max(0, 1 - (time_diff_days / 30))
                    score += 0.2 * time_score
                except:
                    pass  # Ignore time if parsing fails
                
                matrix[i][j] = score
        
        return matrix
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _fetch_all_comments(self, issues: List[Issue]) -> None:
        """Fetch comments for all issues and cache them."""
        for issue in issues:
            if issue.number not in self._comments_cache:
                comments = self._fetch_issue_comments(issue)
                self._comments_cache[issue.number] = comments
    
    def _fetch_issue_comments(self, issue: Issue) -> List[str]:
        """Fetch comments for a single issue from GitHub API."""
        if not self.github_client:
            return []
        
        try:
            # Use the GitHub client's _get method
            url = f"/repos/{issue.repo}/issues/{issue.number}/comments"
            resp = self.github_client._get(url)  # type: ignore[attr-defined]
            resp.raise_for_status()
            comments_data = resp.json()
            
            # Extract comment bodies, limit to first 5 to avoid token overflow
            comments = []
            for comment in comments_data[:5]:
                body = comment.get("body", "")
                if body and len(body.strip()) > 0:
                    # Truncate very long comments
                    if len(body) > 500:
                        body = body[:500] + "..."
                    comments.append(body)
            
            return comments
        except Exception as e:
            logger.warning("Failed to fetch comments for issue #%d: %s", issue.number, e)
            return []
    
    def _find_duplicates_with_llm(
        self,
        issues: List[Issue],
        threshold: float
    ) -> List[tuple[int, int, float]]:
        """Use LLM to find duplicate pairs with semantic understanding.
        
        Uses a two-stage approach for efficiency with large datasets:
        1. Generate embeddings for all issues (cheap, fast)
        2. Use embeddings to find candidate pairs (cosine similarity)
        3. Use LLM to verify only the most promising candidates
        
        This scales to hundreds of issues by avoiding O(n²) LLM calls.
        """
        pairs: List[tuple[int, int, float]] = []
        
        if not self.llm_client:
            return pairs
        
        n = len(issues)
        logger.info("Generating embeddings for %d issues...", n)
        
        # Stage 1: Get embeddings for all issues
        embeddings = self._get_embeddings_batch(issues)
        
        if not embeddings or len(embeddings) != n:
            logger.warning("Failed to get embeddings, falling back to batch comparison")
            # Fallback to old batch method for small datasets
            if n <= 20:
                return self._compute_llm_duplicates_batch(issues, 0, threshold)
            return pairs
        
        # Stage 2: Find candidate pairs using embedding similarity
        import numpy as np
        embeddings_array = np.array(embeddings)
        
        # Compute cosine similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings_array)
        
        # Find candidate pairs with high embedding similarity
        # Use a lower threshold (0.80) to catch more candidates for LLM verification
        # 0.80 is safe because the LLM verification step will filter out false positives
        embedding_threshold = 0.80
        candidates: List[tuple[int, int, float]] = []
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = similarity_matrix[i][j]
                if sim >= embedding_threshold:
                    candidates.append((i, j, float(sim)))
        
        logger.info("Found %d candidate pairs from embeddings (threshold: %.2f)", 
                   len(candidates), embedding_threshold)
        
        # Stage 3: Verify candidates with LLM (only for high-similarity pairs)
        if not candidates:
            return pairs
        
        # Process candidates in batches to avoid token limits
        verified_pairs = self._verify_candidates_with_llm(issues, candidates, threshold)
        pairs.extend(verified_pairs)
        
        logger.info("LLM verified %d duplicate pairs (threshold: %.2f)", len(pairs), threshold)
        return pairs
    
    def _get_embeddings_batch(self, issues: List[Issue]) -> List[List[float]]:
        """Generate embeddings for all issues using sentence-transformers.
        
        Uses the 'all-mpnet-base-v2' model which offers the best quality/speed trade-off.
        - 768 dimensions
        - 512 token limit (vs 256 for MiniLM)
        - Much better semantic understanding
        
        Returns:
            List of embedding vectors, one per issue
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load model (cached after first load)
            # all-mpnet-base-v2: 768 dimensions, ~420MB, best quality
            model = SentenceTransformer('all-mpnet-base-v2')
            
            # Format issues for embedding
            texts = []
            for issue in issues:
                comments = self._comments_cache.get(issue.number, [])
                # Use dedicated formatting for embeddings to optimize for 512 token limit
                text = self._format_issue_for_embedding(issue, comments)
                texts.append(text)
            
            # Generate embeddings (supports batching internally)
            logger.debug("Generating embeddings for %d issues using sentence-transformers...", len(texts))
            embeddings_array = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            
            # Convert numpy array to list of lists
            embeddings: List[List[float]] = embeddings_array.tolist()
            
            logger.debug("Generated %d embeddings", len(embeddings))
            return embeddings
            
        except Exception as e:
            logger.warning("Failed to generate embeddings: %s", e)
            return []
    
    def _verify_candidates_with_llm(
        self,
        issues: List[Issue],
        candidates: List[tuple[int, int, float]],
        threshold: float
    ) -> List[tuple[int, int, float]]:
        """Verify candidate duplicate pairs using LLM.
        
        Args:
            issues: All issues
            candidates: List of (idx1, idx2, embedding_similarity) tuples
            threshold: Minimum confidence threshold
            
        Returns:
            Verified duplicate pairs with LLM confidence scores
        """
        if not self.llm_client or not candidates:
            return []
        
        verified: List[tuple[int, int, float]] = []
        
        # Process candidates in batches
        # Group by pairs for efficient LLM calls
        for idx1, idx2, _emb_sim in candidates:
            issue1 = issues[idx1]
            issue2 = issues[idx2]
            
            # Get LLM verification
            confidence = self._verify_pair_with_llm(issue1, issue2)
            
            if confidence >= threshold:
                verified.append((idx1, idx2, confidence))
                logger.debug(
                    "LLM verified duplicate: #%d and #%d (confidence: %.2f)",
                    issue1.number, issue2.number, confidence
                )
        
        return verified
    
    def _verify_pair_with_llm(self, issue1: Issue, issue2: Issue) -> float:
        """Verify if two issues are duplicates using LLM.
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not self.llm_client:
            return 0.0
        
        comments1 = self._comments_cache.get(issue1.number, [])
        comments2 = self._comments_cache.get(issue2.number, [])
        
        issue1_text = self._format_issue_for_llm(issue1, comments1)
        issue2_text = self._format_issue_for_llm(issue2, comments2)
        
        prompt = f"""Compare these two GitHub issues and determine if they are duplicates.

Issue 1:
{issue1_text}

Issue 2:
{issue2_text}

Consider them duplicates if they:
- Report the same bug/error
- Request the same feature
- Describe the same problem with different wording
- Have similar root causes based on comments

Respond with JSON only:
{{"is_duplicate": boolean, "confidence": float (0.0-1.0), "reason": "brief explanation"}}"""
        
        try:
            client = self.llm_client._get_client()
            model = self.llm_client.deployment
            
            response = client.chat.completions.create(  # type: ignore[attr-defined]
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing GitHub issues."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=200
            )
            
            content = response.choices[0].message.content
            if not content:
                return 0.0
            
            # Parse JSON
            content = content.strip()
            content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
            content = re.sub(r"```\s*$", "", content, flags=re.MULTILINE)
            
            result = json.loads(content.strip())
            
            if result.get("is_duplicate", False):
                return float(result.get("confidence", 0.8))
            
            return 0.0
            
        except Exception as e:
            logger.warning("LLM verification failed for #%d and #%d: %s",
                         issue1.number, issue2.number, e)
            return 0.0
    
    def _compute_llm_duplicates_batch(
        self,
        issues: List[Issue],
        batch_offset: int,
        threshold: float
    ) -> List[tuple[int, int, float]]:
        """Use LLM to identify duplicate pairs in a batch of issues efficiently.
        
        Instead of O(n²) pairwise comparisons, asks the LLM to analyze all issues
        at once and identify which ones are duplicates of each other.
        """
        if not self.llm_client or len(issues) < 2:
            return []
        
        # Format all issues for the prompt
        issues_text = []
        for idx, issue in enumerate(issues):
            comments = self._comments_cache.get(issue.number, [])
            issue_text = self._format_issue_for_llm(issue, comments)
            issues_text.append(f"Issue {idx}:\n{issue_text}")
        
        prompt = f"""You are analyzing GitHub issues to detect duplicates. Below are {len(issues)} issues from a repository.

{chr(10).join(issues_text)}

Analyze ALL issues and identify which ones are duplicates of each other. Consider issues duplicates if they:
- Report the same bug or error
- Request the same feature  
- Describe the same problem even with different wording
- Have similar symptoms and root causes based on comments

Respond with a JSON array of duplicate pairs. Each pair should have:
- "issue1_index": int (index of first issue, 0-based)
- "issue2_index": int (index of second issue, 0-based)
- "confidence": float between 0.0 and 1.0
- "reason": brief explanation

If no duplicates found, return an empty array [].
Respond only with the JSON array, no additional text."""
        
        try:
            client = self.llm_client._get_client()
            model = self.llm_client.deployment
            
            response = client.chat.completions.create(  # type: ignore[attr-defined]
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing GitHub issues to detect duplicates."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=1000
            )
            
            content = response.choices[0].message.content
            if not content:
                return []
            
            content = content.strip()
            
            # Remove markdown code blocks if present
            content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
            content = re.sub(r"```\s*$", "", content, flags=re.MULTILINE)
            content = content.strip()
            
            # Parse JSON response
            pairs_data = json.loads(content)
            
            if not isinstance(pairs_data, list):
                logger.warning("LLM response is not a list, got: %s", type(pairs_data))
                return []
            
            # Convert to tuple format with global indices
            pairs: List[tuple[int, int, float]] = []
            for pair in pairs_data:
                if not isinstance(pair, dict):
                    continue
                
                idx1 = pair.get("issue1_index")
                idx2 = pair.get("issue2_index")
                confidence = pair.get("confidence", 0.8)
                
                if idx1 is None or idx2 is None:
                    continue
                
                if not (0 <= idx1 < len(issues) and 0 <= idx2 < len(issues)):
                    logger.warning("Invalid indices in LLM response: %d, %d", idx1, idx2)
                    continue
                
                if idx1 == idx2:
                    continue
                
                # Ensure idx1 < idx2 for consistency
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                
                if confidence >= threshold:
                    global_idx1 = batch_offset + idx1
                    global_idx2 = batch_offset + idx2
                    pairs.append((global_idx1, global_idx2, float(confidence)))
                    
                    reason = pair.get("reason", "")
                    logger.debug(
                        "LLM found duplicate: #%d and #%d (confidence: %.2f, reason: %s)",
                        issues[idx1].number, issues[idx2].number, confidence, reason
                    )
            
            return pairs
            
        except Exception as e:
            logger.warning("LLM batch duplicate detection failed: %s", e)
            return []
    
    @staticmethod
    def _format_issue_for_embedding(issue: Issue, comments: List[str]) -> str:
        """Format an issue for embedding generation (optimized for 512 token limit)."""
        # Prioritize Title and Body start
        parts = [
            f"Title: {issue.title}",
            f"Labels: {', '.join(issue.labels) if issue.labels else 'none'}",
        ]
        
        if issue.body:
            # Truncate body to ~1000 chars (approx 250 tokens) to leave room for comments
            # The beginning of the body usually contains the most relevant info (error msg, etc)
            body = issue.body[:1000]
            parts.append(f"Description: {body}")
        
        if comments:
            # Include top 2 comments, truncated
            parts.append("Comments:")
            for comment in comments[:2]:
                # Truncate each comment to 200 chars
                parts.append(f"- {comment[:200]}")
        
        return "\n".join(parts)

    @staticmethod
    def _format_issue_for_llm(issue: Issue, comments: List[str]) -> str:
        """Format an issue with comments for LLM analysis."""
        parts = [
            f"Title: {issue.title}",
            f"Number: #{issue.number}",
            f"Labels: {', '.join(issue.labels) if issue.labels else 'none'}",
        ]
        
        if issue.body:
            # Truncate long bodies
            body = issue.body[:1000] if len(issue.body) > 1000 else issue.body
            parts.append(f"Description:\n{body}")
        
        if comments:
            parts.append(f"\nComments ({len(comments)}):")
            for i, comment in enumerate(comments[:3], 1):  # Max 3 comments
                parts.append(f"{i}. {comment}")
        
        return "\n".join(parts)
    
    @staticmethod
    def _clean_body_text(text: str) -> str:
        """Clean body text by removing noise."""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Remove code blocks (markdown style)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove file paths
        text = re.sub(r'[/\\][\w/\\.-]+', '', text)
        # Remove version numbers
        text = re.sub(r'\bv?\d+\.\d+(\.\d+)?(-\w+)?\b', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


__all__ = ["DuplicateDetector", "DuplicateGroup"]
