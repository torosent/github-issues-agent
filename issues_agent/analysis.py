import json
import logging
from typing import List, Dict, Any
from issues_agent.llm_client import AzureOpenAIClient

logger = logging.getLogger(__name__)

def analyze_issues(client: AzureOpenAIClient, issues: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    """
    Analyzes a list of issues to prioritize them and find duplicates using an LLM.

    Args:
        client: The AzureOpenAIClient wrapper.
        issues: A list of issue dictionaries (must contain number, title, body, reactions, created_at).
        model: The model name to use (e.g. "gpt-4", "gpt-5.1").

    Returns:
        A dictionary containing "priorities" and "duplicates".
    """
    if not issues:
        return {"priorities": [], "duplicates": []}

    # Format issues for the prompt
    issues_text = ""
    for issue in issues:
        issues_text += f"ID: {issue.get('number')}\n"
        issues_text += f"Title: {issue.get('title')}\n"
        issues_text += f"Body: {issue.get('body')}\n"
        issues_text += f"Reactions: {issue.get('reactions')}\n"
        issues_text += f"Comments: {issue.get('comments')}\n"
        issues_text += f"Created At: {issue.get('created_at')}\n"
        issues_text += "---\n"

    system_prompt = (
        "You are an expert software project manager and QA lead. "
        "Your task is to analyze a list of GitHub issues to prioritize them and identify duplicates.\n\n"
        "For each issue, assign a priority (P0, P1, P2) and provide a brief reasoning based on the impact, urgency, and severity implied by the title, body, and comments.\n"
        " - P0: Critical/Urgent (System down, data loss, blocking bugs)\n"
        " - P1: High Priority (Major functionality broken, important feature requests)\n"
        " - P2: Normal Priority (Minor bugs, nice-to-have features, cosmetic issues)\n\n"
        "Also, identify any issues that appear to be duplicates of each other. Group duplicate issue IDs together.\n\n"
        "Finally, select the top 20 most urgent issues that require immediate attention. List their IDs in order of urgency.\n\n"
        "You must output your analysis in valid JSON format with the following structure:\n"
        "{\n"
        '  "priorities": [\n'
        '    {"issue_id": <number>, "priority": "<P0/P1/P2>", "reasoning": "<text>"},\n'
        '    ...\n'
        '  ],\n'
        '  "duplicates": [\n'
        '    [<issue_id_1>, <issue_id_2>], ...\n'
        '  ],\n'
        '  "top_urgent_issues": [<issue_id_1>, <issue_id_2>, ...]\n'
        "}\n"
        "Ensure the JSON is valid and strictly follows this schema."
    )

    user_prompt = f"Here are the issues to analyze:\n\n{issues_text}"

    openai_client: Any = client.get_client()
    
    try:
        # Use the Responses API with 'input' parameter
        # The Responses API accepts input as a string or list of message objects
        response = openai_client.responses.create(
            model=model,
            input=[
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
            ],
            reasoning={"effort": "high"},
        )
        
        # Responses API uses output_text instead of choices
        content = response.output_text if hasattr(response, 'output_text') else response.choices[0].message.content
        if not content:
            logger.warning("LLM returned empty content.")
            return {"priorities": [], "duplicates": []}

        result = json.loads(content)
        return result

    except AttributeError:
        # Fallback to chat completions if responses API is not available (e.g. older SDK)
        logger.info("Responses API not found, falling back to Chat Completions.")
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1, 
            reasoning={"effort": "high"},
        )
        content = response.choices[0].message.content
        if not content:
            return {"priorities": [], "duplicates": []}
        return json.loads(content)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return {"priorities": [], "duplicates": []}
