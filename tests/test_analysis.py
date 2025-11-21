import json
import pytest
from unittest.mock import MagicMock
from issues_agent.analysis import analyze_issues
from issues_agent.llm_client import AzureOpenAIClient

@pytest.fixture
def mock_openai_client():
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_response_data = json.dumps({
        "priorities": [
            {"issue_id": 1, "priority": "P0", "reasoning": "Critical bug"},
            {"issue_id": 2, "priority": "P2", "reasoning": "Minor cosmetic issue"}
        ],
        "duplicates": [
            [1, 3]
        ],
        "top_urgent_issues": [1]
    })
    mock_completion.choices[0].message.content = mock_response_data
    mock_client.chat.completions.create.return_value = mock_completion
    # Mock responses API with output_text
    mock_responses_obj = MagicMock()
    mock_responses_obj.output_text = mock_response_data
    mock_client.responses.create.return_value = mock_responses_obj
    return mock_client

@pytest.fixture
def mock_azure_client(mock_openai_client):
    client_wrapper = MagicMock(spec=AzureOpenAIClient)
    client_wrapper.get_client.return_value = mock_openai_client
    client_wrapper.deployment = "gpt-4"
    return client_wrapper

def test_analyze_issues(mock_azure_client, mock_openai_client):
    issues = [
        {"number": 1, "title": "Bug 1", "body": "Critical failure", "reactions": {}, "comments": 5, "created_at": "2023-01-01"},
        {"number": 2, "title": "Feature 2", "body": "Nice to have", "reactions": {}, "comments": 0, "created_at": "2023-01-02"},
        {"number": 3, "title": "Bug 1 again", "body": "Same as bug 1", "reactions": {}, "comments": 1, "created_at": "2023-01-03"},
    ]
    
    result = analyze_issues(mock_azure_client, issues, model="gpt-4")
    
    assert "priorities" in result
    assert len(result["priorities"]) == 2
    assert result["priorities"][0]["issue_id"] == 1
    assert result["priorities"][0]["priority"] == "P0"
    
    assert "duplicates" in result
    assert len(result["duplicates"]) == 1
    assert result["duplicates"][0] == [1, 3]

    # Verify the request was constructed correctly using Responses API
    mock_openai_client.responses.create.assert_called_once()
    call_args = mock_openai_client.responses.create.call_args
    assert call_args.kwargs["model"] == "gpt-4"
    assert call_args.kwargs["reasoning"] == {"effort": "high"}
    input_messages = call_args.kwargs["input"]
    assert isinstance(input_messages, list)
    assert "Bug 1" in input_messages[0]["content"]
    assert "Feature 2" in input_messages[0]["content"]

def test_analyze_issues_fallback(mock_azure_client, mock_openai_client):
    # Simulate Responses API missing (AttributeError)
    del mock_openai_client.responses
    
    issues = [{"number": 1, "title": "Test", "body": "Test", "reactions": {}, "created_at": "2023-01-01"}]
    result = analyze_issues(mock_azure_client, issues, model="gpt-4")
    
    # Should fallback to chat completions
    mock_openai_client.chat.completions.create.assert_called_once()
    assert "priorities" in result

def test_analyze_issues_json_error(mock_azure_client, mock_openai_client):
    # Mock invalid JSON response
    mock_openai_client.responses.create.return_value.output_text = "Invalid JSON"
    
    issues = [{"number": 1, "title": "Test", "body": "Test", "reactions": {}, "created_at": "2023-01-01"}]
    result = analyze_issues(mock_azure_client, issues, model="gpt-4")
    
    # Expect empty result on error
    assert result == {"priorities": [], "duplicates": []}

def test_analyze_issues_api_error(mock_azure_client, mock_openai_client):
    # Mock API error
    mock_openai_client.responses.create.side_effect = Exception("API Error")
    
    issues = [{"number": 1, "title": "Test", "body": "Test", "reactions": {}, "created_at": "2023-01-01"}]
    
    with pytest.raises(Exception, match="API Error"):
        analyze_issues(mock_azure_client, issues, model="gpt-4")
