# GitHub Issues AI Agent

A lightweight, AI-powered command line tool that fetches GitHub issues, prioritizes them using Azure OpenAI (GPT-5/GPT-5.1), and identifies duplicates.

## Features
- **Multi-repo support**: Fetch issues from one or more repositories.
- **Smart Prioritization**: Uses LLM reasoning to assign P0 (Critical), P1 (High), or P2 (Normal) priority.
- **Duplicate Detection**: Identifies potential duplicate issues in the same analysis pass.
- **Concise Reporting**: Generates a clean Markdown report with priorities and reasoning.

## Installation

Requires Python >= 3.11.

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/github-issues-agent.git
   cd github-issues-agent
   ```

2. Create & activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # On Windows: .venv\Scripts\activate
   ```

3. Install the package:
   ```bash
   pip install -e .
   ```

## Configuration

Create a `.env` file with your credentials:

```env
GITHUB_TOKEN=your_github_pat
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_DEPLOYMENT=gpt-5.1
AZURE_OPENAI_API_VERSION=2025-04-01-preview
```

## Usage

Run the agent via the command line:

```bash
issues-agent --repos owner/repo --output report.md
```

Alternatively, you can still use:
```bash
python -m issues_agent --repos owner/repo --output report.md
```

### Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--repos` | Comma-separated list of repositories (owner/repo) | **Required** |
| `--output` | Output markdown file path | `report.md` |
| `--limit` | Max issues to fetch per repo | `100` |
| `--since` | Fetch issues updated since (e.g. `2y`, `6m`, `30d`) | `2y` |
| `--model` | LLM model name to use | `gpt-5.1` |
| `--dry-run` | Print report to stdout instead of writing to file | `False` |

### Examples

**Analyze a single repo:**
```bash
issues-agent --repos microsoft/vscode --limit 50
```

**Analyze multiple repos with a specific model:**
```bash
issues-agent --repos owner/repo1,owner/repo2 --model gpt-4o --output triage.md
```

**Fetch only recent issues:**
```bash
issues-agent --repos owner/repo --since 30d
```

## Project Structure

```
github-issues-agent/
├── issues_agent/
│   ├── __init__.py
│   ├── __main__.py          # Entry point
│   ├── cli.py               # CLI entry point
│   ├── config.py            # Config loading
│   ├── github_client.py     # GitHub API client
│   ├── llm_client.py        # Azure OpenAI client
│   ├── analysis.py          # LLM analysis logic
│   ├── report.py            # Report generator
│   └── logging.py           # Logging setup
├── tests/                   # Tests
├── .env.example             # Env template
├── README.md
└── requirements.txt
```

## License

MIT



