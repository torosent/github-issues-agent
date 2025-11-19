# GitHub Issues AI Agent

An AI-assisted command line tool that fetches GitHub issues across one or more repositories, classifies them with Azure OpenAI, assigns composite scores and P0/P1/P2 priorities, and produces a concise Markdown report for triage and planning. It emphasizes test-driven phases, deterministic scoring, and customizable categories.

## Features
- **Multi-repo issue fetching** (excludes PRs, supports pagination)
- **Date filtering** - fetch only recent issues (e.g., last 2 years, 6 months, 30 days)
- **Azure OpenAI GPT-5 classification** with JSON output parsing and automatic repair
- **Custom category mapping** via YAML/JSON file
- **Composite scoring** with priority bands (P0 / P1 / P2)
- **Duplicate issue detection** using local embeddings and LLM verification
- **Markdown report generation** with summary metrics, tables, and detailed rationales
- **Dry-run mode** to preview reports without writing files
- **Configurable reasoning effort & text verbosity** for Azure OpenAI Responses API

## Installation
Requires Python >= 3.11.

Clone the repository:
```bash
git clone https://github.com/your-org/github-issues-agent.git
cd github-issues-agent
```

Create & activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# On Windows: .venv\\Scripts\\activate
```

Install dependencies (choose one):
```bash
pip install -r requirements.txt
# OR
pip install .
```

## Configuration
Set the following environment variables (see `.env.example`):
- `GITHUB_TOKEN`: Personal access token with `repo` scope for reading issues.
- `AZURE_OPENAI_ENDPOINT`: Base endpoint for your Azure OpenAI resource.
- `AZURE_OPENAI_API_KEY`: Key for Azure OpenAI calls.
- `AZURE_OPENAI_DEPLOYMENT`: Deployment name (e.g. `gpt-5`).
- `AZURE_OPENAI_API_VERSION`: API version (e.g. `2024-02-15-preview`).

You can export them manually or use a `.env` file with `python-dotenv` auto-loading.

## Usage

### Basic Commands

**Single repository (default: last 2 years):**
```bash
python -m issues_agent --repos owner/repo --output report.md
```

**Multiple repositories:**
```bash
python -m issues_agent --repos owner/repo1,owner/repo2 --output triage.md
```

**Filter by date (issues updated in last 6 months):**
```bash
python -m issues_agent --repos owner/repo --since 6m --output report.md
```

**Filter by date (last 30 days):**
```bash
python -m issues_agent --repos owner/repo --since 30d --output report.md
```

**Limit number of issues:**
```bash
python -m issues_agent --repos owner/repo --limit 50 --output report.md
```

**Custom categories file:**
```bash
python -m issues_agent --repos owner/repo --categories examples/categories.yaml --output report.md
```

**Check for duplicates (TF-IDF mode):**
```bash
python -m issues_agent --repos owner/repo --check-duplicates --output report.md
```

**Check for duplicates (LLM + Embeddings mode):**
```bash
python -m issues_agent --repos owner/repo --check-duplicates --use-llm-for-duplicates --output report.md
```

**Check for duplicates ONLY (skip classification):**
```bash
python -m issues_agent --repos owner/repo --check-duplicates --skip-classifier --output report.md
```

**Dry run (preview without writing file):**
```bash
python -m issues_agent --repos owner/repo --output report.md --dry-run
```

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--repos` | Comma-separated list of repositories (owner/name) | **Required** |
| `--output` | Output markdown file path | `report.md` |
| `--limit` | Maximum number of issues to fetch across all repos | `300` |
| `--since` | Fetch issues updated since (e.g., '2y', '6m', '30d') | `2y` |
| `--categories-file` | Optional path to YAML/JSON categories file | Default categories |
| `--batch-size` | Classifier batch size for Azure OpenAI | `10` |
| `--check-duplicates` | Enable duplicate issue detection | `false` |
| `--use-llm-for-duplicates` | Use LLM + Embeddings for duplicates (requires `--check-duplicates`) | `false` |
| `--duplicate-threshold` | Similarity threshold for duplicates (0.0-1.0) | `0.8` |
| `--skip-classifier` | Skip classification and scoring (useful for duplicate checks) | `false` |
| `--dry-run` | Preview report without writing file | `false` |
| `--reasoning-effort` | Reasoning effort hint for model (`low`, `medium`, `high`) | `medium` |
| `--text-verbosity` | Verbosity of textual output (`low`, `medium`, `high`) | `low` |

Get help:
```bash
python -m issues_agent --help
```

### Advanced Classification Options

This tool uses the Azure OpenAI **Responses API** (with a legacy chat completions fallback) to classify issues in batches. Two optional knobs let you tune model behavior:

- `--reasoning-effort` (default: `medium`): Indicates how much internal reasoning the model should apply when producing structured JSON classifications. Lower values may reduce latency/cost; higher values can improve categorization quality consistency.
- `--text-verbosity` (default: `low`): Controls how verbose any generated textual rationale is. Keeping this at `low` biases toward concise rationales, which helps maintain prompt and report compactness.

Example using both flags:
```bash
python -m issues_agent --repos owner/repo \
  --reasoning-effort medium \
  --text-verbosity high \
  --since 6m --output triage.md
```

If the deployed SDK or a test double does not support these parameters, the agent automatically retries without them to preserve compatibility.


## Sample Report

Here's an example of the generated Markdown report:

```markdown
# GitHub Issues Prioritization Report
Generated: 2025-11-07T22:13:10.612546Z
Repositories: microsoft/durabletask-dotnet
- Total Issues: 10
- Repos Count: 1
- Categories Count: 3
- Top Priority (P0) Count: 0
- Average Score: 0.374

Category | Count
--- | ---
bug | 6
documentation | 1
feature | 3

## All Issues (Sorted by Priority Score)

Number | Title | Category | Priority | Score | Labels | URL
--- | --- | --- | --- | --- | --- | ---
465 | Activity is randomly restarted after successful completion | bug | P1 | 0.531 | needs: author response | https://github.com/microsoft/durabletask-dotnet/issues/465
484 | Breaking change of TaskFailureDetails causes incompatibility | bug | P1 | 0.494 | P1 | https://github.com/microsoft/durabletask-dotnet/issues/484
453 | Analyzer Rule DURABLE2001 False Positives | bug | P2 | 0.478 | P2 | https://github.com/microsoft/durabletask-dotnet/issues/453
454 | Durable Function never completed | bug | P1 | 0.412 | Needs: Triage :mag: | https://github.com/microsoft/durabletask-dotnet/issues/454
480 | Question: Would a `TimerCanceledEvent` make sense? | feature | P2 | 0.358 |  | https://github.com/microsoft/durabletask-dotnet/issues/480

## Category: bug
- #465 [Activity is randomly restarted after successful completion](https://github.com/microsoft/durabletask-dotnet/issues/465) (P1, score 0.531) - Activity re-executes after successful completion; indicates potential duplicate execution issue.
- #484 [Breaking change of TaskFailureDetails causes incompatibility](https://github.com/microsoft/durabletask-dotnet/issues/484) (P1, score 0.494) - Breaking change in TaskFailureDetails causes incompatibility with current Functions packages.
- #453 [Analyzer Rule DURABLE2001 False Positives](https://github.com/microsoft/durabletask-dotnet/issues/453) (P2, score 0.478) - Analyzer rule DURABLE2001 reports false positives; incorrect diagnostics.

## Category: feature
- #480 [Question: Would a `TimerCanceledEvent` make sense?](https://github.com/microsoft/durabletask-dotnet/issues/480) (P2, score 0.358) - Proposes a TimerCanceledEvent to align worker/runtime semantics; design enhancement.
```

The report includes:
- **Summary metrics** - total issues, categories, P0 count, average score
- **Category distribution** - count of issues per category
- **All issues table** - sorted by priority score with clickable links
- **Category sections** - detailed rationales for each issue grouped by category

## Scoring Algorithm

Issues are scored using a composite algorithm that combines multiple factors:

**Score Formula:**
```
score = 0.4 × severity + 0.2 × recency + 0.2 × comments + 0.2 × reactions
```

**Components:**
- **Severity**: Based on AI-assigned priority level (P0=3, P1=2, P2=1), normalized
- **Recency**: Decay function based on how recently the issue was updated
- **Comments**: Logarithmic scale of comment count
- **Reactions**: Logarithmic scale of positive reactions (+1, heart, hooray, rocket, eyes)

All components are normalized to [0, 1] range before weighting.

## Duplicate Detection

The agent includes a sophisticated duplicate detection system with two modes:

1.  **TF-IDF Mode (Default)**:
    *   Fast and offline-capable.
    *   Combines Title Similarity (Sequence Matching), Body Similarity (TF-IDF), and Structural Similarity (labels, creation time).

2.  **LLM + Embeddings Mode (`--use-llm-for-duplicates`)**:
    *   **Stage 1: Embeddings**: Generates semantic embeddings for ALL issues using `sentence-transformers` (model: `all-mpnet-base-v2`). This runs locally and is free.
    *   **Stage 2: Candidate Filtering**: Uses cosine similarity to find potential duplicate pairs across the entire dataset (solving the "batch boundary" problem).
    *   **Stage 3: LLM Verification**: Sends only high-confidence candidates to Azure OpenAI for final verification, checking for semantic equivalence (e.g., "same bug, different wording").

## Custom Categories

Create a YAML or JSON file to define custom categories:

**categories.yaml:**
```yaml
categories:
  - bug
  - feature
  - documentation
  - performance
  - security
  - refactor
  - test
  - infrastructure
```

**categories.json:**
```json
{
  "categories": [
    "bug",
    "feature",
    "documentation",
    "performance",
    "security"
  ]
}
```

Default categories if not specified: `bug`, `feature`, `documentation`, `refactor`, `security`, `performance`, `test`, `other`

## Development & Testing

Run the test suite:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=issues_agent --cov-report=html
```

All 40 tests should pass:
- Config loading and validation
- GitHub API client with pagination
- Data models and category loading
- Azure OpenAI classifier with JSON repair
- Scoring algorithm with normalization
- Markdown report generation
- CLI integration and error handling
- End-to-end integration tests

## Project Structure

```
github-issues-agent/
├── issues_agent/
│   ├── __init__.py
│   ├── __main__.py          # Entry point
│   ├── cli.py               # Command-line interface
│   ├── config.py            # Configuration loading
│   ├── github_client.py     # GitHub API client
│   ├── models.py            # Data models
│   ├── classifier.py        # Azure OpenAI classifier
│   ├── duplicates.py        # Duplicate detection logic
│   ├── llm_client.py        # Shared Azure OpenAI client
│   ├── scoring.py           # Scoring algorithm
│   ├── report.py            # Markdown report generator
│   └── logging.py           # Logging setup
├── tests/                   # 40+ comprehensive tests
├── examples/
│   └── categories.yaml      # Sample custom categories
├── .env.example             # Environment variables template
├── README.md
├── pyproject.toml           # Package configuration
└── requirements.txt         # Dependencies
```

## License

MIT



