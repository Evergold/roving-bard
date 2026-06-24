# music-player-agent

Simple ReAct agent
Agent generated with `agents-cli` version `0.5.0`

## Project Structure

```
music-player-agent/
├── app/         # Core agent code
│   ├── agent.py               # Main agent logic
│   └── app_utils/             # App utilities and helpers
├── tests/                     # Unit, integration, and load tests
├── GEMINI.md                  # AI-assisted development guide
└── pyproject.toml             # Project dependencies
```

> 💡 **Tip:** Use [Gemini CLI](https://github.com/google-gemini/gemini-cli) for AI-assisted development - project context is pre-configured in `GEMINI.md`.

## Requirements

Before you begin, ensure you have:
- **uv**: Python package manager (used for all dependency management in this project) - [Install](https://docs.astral.sh/uv/getting-started/installation/) ([add packages](https://docs.astral.sh/uv/concepts/dependencies/) with `uv add <package>`)
- **agents-cli**: Agents CLI - Install with `uv tool install google-agents-cli`
- **Google Cloud SDK**: For GCP services - [Install](https://cloud.google.com/sdk/docs/install)


## Quick Start

Install required packages:

```bash
agents-cli install
```

This project includes a simulated game client and a music player runner for local testing.

### 1. Launch the Simulated Game Client
Run the simulated game client in a separate terminal:
```bash
uv run python simulated_game.py
```
This opens a Tkinter GUI representing the game screen with a mini-map widget showing location/coordinate labels in the top-right corner. You can click on region buttons (Town, Forest, Boss Arena, Cave) to update the current in-game coordinates.

### 2. Start the Music Player Runner
In another terminal, start the automatic screen monitoring loop:
```bash
uv run python run_player.py
```
The runner will:
- Auto-generate test audio files (`town.wav`, `forest.wav`, `boss.wav`, `cave.wav`) in the `music/` directory.
- Capture the screen, crop it to the mini-map area, and run local OCR (Tesseract) to parse location and coordinates.
- Smoothly transition the background music tracks based on the active region using `pygame.mixer`.
- Automatically fallback to Gemini Vision via LiteLLM if local OCR is inconclusive.

### 3. Interactive ADK Playground
To talk directly with the agent (which has tools to manually play/stop music, set volume, check screen, etc.), run:
```bash
agents-cli playground
```
This launches a web browser playground where you can interact with the agent.


## Commands

| Command              | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| `agents-cli install` | Install dependencies using uv                                                         |
| `agents-cli playground` | Launch local development environment                                                  |
| `agents-cli lint`    | Run code quality checks                                                               |
| `agents-cli eval`    | Evaluate agent behavior (generate, grade, analyze, and more — see `agents-cli eval --help`) |
| `uv run pytest tests/unit tests/integration` | Run unit and integration tests                                                        |

## 🛠️ Project Management

| Command | What It Does |
|---------|--------------|
| `agents-cli scaffold enhance` | Add CI/CD pipelines and Terraform infrastructure |
| `agents-cli infra cicd` | One-command setup of entire CI/CD pipeline + infrastructure |
| `agents-cli scaffold upgrade` | Auto-upgrade to latest version while preserving customizations |

---

## Development

Edit your agent logic in `app/agent.py` and test with `agents-cli playground` - it auto-reloads on save.

## Deployment

```bash
gcloud config set project <your-project-id>
agents-cli deploy
```

To add CI/CD and Terraform, run `agents-cli scaffold enhance`.
To set up your production infrastructure, run `agents-cli infra cicd`.

## Observability

Built-in telemetry exports to Cloud Trace, BigQuery, and Cloud Logging.
