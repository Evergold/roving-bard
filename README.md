# Roving Bard: Adaptive, location-aware audio in LOTRO

Roving Bard is an autonomous, screen-aware agent built with the Google Agent Development Kit (ADK). It monitors a video game screen in real-time, extracts in-game locations and coordinates, and dynamically transitions background music tracks (using crossfade effects) to match the player's current location or coordinate hotspot.

The agent uses local OCR (Tesseract) for latency-sensitive local processing and automatically falls back to Gemini Vision for complex screen layouts.

---

## 📂 Project Structure

```text
capstone/
├── .agents-cli-spec.md  # Original agent project specification
├── README.md            # Root GitHub repository documentation (this file)
└── roving-bard/         # Core agent project folder
    ├── app/             # Agent codebase (agent, tools, player logic, FastAPI backend)
    ├── tests/           # Unit, integration, and end-to-end server tests
    ├── pyproject.toml   # Python project dependencies and configuration
    ├── mapping.yaml     # Coordinate-to-music mapping configuration
    ├── run_player.py    # Main screen monitoring loop runner
    └── simulated_game.py# Tkinter GUI simulating a video game screen for local testing
```

---

## 🚀 Key Features

*   **Real-time Screen Monitoring**: Captures the game window and crops to the mini-map/coordinate HUD to ensure user privacy.
*   **Hybrid OCR/Vision Pipeline**:
    *   *Primary (Local OCR)*: Preprocesses and parses coordinate text locally via Tesseract for speed.
    *   *Fallback (Gemini Vision)*: Leverages Gemini multimodal models if local OCR fails to parse the coordinates.
*   **Audio Crossfade Engine**: Smoothly transitions tracks (`fade-in` and `fade-out`) using `pygame.mixer` based on custom coordinate zones or location keywords.
*   **Secure Authentication**: Fully decoupled from hard-coded keys—uses standard environment variables for backend API access.

---

## 🛠️ Prerequisites

Before you begin, ensure you have installed:
1.  **Python 3.11 - 3.13**
2.  **uv** (Python packaging & dependency manager): [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
3.  **Tesseract OCR** (Required for local OCR parsing):
    *   *Ubuntu/Debian*: `sudo apt install tesseract-ocr`
    *   *macOS*: `brew install tesseract`
    *   *Windows*: Download binary from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).

---

## 📦 Setup & Installation

1.  Clone the repository:
    ```bash
    git clone <your-repo-url>
    cd capstone/roving-bard
    ```
2.  Install the **Agents CLI** globally:
    ```bash
    uv tool install google-agents-cli
    ```
3.  Install project dependencies:
    ```bash
    agents-cli install
    ```

---

## 🎮 How to Run (Local Testing)

This project contains a built-in game simulator and a background music runner for testing:

### 1. Start the Game Simulator
In one terminal, run:
```bash
cd roving-bard
uv run python simulated_game.py
```
This launches a Tkinter window showing coordinates and region buttons (Town, Forest, Boss Arena, Cave) to simulate moving around in-game.

### 2. Start the Music Player Runner
In another terminal, export your API key and launch the screen monitor:
```bash
export GEMINI_API_KEY="your-gemini-api-key"  # Or GOOGLE_API_KEY
cd roving-bard
uv run python run_player.py
```
*   *Note: This automatically creates mock audio files (`town.wav`, `forest.wav`, `boss.wav`, `cave.wav`) inside the `music/` directory if they don't already exist.*
*   Now, click on different regions in the simulator GUI. You will see the runner capture the screen, parse the coordinates, and trigger smooth crossfades between the tracks.

### 3. Launch the Interactive Playground
To interact with the agent manually and inspect its status or tools, run:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
cd roving-bard
agents-cli playground
```
This opens a local web UI playground at `http://localhost:8000`.

---

## 🧪 Running Tests

To run the full test suite (unit and integration tests):
```bash
export GEMINI_API_KEY="your-gemini-api-key"  # Required for auth validation tests
cd roving-bard
uv run pytest tests/unit tests/integration
```

---

## ⚙️ Configuration (`mapping.yaml`)

Specify transitions, bounding boxes, and locations in `roving-bard/mapping.yaml`:
```yaml
minimap_bounds:
  x: 0.8         # Top-left corner coordinates (0.0 to 1.0)
  y: 0.05
  width: 0.15    # Dimensions of the cropped mini-map widget
  height: 0.15

transitions:
  fade_out_ms: 1500
  fade_in_ms: 1500

playlist_directory: "audio"
model_name: "gemini/gemini-1.5-flash"
polling_interval: 2.0

mappings:
  - location_name: "Town"
    track_file: "town.wav"
  - location_name: "Forest"
    track_file: "forest.wav"
  - ns_min: 10.0
    ns_max: 20.0
    ew_min: -80.0
    ew_max: -60.0
    track_file: "cave.wav"
```

---

## 🔒 Security & Environment Variables

The agent and server look for the following environment variables.

*   `GEMINI_API_KEY` or `GOOGLE_API_KEY`: Required for visual fallback queries to Gemini models and playground interaction.
*   `AGENT_API_KEY`: Custom API key that can be set to override frontend client authorization.
